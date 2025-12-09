import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (global, lightweight)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> returns x * w with w in [0,1], shape [B,C,1,1]
        w = self.fc(x)
        return x * w, w  # return both feature and the gating weights


class Decoder(nn.Module):
    """
    Decoder that fuses encoder pixel features L_p and GNN pixel-mapped features L_sp
    via a learned pixel-wise gate:
        gate = sigmoid( proj_enc(L_p) + proj_gnn(L_sp) )
    Fusion:
        fused = gate * proj_gnn(L_sp) + (1 - gate) * proj_enc(L_p)
    Optional channel attention (SE) is applied to L_p before projection.

    Args (key):
      fp_channels: channels of encoder feature L_p (e.g. 96)
      fsp_channels: channels of gnn pixel feature L_sp (e.g. 128)
      hidden_channels: decoder mid channels (fused channels, e.g. 256)
      num_classes: segmentation classes
      use_se: whether to apply SE on L_p
      se_reduction: reduction ratio for SE
      dropout: dropout prob inside refine convs
      use_gn: use GroupNorm in refine convs when True (helpful for small batch)
    """
    def __init__(self,
                 fp_channels: int = 96,
                 fsp_channels: int = 128,
                 hidden_channels: int = 256,
                 num_classes: int = 19,
                #  use_se: bool = True,
                #  se_reduction: int = 16,
                 dropout: float = 0.1,
                 use_gn: bool = False):
        super().__init__()
        assert hidden_channels % 2 == 0, "mid_ch should be divisible by 2"
        self.fp_channels = fp_channels
        self.fsp_channels = fsp_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        # self.use_se = use_se

        # # optional channel attention on encoder features
        # if use_se:
        #     self.se = SEBlock(fp_channels, reduction=se_reduction)
        # else:
        #     self.se = None

        # 1x1 projectors to bring both branches to mid_ch/2 each
        self.enc_proj = nn.Conv2d(fp_channels, hidden_channels // 2, kernel_size=1, bias=False)
        self.gnn_proj = nn.Conv2d(fsp_channels, hidden_channels // 2, kernel_size=1, bias=False)

        # small convs for gate computation (just linear-project, no bias required)
        self.gate_enc = nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=1, bias=False)
        self.gate_gnn = nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=1, bias=False)
        self.gate_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # # channel attention on gnn-projected feature (optional and light)
        # # we reuse a small SE-like path but with convolutional implementation
        # self.gnn_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(hidden_channels // 2, max(1, (hidden_channels // 2) // 8), 1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(max(1, (hidden_channels // 2) // 8), hidden_channels // 2, 1, bias=True),
        #     nn.Sigmoid()
        # )

        # refinement convs after fusion
        if use_gn:
            norm_layer = lambda c: nn.GroupNorm(32 if c >= 32 else 8, c)
        else:
            norm_layer = lambda c: nn.BatchNorm2d(c)

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, L_p: torch.Tensor, L_sp: torch.Tensor):
        """
        Args:
          L_p: [B, C_cnn, H, W]   encoder pixel features
          L_sp: [B, C_gnn, H, W]  gnn pixel-mapped features
        Returns:
          logits: [B, num_classes, H, W]
          aux: dict with 'gate' and optionally 'se_weights'
        """
        # 1) channel attention on encoder features
        # se_weights = None
        # if self.se is not None:
        #     L_p, se_weights = self.se(L_p)  # L_p: [B,C,H,W], se_weights: [B,C,1,1]
        # else:
        #     L_p = L_p

        # 2) project both to mid/2 channels
        enc = self.enc_proj(L_p)   # [B, mid/2, H, W]
        gnn = self.gnn_proj(L_sp)     # [B, mid/2, H, W]

        # 3) channel attention on gnn-proj (light)
        # produce scale in [0,1] per channel (global)
        # g_scale = self.gnn_se(gnn)    # [B, mid/2, 1, 1]
        # gnn = gnn * g_scale

        # 4) compute spatial gate = sigmoid( proj(enc) + proj(gnn) )
        ge = self.gate_enc(enc)  # [B, mid/4, H, W]
        gg = self.gate_gnn(gnn)  # [B, mid/4, H, W]
        gate = self.gate_out(ge + gg)  # [B,1,H,W] in (0,1)

        # 5) fusion: gate weights gnn, (1-gate) weights encoder
        g_weighted = gnn * gate
        enc_weighted = enc * (1.0 - gate)
        fused = torch.cat([enc_weighted, g_weighted], dim=1)  # [B, mid, H, W]

        # 6) refine + classify
        x = self.refine(fused)
        logits = self.classifier(x)

        return logits
