from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from util.util import generate_graph_nodes_hard
from .sfcn_loss import compute_semantic_pos_loss


def one_hot_from_labels(label_idx: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    Convert label indices [B, H, W] to one-hot [B, C, H, W], with ignore_index positions zeroed.
    """
    B, H, W = label_idx.shape
    clamped = label_idx.clone().clamp(0, num_classes - 1)
    one_hot = F.one_hot(clamped, num_classes=num_classes)   # [B, H, W, C]
    one_hot = one_hot.permute(0, 3, 1, 2).float()           # [B, C, H, W]
    mask = (label_idx == ignore_index).unsqueeze(1)         # [B,1,H,W]
    one_hot = one_hot.masked_fill(mask, 0.0)
    return one_hot


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = 255, weight: list[float] = None, device = torch.device('cuda')):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.device = device
        self.weight = torch.tensor(weight).to(self.device) if weight is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], targets: [B, H, W]
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # mask out ignore_index positions
        mask = (targets != self.ignore_index).float()
        focal = focal * mask
        return focal.sum() / mask.sum()

class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        ignore_index: int = 255,
        include_classes: Optional[Sequence[int]] = None,
        exclude_background: bool = True,
        background_index: int = 0,
        batch_average: bool = True,  # True: 聚合整个 batch 后按类算 dice；False: per-sample then平均
    ):
        """
        include_classes: 如果给出，只计算这些类（索引列表）。优先级高于 exclude_background。
        exclude_background: 是否排除 background_index（比如 0）。
        batch_average: 如果 True, 在 batch 和空间维度上求和再计算每类 dice（更稳定）。
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.include_classes = None if include_classes is None else torch.tensor(include_classes, dtype=torch.long)
        self.exclude_background = exclude_background
        self.background_index = background_index
        self.batch_average = batch_average

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], targets: [B, H, W]
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # make one-hot (note: function above maps ignore_index temporarily to 0, we'll mask later)
        one_hot = one_hot_from_labels(targets, C, self.ignore_index)  # [B, C, H, W]

        # mask for ignore_index pixels
        valid_mask = (targets != self.ignore_index).view(B, 1, H, W)  # [B,1,H,W]
        valid_mask = valid_mask.float()

        # apply mask to both probs and one_hot so ignored pixels contribute zero
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        # optionally select classes to include
        if self.include_classes is not None:
            class_idx = self.include_classes.to(logits.device)
        else:
            # all classes, maybe exclude background
            class_idx = torch.arange(C, device=logits.device)
            if self.exclude_background:
                class_idx = class_idx[class_idx != self.background_index]

        # reduce dims
        if self.batch_average:
            # sum over batch and spatial dims -> per-class scalars
            probs_sum = probs.sum(dim=(0, 2, 3))      # [C]
            one_hot_sum = one_hot.sum(dim=(0, 2, 3))  # [C]
            inter = (probs * one_hot).sum(dim=(0, 2, 3))  # [C]
            union = probs_sum + one_hot_sum  # [C]
            dice_per_class = (2.0 * inter + self.smooth) / (union + self.smooth)  # [C]
            # select wanted classes
            dice_sel = dice_per_class[class_idx]
            # 可能存在某些类在整个 batch 中完全缺失（就用 smooth 避免除0）
            loss = 1.0 - dice_sel.mean()
            return loss * 0.23
        else:
            # per-sample per-class dice: results shape [B, C]
            probs_flat = probs.view(B, C, -1)
            one_hot_flat = one_hot.view(B, C, -1)
            inter = (probs_flat * one_hot_flat).sum(-1)   # [B, C]
            union = probs_flat.sum(-1) + one_hot_flat.sum(-1)  # [B, C]
            dice = (2.0 * inter + self.smooth) / (union + self.smooth)  # [B, C]
            dice_sel = dice[:, class_idx]  # [B, K]
            loss = 1.0 - dice_sel.mean()
            return loss * 0.23

class Loss(nn.Module):

    def __init__(
            self,
            lambda_main: float = 1.0,
            lambda_spix: float = 0.6,
            lambda_sfcn: float = 0.4,
            cell_size: int = 16,
            class_weight: list[float] = None,
            ignore_index: int = 255,
            device: torch.device = torch.device('cuda'),
            pix_loss_type: str = 'ce',  # options: 'ce', 'focal', 'dice'
    ):
        super().__init__()
        self.lambda_main = lambda_main
        self.lambda_spix = lambda_spix
        self.lambda_sfcn = lambda_sfcn
        self.sfcn_cell_size = cell_size
        self.ignore_index = ignore_index
        self.device = device

        self.spix_loss = FocalLoss(ignore_index=ignore_index, device=device)

        # Pixel-level loss
        if pix_loss_type == 'focal':
            self.pix_loss = FocalLoss(ignore_index=ignore_index, weight=class_weight, device=device)
        elif pix_loss_type == 'dice':
            self.pix_loss = DiceLoss(ignore_index=ignore_index)
        else:
            # default cross-entropy
            if class_weight is not None:
                cw = torch.tensor(class_weight, dtype=torch.float, device=device)
                self.pix_loss = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)
            else:
                self.pix_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self,
                Q: torch.Tensor,
                C: torch.Tensor,
                G: torch.Tensor,
                Ln: torch.Tensor,
                Lsp: torch.Tensor,
                L: torch.Tensor,
                Graph: dgl.DGLGraph,
                GT: torch.Tensor) -> torch.Tensor:
        """
        Args:
          Q:     [B, 9, H, W] sfcn output, affinity probs of pixel to adjacent superpixels
          C:     [B, Channels, H, W] features input to sfcn
          G:     [B, H, W] hard mapping of pixels to superpixels
          Ln:    [N, Classes] logit in superpixel vertex level
          Lsp:   [B, Classes, H, W] logit in superpixel vertex level
          L:     [B, Classes, H, W] logit in pixel level
          Graph: batched DGLGraph
          GT:    [B, H, W] ground truth labels
        Returns:
          total, l_pix, l_spix, l_sfcn
        """
        B, _, H, W = Q.shape
        Classes = L.shape[1]
        gt_one_hot = one_hot_from_labels(GT.long(), Classes, self.ignore_index) # [B, Classes, H, W]

        ##### 1. pixel-level loss #####
        gt_labels = GT.long()
        l_pix = self.pix_loss(L, gt_labels)
        # l_spix = self.pix_loss(Lsp, gt_labels)

        ##### 2. CE for spixl prediction Ln #####
        if False:
            gt_node, _ = generate_graph_nodes_hard(
                gt_one_hot,
                G,
                cell_size=self.sfcn_cell_size,
                device=self.device) # [B, Classes, N_i]
            gt_node = gt_node.permute(0, 2, 1).contiguous().view(-1, Classes) # [N, Classes]
            l_spix = self.spix_loss(Ln, gt_node.argmax(dim=1))

        ##### 2. CE for spixl prediction Lsp #####
        if False:
            l_spix = self.spix_loss(Lsp, gt_labels)

        ##### 2. KL divergence for spixl prediction L_spix #####
        if True:
            flat_gt = gt_one_hot.view(B, Classes, -1)   # [B, Classes, H*W]
            spix_ids = G.long().view(B, -1)          # [B, H*W]
            node_gt_list = []

            # vertex amount of each graph
            lengths = Graph.batch_num_nodes()   # List<int>

            for b in range(B):
                N_b = lengths[b]

                sum_feats = torch.zeros(N_b, Classes, device=self.device)  # [N_b, C]
                count     = torch.zeros(N_b, 1, device=self.device)  # [N_b, 1]

                ids_b = spix_ids[b]               # [H*W]
                gt_b  = flat_gt[b].permute(1,0)   # [H*W, C]
                # sum classifier probs to each spixl vertex
                sum_feats = sum_feats.scatter_add(
                    dim=0,
                    index=ids_b.unsqueeze(1).expand(-1, Classes),
                    src=gt_b
                )
                # count pixel amount of each spixl vertex
                count = count.scatter_add(
                    dim=0,
                    index=ids_b.unsqueeze(1),
                    src=torch.ones_like(ids_b.unsqueeze(1), dtype=torch.float, device=self.device)
                )
                # evaluate mean classifier probs of the spixl vertex
                node_gt = sum_feats / (count + 1e-8)  # [N_b, C]
                node_gt_list.append(node_gt)

            # ground truth of spixl vertexs
            node_gt = torch.cat(node_gt_list, dim=0) # [N, C]

            # evaluate KL of Lsp and GTnode
            logp   = F.log_softmax(Ln, dim=1)
            l_spix = F.kl_div(logp, node_gt, reduction='batchmean')

        ##### 3. sfcn loss #####
        loss_sem_pos, loss_sem, loss_pos = compute_semantic_pos_loss(
            prob_in=Q,
            labxy_feat=C,
            pos_weight=0.003,
            kernel_size=self.sfcn_cell_size,
            device=self.device
        )

        # total
        total = (
            self.lambda_main * l_pix +
            self.lambda_spix * l_spix +
            self.lambda_sfcn * loss_sem_pos * 0.013
        )
        return total, l_pix, l_spix, loss_sem_pos * 0.013
