import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6,12,18)):
        super().__init__()
        rates = tuple(atrous_rates)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # image pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        return x  # [B, out_channels, H, W]

class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super(Encoder, self).__init__()
        mb = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        lowfeat_layer = 2
        highfeat_layer = 12
        mb_out_channel = [-1, 16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576]
        self.body_low = nn.Sequential(*(list(mb.features.children()))[:lowfeat_layer])
        self.body_high = nn.Sequential(*(list(mb.features.children()))[:highfeat_layer])

        aspp_channels=256
        low_level_channels=48

        self.aspp = ASPP(in_channels=mb_out_channel[highfeat_layer], out_channels=aspp_channels)

        self.reduce_low = nn.Sequential(
            nn.Conv2d(mb_out_channel[lowfeat_layer], low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + low_level_channels, aspp_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_channels, aspp_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(aspp_channels, out_channels, kernel_size=1)
        )


    def forward(self, x):
        # x shape: [B, C_in, H, W]
        H, W = x.shape[-2], x.shape[-1]
        low_feat_raw = x
        for layer in self.body_low:
            low_feat_raw = layer(low_feat_raw)
        high_feat_raw = x
        for layer in self.body_high:
            high_feat_raw = layer(high_feat_raw)
        aspp_feat_raw = self.aspp(high_feat_raw)

        low_feat = self.reduce_low(low_feat_raw)
        aspp_feat = F.interpolate(aspp_feat_raw, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([aspp_feat, low_feat], dim=1)
        y = self.last_conv(y)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        return y  # [B, C, H, W]
