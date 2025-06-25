# student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Conv Block
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# Conv Block with SE and DepthwiseConv
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseConv(in_channels, out_channels)
        self.conv2 = DepthwiseConv(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return out + identity

# PixelShuffle Upsample Block
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))

# Final Student U-Net
class StudentUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = ConvBlock(64, 128)

        self.up1 = UpsampleBlock(128, 64)
        self.decoder1 = ConvBlock(128, 64)

        self.up2 = UpsampleBlock(64, 32)
        self.decoder2 = ConvBlock(64, 32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()  # Ensure output is in [0,1] for image comparison
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        mid = self.middle(self.pool2(enc2))

        dec1 = self.decoder1(torch.cat([self.up1(mid), enc2], dim=1))
        dec2 = self.decoder2(torch.cat([self.up2(dec1), enc1], dim=1))
        return self.out_conv(dec2)
