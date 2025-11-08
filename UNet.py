import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(  # 两个卷积块
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 卷积层
            nn.BatchNorm2d(out_channels),  # 批归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 卷积层
            nn.BatchNorm2d(out_channels),  # 批归一化
            nn.ReLU(inplace=True)  # 激活函数
        )

    def forward(self, x):
        return self.double_conv(x)  # 前向传播


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(  # 下采样块
            nn.MaxPool2d(2),  # 最大池化
            DoubleConv(in_channels, out_channels)  # 两个卷积
        )

    def forward(self, x):
        return self.maxpool_conv(x)  # 前向传播


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 双线性插值上采样
            self.conv = DoubleConv(in_channels, out_channels)  # 卷积
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 转置卷积上采样
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样
        diffY = x2.size()[2] - x1.size()[2]  # 计算高度差
        diffX = x2.size()[3] - x1.size()[3]  # 计算宽度差
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # 拼接特征图
        return self.conv(x)  # 前向传播


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.out_channels = out_channels

        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.phi = nn.Conv2d(gating_channels, out_channels, kernel_size=1)
        self.psi = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x g的尺寸要一致
        if x.size() != g.size():
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = theta_x + phi_g
        f = self.relu(f)
        psi_f = self.psi(f)
        attn = self.sigmoid(psi_f)
        return x * attn


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积

    def forward(self, x):
        return self.conv(x)  # 前向传播


class AttentionUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024 + 512, 512, bilinear)
        self.up2 = Up(512 + 256, 256, bilinear)
        self.up3 = Up(256 + 128, 128, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)

        # 注意力机制
        self.att1 = AttentionGate(512, 1024, 512)
        self.att2 = AttentionGate(256, 512, 256)
        self.att3 = AttentionGate(128, 256, 128)
        self.att4 = AttentionGate(64, 128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(self.att1(x, x5), x3)
        x = self.up3(self.att2(x, x4), x2)
        x = self.up4(self.att3(x, x3), x1)

        logits = self.outc(x)
        return torch.sigmoid(logits)
