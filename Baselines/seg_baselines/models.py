import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=64):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.down2 = DoubleConv(base, base*2)
        self.down3 = DoubleConv(base*2, base*4)
        self.down4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))

        x = self.up3(c4)
        x = self.conv3(torch.cat([x, c3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, c2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, c1], dim=1))
        return self.out(x)

# --- Minimal U-Net++ (Nested U-Net) ---
class UNetPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        # encoder
        self.conv00 = DoubleConv(in_channels, base)
        self.conv10 = DoubleConv(base, base*2)
        self.conv20 = DoubleConv(base*2, base*4)
        self.conv30 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)

        # decoder (nested)
        self.up01 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.conv01 = DoubleConv(base + base, base)

        self.up11 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.conv11 = DoubleConv(base*2 + base*2, base*2)

        self.up21 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.conv21 = DoubleConv(base*4 + base*4, base*4)

        self.up02 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.conv02 = DoubleConv(base + base + base, base)

        self.up12 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.conv12 = DoubleConv(base*2 + base*2 + base*2, base*2)

        self.up03 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.conv03 = DoubleConv(base + base + base + base, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))

        x01 = self.conv01(torch.cat([x00, self.up01(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up21(x30)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], dim=1))
        return self.out(x03)
