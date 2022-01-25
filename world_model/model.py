import torch.nn.functional as F

from gan_utils import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_droput = nn.Dropout(0.8)

        # # Unet
        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        #
        # self.up1 = Up(192, 64, bilinear)
        # # self.up2 = Up(512, 128, bilinear)
        # # self.up3 = Up(256, 64, bilinear)
        # # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 32)
        self.down2 = Down(128, 256)

        self.up1 = Up(96, 16, bilinear)
        self.up2 = Up(320, 64, bilinear)

        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        evil_twin_of_x = x

        x1 = self.inc(x)

        x2 = self.down1(x1)
        #x3 = self.down2(x2)

        x = self.up1(x2, x1)
        # x = self.up2(x, x1)

        logits = self.outc(x)

        # reward = F.relu(self.bn1(self.conv1(evil_twin_of_x)))
        #
        # reward = F.relu(self.bn2(self.conv2(reward)))
        #
        # reward = F.relu(self.bn3(self.conv3(reward)))
        #
        # reward = F.relu(self.bn4(self.conv4  (reward)))




        #reward = F.relu(self.bn2(self.conv5(reward)))

        return logits
