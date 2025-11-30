import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_blocks import Conv, CSP, SPPF, C2F, Head
from constant import P1_CHANNELS, P2_CHANNELS, P3_CHANNELS, P4_CHANNELS, P5_CHANNELS
from constant import D, W, R


class YoloV8(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head()

    def forward(self, activation):
        small, medium, large = self.backbone(activation)
        small, medium, large = self.neck(small, medium, large)
        return self.head(small, medium, large)
        

class Backbone(nn.Module):
    def __init__(self, d=D, w=W, ratio=R):
        super().__init__()
        # 640x640x3
        self.layer0 = Conv(k=3, s=2, p=1, c_out=P1_CHANNELS) # P1
        # 320x320x64
        self.layer1 = Conv(k=3, s=2, p=1, c_out=P2_CHANNELS) # P2
        # 160x160x128
        self.layer2 = C2F(c_out=P2_CHANNELS, shortcut=True, bottlenecks=3*d)
        self.layer3 = Conv(k=3, s=2, p=1, c_out=P3_CHANNELS) # P3
        # 80x80x256
        self.layer4 = C2F(c_out=P3_CHANNELS, shortcut=True, bottlenecks=6*d)
        self.layer5 = Conv(k=3, s=2, p=1, c_out=P4_CHANNELS) # P4
        # 40x40x512
        self.layer6 = C2F(c_out=P4_CHANNELS, shortcut=True, bottlenecks=6*d)
        self.layer7 = Conv(k=3, s=2, p=1, c_out=P5_CHANNELS) # P5
        # 20x20x512
        self.layer8 = C2F(c_out=P5_CHANNELS, shortcut=True, bottlenecks=3*d)
        self.layer9 = SPPF()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        large = self.layer4(x)
        x = self.layer5(large)
        medium = self.layer6(x)
        x = self.layer7(medium)
        x = self.layer8(x)
        small = self.layer9(x)
        return small, medium, large


class Neck(nn.Module):
    def __init__(self, d=D, w=W, ratio=R):
        super().__init__()
        # layer 10 and 13
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # layer 11 is a Concat (implemented in forward) - P3
        self.layer12 = C2F(c_out=P3_CHANNELS, shortcut=False, bottlenecks=3*d)
        # layer 14 is a Concat (implemented in forward) - P3
        self.layer15 = C2F(c_out=P3_CHANNELS, shortcut=False, bottlenecks=3*d)
        self.layer16 = Conv(k=3, s=2, p=1, c_out=P3_CHANNELS)
        # layer 17 is a Concat (implemented in forward) - P4
        self.layer18 = C2F(c_out=P4_CHANNELS, shortcut=False, bottlenecks=3*d)
        self.layer19 = Conv(k=3, s=2, p=1, c_out=P4_CHANNELS)
        # layer 20 is a Concat (implemented in forward) - P5
        self.layer21 = C2F(c_out=P5_CHANNELS, shortcut=False, bottlenecks=3*d)

    def forward(self, small, medium, large):
        x = self.upsample(small)
        x = torch.cat((x, medium), dim=1)
        medium_c2f_out = self.layer12(x)
        x = self.upsample(medium_c2f_out)
        x = torch.cat((x, large), dim=1)
        large_out = self.layer15(x)
        x = self.layer16(x)
        x = torch.cat((x, medium_c2f_out), dim=1)
        medium_out = self.layer18(x)
        x = self.layer19(medium_out)
        x = torch.cat((x, small), dim=1)
        small_out = self.layer21(x)
        return small_out, medium_out, large_out


class Detect(nn.Module):
    def __init__(self):
        super().__init__()
        self.small_detect = Head()
        self.medium_detect = Head()
        self.large_detect = Head()

    def forward(self, small, medium, large):
        small_out = self.small_detect.forward(small)
        medium_out = self.medium_detect.forward(medium)
        large_out = self.large_detect.forward(large)
        x = nn.cat((small_out,medium_out,large_out),dim=0)
        

        return x

