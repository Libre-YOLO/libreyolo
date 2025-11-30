import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, k, s, p, c_out):
        super().__init__()
        self.convolution = nn.LazyConv2d(out_channels=c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.normalization = nn.BatchNorm2d(num_features=c_out)
        self.activation_function = nn.SiLU()
    def forward(self, activation):
        x = self.convolution(activation)
        x = self.normalization(x)
        x = self.activation_function(x)
        return x

class C2F(nn.Module):
    def __init__(self, c_out, shortcut, bottlenecks=3):
        super().__init__()
        self.convolution1 = Conv(k=1, s=1, p=0, c_out=c_out)
        self.convolution2 = Conv(k=1, s=1, p=0, c_out=c_out)
        self.bottleneck = Bottleneck(k=3, s=1, p=1, c=c_out, shortcut=shortcut)
        self.nb_bottlenecks = bottlenecks

    def forward(self,activation):
        x = self.convolution(activation)
        x1,x2 = torch.tensor_split(x)
        # x = torch.cat(x1,x2)
        for i in range(self.nb_bottlenecks - 1):
            x2 = self.bottleneck(x2)
            x = torch.cat(x,x2)
        x = self.convolution(activation)
        return x

class CSP(nn.Module):
    def __init__(self):
        super().__init__()
        # Branch A (light)
        self.branch_A = Conv(k=1, s=1, p=0)
        # Branch B (two Conv)
        self.branch_B_1= Conv(k=1, s=1, p=0)
        self.branch_B_2= Conv(k=1, s=1, p=0)
        # Fuse both branches
        self.fuse = Conv(k=1, s=1, p=0, c_in=128)

    def forward(self, activations):
        a = self.branch_A(activations)
        b = self.branch_B_1(activations)
        b = self.branch_B_2(b)
        x = torch.cat((a,b), dim=1)
        x = self.fuse(x)
        return x


class SPPF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv(k=1, s=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.fuse = Conv(k=1, s=1, p=0)
    def forward(self, activations):
        x0 = self.conv(activations)
        x1 = self.pool(x0)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        return self.conv(x)
        # return self.fuse(y)


class Bottleneck(torch.nn.Module):
    def __init__(self, k, s, p, c, shortcut=False):
        super().__init__()
        """
        # Old Conv implementation with c_in
        self.layer1 = Conv(k=k,s=s,p=p,c_in=c)
        self.layer2 = Conv(k=k,s=s,p=p,c_in=c)
        """
        
        self.layer1 = Conv(k=k, s=s, p=p) # Aqui falta c_out
        self.layer2 = Conv(k=k, s=s, p=p) # Aqui falta c_out
        self.use_shortcut = shortcut

    def forward(self, activations):
        x = self.layer1(activations)
        x = self.layer2(x)
        if self.use_shortcut:
            x += activations
        return x


class Head(nn.Module):
    def __init__(self, nc=80, reg_max=1):
        super().__init__()
        self.conv = Conv(k=3, s=1, p=1)
        self.bb_conv2d  = nn.LazyConv2d(out_channels=4*reg_max, kernel_size=1, stride=1, padding=0)
        self.cls_conv2d  = nn.LazyConv2d(out_channels=nc, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # box path
        b = self.conv(x)
        b = self.conv(b)
        b = self.bb_conv2d(b)

        # class path
        c = self.conv(x)
        c = self.conv(c)
        c = self.cls_conv2d(c)

        return b, c


class BDDCollapse(nn.Module):
    """
    The goal of decode_bbox is to input the prediction ditribution tensor (16 per side following DFL)
    and to output a single bounding box.
    Prediction distribution tensor shape:(B,4*16,feature_map_h,feature_map_w)
    INPUTS: [B,8400,64]
    OUTPUTS: [B,8400,4]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, 16, 1, 1))
        self.c1 = 16

    def forward(self,x):
        if x.ndim != 3 or x.shape[1:] != (8400, 64): #check if the input shape is correct (B,8400,64)
            raise ValueError(f"Expected input of shape [B,8400,64], got {tuple(x.shape)}")
        b, c, h, w = x.shape
        # Reshape to (Batch, 4, 16, H, W) and transpose to (Batch, 16, 4, H, W)
        x = x.view(b, 4, self.c1, h, w).transpose(2, 1)
        # Apply softmax to get probability distribution
        x = F.softmax(x, dim=1)
        # Apply the convolution (weighted sum) to reduce 16 channels to 1
        # Reshape to (Batch, 16, 4*H, W) so Conv2d sees a valid 4D tensor
        x = self.conv(x.reshape(b, self.c1, 4 * h, w))
        # Reshape back to (Batch, 4, H, W)
        return x.view(b, 4, h, w)
        

        