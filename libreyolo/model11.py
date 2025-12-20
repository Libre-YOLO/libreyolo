"""
Neural network architecture for Libre YOLO11.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    Converts the distribution of 4*reg_max channels into the final 4 coordinates (x, y, w, h).
    """
    def __init__(self, c1=16):
        super().__init__()
        # 1x1 convolution with fixed weights 0..c1-1 to calculate the expectation
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # Input x: (Batch, 64, H, W) for reg_max=16
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


def autopad(k, p=None, d=1):
    """Pad to 'same' output shape."""
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = tuple(d * (x - 1) + 1 for x in k)
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = tuple(x // 2 for x in k)
    return p


class Conv(nn.Module):
    """Standard convolution block: Conv2d + BatchNorm + SiLU"""
    def __init__(self, c_out, c_in, k, s, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = autopad(k, p, d)
        self.cnn = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p, groups=g, dilation=d, bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=c_out, eps=0.001, momentum=0.03)
        self.silu = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.batchnorm(x)
        x = self.silu(x)
        return x


class DWConv(Conv):
    """Depthwise convolution"""
    def __init__(self, c_out, c_in, k=3, s=1, d=1, act=True):
        super().__init__(c_out, c_in, k=k, s=s, p=None, g=math.gcd(c_in, c_out), d=d, act=act)


class Bottleneck(nn.Module):
    """Standard bottleneck block"""
    def __init__(self, c_out, c_in, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c_out * e)
        self.conv1 = Conv(c_out=c_, c_in=c_in, k=k[0], s=1, p=autopad(k[0]))
        self.conv2 = Conv(c_out=c_out, c_in=c_, k=k[1], s=1, p=autopad(k[1]), g=g)
        self.add = shortcut and c_in == c_out
        
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck with 3 convs (used by C3k)."""
    def __init__(self, c_out, c_in, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c_out * e)
        self.conv1 = Conv(c_out=c_, c_in=c_in, k=1, s=1, p=0)
        self.conv2 = Conv(c_out=c_, c_in=c_in, k=1, s=1, p=0)
        self.conv3 = Conv(c_out=c_out, c_in=2 * c_, k=1, s=1, p=0)
        self.bottlenecks = nn.Sequential(*(Bottleneck(c_out=c_, c_in=c_, shortcut=shortcut, g=g, k=(1, 3), e=1.0) for _ in range(n)))
        
    def forward(self, x):
        return self.conv3(torch.cat((self.bottlenecks(self.conv1(x)), self.conv2(x)), dim=1))


class C3k(C3):
    """C3 with customizable kernel-size in bottlenecks."""
    def __init__(self, c_out, c_in, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c_out, c_in, n=n, shortcut=shortcut, g=g, e=e)
        c_ = int(c_out * e)
        self.bottlenecks = nn.Sequential(*(Bottleneck(c_out=c_, c_in=c_, shortcut=shortcut, g=g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(nn.Module):
    """
    YOLO11 C3k2: "C2f-like" block with optional C3k inner blocks.
    The concat always includes the 2 split chunks + n outputs -> (2+n)*c hidden channels.
    """
    def __init__(self, c_out, c_in, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c_out * e)  # hidden channels
        self.conv1 = Conv(c_out=2 * self.c, c_in=c_in, k=1, s=1, p=0)
        self.conv2 = Conv(c_out=c_out, c_in=(2 + n) * self.c, k=1, s=1, p=0)
        self.bottlenecks = nn.ModuleList(
            C3k(c_out=self.c, c_in=self.c, n=2, shortcut=shortcut, g=g) if c3k 
            else Bottleneck(c_out=self.c, c_in=self.c, shortcut=shortcut, g=g) 
            for _ in range(n)
        )
        
    def forward(self, x):
        y = list(self.conv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.conv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast"""
    def __init__(self, c_out, c_in, k=5):
        super().__init__()
        c_ = c_in // 2
        self.conv1 = Conv(c_out=c_, c_in=c_in, k=1, s=1, p=0)
        self.conv2 = Conv(c_out=c_out, c_in=c_ * 4, k=1, s=1, p=0)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), dim=1))


class Attention(nn.Module):
    """Attention mechanism (used by C2PSA)."""
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5

        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(c_out=h, c_in=dim, k=1, s=1, p=0, act=False)
        self.proj = Conv(c_out=dim, c_in=dim, k=1, s=1, p=0, act=False)
        self.pe = Conv(c_out=dim, c_in=dim, k=3, s=1, p=1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class PSABlock(nn.Module):
    """PSA Block with Attention and FFN"""
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            Conv(c_out=c * 2, c_in=c, k=1, s=1, p=0),
            Conv(c_out=c, c_in=c * 2, k=1, s=1, p=0, act=False)
        )
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2 + PSA (YOLO11)."""
    def __init__(self, c_out, c_in, n=1, e=0.5):
        super().__init__()
        if c_in != c_out:
            raise ValueError("C2PSA requires c_in == c_out")
        self.c = int(c_in * e)
        self.conv1 = Conv(c_out=2 * self.c, c_in=c_in, k=1, s=1, p=0)
        self.conv2 = Conv(c_out=c_in, c_in=2 * self.c, k=1, s=1, p=0)
        # num_heads=self.c // 64 (floor). Ensure at least 1.
        nh = max(1, self.c // 64)
        self.bottlenecks = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=nh) for _ in range(n)))

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, dim=1)
        b = self.bottlenecks(b)
        return self.conv2(torch.cat((a, b), dim=1))


class Backbone(nn.Module):
    """Feature extraction backbone for YOLO11"""
    def __init__(self, config):
        super().__init__()
        
        # Channel configurations for each model size
        # Format: {size: (p1, p2, c2f1, c2f2, c2f3, c2f4, sppf, c2psa, n_bottlenecks, use_c3k_everywhere)}
        # use_c3k_everywhere: For m/l/x models, ALL C3k2 blocks use C3k structure
        self.channel_config = {
            'n': (16, 32, 64, 128, 128, 256, 256, 256, 1, False),     # w=0.25
            's': (32, 64, 128, 256, 256, 512, 512, 512, 1, False),    # w=0.5
            'm': (64, 128, 256, 512, 512, 512, 512, 512, 1, True),    # w=1.0, max_ch=512, c3k everywhere, n=1
            'l': (64, 128, 256, 512, 512, 512, 512, 512, 2, True),    # w=1.0, max_ch=512, c3k everywhere, n=2
            'x': (96, 192, 384, 768, 768, 768, 768, 768, 2, True),    # w=1.5, max_ch=768, c3k everywhere, n=2
        }
        
        c1, c2, c3, c4_out, c4_mid, c5_out, c5_sppf, c5_psa, n, c3k_all = self.channel_config[config]
        
        # Layer 0: Conv 3->c1
        self.p1 = Conv(c_out=c1, c_in=3, k=3, s=2, p=1)
        # Layer 1: Conv c1->c2
        self.p2 = Conv(c_out=c2, c_in=c1, k=3, s=2, p=1)
        # Layer 2: C3k2 c2->c3 (C3k for m/l/x, Bottleneck for n/s)
        self.c2f1 = C3k2(c_out=c3, c_in=c2, n=n, c3k=c3k_all, e=0.25, shortcut=True)
        # Layer 3: Conv c3->c3
        self.p3 = Conv(c_out=c3, c_in=c3, k=3, s=2, p=1)
        # Layer 4: C3k2 c3->c4_out (C3k for m/l/x, Bottleneck for n/s)
        self.c2f2 = C3k2(c_out=c4_out, c_in=c3, n=n, c3k=c3k_all, e=0.25, shortcut=True)
        # Layer 5: Conv c4_out->c4_mid
        self.p4 = Conv(c_out=c4_mid, c_in=c4_out, k=3, s=2, p=1)
        # Layer 6: C3k2 c4_mid->c4_mid (always C3k) - uses e=0.5
        self.c2f3 = C3k2(c_out=c4_mid, c_in=c4_mid, n=n, c3k=True, e=0.5, shortcut=True)
        # Layer 7: Conv c4_mid->c5_out
        self.p5 = Conv(c_out=c5_out, c_in=c4_mid, k=3, s=2, p=1)
        # Layer 8: C3k2 c5_out->c5_out (always C3k) - uses e=0.5
        self.c2f4 = C3k2(c_out=c5_out, c_in=c5_out, n=n, c3k=True, e=0.5, shortcut=True)
        # Layer 9: SPPF c5_out->c5_sppf
        self.sppf = SPPF(c_out=c5_sppf, c_in=c5_out, k=5)
        # Layer 10: C2PSA c5_sppf->c5_psa
        self.c2psa = C2PSA(c_out=c5_psa, c_in=c5_sppf, n=n, e=0.5)
        
    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.c2f1(x)
        x = self.p3(x)
        x = self.c2f2(x)
        x8 = x.clone()      # P3 (Stride 8)
        
        x = self.p4(x)
        x = self.c2f3(x)
        x16 = x.clone()     # P4 (Stride 16)
        
        x = self.p5(x)
        x = self.c2f4(x)
        x = self.sppf(x)
        x32 = self.c2psa(x) # P5 (Stride 32)
        
        return x8, x16, x32


class Neck(nn.Module):
    """Feature pyramid network neck for YOLO11"""
    def __init__(self, config):
        super().__init__()
        
        # Channel configurations for neck
        # Format: {size: (bb_p3, bb_p4, bb_p5, neck_p3, neck_p4, neck_p5, n, use_c3k_everywhere)}
        # bb_p3/p4/p5 = backbone outputs (c2f2, c2f3, c2psa)
        # neck_p3/p4/p5 = neck outputs
        # use_c3k_everywhere: For m/l/x models, ALL C3k2 blocks use C3k structure
        self.channel_config = {
            'n': (128, 128, 256, 64, 128, 256, 1, False),    # w=0.25
            's': (256, 256, 512, 128, 256, 512, 1, False),   # w=0.5
            'm': (512, 512, 512, 256, 512, 512, 1, True),    # w=1.0, max_ch=512, c3k everywhere, n=1
            'l': (512, 512, 512, 256, 512, 512, 2, True),    # w=1.0, max_ch=512, c3k everywhere, n=2
            'x': (768, 768, 768, 384, 768, 768, 2, True),    # w=1.5, max_ch=768, c3k everywhere, n=2
        }
        
        bb_p3, bb_p4, bb_p5, neck_p3, neck_p4, neck_p5, n, c3k_all = self.channel_config[config]

        # Layer 11: Upsample
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        # Layer 12: Concat (bb_p5 + bb_p4)
        # Layer 13: C3k2 -> c2f21_out (intermediate P4, C3k for m/l/x)
        c2f21_out = neck_p4  # Uses neck_p4 as intermediate output
        self.c2f21 = C3k2(c_out=c2f21_out, c_in=bb_p5 + bb_p4, n=n, c3k=c3k_all, e=0.5, shortcut=True)
        
        # Layer 14: Upsample
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # Layer 15: Concat (c2f21_out + bb_p3)
        # Layer 16: C3k2 -> neck_p3 (final P3 output, C3k for m/l/x)
        self.c2f11 = C3k2(c_out=neck_p3, c_in=c2f21_out + bb_p3, n=n, c3k=c3k_all, e=0.5, shortcut=True)
        
        # Layer 17: Conv neck_p3->neck_p3
        self.conv1 = Conv(c_out=neck_p3, c_in=neck_p3, k=3, s=2, p=1)
        # Layer 18: Concat (neck_p3 + c2f21_out)
        # Layer 19: C3k2 -> neck_p4 (final P4 output, C3k for m/l/x)
        self.c2f12 = C3k2(c_out=neck_p4, c_in=neck_p3 + c2f21_out, n=n, c3k=c3k_all, e=0.5, shortcut=True)
        
        # Layer 20: Conv neck_p4->neck_p4
        self.conv2 = Conv(c_out=neck_p4, c_in=neck_p4, k=3, s=2, p=1)
        # Layer 21: Concat (neck_p4 + bb_p5)
        # Layer 22: C3k2 -> neck_p5 (final P5 output, always C3k)
        self.c2f22 = C3k2(c_out=neck_p5, c_in=neck_p4 + bb_p5, n=n, c3k=True, e=0.5, shortcut=True)
        
    def forward(self, x8, x16, x32):
        # 1. Upsample P5 (x32) to P4 size
        aux1 = self.upsample1(x32)
        x16 = torch.cat((aux1, x16), dim=1)
        x16 = self.c2f21(x16)
        
        # 2. Upsample P4 (x16) to P3 size
        aux2 = self.upsample2(x16)
        x8 = torch.cat((aux2, x8), dim=1)
        x8 = self.c2f11(x8)
        
        # 3. Downsample P3 (x8) to P4 size
        aux3 = self.conv1(x8)
        x16 = torch.cat((aux3, x16), dim=1)
        x16 = self.c2f12(x16)
        
        # 4. Downsample P4 (x16) to P5 size
        aux4 = self.conv2(x16)
        x32 = torch.cat((aux4, x32), dim=1)
        x32 = self.c2f22(x32)
        
        return x8, x16, x32


class Head(nn.Module):
    """Decoupled detection head for YOLO11"""
    def __init__(self, c_in, c_box, c_cls, reg_max, nb_classes):
        super().__init__()
        
        # Box Branch: c_in -> c_box -> c_box -> 4*reg_max
        # YOLO11: Sequential(Conv, Conv, Conv2d)
        self.conv11 = Conv(c_out=c_box, c_in=c_in, k=3, s=1, p=1)
        self.conv12 = Conv(c_out=c_box, c_in=c_box, k=3, s=1, p=1)
        self.cnn1 = nn.Conv2d(in_channels=c_box, out_channels=4*reg_max, kernel_size=1, stride=1, padding=0)
        
        # Class Branch: c_in -> c_cls -> c_cls -> nb_classes
        # YOLO11: Sequential(Sequential(DWConv, Conv), Sequential(DWConv, Conv), Conv2d)
        # Map to: conv21 = Sequential(DWConv, Conv), conv22 = Sequential(DWConv, Conv)
        self.conv21 = nn.Sequential(
            DWConv(c_out=c_in, c_in=c_in, k=3, s=1),
            Conv(c_out=c_cls, c_in=c_in, k=1, s=1, p=0)
        )
        self.conv22 = nn.Sequential(
            DWConv(c_out=c_cls, c_in=c_cls, k=3, s=1),
            Conv(c_out=c_cls, c_in=c_cls, k=1, s=1, p=0, act=True)
        )
        self.cnn2 = nn.Conv2d(in_channels=c_cls, out_channels=nb_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        box = self.cnn1(self.conv12(self.conv11(x)))
        cls = self.cnn2(self.conv22(self.conv21(x)))
        return box, cls


class LibreYOLO11Model(nn.Module):
    """Main YOLOv11 model"""
    def __init__(self, config, reg_max, nb_classes):
        super().__init__()
        
        # Head channel configurations
        # Format: {size: (neck_p3, neck_p4, neck_p5, c_box)}
        self.head_config = {
            'n': (64, 128, 256, 64),     # w=0.25
            's': (128, 256, 512, 64),    # w=0.5
            'm': (256, 512, 512, 64),    # w=1.0, max_ch=512
            'l': (256, 512, 512, 64),    # w=1.0, max_ch=512
            'x': (384, 768, 768, 96),    # w=1.5, max_ch=768
        }
        
        c_p3, c_p4, c_p5, c_box = self.head_config[config]
        
        self.backbone = Backbone(config=config)
        self.neck = Neck(config=config)
        
        # --- HEAD CONFIGURATION ---
        # Cls channels: max(c_p3, min(nb_classes, 100))
        c_cls = max(c_p3, min(nb_classes, 100))
        
        # Head inputs: P3, P4, P5 channels
        self.head8 = Head(c_in=c_p3, c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        self.head16 = Head(c_in=c_p4, c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        self.head32 = Head(c_in=c_p5, c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        
        self.dfl = DFL(c1=reg_max)

    def forward(self, x):
        x8, x16, x32 = self.backbone(x)
        x8, x16, x32 = self.neck(x8, x16, x32)
        
        box8, cls8 = self.head8(x8)
        box16, cls16 = self.head16(x16)
        box32, cls32 = self.head32(x32)
        
        decoded_box8 = self.dfl(box8)
        decoded_box16 = self.dfl(box16)
        decoded_box32 = self.dfl(box32)
        
        out = {
            'x8': {'box': decoded_box8, 'cls': cls8},
            'x16': {'box': decoded_box16, 'cls': cls16},
            'x32': {'box': decoded_box32, 'cls': cls32}
        }
        
        return out
