"""
Neural network architecture for Libre YOLO11.
Matches the actual YOLO11 architecture from Ultralytics.
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
        # Match Ultralytics BatchNorm parameters: eps=0.001, momentum=0.03
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
    """Ultralytics-style attention (used by C2PSA)."""
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
        # In Ultralytics: num_heads=self.c // 64 (floor). Ensure at least 1.
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
        
        self.configuration = {
            'n': {'d': 0.33, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.33, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.67, 'w': 0.75, 'r': 1.5},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.25, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']
        
        # YOLO11n structure (matching actual Ultralytics model)
        # Layer 0: Conv 3->16
        self.p1 = Conv(c_out=16, c_in=3, k=3, s=2, p=1)
        # Layer 1: Conv 16->32
        self.p2 = Conv(c_out=32, c_in=16, k=3, s=2, p=1)
        # Layer 2: C3k2 32->64 (with Bottleneck, not C3k)
        self.c2f1 = C3k2(c_out=64, c_in=32, n=1, c3k=False, e=0.25, shortcut=True)
        # Layer 3: Conv 64->64
        self.p3 = Conv(c_out=64, c_in=64, k=3, s=2, p=1)
        # Layer 4: C3k2 64->128 (with Bottleneck)
        self.c2f2 = C3k2(c_out=128, c_in=64, n=1, c3k=False, e=0.25, shortcut=True)
        # Layer 5: Conv 128->128
        self.p4 = Conv(c_out=128, c_in=128, k=3, s=2, p=1)
        # Layer 6: C3k2 128->128 (with C3k nested) - uses e=0.5
        self.c2f3 = C3k2(c_out=128, c_in=128, n=1, c3k=True, e=0.5, shortcut=True)
        # Layer 7: Conv 128->256
        self.p5 = Conv(c_out=256, c_in=128, k=3, s=2, p=1)
        # Layer 8: C3k2 256->256 (with C3k nested) - uses e=0.5
        self.c2f4 = C3k2(c_out=256, c_in=256, n=1, c3k=True, e=0.5, shortcut=True)
        # Layer 9: SPPF 256->256
        self.sppf = SPPF(c_out=256, c_in=256, k=5)
        # Layer 10: C2PSA 256->256
        self.c2psa = C2PSA(c_out=256, c_in=256, n=1, e=0.5)
        
    def forward(self, x):
        x = self.p1(x)      # 16
        x = self.p2(x)      # 32
        x = self.c2f1(x)    # 64
        x = self.p3(x)      # 64
        x = self.c2f2(x)    # 128
        x8 = x.clone()      # P3 (Stride 8), 128 channels
        
        x = self.p4(x)      # 128
        x = self.c2f3(x)    # 128
        x16 = x.clone()     # P4 (Stride 16), 128 channels
        
        x = self.p5(x)      # 256
        x = self.c2f4(x)    # 256
        x = self.sppf(x)    # 256
        x32 = self.c2psa(x) # P5 (Stride 32), 256 channels
        
        return x8, x16, x32


class Neck(nn.Module):
    """Feature pyramid network neck for YOLO11"""
    def __init__(self, config):
        super().__init__()
        
        self.configuration = {
            'n': {'d': 0.33, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.33, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.67, 'w': 0.75, 'r': 1.5},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.25, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']

        # YOLO11n neck structure (matching actual Ultralytics model)
        # Layer 11: Upsample
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        # Layer 12: Concat (256+128=384)
        # Layer 13: C3k2 384->128 (with Bottleneck) - uses e=0.5
        self.c2f21 = C3k2(c_out=128, c_in=384, n=1, c3k=False, e=0.5, shortcut=True)
        
        # Layer 14: Upsample
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # Layer 15: Concat (128+128=256)
        # Layer 16: C3k2 256->64 (with Bottleneck) - uses e=0.5
        self.c2f11 = C3k2(c_out=64, c_in=256, n=1, c3k=False, e=0.5, shortcut=True)
        
        # Layer 17: Conv 64->64
        self.conv1 = Conv(c_out=64, c_in=64, k=3, s=2, p=1)
        # Layer 18: Concat (64+128=192)
        # Layer 19: C3k2 192->128 (with Bottleneck) - uses e=0.5
        self.c2f12 = C3k2(c_out=128, c_in=192, n=1, c3k=False, e=0.5, shortcut=True)
        
        # Layer 20: Conv 128->128
        self.conv2 = Conv(c_out=128, c_in=128, k=3, s=2, p=1)
        # Layer 21: Concat (128+256=384)
        # Layer 22: C3k2 384->256 (with C3k nested) - uses e=0.5
        self.c2f22 = C3k2(c_out=256, c_in=384, n=1, c3k=True, e=0.5, shortcut=True)
        
    def forward(self, x8, x16, x32):
        # 1. Upsample P5 (x32=256) to P4 size
        aux1 = self.upsample1(x32)  # 256
        x16 = torch.cat((aux1, x16), dim=1)  # 256+128=384
        x16 = self.c2f21(x16)  # 128
        
        # 2. Upsample P4 (x16=128) to P3 size
        aux2 = self.upsample2(x16)  # 128
        x8 = torch.cat((aux2, x8), dim=1)  # 128+128=256
        x8 = self.c2f11(x8)  # 64
        
        # 3. Downsample P3 (x8=64) to P4 size
        aux3 = self.conv1(x8)  # 64
        x16 = torch.cat((aux3, x16), dim=1)  # 64+128=192
        x16 = self.c2f12(x16)  # 128
        
        # 4. Downsample P4 (x16=128) to P5 size
        aux4 = self.conv2(x16)  # 128
        x32 = torch.cat((aux4, x32), dim=1)  # 128+256=384
        x32 = self.c2f22(x32)  # 256
        
        return x8, x16, x32


class Head(nn.Module):
    """Decoupled detection head for YOLO11 - matches Ultralytics structure"""
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
            Conv(c_out=c_cls, c_in=c_cls, k=1, s=1, p=0, act=True)  # Note: Ultralytics has activation here
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
        
        self.configuration = {
            'n': {'d': 0.33, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.33, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.67, 'w': 0.75, 'r': 1.5},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.25, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']
        
        self.backbone = Backbone(config=config)
        self.neck = Neck(config=config)
        
        # --- HEAD CONFIGURATION (Matches Ultralytics 'Detect' Logic) ---
        # For YOLO11n: P3=64, P4=128, P5=256
        # Box channels: max(16, c_p3 // 4, reg_max * 4)
        c_box = max(16, 64 // 4, reg_max * 4)  # max(16, 16, 64) = 64
        # Cls channels: max(c_p3, min(nb_classes, 100))
        c_cls = max(64, min(nb_classes, 100))  # max(64, min(80, 100)) = 80
        
        # Head inputs: P3=64, P4=128, P5=256
        self.head8 = Head(c_in=64, c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        self.head16 = Head(c_in=128, c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        self.head32 = Head(c_in=256, c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        
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
