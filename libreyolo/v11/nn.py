"""
Neural network architecture for Libre YOLO11.

Layer naming convention (must match converted weights):
- backbone: p1, p2, c2f1, p3, c2f2, p4, c2f3, p5, c2f4, sppf, c2psa
- neck: c2f21, c2f11, conv1, c2f12, conv2, c2f22  
- head: head8, head16, head32 with conv11, conv12, conv21, conv22, cnn1, cnn2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DFL(nn.Module):
    """Distribution Focal Loss module."""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, 4, self.c1, h, w).transpose(2, 1)
        x = F.softmax(x, dim=1)
        x = self.conv(x.reshape(b, self.c1, 4 * h, w))
        return x.view(b, 4, h, w)


class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm + SiLU (optional)"""
    def __init__(self, c_out, c_in, k, s, p, g=1, act=True):
        super().__init__()
        self.cnn = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        # Use eps=0.001 to match training configuration
        self.batchnorm = nn.BatchNorm2d(c_out, eps=0.001, momentum=0.03)
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.batchnorm(self.cnn(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c_out, c_in, res_connection, e=0.5, k=(3, 3)):
        super().__init__()
        c_ = int(c_out * e)  # hidden channels
        if isinstance(k, int):
            k = (k, k)
        self.conv1 = Conv(c_, c_in, k[0], 1, k[0]//2)
        self.conv2 = Conv(c_out, c_, k[1], 1, k[1]//2)
        self.res_connection = res_connection and c_out == c_in
        
    def forward(self, x):
        if self.res_connection:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class C3k(nn.Module):
    """C3k module, a CSP bottleneck with 3 convolutions"""
    def __init__(self, c_out, c_in, n=1, shortcut=True, e=0.5, k=3):
        super().__init__()
        c_ = int(c_out * e)  # hidden channels
        self.conv1 = Conv(c_, c_in, 1, 1, 0)
        self.conv2 = Conv(c_, c_in, 1, 1, 0)
        self.conv3 = Conv(c_out, 2 * c_, 1, 1, 0)
        
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c_, c_, shortcut, e=1.0, k=k)
            for _ in range(n)
        ])
        
    def forward(self, x):
        return self.conv3(torch.cat((self.bottlenecks(self.conv1(x)), self.conv2(x)), dim=1))


class C3k2(nn.Module):
    """
    C3k2 module (CSP Bottleneck with 2 convolutions)
    Adapts structure based on c3k flag:
    - c3k=True: Uses C3k blocks internally (for m/l/x models)
    - c3k=False: Uses Bottleneck blocks internally (for n/s models)
    """
    def __init__(self, c_out, c_in, n=1, c3k=False, e=0.5, shortcut=False):
        super().__init__()
        self.c_ = int(c_out * e)  # hidden channels
        # Input splits into two branches (hence 2 * c_)
        self.conv1 = Conv(2 * self.c_, c_in, 1, 1, 0)
        # Output concatenates 2 branches + n bottleneck outputs (hence (2 + n) * c_)
        self.conv2 = Conv(c_out, (2 + n) * self.c_, 1, 1, 0)
        
        self.bottlenecks = nn.ModuleList([
            C3k(self.c_, self.c_, 2, shortcut) if c3k 
            else Bottleneck(self.c_, self.c_, shortcut, e=0.5)
            for _ in range(n)
        ])
        
    def forward(self, x):
        # Split input into two branches
        y = list(self.conv1(x).chunk(2, 1))
        # Extend with bottleneck outputs
        y.extend(m(y[-1]) for m in self.bottlenecks)
        # Concatenate all and project
        return self.conv2(torch.cat(y, 1))


class Attention(nn.Module):
    """Multi-head attention module for C2PSA"""
    def __init__(self, c, num_heads=None, attn_ratio=0.5):
        super().__init__()
        # Calculate num_heads based on channel count (same as original)
        if num_heads is None:
            num_heads = max(1, c // 64)
        
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        # QKV projection: outputs dim + 2*key_dim*num_heads
        # Split: Q (key_dim per head), K (key_dim per head), V (head_dim per head)
        # Note: All attention convs have NO activation
        nh_kd = self.key_dim * num_heads
        h = c + nh_kd * 2  # V channels + Q channels + K channels
        self.qkv = Conv(h, c, 1, 1, 0, act=False)
        self.proj = Conv(c, c, 1, 1, 0, act=False)
        # Depthwise conv for positional encoding
        self.pe = Conv(c, c, 3, 1, 1, g=c, act=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # QKV projection
        qkv = self.qkv(x)
        
        # Reshape and split into Q, K, V per head
        # Shape: (B, num_heads, key_dim*2 + head_dim, N)
        qkv = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        
        # Attention: (B, heads, key_dim, N) @ (B, heads, key_dim, N).T -> (B, heads, N, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to V: (B, heads, head_dim, N) @ (B, heads, N, N).T -> (B, heads, head_dim, N)
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        
        # Add positional encoding and project
        out = out + self.pe(v.reshape(B, C, H, W))
        return self.proj(out)


class PSABlock(nn.Module):
    """PSA Block with attention and FFN"""
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c)
        self.ffn = nn.Sequential(
            Conv(c * 2, c, 1, 1, 0),           # expand with activation
            Conv(c, c * 2, 1, 1, 0, act=False)  # contract without activation
        )
        
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2PSA module with attention for YOLO11"""
    def __init__(self, c_out, c_in, n=1):
        super().__init__()
        self.c_ = c_out // 2
        self.conv1 = Conv(c_out, c_in, 1, 1, 0)
        self.conv2 = Conv(c_out, c_out, 1, 1, 0)
        
        self.bottlenecks = nn.ModuleList([
            PSABlock(self.c_)
            for _ in range(n)
        ])
        
    def forward(self, x):
        x = self.conv1(x)
        a, b = x.chunk(2, dim=1)
        
        for bn in self.bottlenecks:
            b = bn(b)
        
        x = torch.cat([a, b], dim=1)
        return self.conv2(x)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast"""
    def __init__(self, c_out, c_in):
        super().__init__()
        c_ = c_out // 2
        self.conv1 = Conv(c_, c_in, 1, 1, 0)
        self.maxpool = nn.MaxPool2d(5, 1, 2)
        self.conv2 = Conv(c_out, c_ * 4, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        return self.conv2(torch.cat((x, x1, x2, x3), 1))


class Backbone(nn.Module):
    """YOLO11 Backbone"""
    def __init__(self, config):
        super().__init__()
        
        # YOLO11 configurations (r = ratio for P5 scale channels)
        self.configuration = {
            'n': {'d': 0.50, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.50, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.50, 'w': 1.00, 'r': 1.0},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.50, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']
        
        # c3k_setting: True for m/l/x, False for n/s
        c3k_setting = config in ('m', 'l', 'x')
                
        # Stem
        self.p1 = Conv(int(64*w), 3, 3, 2, 1)
        self.p2 = Conv(int(128*w), int(64*w), 3, 2, 1)
        
        # Stage 1-2: C3k2 with e=0.25 (all models), c3k varies by model size
        self.c2f1 = C3k2(int(256*w), int(128*w), n=max(1, int(round(2*d))), c3k=c3k_setting, e=0.25, shortcut=True)
        self.p3 = Conv(int(256*w), int(256*w), 3, 2, 1)
        self.c2f2 = C3k2(int(512*w), int(256*w), n=max(1, int(round(2*d))), c3k=c3k_setting, e=0.25, shortcut=True)
        
        # Stage 3-4: C3k2 with c3k=True always, e=0.5
        self.p4 = Conv(int(512*w), int(512*w), 3, 2, 1)
        self.c2f3 = C3k2(int(512*w), int(512*w), n=max(1, int(round(2*d))), c3k=True, e=0.5, shortcut=True)
        
        self.p5 = Conv(int(512*w*r), int(512*w), 3, 2, 1)
        self.c2f4 = C3k2(int(512*w*r), int(512*w*r), n=max(1, int(round(2*d))), c3k=True, e=0.5, shortcut=True)
        
        # SPPF + C2PSA
        self.sppf = SPPF(int(512*w*r), int(512*w*r))
        self.c2psa = C2PSA(int(512*w*r), int(512*w*r), max(1, int(round(2*d))))
        
    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.c2f1(x)
        x = self.p3(x)
        x = self.c2f2(x)
        x8 = x.clone()  # P3
        
        x = self.p4(x)
        x = self.c2f3(x)
        x16 = x.clone()  # P4
        
        x = self.p5(x)
        x = self.c2f4(x)
        x = self.sppf(x)
        x32 = self.c2psa(x)  # P5
        
        return x8, x16, x32


class Neck(nn.Module):
    """YOLO11 Neck (PANet)"""
    def __init__(self, config):
        super().__init__()
        
        self.configuration = {
            'n': {'d': 0.50, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.50, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.50, 'w': 1.00, 'r': 1.0},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.50, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']
        
        # c3k_setting: True for m/l/x, False for n/s
        c3k_setting = config in ('m', 'l', 'x')

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')        
        # All neck C3k2 blocks use shortcut=True
        self.c2f21 = C3k2(int(512*w), int(512*w*(1+r)), n=max(1, int(round(2*d))), c3k=c3k_setting, e=0.5, shortcut=True)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f11 = C3k2(int(256*w), int(512*w + 512*w), n=max(1, int(round(2*d))), c3k=c3k_setting, e=0.5, shortcut=True)    
        
        self.conv1 = Conv(int(256*w), int(256*w), 3, 2, 1)
        self.c2f12 = C3k2(int(512*w), int(256*w + 512*w), n=max(1, int(round(2*d))), c3k=c3k_setting, e=0.5, shortcut=True)
        
        self.conv2 = Conv(int(512*w), int(512*w), 3, 2, 1)
        # Last C2F block: c3k=True always
        self.c2f22 = C3k2(int(512*w*r), int(512*w + 512*w*r), n=max(1, int(round(2*d))), c3k=True, e=0.5, shortcut=True)
        
    def forward(self, x8, x16, x32):
        aux1 = self.upsample1(x32)
        x16 = torch.cat((aux1, x16), dim=1) 
        x16 = self.c2f21(x16)
        
        aux2 = self.upsample2(x16)
        x8 = torch.cat((aux2, x8), dim=1)   
        x8 = self.c2f11(x8)
        
        aux3 = self.conv1(x8)
        x16 = torch.cat((aux3, x16), dim=1) 
        x16 = self.c2f12(x16)
        
        aux4 = self.conv2(x16)
        x32 = torch.cat((aux4, x32), dim=1) 
        x32 = self.c2f22(x32)
        
        return x8, x16, x32


class Head(nn.Module):
    """YOLO11 Detection Head with depthwise separable convs"""
    def __init__(self, c_in, c_box, c_cls, reg_max, nb_classes):
        super().__init__()
        
        # Box branch: standard convs
        self.conv11 = Conv(c_box, c_in, 3, 1, 1)
        self.conv12 = Conv(c_box, c_box, 3, 1, 1)
        
        # Class branch: depthwise separable convs
        self.conv21 = nn.Sequential(
            Conv(c_in, c_in, 3, 1, 1, g=c_in),  # depthwise
            Conv(c_cls, c_in, 1, 1, 0),  # pointwise
        )
        self.conv22 = nn.Sequential(
            Conv(c_cls, c_cls, 3, 1, 1, g=c_cls),  # depthwise
            Conv(c_cls, c_cls, 1, 1, 0),  # pointwise
        )
        
        self.cnn1 = nn.Conv2d(c_box, 4*reg_max, 1, 1, 0)
        self.cnn2 = nn.Conv2d(c_cls, nb_classes, 1, 1, 0)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in [self.cnn1, self.cnn2]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

        pi = 0.01
        bias_val = -math.log((1 - pi) / pi)
        nn.init.constant_(self.cnn2.bias, bias_val)
        
    def forward(self, x):
        box = self.cnn1(self.conv12(self.conv11(x)))
        cls = self.cnn2(self.conv22(self.conv21(x)))
        return box, cls


class LibreYOLO11Model(nn.Module):
    """Main Libre YOLO11 model"""
    def __init__(self, config='n', reg_max=16, nb_classes=80, img_size=640):
        super().__init__()
        
        self.nc = nb_classes
        self.img_size = img_size
        self.configuration = {
            'n': {'d': 0.50, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.50, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.50, 'w': 1.00, 'r': 1.0},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.50, 'r': 1.0}
        }

        w = self.configuration[config]['w']
        r = self.configuration[config]['r']
        
        self.backbone = Backbone(config)
        self.neck = Neck(config)
        
        # Head configuration
        c_p3 = int(256 * w)
        c_box = max(64, c_p3 // 4)
        c_cls = max(c_p3, min(nb_classes, 100))
        
        self.head8 = Head(int(256*w), c_box, c_cls, reg_max, nb_classes)
        self.head16 = Head(int(512*w), c_box, c_cls, reg_max, nb_classes)
        self.head32 = Head(int(512*w*r), c_box, c_cls, reg_max, nb_classes)
        
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
        
        return {
            'x8': {'box': decoded_box8, 'cls': cls8, 'raw_box': box8},
            'x16': {'box': decoded_box16, 'cls': cls16, 'raw_box': box16},
            'x32': {'box': decoded_box32, 'cls': cls32, 'raw_box': box32}
        }
