# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from contextlib import nullcontext
if os.environ.get("ULTRALYTICS_DISABLE_MMCV", "0") == "1":
    MMCVDeformConv2dPack = None
    MMCVModulatedDeformConv2dPack = None
else:
    try:
        from mmcv.ops import DeformConv2dPack as MMCVDeformConv2dPack
        from mmcv.ops import ModulatedDeformConv2dPack as MMCVModulatedDeformConv2dPack
    except Exception:
        MMCVDeformConv2dPack = None
        MMCVModulatedDeformConv2dPack = None

# Optional local DCNv2 backend (DCNv2/_ext) to avoid mmcv dependency.
_DCNV2_CONV = None
if os.environ.get("ULTRALYTICS_USE_LOCAL_DCNV2", "0") == "1":
    _dcnv2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "DCNv2"))
    if os.path.isdir(_dcnv2_root) and _dcnv2_root not in sys.path:
        sys.path.insert(0, _dcnv2_root)
    try:
        from dcn_v2 import dcn_v2_conv as _DCNV2_CONV
    except Exception:
        _DCNV2_CONV = None
try:
    from torchvision.ops import DeformConv2d
    from torchvision.ops import deform_conv2d as tv_deform_conv2d
except ImportError:
    DeformConv2d = None
    tv_deform_conv2d = None
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from ultralytics.nn.backbone.Unet_1 import UNet
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock



__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C3k2_CKA",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "UNetV1",
    "DWBConv",
    "NonLocalBlockND",
    "DWBConv_02",
    "CKASnet",
    "CKAnet",
    "CKACAnet",
    "CKAPEnet",
    "CKASEnet",
    "CKACA_SPDnet",
    "SPDConv",
    "CKACAnetv2",
    "CKASAPE",
    "CKAS_IM",
    "CKAS_DAF",
    "CKAS_DAF_2",
    "CKAS_PConv",
    "DAFusion",
    "CKASAPE_75",
    "CKASAPE_dili",
    "CKASAPE_d",
    "CKASAPE_dili_v2",
    "CKAS_IM_v3",
    "CKAS_IM_v2",
    "CKAS_IM_IP",
    "CKAS_59",
    "CKAS_DAF_V2",
    "C2PSA_1",
    "C2PSA_1_CondConv",
    "C2PSA_1_DCNv1",
    "C2PSA_1_DCNv2",
    "C2PSA_1_ODConv",
    "C2PSA_1_DyConv",
    "C2PSA_1_DSConv",
    "C2PSA_1_Conv",
    "CKABS",
    "C2PSA_2",
    "C2PSA_4",
    "C3k2_CKA_2",
    "C2PSA_3",
    "C3k2_CKA_3",
    "CKAS_DAF_3x3",
    "CKAS_DAF_5x5",
    "CKAS_DAF_7x7",
    "CKAS_DAF_3_5",
    "CKAS_DAF_3_7",
    "C3k2_CKA_4",
    "MSWAC",
    "MSWAC_CondConv",
    "MSWAC_DCNv1",
    "MSWAC_DCNv2",
    "MSWAC_ODConv",
    "MSWAC_DyConv",
    "MSWAC_DSConv",
    "MSWAC_Conv",
    "CKAS_DAF_dilation"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))




class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class Bottleneck_CKA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = CKAConv(c_,c2)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # if self.add:
        #     print(x.shape)
        #     r = self.cv1(x)
        #     print(r.shape)
        #     r = self.cv2(r)
        #     print(r.shape)
        #     r = r + x
        # else:
        #     r = self.cv2(self.cv1(x))
        # return r
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
    
class C2f_CKA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CKA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        
class C3k_CKA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_CKA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_CKA(C2f_CKA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_CKA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CKA(self.c, self.c, shortcut, g) for _ in range(n)
        )

# CKA_2

class Bottleneck_CKA_2(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = CKAConv_2(c_,c2)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # if self.add:
        #     print(x.shape)
        #     r = self.cv1(x)
        #     print(r.shape)
        #     r = self.cv2(r)
        #     print(r.shape)
        #     r = r + x
        # else:
        #     r = self.cv2(self.cv1(x))
        # return r
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C2f_CKA_2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CKA_2(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k_CKA_2(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_CKA_2(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_CKA_2(C2f_CKA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_CKA_2(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CKA_2(self.c, self.c, shortcut, g) for _ in range(n)
        )


# CKA_3

class Bottleneck_CKA_3(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = CKAConv_3(c_,c2)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # if self.add:
        #     print(x.shape)
        #     r = self.cv1(x)
        #     print(r.shape)
        #     r = self.cv2(r)
        #     print(r.shape)
        #     r = r + x
        # else:
        #     r = self.cv2(self.cv1(x))
        # return r
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C2f_CKA_3(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CKA_3(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3_CKA_3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_CKA_3(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k_CKA_3(C3_CKA_3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_CKA_3(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_CKA_3(C2f_CKA_3):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_CKA_3(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CKA_3(self.c, self.c, shortcut, g) for _ in range(n)
        )



# CKA_4
class Bottleneck_CKA_4(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = CKAConv(c_,c2)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # if self.add:
        #     print(x.shape)
        #     r = self.cv1(x)
        #     print(r.shape)
        #     r = self.cv2(r)
        #     print(r.shape)
        #     r = r + x
        # else:
        #     r = self.cv2(self.cv1(x))
        # return r
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C2f_CKA_4(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CKA_4(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3_CKA_4(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_CKA_4(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k_CKA_4(C3_CKA_4):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_CKA_4(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_CKA_4(C2f_CKA_4):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_CKA_4(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CKA_4(self.c, self.c, shortcut, g) for _ in range(n)
        )


# -----------------------------------------

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))

class UNetV1(nn.Module):
    def __init__(self, slice):
        super(UNetV1, self).__init__()
        self.model = None
        if slice == 0:
            self.model = UNet().features[:4]
        elif slice == 1:
            self.model = UNet().features[4:6]
        elif slice == 2:
            self.model = UNet().features[6:8]
        elif slice == 3:
            self.model = UNet().features[8:]

    def forward(self, x):
        return self.model(x)


class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, p=1,  square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
 
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
 
        self.Conv = Conv(in_channels, out_channels, square_kernel_size,p)
 
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
        return self.Conv(x)

class ConvCrossAttention(nn.Module):
    def __init__(self, dims):
        super().__init__()
        
        # d_in 是输入特征维度，d_out_kq 是查询和键的输出维度，d_out_v 是值的输出维度
        self.d_out_kq = dims
        self.W_query = nn.Parameter(torch.rand(dims, dims))  # 查询权重
        self.W_key = nn.Parameter(torch.rand(dims, dims))  # 键权重
        self.W_value = nn.Parameter(torch.rand(dims, dims))  # 值权重

    def forward(self, x1, x2):

        b, c, h, w = x1.shape
        
        x1 = x1.view(b, c, h*w)
        x2 = x2.view(b, c, h*w)

        queries_1 = torch.matmul(x1, self.W_query)  
        keys_2 = torch.matmul(x2, self.W_key)  
        values_2 = torch.matmul(x2, self.W_value) 
 

        attn_scores = torch.matmul(queries_1, keys_2.transpose(-2, -1))  # (b, a, a)

        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)  # (b, a, a)

        context_vec = torch.matmul(attn_weights, values_2) 

        # context_vec = context_vec.permute(2, 1, 0).contiguous()
        
        context_vec = context_vec.view(b, c, h, w)

        return context_vec

class ConvCrossAttention_conv(nn.Module):
    def __init__(self, dims):
        super().__init__()
        
        # d_in 是输入特征维度，d_out_kq 是查询和键的输出维度，d_out_v 是值的输出维度
        self.d_out_kq = dims
        self.W_query = nn.Parameter(torch.rand(dims, dims))  # 查询权重
        self.W_key = nn.Parameter(torch.rand(dims, dims))  # 键权重
        self.W_value = nn.Parameter(torch.rand(dims, dims))  # 值权重

    def forward(self, x1, x2):

        b, c, h, w = x1.shape
        
        x1 = x1.view(b, c, h*w).permute(0, 2, 1)
        x2 = x2.view(b, c, h*w).permute(0, 2, 1)

        queries_1 = torch.matmul(x1, self.W_query)  
        keys_2 = torch.matmul(x2, self.W_key)  
        values_2 = torch.matmul(x2, self.W_value) 
 

        attn_scores = torch.matmul(queries_1, keys_2.transpose(-2, -1))  # (b, a, a)

        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)  # (b, a, a)

        context_vec = torch.matmul(attn_weights, values_2) 

        # context_vec = context_vec.permute(2, 1, 0).contiguous()
        
        context_vec = context_vec.view(b, c, h, w)

        return context_vec
class CKASEnet(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
        #     nn.ReLU(),  # 激活函数
        #     nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
        #     nn.ReLU(),
        #     nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
        #     nn.ReLU()
        # )
        # self.conv7 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        # self.w_attention_7 = ConvWeightAttention_PE(dims, num_heads=1,kernel_size=3,kernel_size_pe=7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )
        
        self.conv4 = Conv(dims, c2, k,s)
        self.sed = SED()    

    def forward(self, x):
        # r3 = self.conv2(x)
        r3 = self.conv2(x)
        r3 = self.sed(r3) + x
        
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)
        # w7 = self.se(W7)
        
        w5 = F.interpolate(w5, size=7, mode="bilinear", align_corners=False)
        w7 = w5 + w7

        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        r = torch.cat([r3, r7], dim=1)

        r_end = self.conv(r)
        c = self.conv4(r_end)
        
        # return self.conv(r)
        return c   


class CKAPEnet(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
        #     nn.ReLU(),  # 激活函数
        #     nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
        #     nn.ReLU(),
        #     nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
        #     nn.ReLU()
        # )
        # self.conv7 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)
        self.w_attention = ConvWeightAttention_PE(dims, num_heads=1,kernel_size=3,kernel_size_pe=7)
        # self.w_attention_7 = ConvWeightAttention_PE(dims, num_heads=1,kernel_size=3,kernel_size_pe=7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )
        
        self.conv4 = Conv(dims, c2, k,s)    

    def forward(self, x):
        # r3 = self.conv2(x)
        r3 = self.conv2(x) + x

        w = F.interpolate(self.conv2.weight.data, size=3, mode="bilinear", align_corners=False)
        w = self.w_attention(w)

        r7 = F.conv2d(x, w, padding=3, groups=self.dims) + x
        r = torch.cat([r3, r7], dim=1)
        # print(r7.shape)
        # print(r3.shape)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end    

class ConvWeightAttention_PE(nn.Module):
    def __init__(self, dims, num_heads, kernel_size, kernel_size_pe):
        super().__init__()
        embed_dim = 1
        self.num_heads = num_heads
        self.head_dim = head_dim = dims // num_heads
        all_head_dim = head_dim * self.num_heads
        kernel_size_proj = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.pe = nn.Conv2d(dims, dims, kernel_size=kernel_size_pe, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.pe = nn.Parameter(self.pe.weight.data)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim, kernel_size_pe, kernel_size_pe])
        self.se = SEAtt(dims)
    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        weight_7 = F.interpolate(weight, size=7, mode="bilinear", align_corners=False)
        pe = torch.matmul(weight_7, self.pe)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        res = F.interpolate(res, size=7, mode="bilinear", align_corners=False)
        res = res + pe
        res = self.se(res)
        weight = F.interpolate(weight, size=7, mode="bilinear", align_corners=False)
        return self.ln(res + weight)

class CKACA_SPDnet(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv3 = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
            nn.ReLU(),  # 激活函数
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
            nn.ReLU(),
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
            nn.ReLU()
        )
        # self.conv7 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, 5)
        self.w_attention_7 = ConvWeightAttention_CKAS(dims, 7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )
        self.spd = SPDConv(inc=dims,k1=3,s1=1)
        self.ca = ConvCrossAttention(49)
        # self.conv4 = InceptionDWConv2d(dims,c2)
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):
        r3 = self.conv3(x)
        # r3 = self.spd(r3) + x
    
        r3 = self.conv3(x) + x 

        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)
        # w7 = self.se(W7)
        
        w5 = F.interpolate(w5, size=7, mode="bilinear", align_corners=False)
        w7 = self.ca(w7,w5)

        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        r = torch.cat([r3, r7], dim=1)

        r_end = self.conv(r)
        c = self.conv4(r_end)
        return c   


class CKACAnetv2(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_5 = ConvWeightAttention_CKAS(dims, 5)
        self.w_attention_7 = ConvWeightAttention_CKAS(dims, 7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        self.post_process = nn.Sequential(
            nn.Conv2d(2*dims, dims, 1),  # Concat后通道数变为3倍
            nn.BatchNorm2d(dims),
            nn.SiLU(),
            nn.Conv2d(dims, dims, 1),
            nn.BatchNorm2d(dims),
            nn.SiLU()
        )
        self.se = SEAtt(dims)
        self.conv4 = Conv(dims, c2, k,s)
        print(CKACAnetv2)
    def forward(self, x):

        
        r3 = self.conv2(x) + x 

        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)
        w5 = self.se(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)
        w7 = self.se(w7)

        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) 
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims)

        r4 = torch.cat([r5, r7], dim=1)
        r4 = self.post_process(r4) + x

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        c = self.conv4(r_end)
        
        return c 


class Fusion_3(nn.Module):
    def __init__(self,  ouc) -> None:
        super().__init__()
        
        # self.conv_align1 = Conv(inc, ouc, 1)
        # self.conv_align2 = Conv(inc, ouc, 1)
        
        # self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        # self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x3_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)
        
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        
        x1, x2, x3 = x

        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param + x3 *self.x3_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign
            
class Fusion_75(nn.Module):
    def __init__(self,  ouc) -> None:
        super().__init__()
        
        # self.conv_align1 = Conv(inc, ouc, 1)
        # self.conv_align2 = Conv(inc, ouc, 1)
        
        # self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        # self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)
        
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        
        x1, x2 = x

        
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign

class DAFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.conv_align1 = Conv(inc, ouc, 1)
        self.conv_align2 = Conv(inc, ouc, 1)
        
        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)
        
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        
        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight
        
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign

class DAFusion_3(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.conv_align1 = Conv(inc, ouc, 1)
        self.conv_align2 = Conv(inc, ouc, 1)
        self.conv_align3 = Conv(inc, ouc, 1)
        
        self.conv_concat = Conv(ouc * 3, ouc * 3, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x3_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)
        
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        self._clamp_abs(self.x3_param.data, 1.0)
        
        x1, x2, x3 = x
        x1, x2, x3 = self.conv_align1(x1), self.conv_align2(x2), self.conv_align3(x3)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2, x3], dim=1)))
        x1_weight, x2_weight, x3_weight = torch.chunk(x_concat, 3, dim=1)
        x1, x2, x3 = x1 * x1_weight, x2 * x2_weight, x3 * x3_weight
        
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param + x3 * self.x3_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign


class Triad_Weighted_Fusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        scale = 4
        group = 2
        self.conv_align1 = Conv(inc, ouc, 1)
        self.conv_align2 = Conv(inc, ouc, 1)
        
        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * inc),
            nn.Conv2d(group * inc, scale * inc, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale *inc,ouc, kernel_size=1)
        )
        
    def forward(self, r5,r7,r3):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        
        x1, x2 = self.conv_align1(r5), self.conv_align2(r7)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight
        
        x = self.conv_final(x1 * self.x1_param + x2 * self.x2_param)
        r = torch.cat[x,r3]
        r = self.conv(r)
        return r
    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign

class DynamicAlignFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.conv_align1 = Conv(inc, ouc, 1)
        self.conv_align2 = Conv(inc, ouc, 1)
        
        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)
        
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        
        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight
        
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign
    
class CKAS_PConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k,s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        # self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        # self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c1*4, c1, 2, s=1, p=0)
        self.convw = nn.Conv2d(c1, c1, (1,k), stride=1, padding=0, groups=c1)
        self.convh = nn.Conv2d(c1, c1, (k,1), stride=1, padding=0, groups=c1)
        self.w = ConvWeightAttention_PC(c1, k)
        self.h = ConvWeightAttention_PC(c1, k)
        self.c1 = c1
        self.conv4 = Conv(c1, c2, k,s)
    def forward(self, x):
        ww = self.convw.weight.data
        wh = self.convh.weight.data
        ww = self.w(ww)
        wh = self.w(wh)
        yw0 = F.conv2d(self.pad[0](x), ww, padding=0, groups=self.c1)
        yw1 = F.conv2d(self.pad[1](x), ww, padding=0, groups=self.c1)
        yh0 = F.conv2d(self.pad[2](x), wh, padding=0, groups=self.c1)
        yh1 = F.conv2d(self.pad[3](x), wh, padding=0,groups=self.c1)
        # yw0 = self.cw(self.pad[0](x))
        # yw1 = self.cw(self.pad[1](x))
        # yh0 = self.ch(self.pad[2](x))
        # yh1 = self.ch(self.pad[3](x))
        r_end = self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
        # r_end = self.conv4(r_end)
        return r_end

class ConvWeightAttention_PC(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.ln_1 = nn.LayerNorm([embed_dim, kernel_size, 1])
        self.ln_2 = nn.LayerNorm([embed_dim, 1, kernel_size])

    def forward(self, weight):
        b, c, h, w = weight.shape
        if w > h:
            q = torch.matmul(weight, self.qw)
            k = torch.matmul(weight, self.kw)
            v = torch.matmul(weight, self.vw)
        else:        
            q = torch.matmul(self.qw,weight)
            k = torch.matmul(self.kw,weight)
            v = torch.matmul(self.vw,weight)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        if w > h:
            res = self.ln_2(res + weight)
        else:
            res = self.ln_1(res + weight)
        # return self.ln(res + weight)
        return res


class CKAS_DAF_35(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)
        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
        # r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

        rl = [r5,r3]
        r4 = self.daf(rl)

        # r = torch.cat([r3, r4], dim=1)
        # r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_37(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        # self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)
        
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        # r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

        rl = [r7,r3]
        r4 = self.daf(rl)

        # r = torch.cat([r3, r4], dim=1)
        # r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_37(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        # self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)
        
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        # r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

        rl = [r7,r3]
        r4 = self.daf(rl)

        # r = torch.cat([r3, r4], dim=1)
        # r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 



class CKAS_DAF(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

        rl = [r5,r7]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 


class CKAS_DAF_dilation(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 2
        group = 1
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        # self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        # self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion_3(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        # w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        # w5 = self.w_attention_5(w5)

        # w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        # w7 = self.w_attention_7(w7)
        w3 = self.w_attention_3(self.conv2.weight.data)
        # r3 = F.conv2d(x, w3, padding=1, group=self.dims)
        # r3 = F.conv2d(x, w3, padding=1, groups=self.dims) + x
        r5 = F.conv2d(x, w3, padding=2, groups=self.dims, dilation=2)
        r7 = F.conv2d(x, w3, padding=3, groups=self.dims, dilation=3)
        # print(r3.shape)
        rl = [r3,r5,r7]
        r = self.daf(rl)
        # print(r.shape)
        # r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CondConv2d(nn.Module):
    """Per-sample conditional convolution with expert kernels."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_experts=4,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.num_experts = num_experts
        self.in_channels_per_group = in_channels // groups

        kh, kw = self.kernel_size
        self.weight = nn.Parameter(
            torch.empty(num_experts, out_channels, self.in_channels_per_group, kh, kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_channels))
        else:
            self.register_parameter("bias", None)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.routing = nn.Linear(in_channels, num_experts)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels_per_group * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # Start from uniform routing; sigmoid(0)=0.5 for all experts.
        nn.init.zeros_(self.routing.weight)
        nn.init.zeros_(self.routing.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        routing_weights = torch.sigmoid(self.routing(self.pool(x).flatten(1)))  # (B, E)

        weight = torch.matmul(routing_weights, self.weight.view(self.num_experts, -1))
        weight = weight.view(
            b * self.out_channels,
            self.in_channels_per_group,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias).reshape(-1)

        x = x.reshape(1, b * c, h, w)
        y = F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=b * self.groups,
        )
        return y.view(b, self.out_channels, y.shape[-2], y.shape[-1])


class ODConvAttention(nn.Module):
    """Attention generator for ODConv2dLocal."""

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        return torch.sigmoid(spatial_attention / self.temperature)

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        return F.softmax(kernel_attention / self.temperature, dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2dLocal(nn.Module):
    """ODConv2d (embedded local implementation)."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        reduction=0.0625,
        kernel_num=4,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = ODConvAttention(
            in_planes, out_planes, kernel_size, groups=groups, reduction=reduction, kernel_num=kernel_num
        )
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True
        )
        self._initialize_weights()
        self._forward_impl = self._forward_impl_pw1x if (kernel_size == 1 and kernel_num == 1) else self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode="fan_out", nonlinearity="relu")

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, _, h, w = x.size()
        x = (x * channel_attention).reshape(1, -1, h, w)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            -1, self.in_planes // self.groups, self.kernel_size, self.kernel_size
        )
        output = F.conv2d(
            x,
            weight=aggregate_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * batch_size,
        )
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output * filter_attention

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, _, _ = self.attention(x)
        output = F.conv2d(
            x * channel_attention,
            weight=self.weight.squeeze(dim=0),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output * filter_attention

    def forward(self, x):
        return self._forward_impl(x)


class DyAttention2d(nn.Module):
    """Attention head for DynamicConv2dLocal."""

    def __init__(self, in_planes, ratio=0.25, kernel_num=4, temperature=34):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_planes = int(in_planes * ratio) + 1 if in_planes != 3 else kernel_num
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, kernel_num, 1, bias=True)
        self.temperature = temperature
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, dim=1)


class DynamicConv2dLocal(nn.Module):
    """Dynamic convolution with K kernels (local implementation)."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        ratio=0.25,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        kernel_num=4,
        temperature=34,
    ):
        super().__init__()
        if in_planes % groups != 0:
            raise ValueError("in_planes must be divisible by groups")
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = DyAttention2d(in_planes, ratio=ratio, kernel_num=kernel_num, temperature=temperature)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(kernel_num, out_planes)) if bias else None
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))

    def forward(self, x):
        softmax_attention = self.attention(x)  # (B, K)
        b, c, h, w = x.size()
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.kernel_num, -1)
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            b * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size
        )
        aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1) if self.bias is not None else None
        out = F.conv2d(
            x,
            weight=aggregate_weight,
            bias=aggregate_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * b,
        )
        return out.view(b, self.out_planes, out.size(-2), out.size(-1))


class MSWAC(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_SE(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention_SE(dims, kernel_size=5)
        # self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        # self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims)
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims)

        rl = [r5,r7]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class MSWAC_CondConv(nn.Module):
    """MSWAC ablation variant: replace r5/r7 dynamic kernels with CondConv branches."""

    def __init__(self, dims, c2, k, s, num_experts=4):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.condconv5 = CondConv2d(
            dims, dims, kernel_size=5, num_experts=num_experts, padding=2, groups=dims, bias=False
        )
        self.condconv7 = CondConv2d(
            dims, dims, kernel_size=7, num_experts=num_experts, padding=3, groups=dims, bias=False
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)

    def forward(self, x):
        r3 = self.conv2(x) + x
        r5 = self.condconv5(x)
        r7 = self.condconv7(x)

        r4 = self.daf([r5, r7])
        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)
        return r_end

class MSWAC_DCNv1(nn.Module):
    """MSWAC ablation variant: replace r5/r7 branches with DCNv1."""

    def __init__(self, dims, c2, k, s):
        super().__init__()
        if MMCVDeformConv2dPack is None and DeformConv2d is None:
            raise ImportError(
                "MSWAC_DCNv1 requires mmcv.ops.DeformConv2dPack or torchvision.ops.DeformConv2d, but neither is installed."
            )

        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        # Force torchvision backend to avoid mmcv CUDA extension mismatch/segfaults.
        self.use_mmcv = False
        self.offset5 = nn.Conv2d(dims, 2 * 5 * 5, kernel_size=3, padding=1)
        self.offset7 = nn.Conv2d(dims, 2 * 7 * 7, kernel_size=3, padding=1)
        self.dcn5 = DeformConv2d(dims, dims, kernel_size=5, padding=2, bias=False)
        self.dcn7 = DeformConv2d(dims, dims, kernel_size=7, padding=3, bias=False)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)

    def forward(self, x):
        # Keep the DCN branch in fp32 and disable autocast to avoid unstable CUDA kernels.
        input_dtype = x.dtype
        amp_off = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()
        with amp_off:
            x = x.float().contiguous()
            r3 = self.conv2(x) + x
            if x.is_cuda:
                if self.use_mmcv:
                    r5 = self.dcn5(x)
                    r7 = self.dcn7(x)
                else:
                    r5 = self.dcn5(x, self.offset5(x).contiguous())
                    r7 = self.dcn7(x, self.offset7(x).contiguous())
            else:
                # CPU fallback (used during stride inference)
                r5 = F.conv2d(x, self.dcn5.weight, None, stride=1, padding=2)
                r7 = F.conv2d(x, self.dcn7.weight, None, stride=1, padding=3)

            r4 = self.daf([r5, r7])
            r = torch.cat([r3, r4], dim=1)
            r_end = self.conv(r)
            r_end = self.conv4(r_end)
        return r_end.to(input_dtype)

class ModulatedDeformConv2d(nn.Module):
    """DCNv2-style modulated deformable convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        if _DCNV2_CONV is None and tv_deform_conv2d is None:
            raise ImportError("ModulatedDeformConv2d requires torchvision.ops.deform_conv2d, but torchvision is not installed.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        kh, kw = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, offset, mask):
        if not x.is_cuda:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        if _DCNV2_CONV is not None:
            bias = self.bias if self.bias is not None else x.new_zeros(self.out_channels)
            return _DCNV2_CONV(
                x,
                offset,
                mask,
                self.weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                1,  # deformable_groups
            )
        return tv_deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class MSWAC_DCNv2(nn.Module):
    """MSWAC ablation variant: replace r5/r7 branches with DCNv2."""

    def __init__(self, dims, c2, k, s):
        super().__init__()
        if MMCVModulatedDeformConv2dPack is None and tv_deform_conv2d is None:
            raise ImportError(
                "MSWAC_DCNv2 requires mmcv.ops.ModulatedDeformConv2dPack or torchvision.ops.deform_conv2d, but neither is installed."
            )

        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        # Force torchvision backend to avoid mmcv CUDA extension mismatch/segfaults.
        self.use_mmcv = False
        self.offset5 = nn.Conv2d(dims, 2 * 5 * 5, kernel_size=3, padding=1)
        self.offset7 = nn.Conv2d(dims, 2 * 7 * 7, kernel_size=3, padding=1)
        self.mask5 = nn.Sequential(nn.Conv2d(dims, 5 * 5, kernel_size=3, padding=1), nn.Sigmoid())
        self.mask7 = nn.Sequential(nn.Conv2d(dims, 7 * 7, kernel_size=3, padding=1), nn.Sigmoid())
        self.dcn5 = ModulatedDeformConv2d(dims, dims, kernel_size=5, padding=2, bias=False)
        self.dcn7 = ModulatedDeformConv2d(dims, dims, kernel_size=7, padding=3, bias=False)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)

    def forward(self, x):
        # Keep the DCN branch in fp32 and disable autocast to avoid unstable CUDA kernels.
        input_dtype = x.dtype
        amp_off = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()
        with amp_off:
            x = x.float().contiguous()
            r3 = self.conv2(x) + x
            if self.use_mmcv and x.is_cuda:
                r5 = self.dcn5(x)
                r7 = self.dcn7(x)
            else:
                r5 = self.dcn5(x, self.offset5(x).contiguous(), self.mask5(x).contiguous())
                r7 = self.dcn7(x, self.offset7(x).contiguous(), self.mask7(x).contiguous())

            r4 = self.daf([r5, r7])
            r = torch.cat([r3, r4], dim=1)
            r_end = self.conv(r)
            r_end = self.conv4(r_end)
        return r_end.to(input_dtype)


class MSWAC_ODConv(nn.Module):
    """MSWAC ablation variant: replace r5/r7 branches with ODConv."""

    def __init__(self, dims, c2, k, s, kernel_num=4):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.odconv5 = ODConv2dLocal(
            dims, dims, kernel_size=5, padding=2, groups=dims, reduction=0.0625, kernel_num=kernel_num
        )
        self.odconv7 = ODConv2dLocal(
            dims, dims, kernel_size=7, padding=3, groups=dims, reduction=0.0625, kernel_num=kernel_num
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)

    def forward(self, x):
        r3 = self.conv2(x) + x
        r5 = self.odconv5(x)
        r7 = self.odconv7(x)

        r4 = self.daf([r5, r7])
        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)
        return r_end


class MSWAC_DyConv(nn.Module):
    """MSWAC ablation variant: replace r5/r7 branches with DynamicConv."""

    def __init__(self, dims, c2, k, s, kernel_num=4):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.dyconv5 = DynamicConv2dLocal(
            dims, dims, kernel_size=5, stride=1, padding=2, groups=dims, bias=False, kernel_num=kernel_num
        )
        self.dyconv7 = DynamicConv2dLocal(
            dims, dims, kernel_size=7, stride=1, padding=3, groups=dims, bias=False, kernel_num=kernel_num
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)

    def forward(self, x):
        r3 = self.conv2(x) + x
        r5 = self.dyconv5(x)
        r7 = self.dyconv7(x)

        r4 = self.daf([r5, r7])
        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)
        return r_end


class MSWAC_DSConv(nn.Module):
    """MSWAC ablation variant: replace r5/r7 branches with standard depthwise separable conv."""

    def __init__(self, dims, c2, k, s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.dsconv5 = nn.Sequential(Conv(dims, dims, 5, 1, g=dims), Conv(dims, dims, 1, 1))
        self.dsconv7 = nn.Sequential(Conv(dims, dims, 7, 1, g=dims), Conv(dims, dims, 1, 1))
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)

    def forward(self, x):
        r3 = self.conv2(x) + x
        r5 = self.dsconv5(x)
        r7 = self.dsconv7(x)

        r4 = self.daf([r5, r7])
        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)
        return r_end


class MSWAC_Conv(nn.Module):
    """MSWAC ablation variant: replace r5/r7 branches with standard convolutions."""

    def __init__(self, dims, c2, k, s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv5 = Conv(dims, dims, 5, 1)
        self.conv7 = Conv(dims, dims, 7, 1)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)

    def forward(self, x):
        r3 = self.conv2(x) + x
        r5 = self.conv5(x)
        r7 = self.conv7(x)

        r4 = self.daf([r5, r7])
        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)
        return r_end

class CKAS_DAF_3x3(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        # self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        # w3 = F.interpolate(self.conv2.weight.data, size=3, mode="bilinear", align_corners=False)
        w3 = self.w_attention_3(self.conv2.weight.data)

        # w3_ = F.interpolate(self.conv2.weight.data, size=3, mode="bilinear", align_corners=False)
        # w7 = self.w_attention_3(w3)

        
        r3 = F.conv2d(x, w3, padding=1, groups=self.dims) + x
        # r7 = F.conv2d(x, w3, padding=1, groups=self.dims) + x

        rl = [r3,r3]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_5x5(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        # self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        # w3 = self.w_attention_3(self.conv2.weight.data)

        # w3_ = F.interpolate(self.conv2.weight.data, size=3, mode="bilinear", align_corners=False)
        # w7 = self.w_attention_3(w3)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
        # r7 = F.conv2d(x, w5, padding=2, groups=self.dims) + x

        rl = [r5,r5]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_7x7(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        # self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        # w3 = self.w_attention_3(self.conv2.weight.data)

        # w3_ = F.interpolate(self.conv2.weight.data, size=3, mode="bilinear", align_corners=False)
        # w7 = self.w_attention_3(w3)

        
        # r5 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

        rl = [r7,r7]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_3_5(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w3 = self.conv2.weight.data
        w3 = self.w_attention_3(w3)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
        r3 = F.conv2d(x, w3, padding=1, groups=self.dims) + x

        rl = [r5,r3]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_3_7(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)

        w3 = self.conv2.weight.data
        w3 = self.w_attention_3(w3)

        
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        r3 = F.conv2d(x, w3, padding=1, groups=self.dims) + x

        rl = [r3,r7]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_2(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKA(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention_CKA(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

        rl = [r5,r7]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 

class CKAS_DAF_V2(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        # self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.daf = DAFusion(dims,dims) 
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x 
        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

        rl = [r5,r7]
        r4 = self.daf(rl)
        # r4 = r4
        # r4 = torch.cat([r5, r7], dim=1)
        # r4 = self.conv1(r4) + x

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end 


# class CKAS_DAF_75(nn.Module):
#     def __init__(self, dims,c2,k,s):
#         super().__init__()
#         scale = 4
#         group = 2
#         self.dims = dims
#         self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

#         self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
#         self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
#         self.conv = nn.Sequential(
#             nn.BatchNorm2d(group * dims),
#             nn.Conv2d(group * dims, scale * dims, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(scale * dims, dims, kernel_size=1)
#         )

#         self.daf = DAFusion(dims,dims) 
#         self.conv4 = Conv(dims, c2, k,s)

#     def forward(self, x):

#         r3 = self.conv2(x) + x 
#         w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
#         w5 = self.w_attention_5(w5)

#         w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
#         w7 = self.w_attention_7(w7)

#         r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
#         r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x

#         rl = [r5,r7]
#         r4 = self.daf(rl)

#         r = torch.cat([r3, r4], dim=1)
#         r_end = self.conv(r)
#         r_end = self.conv4(r_end)

#         return r_end 

class CKAS_75_DAF(nn.Module):
    def __init__(self, dims,c2, k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv_s1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1,groups=dims, dilation=1)  # atrous=1
        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)

        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        # self.fusion = Fusion_3(dims)
        self.omega = nn.Parameter(torch.ones(1) * 0.5)  # 基础权重
        self.delta_omega1 = nn.Parameter(torch.zeros(1),requires_grad=True) 
        self.delta_omega2 = nn.Parameter(torch.zeros(1),requires_grad=True)  
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):
        # print(x.shape)
        # print(self.conv2)
        # print("75")
        r3 = self.conv2(x)

        w3 = self.conv_s1.weight.data
        w3 = self.w_attention_3(w3)
        r5 = F.conv2d(x, w3, padding=5, groups=self.dims,dilation=5)
        r7 = F.conv2d(x, w3, padding=9, groups=self.dims, dilation=9)
        r = r3* self.omega + r5 * (self.omega + self.delta_omega1) + r7 * (self.omega + self.delta_omega2)
        r_end = self.conv_G(r)
        c = self.conv4(r_end)
        return c

class CKACAnet(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv3 = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
            nn.ReLU(),  # 激活函数
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
            nn.ReLU(),
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims),
            nn.ReLU()
        )
        # self.conv7 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, 5)
        self.w_attention_7 = ConvWeightAttention_CKAS(dims, 7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )
        self.ca = ConvCrossAttention(49)
        # self.conv4 = InceptionDWConv2d(dims,c2)
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):
        # r3 = self.conv3(x)
        # r3 = self.spd(r3) + x
        
        r3 = self.conv3(x) + x 

        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)
        # w7 = self.se(W7)
        
        w5 = F.interpolate(w5, size=7, mode="bilinear", align_corners=False)
        w7 = self.ca(w7,w5)

        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        r = torch.cat([r3, r7], dim=1)

        r_end = self.conv(r)
        c = self.conv4(r_end)
        
        # return self.conv(r)
        return c   

class LN(nn.Module):
    def __init__(self, channels):
        super(LN, self).__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return x

# Feed-Forward Network (FFN)
class FFN(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super(FFN, self).__init__()
        hidden_channels = hidden_channels or 4 * in_channels  # 默认扩展 4 倍
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class ConvWeightAttention_APE_1(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.ape = nn.Parameter(torch.zeros(dims, kernel_size*kernel_size, embed_dim))
        torch.nn.init.trunc_normal_(self.ape.data, std=.02)  # 直接修改参数的数据，不重新赋值
    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0] + self.ape
        res = res.permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        res = self.ln(res + weight)
        return res


class ConvLNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True):
        super(ConvLNACT, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return self.act(x)
    
class CKASBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        # self.conv7 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)
        self.w_attention = ConvWeightAttention_APE_1(dims, 7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

    def forward(self, x):
        r3 = self.conv3(x) + x

        w7 = F.interpolate(self.conv3.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention(w7)
        # self.conv7.weight.data = w7
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        r = torch.cat([r3, r7], dim=1)
        r_end = self.conv(r)

        return r_end


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # 假设 x 的形状是 (batch_size, channels, height, width)
        mean = x.mean(dim=(-1, -2), keepdim=True)  # 计算 height 和 width 维度上的均值
        std = x.std(dim=(-1, -2), keepdim=True)    # 计算 height 和 width 维度上的标准差

        # 通过 unsqueeze 调整 weight 和 bias 的形状，确保广播
        weight = self.weight.view(1, self.channels, 1, 1)
        bias = self.bias.view(1, self.channels, 1, 1)

        return weight * (x - mean) / (std + self.eps) + bias


# class CKAS_IM_v3(nn.Module):
#     def __init__(self, in_channels, c2,k,s):
#         super().__init__()
#         self.in_channels = in_channels
#         # self.ln1 = LN(in_channels)
#         self.ln1 = DynamicTanh(in_channels,False)
#         self.ffn = FFN(in_channels)
#         # self.ln2 = LN(in_channels)
#         self.ln2 = DynamicTanh(in_channels,False)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
#         self.ckas_pc = CKAS_PConv(in_channels,in_channels,k,s)
#         self.ckas = CKASAPE_dili_v2(in_channels,c2,k,s)
#         self.conv4 = Conv(in_channels, c2, k,s)
#         # self.mlp = nn.Sequential(
#         #     nn.Conv2d(in_channels, c2, kernel_size=1),
#         #     nn.BatchNorm2d(c2),
#         #     nn.SiLU(inplace=True)
#         # )
#     def forward(self, x):
#         residual_1 = x
#         x = self.ckas_pc(x)
#         x = self.ckas(x)
#         x = self.ln1(x)
#         x = x + residual_1 
#         residual_2 = x
#         x = self.ffn(x)
#         x = self.ln1(x)
#         x = x + residual_2  # FFN 残差
#         x = self.conv4(x)
#         return x


class CKAS_IM_v3(nn.Module):
    def __init__(self, in_channels, c2,k,s):
        super().__init__()
        self.in_channels = in_channels
        # self.ln1 = LN(in_channels)
        self.ln1 = DynamicTanh(in_channels,False)
        self.ffn = FFN(in_channels)
        # self.ln2 = LN(in_channels)
        self.ln2 = DynamicTanh(in_channels,False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # self.ckas_pc = CKAS_PConv(in_channels,in_channels,k,s)
        self.ckas = CKASAPE_75(in_channels,c2,k,s)
        # self.cv1 = InceptionDWConv2d(in_channels,c2)
        self.conv4 = Conv(in_channels, c2, k,s)
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(in_channels, c2, kernel_size=1),
        #     nn.BatchNorm2d(c2),
        #     nn.SiLU(inplace=True)
        # )
    def forward(self, x):
        residual_1 = x
        # x = self.ckas_pc(x)
        x = self.ckas(x)
        x = self.ln1(x)
        x = x + residual_1 
        residual_2 = x
        x = self.ffn(x)
        x = self.ln1(x)
        x = x + residual_2  # FFN 残差
        x = self.conv4(x)
        # x = self.cv1(x)
        return x

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

# 基本块
# class CKAS_IM(nn.Module):
#     def __init__(self, in_channels, c2,k,s):
#         super().__init__()
#         self.in_channels = in_channels
#         # self.ln1 = LN(in_channels)
#         self.ln1 = LN(in_channels)
#         self.ffn = FFN(in_channels)
#         # self.ln2 = LN(in_channels)
#         self.ln2 = LN(in_channels)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
#         self.ckas = CKAS_PConv(in_channels,in_channels,k,s)
#         self.conv4 = Conv(in_channels, c2, k,s)
#         # self.mlp = nn.Sequential(
#         #     nn.Conv2d(in_channels, c2, kernel_size=1),
#         #     nn.BatchNorm2d(c2),
#         #     nn.SiLU(inplace=True)
#         # )
#     def forward(self, x):
#         residual_1 = x
#         x = self.ckas(x)
#         x = self.ln1(x)
#         x = x + residual_1 
#         residual_2 = x
#         x = self.ffn(x)
#         x = self.ln1(x)
#         x = x + residual_2  # FFN 残差
#         x = self.conv4(x)
#         return x

class CKAS_IM(nn.Module):
    def __init__(self, in_channels, c2,k,s):
        super().__init__()
        self.in_channels = in_channels
        # self.ln1 = LN(in_channels)
        self.ln1 = LN(in_channels)
        self.ffn = FFN(in_channels)
        # self.ln2 = LN(in_channels)
        self.ln2 = LN(in_channels)
        # self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.ckas = CKAS_PConv(in_channels,in_channels,k,s)
        # self.conv4 = Conv(in_channels, c2, k,s)
        self.cv1 = InceptionDWConv2d(in_channels,c2)
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(in_channels, c2, kernel_size=1),
        #     nn.BatchNorm2d(c2),
        #     nn.SiLU(inplace=True)
        # )
    def forward(self, x):
        residual_1 = x
        x = self.ckas(x)
        x = self.ln1(x)
        x = x + residual_1 
        residual_2 = x
        x = self.ffn(x)
        x = self.ln1(x)
        x = x + residual_2  # FFN 残差
        # x = self.conv4(x)
        x = self.cv1(x)
        return x

class CKAS_IM_IP(nn.Module):
    def __init__(self, in_channels, c2,k,s):
        super().__init__()
        self.in_channels = in_channels
        # self.ln1 = LN(in_channels)
        self.ln1 = LN(in_channels)
        self.ffn = FFN(in_channels)
        # self.ln2 = LN(in_channels)
        self.ln2 = LN(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.ckas = CKAS_PConv(in_channels,in_channels,k,s)
        # self.conv4 = Conv(in_channels, c2, k,s)
        self.cv1 = InceptionDWConv2d(in_channels,c2)
        
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(in_channels, c2, kernel_size=1),
        #     nn.BatchNorm2d(c2),
        #     nn.SiLU(inplace=True)
        # )
    def forward(self, x):
        residual_1 = x
        x = self.ckas(x)
        x = self.ln1(x)
        x = x + residual_1 
        residual_2 = x
        x = self.ffn(x)
        x = self.ln1(x)
        x = x + residual_2  # FFN 残差
        x = self.cv1(x)
        return x

class CKAS_IM_v2(nn.Module):
    def __init__(self, in_channels, c2,k,s):
        super().__init__()
        self.in_channels = in_channels
        # self.ln1 = LN(in_channels)
        self.ln1 = DynamicTanh(in_channels,False)
        self.ffn = FFN(in_channels)
        # self.ln2 = LN(in_channels)
        self.ln2 = DynamicTanh(in_channels,False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.ckas = CKAS_PConv(in_channels,in_channels,k,s)
        self.conv4 = Conv(in_channels, c2, k,s)
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(in_channels, c2, kernel_size=1),
        #     nn.BatchNorm2d(c2),
        #     nn.SiLU(inplace=True)
        # )
    def forward(self, x):
        residual_1 = x
        x = self.ckas(x)
        x = self.ln1(x)
        x = x + residual_1 
        residual_2 = x
        x = self.ffn(x)
        x = self.ln1(x)
        x = x + residual_2  # FFN 残差
        x = self.conv4(x)
        return x

class CKASnet(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        # self.conv7 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)
        self.w_attention = ConvWeightAttention_CKAS(dims, 7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):
        
        r3 = self.conv3(x) + x

        w7 = F.interpolate(self.conv3.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention(w7)
        # self.conv7.weight.data = w7

        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        # r7 = self.conv7(x)+ x
        r = torch.cat([r3, r7], dim=1)

        r_end = self.conv(r)
        c = self.conv4(r_end)
        
        # return self.conv(r)
        return c        

class CKASAPE_75(nn.Module):
    def __init__(self, dims,c2, k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv_s1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1,groups=dims, dilation=1)  # atrous=1
        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)

        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        # self.fusion = Fusion_3(dims)
        self.omega = nn.Parameter(torch.ones(1) * 0.5)  # 基础权重
        self.delta_omega1 = nn.Parameter(torch.zeros(1),requires_grad=True) 
        self.delta_omega2 = nn.Parameter(torch.zeros(1),requires_grad=True)  
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x)

        w3 = self.conv_s1.weight.data
        w3 = self.w_attention_3(w3)
        r5 = F.conv2d(x, w3, padding=5, groups=self.dims,dilation=5)
        r7 = F.conv2d(x, w3, padding=9, groups=self.dims, dilation=9)
        r = r3* self.omega + r5 * (self.omega + self.delta_omega1) + r7 * (self.omega + self.delta_omega2)
        r_end = self.conv_G(r)
        c = self.conv4(r_end)
        return c


class CKAS_59(nn.Module):
    def __init__(self, dims,c2, k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv_s1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1,groups=dims, dilation=1)  # atrous=1
        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)

        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        # self.fusion = Fusion_3(dims)
        self.omega = nn.Parameter(torch.ones(1) * 0.5)  # 基础权重
        self.delta_omega1 = nn.Parameter(torch.zeros(1),requires_grad=True) 
        self.delta_omega2 = nn.Parameter(torch.zeros(1),requires_grad=True)  
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        r3 = self.conv2(x) + x

        w3 = self.conv_s1.weight.data
        w3 = self.w_attention_3(w3)
        r5 = F.conv2d(x, w3, padding=5, groups=self.dims,dilation=5) + x
        r7 = F.conv2d(x, w3, padding=9, groups=self.dims, dilation=9) + x
        r = r3* self.omega + r5 * (self.omega + self.delta_omega1) + r7 * (self.omega + self.delta_omega2)
        r_end = self.conv_G(r)
        c = self.conv4(r_end)
        return c


class MS_DCKA(nn.Module):
    def __init__(self, dims,c2, k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv_s1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1,groups=dims, dilation=1)  # atrous=1
        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)

        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        # self.fusion = Fusion_3(dims)
        self.omega = nn.Parameter(torch.ones(1) * 0.5)  # 基础权重
        self.delta_omega1 = nn.Parameter(torch.zeros(1),requires_grad=True) 
        self.delta_omega2 = nn.Parameter(torch.zeros(1),requires_grad=True)  
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        
        r3 = self.conv2(x)

        w3 = self.conv_s1.weight.data
        w3 = self.w_attention_3(w3)
        r5 = F.conv2d(x, w3, padding=5, groups=self.dims,dilation=5)
        r7 = F.conv2d(x, w3, padding=9, groups=self.dims, dilation=9)
        r = r3* self.omega + r5 * (self.omega + self.delta_omega1) + r7 * (self.omega + self.delta_omega2)
        r_end = self.conv_G(r)
        c = self.conv4(r_end)
        return c


class CKASAPE_d(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv7 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)
        self.conv5 = nn.Conv2d(dims, dims, kernel_size=5, padding=2, groups=dims)
        # self.w_attention_5 = ConvWeightAttention(dims, 5)
        # self.w_attention_7 = ConvWeightAttention(dims, 7)
        # self.w_attention_7 = ConvWeightAttention_APE(dims, kernel_size=7)
        # self.w_attention_5 = ConvWeightAttention_APE(dims, kernel_size=5)       
        self.w_attention_7 = ConvWeightAttention(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention(dims, kernel_size=5)
        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        self.fusion = Fusion_75(dims)
        self.conv4 = Conv(dims, c2, k,s)
        # self.se = SEAtt(dims)

    def forward(self, x):

        
        r3 = self.conv2(x)

        w5 = self.conv5.weight.data
        w5 = self.w_attention_5(w5)

        w7 = self.conv7.weight.data
        w7 = self.w_attention_7(w7)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) 
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims)

        r4 = self.fusion([r5, r7])
        # r4 = torch.cat([r5, r7], dim=1)
        r = r4 + r3 + x

        # r = torch.cat([r3, r4], dim=1)
        r_end = self.conv_G(r)
        c = self.conv4(r_end)
        return c 

class CKASAPE_dili(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        # self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv_s1 = nn.Conv2d(dims, dims, kernel_size=3, padding=1,groups=dims)  # atrous=1
        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)

        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        self.fusion = Fusion_3(dims)
        self.conv4 = Conv(dims, c2, k,s)
        # self.cv1 = InceptionDWConv2d(dims,c2)

    def forward(self, x):

        
        r3 = self.conv_s1(x)

        w3 = self.conv_s1.weight.data
        w3 = self.w_attention_3(w3)
        r5 = F.conv2d(x, w3, padding=5, groups=self.dims,dilation=5)
        r9 = F.conv2d(x, w3, padding=9, groups=self.dims, dilation=9)

        r = self.fusion([r3, r5, r9])
        r_end = self.conv_G(r)
        c = self.conv4(r_end)
        # c = self.cv1(r_end)
        return c


class CKASAPE_dili(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 3
        group = 2
        self.scale =scale
        self.dims = dims
        # self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv_s1 = nn.Conv2d(dims, dims, kernel_size=3, padding=1,groups=dims)  # atrous=1
        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)

        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims*scale),
            nn.Conv2d(dims * scale, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        self.fusion = Fusion_3(dims)
        # self.omega = nn.Parameter(torch.ones(1) * 0.5)  # 基础权重
        # self.delta_omega1 = nn.Parameter(torch.zeros(1)) 
        # self.delta_omega2 = nn.Parameter(torch.zeros(1))  
        self.conv4 = Conv(dims, c2, k,s)
        # self.cv1 = InceptionDWConv2d(dims,c2)

    def forward(self, x):

        
        r3 = self.conv_s1(x) + x

        w3 = self.conv_s1.weight.data
        w3 = self.w_attention_3(w3)
        r5 = F.conv2d(x, w3, padding=5, groups=self.dims,dilation=5) + x
        r7 = F.conv2d(x, w3, padding=7, groups=self.dims, dilation=7) + x
        # r4 = self.fusion(r5, r7)
        # print(r3.shape, r5.shape, r7.shape, x.shape)

        r = torch.cat([r3, r5, r7], dim=1)
        # print(self.dims*self.scale)
        # print(r.shape)
        r_end = self.conv_G(r)
        # c = self.cv1(r_end)
        c = self.conv4(r_end)
        return c

class CKASAPE_dili_v2(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv_s1 = nn.Conv2d(dims, dims, kernel_size=3, stride=1,groups=dims, dilation=1)   # atrous=1
        self.w_attention_3 = ConvWeightAttention_CKAS(dims, kernel_size=3)
        

        # self.conv_B = nn.Sequential(
        #     nn.BatchNorm2d(group * dims),
        #     nn.Conv2d(group * dims, scale * dims, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(scale * dims, dims, kernel_size=1)
        # )
        self.conv_G = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        self.fusion = Fusion_3(dims)
        self.conv4 = Conv(dims, c2, k,s)

    def forward(self, x):

        
        r3 = self.conv2(x)

        w3 = self.conv_s1.weight.data
        w3 = self.w_attention_3(w3)
        r5 = F.conv2d(x, w3, padding=5, groups=self.dims,dilation=5)
        r9 = F.conv2d(x, w3, padding=9, groups=self.dims, dilation=9)
        # r4 = self.fusion(r5, r7)
        r = self.fusion([r3, r5, r9])
        
        r_end = self.conv_G(r)
        c = self.conv4(r_end)
        return c

class CKASAPE(nn.Module):
    def __init__(self, dims,c2,k,s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims)

        # self.w_attention_5 = ConvWeightAttention(dims, 5)
        # self.w_attention_7 = ConvWeightAttention(dims, 7)
        self.w_attention_7 = ConvWeightAttention_APE(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention_APE(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )

        self.conv1 = nn.Conv2d(in_channels=2*dims, out_channels=dims, kernel_size=1, stride=1, padding=0)
        self.conv4 = Conv(dims, c2, k,s)
    def forward(self, x):

        
        r3 = self.conv2(x) + x 

        w5 = F.interpolate(self.conv2.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(self.conv2.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention_7(w7)

        
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) 
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims)

        r4 = torch.cat([r5, r7], dim=1)
        r4 = self.conv1(r4) + x

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        c = self.conv4(r_end)
        return c 
class ConvWeightAttention_APE(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.ape = nn.Parameter(torch.zeros(dims, kernel_size*kernel_size, embed_dim))
        torch.nn.init.trunc_normal_(self.ape.data, std=.02)  # 直接修改参数的数据，不重新赋值
        self.se = SEAtt(dims)
    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0] + self.ape
        res = res.permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        res = self.ln(res + weight)
        return self.se(res)

class ConvWeightAttention_CKAS(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])

    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        return self.ln(res + weight)

class ConvWeightAttention_SE(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.se = SEAtt(dims)

    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        return self.se(self.ln(res + weight))

class DWBConv(nn.Module):
    def __init__(self,  dims, c2, k, s, scale=4):
        super(DWBConv, self).__init__()
        self.conv3 = SADWConv(dims, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        self.cv1 = InceptionDWConv2d(dims,c2)
        
    def forward(self, x):
        r = self.conv3(x)
        b = self.conv(r) + x
        c = self.cv1(b)
        return c


class DWBConv_02(nn.Module):
    def __init__(self,  dims, c2, k, s, scale=4):
        super(DWBConv_02, self).__init__()
        self.conv3 = SADWConv(dims, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )
        self.cv1 = Conv(dims,c2,k,s)
    def forward(self, x):
        r = self.conv3(x)
        b = self.conv(r) + x
        c = self.cv1(b)
        return c

class SADWConv(nn.Module):
    def __init__(self, dims, kernel_size, padding, stride=1):
        super(SADWConv, self).__init__()
        self.padding = padding
        self.dims = dims
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        # self.weight = self.initialization(dims,kernel_size)
        self.real_w = None
        self.w_attention = ConvWeightAttention(dims, kernel_size)
        self.se = SEAtt(dims)
        self.reset_w()

    def reset_w(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def get_weight(self):
        return self.real_w

    def forward(self, x):
        self.real_w = self.w_attention(self.weight)
        # self.real_w = self.ela(self.real_w)
        self.real_w = self.se(self.real_w)
        out = F.conv2d(x, self.real_w, padding=self.padding, groups=self.dims, stride=self.stride)
        return out


class ConvWeightAttention(nn.Module):
    def __init__(self, dims, kernel_size):
        super(ConvWeightAttention, self).__init__()
        embed_dim = 1
        self.qw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        self.kw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        self.vw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))      
        self.attention = self.Mulatt(embed_dim)
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.reset_w()

    # def initialization(self,dims,kernel_size):
    #     x = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
    #     return x
    
    def Mulatt(self,embed_dim):
        x = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        return x
    def reset_w(self) -> None:
        init.kaiming_uniform_(self.qw, a=math.sqrt(5))
        init.kaiming_uniform_(self.kw, a=math.sqrt(5))
        init.kaiming_uniform_(self.vw, a=math.sqrt(5))

    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw).view(b, c, h * w).permute(0, 2, 1).contiguous()
        k = torch.matmul(weight, self.kw).view(b, c, h * w).permute(0, 2, 1).contiguous()
        v = torch.matmul(weight, self.vw).view(b, c, h * w).permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous().view(b, c, h, w)
        res_out = self.ln(res + weight)
        return res_out



class SEAtt(nn.Module):
    def __init__(self, dims):
        super(SEAtt, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dims, dims // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims // 16, dims, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, weight):
        b, _, h, w = weight.shape
        weight = weight.view(1, b, h, w)
        att = self.se(weight)
        weight = weight * att
        weight = weight.view(b, 1, h, w)
        return weight

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None,  sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample  # 是否进行下采样

        self.in_channels = in_channels  # 输入通道数
        self.inter_channels = inter_channels  # 中间通道数

        # 如果未指定中间通道数，默认为输入通道数的一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        # 定义 g、theta、phi 的卷积层
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        # 定义 W 层，可选择是否使用批归一化
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            # nn.init.constant(self.W[1].weight, 0)  # 初始化权重
            # nn.init.constant(self.W[1].bias, 0)    # 初始化偏置
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            # nn.init.constant(self.W.weight, 0)  # 初始化权重
            # nn.init.constant(self.W.bias, 0)    # 初始化偏置

        # 定义 theta 和 phi 的卷积层
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # 如果进行下采样，则对 g 和 phi 添加池化层
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        # print(x.shape)
        batch_size = x.size(0)  # 获取批量大小

        # 计算 g(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # 变形为 (b, inter_channels, N)
        g_x = g_x.permute(0, 2, 1)  # 调整维度顺序

        # 计算 theta(x) 和 phi(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # 调整维度顺序
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 计算注意力权重
        f = torch.matmul(theta_x, phi_x)  # 矩阵乘法
        f_div_C = F.softmax(f, dim=-1)  # 归一化

        # 加权聚合
        y = torch.matmul(f_div_C, g_x)  # 使用权重对 g_x 加权
        y = y.permute(0, 2, 1).contiguous()  # 调整维度顺序
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # 变形为 (b, inter_channels, t, h, w)

        # 融合输入和输出
        W_y = self.W(y)  # 通过 W 层
        z = W_y + x  # 残差连接

        return z  # 返回最终输出




class SED(nn.Module):
    def __init__(self):
        super(SED, self).__init__()
        
        self.sobel_kernel_x = torch.tensor([[[[-1, 0, 1],
                                              [-2, 0, 2],
                                              [-1, 0, 1]]]], dtype=torch.float32, requires_grad=False)

        self.sobel_kernel_y = torch.tensor([[[[-1, -2, -1],
                                              [0,  0,  0],
                                              [1,  2,  1]]]], dtype=torch.float32, requires_grad=False)
    
    def forward(self, x):
        sobel_kernel_x = self.sobel_kernel_x.to(x.device)
        sobel_kernel_y = self.sobel_kernel_y.to(x.device)

        B, C, H, W = x.shape
        
        sobel_kernel_x = sobel_kernel_x.expand(C, 1, 3, 3)  # [C, 1, 3, 3]
        sobel_kernel_y = sobel_kernel_y.expand(C, 1, 3, 3)  # [C, 1, 3, 3]

        edge_x = F.conv2d(x, sobel_kernel_x.to(x.dtype), padding=1, groups=C)  
        edge_y = F.conv2d(x, sobel_kernel_y.to(x.dtype), padding=1, groups=C)  

        sobel_features = torch.abs(edge_x) + torch.abs(edge_y)

        return sobel_features

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, k1,s1,dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = nn.Conv2d(inc*4, inc, kernel_size=3, padding=1, groups=inc)
    def forward(self, x):
        # print(x.shape)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        return x

class PSABlock_2(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention_2(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA_2(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_2(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Attention_2(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = CKAConv_2(dim, dim)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock_3(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention_3(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA_3(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_3(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Attention_3(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = CKAConv_3(dim, dim)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x



class PSABlock_4(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention_4(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA_4(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_4(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Attention_4(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = CKAConv_5(dim, dim)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x



class PSABlock_1(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, pe_layer=None) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention_1(c, attn_ratio=attn_ratio, num_heads=num_heads, pe_layer=pe_layer)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA_1(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5, pe_layer=None):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(
            *(PSABlock_1(self.c, attn_ratio=0.5, num_heads=self.c // 64, pe_layer=pe_layer) for _ in range(n))
        )

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Attention_1(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5, pe_layer=None):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        pe_layer = CKAConv if pe_layer is None else pe_layer
        self.pe = pe_layer(dim, dim)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class CKABS(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.c1 = c1
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1,groups=c1) 
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.cka_wa = ConvWeightAttention_CKAS(c1,kernel_size=3)
        # self.se = SEAtt(c1)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # r = self.conv2(x) + x
        w3 = self.cka_wa(self.conv2.weight.data)
        r3  = F.conv2d(x, w3, padding=1, groups=self.c1, stride=1) + x
        return self.act(self.bn(self.conv(r3)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class DynamicTanh_ConvWeight(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None, None] + self.bias[:, None, None, None]   
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

class ConvWeightAttention_CKA(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        # self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.ln = DynamicTanh_ConvWeight(dims,False)
        

    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        res = self.ln(res + weight)
        return res

class CKAConv(nn.Module):
    def __init__(self, dims,c2):
        super().__init__()
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.w_attention = ConvWeightAttention_CKA(dims, 3)
        self.se = SEAtt(dims)
        self.conv = Conv(dims, c2,k=3,s=1)

    def forward(self, x):
        
        # r3 = self.conv3(x) + x
        w3 = self.conv3.weight.data
        w3 = self.w_attention(w3)
        w3 = self.se(w3)
        r3 = F.conv2d(x, w3, padding=1, groups=self.dims) 

        r = r3 + x
        r = self.conv(r)
        return r   


class CKAConv_2(nn.Module):
    def __init__(self, dims,c2):
        super().__init__()
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.w_attention = ConvWeightAttention_CKA(dims, 7)
        self.se = SEAtt(dims)
        self.conv = Conv(dims, c2,k=1,s=1)

    def forward(self, x):
        
        # r3 = self.conv3(x) + x
        w7 = F.interpolate(self.conv3.weight.data, size=7, mode="bilinear", align_corners=False)
        w7 = self.w_attention(w7)
        w7 = self.se(w7)
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) 
        r = r7
        r = self.conv(r)
        return r   

# class CKAConv_2(nn.Module):
#     def __init__(self, dims,c2):
#         super().__init__()
#         self.dims = dims
#         scale = 4
#         group = 2
#         self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
#         self.w_attention = ConvWeightAttention_CKA(dims, 7)
#         self.se = SEAtt(dims)
#         self.conv = Conv(dims, c2,k=3,s=1)
#         self.convr = nn.Sequential(
#             nn.BatchNorm2d(group * dims),
#             nn.Conv2d(group * dims, scale * dims, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(scale * dims, dims, kernel_size=1)
#         )
#     def forward(self, x):
        
#         r3 = self.conv3(x) + x

#         w7 = F.interpolate(self.conv3.weight.data, size=7, mode="bilinear", align_corners=False)
#         w7 = self.w_attention(w7)
#         w7 = self.se(w7)
#         r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
#         r = torch.cat([r3, r7], dim=1)
#         r = self.convr(r)
#         r = self.conv(r)
#         return r   

class CKAConv_3(nn.Module):
    def __init__(self, dims,c2):
        super().__init__()
        scale = 4
        group = 1
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.w_attention = ConvWeightAttention_CKA(dims, 3)
        self.se = SEAtt(dims)
        self.conv = Conv(dims, c2,k=3,s=1)
        self.convr = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1)
        )
    def forward(self, x):
        w3 = self.conv3.weight.data
        w3 = self.w_attention(w3)
        w3 = self.se(w3)
        r3 = F.conv2d(x, w3, padding=1, groups=self.dims) 

        r = self.convr(r3) + x
        r = self.conv(r)
        return r   

class CKAConv_5(nn.Module):
    def __init__(self, dims,c2):
        super().__init__()
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.w_attention = ConvWeightAttention_CKA(dims, 5)
        self.se = SEAtt(dims)
        self.conv = Conv(2*dims, c2,k=1,s=1)
    def forward(self, x):
        
        w5= F.interpolate(self.conv3.weight.data, size=5, mode="bilinear", align_corners=False)
        w5 = self.w_attention(w5)
        w5 = self.se(w5)
        r5 = F.conv2d(x, w5, padding=2, groups=self.dims)
        r = torch.cat([r5, x], dim=1)
        r = self.conv(r)
        return r


class CKAConv_CondConv(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        self.cond = CondConv2d(dims, dims, kernel_size=3, num_experts=4, padding=1, groups=dims, bias=False)
        self.conv = Conv(dims, c2, k=3, s=1)

    def forward(self, x):
        r = self.cond(x) + x
        return self.conv(r)


class CKAConv_DCNv1(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        if MMCVDeformConv2dPack is None and DeformConv2d is None:
            raise ImportError(
                "CKAConv_DCNv1 requires mmcv.ops.DeformConv2dPack or torchvision.ops.DeformConv2d, but neither is installed."
            )
        # Force torchvision backend to avoid mmcv CUDA extension mismatch/segfaults.
        self.use_mmcv = False
        self.offset = nn.Conv2d(dims, 2 * 3 * 3, kernel_size=3, padding=1)
        self.dcn = DeformConv2d(dims, dims, kernel_size=3, padding=1, bias=False)
        self.conv = Conv(dims, c2, k=3, s=1)

    def forward(self, x):
        input_dtype = x.dtype
        amp_off = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()
        with amp_off:
            x = x.float().contiguous()
            if x.is_cuda:
                if self.use_mmcv:
                    r = self.dcn(x)
                else:
                    r = self.dcn(x, self.offset(x).contiguous())
            else:
                # CPU fallback (used during stride inference)
                r = F.conv2d(x, self.dcn.weight, None, stride=1, padding=1)
            r = r + x
            out = self.conv(r)
        return out.to(input_dtype)


class CKAConv_DCNv2(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        if MMCVModulatedDeformConv2dPack is None and tv_deform_conv2d is None:
            raise ImportError(
                "CKAConv_DCNv2 requires mmcv.ops.ModulatedDeformConv2dPack or torchvision.ops.deform_conv2d, but neither is installed."
            )
        # Force torchvision backend to avoid mmcv CUDA extension mismatch/segfaults.
        self.use_mmcv = False
        self.offset = nn.Conv2d(dims, 2 * 3 * 3, kernel_size=3, padding=1)
        self.mask = nn.Sequential(nn.Conv2d(dims, 3 * 3, kernel_size=3, padding=1), nn.Sigmoid())
        self.dcn = ModulatedDeformConv2d(dims, dims, kernel_size=3, padding=1, bias=False)
        self.conv = Conv(dims, c2, k=3, s=1)

    def forward(self, x):
        input_dtype = x.dtype
        amp_off = torch.cuda.amp.autocast(enabled=False) if x.is_cuda else nullcontext()
        with amp_off:
            x = x.float().contiguous()
            if self.use_mmcv and x.is_cuda:
                r = self.dcn(x)
            else:
                r = self.dcn(x, self.offset(x).contiguous(), self.mask(x).contiguous())
            r = r + x
            out = self.conv(r)
        return out.to(input_dtype)


class CKAConv_ODConv(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        self.od = ODConv2dLocal(dims, dims, kernel_size=3, padding=1, groups=dims, reduction=0.0625, kernel_num=4)
        self.conv = Conv(dims, c2, k=3, s=1)

    def forward(self, x):
        r = self.od(x) + x
        return self.conv(r)


class CKAConv_DyConv(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        self.dy = DynamicConv2dLocal(
            dims, dims, kernel_size=3, stride=1, padding=1, groups=dims, bias=False, kernel_num=4
        )
        self.conv = Conv(dims, c2, k=3, s=1)

    def forward(self, x):
        r = self.dy(x) + x
        return self.conv(r)


class CKAConv_DSConv(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        self.ds = nn.Sequential(Conv(dims, dims, 3, 1, g=dims), Conv(dims, dims, 1, 1))
        self.conv = Conv(dims, c2, k=3, s=1)

    def forward(self, x):
        r = self.ds(x) + x
        return self.conv(r)


class CKAConv_Conv(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        self.cv = Conv(dims, dims, 3, 1)
        self.conv = Conv(dims, c2, k=3, s=1)

    def forward(self, x):
        r = self.cv(x) + x
        return self.conv(r)


class C2PSA_1_CondConv(C2PSA_1):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e, pe_layer=CKAConv_CondConv)


class C2PSA_1_DCNv1(C2PSA_1):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e, pe_layer=CKAConv_DCNv1)


class C2PSA_1_DCNv2(C2PSA_1):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e, pe_layer=CKAConv_DCNv2)


class C2PSA_1_ODConv(C2PSA_1):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e, pe_layer=CKAConv_ODConv)


class C2PSA_1_DyConv(C2PSA_1):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e, pe_layer=CKAConv_DyConv)


class C2PSA_1_DSConv(C2PSA_1):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e, pe_layer=CKAConv_DSConv)


class C2PSA_1_Conv(C2PSA_1):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e, pe_layer=CKAConv_Conv)
