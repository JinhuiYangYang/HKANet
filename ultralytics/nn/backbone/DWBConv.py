import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class DWBConv(nn.Module):
    def __init__(self, dims,scale = 4):
        super(DWBConv, self).__init__()
        self.conv3 = SADWConv(dims, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )

    def forward(self, x):
        r = self.conv3(x)
        b = self.conv(r) + x
        return b


class SADWConv(nn.Module):
    def __init__(self, dims, kernel_size, padding, stride=1):
        super(SADWConv, self).__init__()
        self.padding = padding
        self.dims = dims
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
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
        self.real_w = self.se(self.real_w)
        return F.conv2d(x, self.real_w, padding=self.padding, groups=self.dims, stride=self.stride)

class ConvWeightAttention(nn.Module):
    def __init__(self, dims, kernel_size, embed_dim=1):
        super(ConvWeightAttention, self).__init__()
        self.qw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        self.kw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        self.vw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.reset_w()

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
        return weight.view(b, 1, h, w)


if __name__ == '__main__':
    model = DWBConv(256)
    x = torch.randn(1, 256, 80, 80)
    y = model(x)
    print(model)
    print(y.shape)