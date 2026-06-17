import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import copy

__all__ = ['UNet_0']

class UNet(nn.Module):
    def __init__(self, in_c=3):
        super(UNet, self).__init__()

        # self.dims = [32, 64, 128, 256, 512]
        self.dims = [32,64, 128, 256, 512, 1024]
        # self.stem_se = StemSE(dims[0])
        self.stem = Stem(in_c, self.dims[0])
        # encoder
        self.e1 = DWBlocks(self.dims[0], 2)
        self.e2 = DWBlocks(self.dims[1], 2)
        self.e3 = DWBlocks(self.dims[2], 2)
        self.e4 = DWBlocks(self.dims[3], 2)
        self.e5 = DWBlocks(self.dims[4], 2)     
        self.down1 = DownSample(self.dims[0], self.dims[1])
        self.down2 = DownSample(self.dims[1], self.dims[2])
        self.down3 = DownSample(self.dims[2], self.dims[3])
        self.down4 = DownSample(self.dims[3], self.dims[4])
        self.down5 = DownSample(self.dims[4], self.dims[5])



    def forward(self, x):

        x = self.stem(x)

        x1 = self.e1(x)
        x1_down = self.down1(x1)
        x2 = self.e2(x1_down)
        x2_down = self.down2(x2)
        x3 = self.e3(x2_down)
        x3_down = self.down3(x3)
        x4 = self.e4(x3_down)
        x4_down = self.down4(x4)
        x5 = self.e5(x4_down)
        x5_down = self.down5(x5)


        outputs = [x2_down,x3_down, x4_down, x5_down]
        return outputs



class Block(nn.Module):
    def __init__(self, dims):
        super(Block, self).__init__()
        scale = 4
        self.conv3 = SADWConv(dims, kernel_size=3, padding=1)


        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=dims),
            nn.Conv2d(dims, dims * scale, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims * scale, dims, kernel_size=1)
        )

    def forward(self, x):
        r = self.conv3(x)
        return self.conv(r) + x


class DWBlocks(nn.Module):
    def __init__(self, dims, times):
        super(DWBlocks, self).__init__()
        self.Blocks = nn.ModuleList([
            Block(dims) for i in range(times)
        ])

    def forward(self, x):
        for block in self.Blocks:
            x = block(x)
        return x


class SADWConv(nn.Module):
    def __init__(self, dims, kernel_size, padding, stride=1):
        super(SADWConv, self).__init__()
        self.padding = padding
        self.dims = dims
        self.stride = stride
        # self.weight = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        self.weight = self.initialization(dims,kernel_size)
        self.real_w = None
        self.w_attention = ConvWeightAttention(dims, kernel_size)
        self.se = SEAtt(dims)
        # ela注意力
        # self.ela = EfficientLocalizationAttention(dims)
        self.reset_w()

    def reset_w(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def initialization(self,dims, kernel_size):
        x = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        return x
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
        # q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        # k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        # v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        # self.qw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        # self.kw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        # self.vw = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))      
        # self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)  
        self.qw = self.initialization(dims,kernel_size)
        self.kw = self.initialization(dims,kernel_size)
        self.vw = self.initialization(dims,kernel_size)
        self.attention = self.Mulatt(embed_dim)
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.reset_w()

    def initialization(self,dims,kernel_size):
        x = nn.Parameter(torch.empty(dims, 1, kernel_size, kernel_size))
        return x
    
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
        return self.ln(res + weight)


class DownSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSample, self).__init__()
        # self.dwConv = SADWConv(in_c, kernel_size=3, padding=1, stride=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_c, out_c, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = F.avg_pool2d(x, kernel_size=2)
        x2 = F.max_pool2d(x, kernel_size=2)
        return self.mlp(torch.cat([x1, x2], dim=1))


class UpSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSample, self).__init__()
        # self.upConv = SADWTransposeConv(in_c, kernel_size=3, padding=1, stride=2)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x1 = self.upConv(x)
        # x1 = F.pad(x1, (0, 1, 0, 1))
        x2 = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # return self.mlp(torch.cat([x1, x2], dim=1))
        return self.mlp(x2)


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

def UNet_0():
    model = UNet(3)
    return model

class Stem(nn.Module):
    def __init__(self, in_c, out_c):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.stem(x)



if __name__ == '__main__':
    model = UNet(3)
    a = torch.randn(1,3,640,640)
    model_2 = copy.deepcopy(model)
    output = model_2(a)
    for i in output: 
        print(i.shape)    
    
    
    
    # model = DWBlocks(3,1)
    # model_2 = Stem(3,32)
    # a = torch.randn(1,3,640,640)
    # b = model_2(a)
    # output = model(a)
    # for i in output: 
    #     print(i.shape)