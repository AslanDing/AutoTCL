import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,stride = 1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,dilation,downsampling=False):
        super().__init__()
        self.inout = (in_channels,out_channels)
        if downsampling:
            self.conv1 = SamePadConv(in_channels, out_channels, kernel_size=3,stride=2, dilation=dilation) # in_channels,out_channels
            self.conv2 = SamePadConv(out_channels, out_channels, kernel_size=3,stride=1, dilation=dilation) # out_channels,out_channels
            self.projector =nn.Conv1d(in_channels, out_channels, 1,stride=2)

        else:
            self.conv1 = SamePadConv(in_channels, out_channels, kernel_size=3,stride=1, dilation=dilation) #out_channels
            self.conv2 = SamePadConv(out_channels, out_channels, kernel_size=3,stride=1, dilation=dilation) #out_channels
            self.projector =None

    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class Convlayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, first_layer = True):
        super().__init__()
        self.inout = (in_channels,out_channels)
        if first_layer:
            self.block1 = ConvBlock(in_channels=in_channels,out_channels=out_channels,dilation=dilation,downsampling=False)
            self.block2 = ConvBlock(in_channels=out_channels,out_channels=out_channels,dilation=dilation,downsampling=False)
        else:
            self.block1 = ConvBlock(in_channels=in_channels,out_channels=out_channels,dilation=dilation,downsampling=True)
            self.block2 = ConvBlock(in_channels=out_channels,out_channels=out_channels,dilation=dilation,downsampling=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        channels = [in_channels,64,64,64,64,out_channels]
        self.net = nn.Sequential(*[
            Convlayer(
                channels[i],
                channels[i+1],
                first_layer=(i==0),
                dilation=2**i,
            )
            for i in range(len(channels)-1)
        ])
        
    def forward(self, x):
        return self.net(x)
    def get_hiddens(self,x):
        hiddens = []
        for layer in self.net:
            x = layer(x)
            hiddens.append(x)
        return hiddens
