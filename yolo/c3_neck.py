import torch
from torch import nn
from conv_bn_silu import ConvBNSiLU


# in the PANET the C3 block is different: no more CSP but a residual block
# composed a sequential branch of n SiLUs and a skipped branch with one SiLU
class C3_NECK(nn.Module):
    def __init__(self, in_channels, out_channels, width, depth):
        super(C3_NECK, self).__init__()
        c_ = int(in_channels * width)
        self.in_channels = in_channels
        self.c_ = c_
        self.out_channels = out_channels
        self.c_skipped = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        self.c_out = ConvBNSiLU(c_ * 2, out_channels, 1, 1, 0)
        self.silu_block = self.make_silu_block(depth)

    def make_silu_block(self, depth):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(ConvBNSiLU(self.in_channels, self.c_, 1, 1, 0))
            elif i % 2 == 0:
                layers.append(ConvBNSiLU(self.c_, self.c_, 3, 1, 1))
            elif i % 2 != 0:
                layers.append(ConvBNSiLU(self.c_, self.c_, 1, 1, 0))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.c_out(
            torch.cat([self.silu_block(x), self.c_skipped(x)], dim=1)
        )
