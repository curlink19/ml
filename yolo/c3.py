import torch
from torch import nn
from conv_bn_silu import ConvBNSiLU
from bottleneck1 import BottleNeck1
from bottleneck2 import BottleNeck2


class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck1
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks1,
                         if False,  it will be composed by bottlenecks2.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        width_multiple=1,
        depth=1,
        backbone=True,
    ):
        super(C3, self).__init__()
        c_ = int(width_multiple * in_channels)

        self.c1 = ConvBNSiLU(
            in_channels, c_, kernel_size=1, stride=1, padding=0
        )
        self.c_skipped = ConvBNSiLU(
            in_channels, c_, kernel_size=1, stride=1, padding=0
        )
        if backbone:
            self.seq = nn.Sequential(
                *[BottleNeck1(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[BottleNeck2(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        self.c_out = ConvBNSiLU(
            c_ * 2, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)
