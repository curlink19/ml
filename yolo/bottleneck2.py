from torch import nn
from conv_bn_silu import ConvBNSiLU


class BottleNeck2(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the model. If closer to 1,
                                the model becomes more complex.
    """

    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(BottleNeck2, self).__init__()
        c_ = int(width_multiple * in_channels)
        self.c1 = ConvBNSiLU(
            in_channels, c_, kernel_size=1, stride=1, padding=0
        )
        self.c2 = ConvBNSiLU(
            c_, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        return self.c2(self.c1(x))
