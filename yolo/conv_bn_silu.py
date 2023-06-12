from torch import nn


class ConvBNSiLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding
    ):
        super(ConvBNSiLU, self).__init__()

        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

        self.cbl = nn.Sequential(
            conv,
            bn,
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.cbl(x)
