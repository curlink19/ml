import torch
from torch import nn
from conv_bn_silu import ConvBNSiLU
from c3 import C3
from sppf import SPPF
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from c3_neck import C3_NECK
from heads import HEADS


class YOLOV5m(nn.Module):
    def __init__(self, first_out, nc=80, anchors=(), ch=(), inference=False):
        super(YOLOV5m, self).__init__()
        self.inference = inference
        self.backbone = nn.ModuleList()
        self.backbone += [
            ConvBNSiLU(
                in_channels=3,
                out_channels=first_out,
                kernel_size=6,
                stride=2,
                padding=2,
            ),
            ConvBNSiLU(
                in_channels=first_out,
                out_channels=first_out * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            C3(
                in_channels=first_out * 2,
                out_channels=first_out * 2,
                width_multiple=0.5,
                depth=2,
            ),
            ConvBNSiLU(
                in_channels=first_out * 2,
                out_channels=first_out * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            C3(
                in_channels=first_out * 4,
                out_channels=first_out * 4,
                width_multiple=0.5,
                depth=4,
            ),
            ConvBNSiLU(
                in_channels=first_out * 4,
                out_channels=first_out * 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            C3(
                in_channels=first_out * 8,
                out_channels=first_out * 8,
                width_multiple=0.5,
                depth=6,
            ),
            ConvBNSiLU(
                in_channels=first_out * 8,
                out_channels=first_out * 16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            C3(
                in_channels=first_out * 16,
                out_channels=first_out * 16,
                width_multiple=0.5,
                depth=2,
            ),
            SPPF(in_channels=first_out * 16, out_channels=first_out * 16),
        ]

        self.neck = nn.ModuleList()
        self.neck += [
            ConvBNSiLU(
                in_channels=first_out * 16,
                out_channels=first_out * 8,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            C3(
                in_channels=first_out * 16,
                out_channels=first_out * 8,
                width_multiple=0.25,
                depth=2,
                backbone=False,
            ),
            ConvBNSiLU(
                in_channels=first_out * 8,
                out_channels=first_out * 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            C3(
                in_channels=first_out * 8,
                out_channels=first_out * 4,
                width_multiple=0.25,
                depth=2,
                backbone=False,
            ),
            ConvBNSiLU(
                in_channels=first_out * 4,
                out_channels=first_out * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            C3(
                in_channels=first_out * 8,
                out_channels=first_out * 8,
                width_multiple=0.5,
                depth=2,
                backbone=False,
            ),
            ConvBNSiLU(
                in_channels=first_out * 8,
                out_channels=first_out * 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            C3(
                in_channels=first_out * 16,
                out_channels=first_out * 16,
                width_multiple=0.5,
                depth=2,
                backbone=False,
            ),
        ]
        self.head = HEADS(nc=nc, anchors=anchors, ch=ch)

    def forward(self, x):
        assert (
            x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0
        ), "Width and Height aren't divisible by 32!"
        backbone_connection = []
        neck_connection = []
        outputs = []
        for idx, layer in enumerate(self.backbone):
            # takes the out of the 2nd and 3rd C3 block and stores it
            x = layer(x)
            if idx in [4, 6]:
                backbone_connection.append(x)

        for idx, layer in enumerate(self.neck):
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = Resize(
                    [x.shape[2] * 2, x.shape[3] * 2],
                    interpolation=InterpolationMode.NEAREST,
                )(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif idx in [4, 6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)

            elif (isinstance(layer, C3_NECK) and idx > 2) or (
                isinstance(layer, C3) and idx > 2
            ):
                x = layer(x)
                outputs.append(x)

            else:
                x = layer(x)

        return self.head(outputs)
