import torch
from torch import nn


class HEADS(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(HEADS, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        #  command+f register_buffer has the same result as
        # self.anchors = anchors but, it's a way to register a buffer (make
        # a variable available in runtime) that should not be
        # considered a model parameter
        self.stride = [8, 16, 32]

        # anchors are divided by the stride
        # (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(
            self.nl, -1, 2
        ) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer("anchors", anchors_)  # shape(nl,na,2)

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=(5 + self.nc) * self.naxs,
                    kernel_size=1,
                )
            ]

    def forward(self, x):
        for i in range(self.nl):
            # performs out_convolution and stores the result in place
            x[i] = self.out_convs[i](x[i])

            bs, _, grid_y, grid_x = x[i].shape
            # reshaping output to be
            # (bs, n_scale_predictions, n_grid_y, n_grid_x, 5 + num_classes)
            # why .permute? Here
            # https://github.com/ultralytics/yolov5/issues/10524#issuecomment-1356822063
            x[i] = (
                x[i]
                .view(bs, self.naxs, (5 + self.nc), grid_y, grid_x)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

        return x
