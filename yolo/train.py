from yolo import YOLOV5m
import config
from utils.utils import try_all_gpus
from torch.optim import Adam
from loss import YOLO_LOSS
import math
from torch import nn
import random


def multi_scale(img, target_shape, max_stride):
    # to make it work with collate_fn of the loader
    # returns a random number between target_shape*0.5 e
    # target_shape*1.5+max_stride, applies an integer
    # division by max stride and multiplies again for max_stride
    # in other words it returns a number between those two
    # interval divisible by 32
    sz = (
        random.randrange(target_shape * 0.5, target_shape + max_stride)
        // max_stride
        * max_stride
    )
    # sf is the ratio between the random number and the max between height
    # and width
    sf = sz / max(img.shape[2:])
    h, w = img.shape[2:]
    # 1) regarding the larger dimension (height or width) it will become the
    # closest divisible by 32 of
    # larger_dimension*sz
    # 2) regarding the smaller dimension (height or width) it will become the
    # closest divisible by 32 of
    # smaller_dimension*sf
    # (random_number_divisible_by_32_within_range/larger_dimension)
    # math.ceil is the opposite of floor, it rounds the floats to the next ints
    ns = [math.ceil(i * sf / max_stride) * max_stride for i in [h, w]]
    # ns are the height,width that the new image will have
    imgs = nn.functional.interpolate(
        img, size=ns, mode="bilinear", align_corners=False
    )
    return imgs


if __name__ == "__main__":
    devices = try_all_gpus()
    print("devices: " + str(devices) + "\n \\/ \n")

    model = YOLOV5m(
        first_out=config.FIRST_OUT,
        nc=config.NC,
        anchors=config.ANCHORS,
        ch=(config.FIRST_OUT * 4, config.FIRST_OUT * 8, config.FIRST_OUT * 16),
        inference=False,
    ).to(devices[0])

    optim = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
