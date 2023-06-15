import albumentations as A

# model
ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32#
]

NC = 2
FIRST_OUT = 48

# optim
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

# loss
CLS_PW = 1.0
OBJ_PW = 1.0

# over
DEVICE = None
IMAGE_SIZE = 640

CONF_THRESHOLD = 0.01  # to get all possible bboxes,
# trade-off metrics/speed --> we choose metrics
NMS_IOU_THRESH = 0.6
# for map 50
MAP_IOU_THRESH = 0.5


TRAIN_TRANSFORMS = A.Compose(
    [
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4
        ),
        A.Transpose(p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-20, 20), p=0.7),
        A.Blur(p=0.05),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ChannelShuffle(p=0.05),
    ],
    bbox_params=A.BboxParams(
        "yolo",
        min_visibility=0.4,
        label_fields=[],
    ),
)
