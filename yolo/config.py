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
