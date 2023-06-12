import torch
from torch import nn
import config
import numpy as np
from torchvision.ops import nms


# ALADDIN'S
def iou_width_height(
    gt_box, anchors, strided_anchors=True, stride=[8, 16, 32]
):
    """
    Parameters:
        gt_box (tensor): width and height of the ground truth box
        anchors (tensor): lists of anchors containing width and height
        strided_anchors (bool): if the anchors are divided by the stride or not
    Returns:
        tensor: Intersection over union between the gt_box
        and each of the n-anchors
    """
    # boxes 1 (gt_box): shape (2,)
    # boxes 2 (anchors): shape (9,2)
    # intersection shape: (9,)
    anchors /= 640
    if strided_anchors:
        anchors = anchors.reshape(9, 2) * torch.tensor(stride).repeat(
            6, 1
        ).T.reshape(9, 2)

    intersection = torch.min(gt_box[..., 0], anchors[..., 0]) * torch.min(
        gt_box[..., 1], anchors[..., 1]
    )
    union = (
        gt_box[..., 0] * gt_box[..., 1]
        + anchors[..., 0] * anchors[..., 1]
        - intersection
    )
    # intersection/union shape (9,)
    return intersection / union


# ALADDIN'S MODIFIED
def intersection_over_union(
    boxes_preds, boxes_labels, box_format="midpoint", GIoU=False, eps=1e-7
):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        GIoU (bool): if True it computed GIoU loss (https://giou.stanford.edu)
        eps (float): for numerical stability

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    else:  # if not midpoints box coordinates are considered
        #   to be in coco format
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    w1, h1, w2, h2 = (
        box1_x2 - box1_x1,
        box1_y2 - box1_y1,
        box2_x2 - box2_x1,
        box2_y2 - box2_y1,
    )
    # Intersection area
    inter = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(
        0
    ) * (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(
            box1_x1, box2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        c_area = cw * ch + eps  # convex height
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


# https://gist.github.com/cbernecker/1ac2f9d45f28b6a4902ba651e3d4fa91#file-coco_to_yolo-py
def coco_to_yolo(bbox, image_w=640, image_h=640):
    x1, y1, w, h = bbox
    # return [((x1 + w)/2)/image_w, ((y1 + h)/2)/image_h, w/image_w, h/image_h]
    return [
        ((2 * x1 + w) / (2 * image_w)),
        ((2 * y1 + h) / (2 * image_h)),
        w / image_w,
        h / image_h,
    ]


def coco_to_yolo_tensors(bbox, w0=640, h0=640):
    x1, y1, w, h = np.split(bbox, 4, axis=1)
    # return [((x1 + w)/2)/image_w, ((y1 + h)/2)/image_h, w/image_w, h/image_h]
    return np.concatenate(
        [((2 * x1 + w) / (2 * w0)), ((2 * y1 + h) / (2 * h0)), w / w0, h / h0],
        axis=1,
    )


# rescales bboxes from an image_size to another image_size
"""def rescale_bboxes(bboxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    new_boxes = []
    for bbox in bboxes:
        x = math.floor(bbox[0] * ew/sw * 100)/100
        y = math.floor(bbox[1] * eh/sh * 100)/100
        w = math.floor(bbox[2] * ew/sw * 100)/100
        h = math.floor(bbox[3] * eh/sh * 100)/100
        new_boxes.append([x, y, w, h])
    return new_boxes"""


def rescale_bboxes(bboxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    y = np.copy(bboxes)

    y[:, 0:1] = np.floor(bboxes[:, 0:1] * ew / sw * 100) / 100
    y[:, 1:2] = np.floor(bboxes[:, 1:2] * eh / sh * 100) / 100
    y[:, 2:3] = np.floor(bboxes[:, 2:3] * ew / sw * 100) / 100
    y[:, 3:4] = np.floor(bboxes[:, 3:4] * eh / sh * 100) / 100

    return y


# ALADDIN'S
def non_max_suppression_aladdin(
    bboxes, iou_threshold, threshold, box_format="corners", max_detections=300
):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    if len(bboxes) > max_detections:
        bboxes = bboxes[:max_detections]

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def non_max_suppression(
    batch_bboxes, iou_threshold, threshold, max_detections=300, tolist=True
):
    """new_bboxes = []
    for box in bboxes:
        if box[1] > threshold:
            box[3] = box[0] + box[3]
            box[2] = box[2] + box[4]
            new_bboxes.append(box)"""

    bboxes_after_nms = []
    for boxes in batch_bboxes:
        boxes = torch.masked_select(
            boxes, boxes[..., 1:2] > threshold
        ).reshape(-1, 6)

        # from xywh to x1y1x2y2

        boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
        boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:] / 2)
        boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
        boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

        indices = nms(
            boxes=boxes[..., 2:] + boxes[..., 0:1],
            scores=boxes[..., 1],
            iou_threshold=iou_threshold,
        )
        boxes = boxes[indices]

        # sorts boxes by objectness score but it's already done
        #  internally by torch metrics's nms
        # _, si = torch.sort(boxes[:, 1], dim=0, descending=True)
        # boxes = boxes[si, :]

        if boxes.shape[0] > max_detections:
            boxes = boxes[:max_detections, :]

        bboxes_after_nms.append(boxes.tolist() if tolist else boxes)

    return bboxes_after_nms if tolist else torch.cat(bboxes_after_nms, dim=0)


class YOLO_LOSS:
    def __init__(
        self,
        model,
        filename=None,
        resume=False,
    ):
        self.mse = nn.MSELoss()
        self.BCE_cls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.CLS_PW)
        )
        self.BCE_obj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.OBJ_PW)
        )
        self.sigmoid = nn.Sigmoid()

        # (https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml)
        # (https://github.com/ultralytics/yolov5/blob/master/utils/loss.py#L170)
        # (https://github.com/ultralytics/yolov5/blob/master/train.py#L232)
        self.lambda_class = 0.5 * (model.head.nc / 80 * 3 / model.head.nl)
        self.lambda_obj = 1 * (
            (config.IMAGE_SIZE / 640) ** 2 * 3 / model.head.nl
        )
        self.lambda_box = 0.05 * (3 / model.head.nl)

        self.balance = [
            4.0,
            1.0,
            0.4,
        ]  # explanation.. https://github.com/ultralytics/yolov5/issues/2026

        self.nc = model.head.nc
        self.anchors_d = model.head.anchors.clone().detach()
        self.anchors = model.head.anchors.clone().detach().to("cpu")

        self.na = self.anchors.reshape(9, 2).shape[0]
        self.num_anchors_per_scale = self.na // 3
        self.S = model.head.stride
        self.ignore_iou_thresh = 0.5
        self.ph = None
        self.pw = None
        self.filename = filename

    def __call__(self, preds, targets, pred_size, batch_idx=None, epoch=None):
        self.batch_idx = batch_idx
        self.epoch = epoch

        targets = [
            self.build_targets(preds, bboxes, pred_size) for bboxes in targets
        ]

        t1 = torch.stack([target[0] for target in targets], dim=0).to(
            config.DEVICE, non_blocking=True
        )
        t2 = torch.stack([target[1] for target in targets], dim=0).to(
            config.DEVICE, non_blocking=True
        )
        t3 = torch.stack([target[2] for target in targets], dim=0).to(
            config.DEVICE, non_blocking=True
        )
        loss = (
            self.compute_loss(
                preds[0],
                t1,
                anchors=self.anchors_d[0],
                balance=self.balance[0],
            )[0]
            + self.compute_loss(
                preds[1],
                t2,
                anchors=self.anchors_d[1],
                balance=self.balance[1],
            )[0]
            + self.compute_loss(
                preds[2],
                t3,
                anchors=self.anchors_d[2],
                balance=self.balance[2],
            )[0]
        )

        return loss

    def build_targets(self, input_tensor, bboxes, pred_size):
        check_loss = True

        if check_loss:
            targets = [
                torch.zeros(
                    (
                        self.num_anchors_per_scale,
                        input_tensor[i].shape[2],
                        input_tensor[i].shape[3],
                        6,
                    )
                )
                for i in range(len(self.S))
            ]

        else:
            targets = [
                torch.zeros(
                    (
                        self.num_anchors_per_scale,
                        int(input_tensor.shape[2] / S),
                        int(input_tensor.shape[3] / S),
                        6,
                    )
                )
                for S in self.S
            ]

        classes = bboxes[:, 0].tolist() if len(bboxes) else []
        bboxes = bboxes[:, 1:] if len(bboxes) else []

        for idx, box in enumerate(bboxes):
            iou_anchors = iou_width_height(
                torch.from_numpy(box[2:4]), self.anchors
            )

            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            (
                x,
                y,
                width,
                height,
            ) = box
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                # i.e if the best anchor idx is 8, num_anchors_per_scale
                # we know that 8//3 = 2 --> the best scale_idx is 2 -->
                # best_anchor belongs to last scale (52,52)
                # scale_idx will be used to slice the variable "targets"
                # another pov: scale_idx searches the best scale of anchors
                scale_idx = torch.div(
                    anchor_idx,
                    self.num_anchors_per_scale,
                    rounding_mode="floor",
                )
                # print(scale_idx)
                # anchor_on_scale searches the idx of the
                #  best anchor in a given scale
                # found via index in the line below
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # slice anchors based on the idx of the best scales of anchors
                if check_loss:
                    scale_y = input_tensor[int(scale_idx)].shape[2]
                    scale_x = input_tensor[int(scale_idx)].shape[3]
                else:
                    S = self.S[scale_idx]
                    scale_y = int(input_tensor.shape[2] / S)
                    scale_x = int(input_tensor.shape[3] / S)

                # S = self.S[int(scale_idx)]
                # another problem: in the labels the coordinates of
                # the objects are set
                # with respect to the whole image, while we need them wrt
                # the corresponding (?) cell
                # next line idk how --> i tells which y cell, j which x cell
                # i.e x = 0.5, S = 13 --> int(S * x) = 6 --> 6th cell
                i, j = int(scale_y * y), int(scale_x * x)  # which cell
                # targets[scale_idx] --> shape (3, 13, 13, 6) best group
                # of anchors
                # targets[scale_idx][anchor_on_scale] --> shape (13,13,6)
                # i and j are needed to slice to the right cell
                # 0 is the idx corresponding to p_o
                # I guess [anchor_on_scale, i, j, 0] equals to
                # [anchor_on_scale][i][j][0]
                # check that the anchor hasn't been already taken by
                # another object (rare)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]
                # if not anchor_taken == if anchor_taken is still == 0 cause
                # in the following
                # lines will be set to one
                # if not has_anchor[scale_idx] --> if this scale has not been
                # already taken
                # by another anchor which were ordered in descending order by
                # iou, hence
                # the previous ones are better
                if not anchor_taken and not has_anchor[scale_idx]:
                    # here below we are going to populate all the
                    # 6 elements of targets[scale_idx][anchor_on_scale, i, j]
                    # setting p_o of the chosen cell = 1 since there is an
                    # object there
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1
                    # setting the values of the coordinates x, y
                    # i.e (6.5 - 6) = 0.5 --> x_coord is in the middle of this
                    # particular cell
                    # both are between [0,1]
                    x_cell, y_cell = (
                        scale_x * x - j,
                        scale_y * y - i,
                    )  # both between [0,1]
                    # width = 0.5 would be 0.5 of the entire image
                    # and as for x_cell we need the measure w.r.t the cell
                    # i.e S=13, width = 0.5 --> 6.5
                    width_cell, height_cell = (
                        width * scale_x,
                        height * scale_y,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][
                        anchor_on_scale, i, j, 0:4
                    ] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(
                        classes[idx]
                    )
                    has_anchor[scale_idx] = True
                # not understood

                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    targets[scale_idx][
                        anchor_on_scale, i, j, 4
                    ] = -1  # ignore prediction

        return targets

    # TRAINING_LOSS
    def compute_loss(self, preds, targets, anchors, balance):
        # originally anchors have shape (3,2) --> 3 set
        # of anchors of width and height
        bs = preds.shape[0]
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        obj = targets[..., 4] == 1

        pxy = (preds[..., 0:2].sigmoid() * 2) - 0.5
        pwh = ((preds[..., 2:4].sigmoid() * 2) ** 2) * anchors
        pbox = torch.cat((pxy[obj], pwh[obj]), dim=-1)
        tbox = targets[..., 0:4][obj]

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        iou = intersection_over_union(
            pbox, tbox, GIoU=True
        ).squeeze()  # iou(prediction, target)
        lbox = (1.0 - iou).mean()  # iou loss

        # ======================= #
        #   FOR OBJECTNESS SCORE    #
        # ======================= #
        iou = iou.detach().clamp(0)
        targets[..., 4][obj] *= iou

        lobj = self.BCE_obj(preds[..., 4], targets[..., 4]) * balance
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        # NB: my targets[...,5:6]) is a vector of size bs, 1,
        # ultralytics targets[...,5:6]) is a matrix of shape bs, num_classes

        tcls = torch.zeros_like(preds[..., 5:][obj], device=config.DEVICE)

        tcls[
            torch.arange(tcls.size(0)), targets[..., 5][obj].long()
        ] = 1.0  # for torch > 1.11.0

        lcls = self.BCE_cls(preds[..., 5:][obj], tcls)  # BCE

        return (
            (
                self.lambda_box * lbox
                + self.lambda_obj * lobj
                + self.lambda_class * lcls
            )
            * bs,
            torch.unsqueeze(
                torch.stack(
                    [
                        self.lambda_box * lbox,
                        self.lambda_obj * lobj,
                        self.lambda_class * lcls,
                    ]
                ),
                dim=0,
            ),
        )
