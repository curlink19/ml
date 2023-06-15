import torch
from torch import nn
import config
from utils.utils import intersection_over_union, iou_width_height


class YOLO_LOSS:
    def __init__(self, model):
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
                # anchor_on_scale searches the idx of the best
                #  anchor in a given scale
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
                # another problem: in the labels the
                # coordinates of the objects are set
                # with respect to the whole image, while we need them
                # wrt the corresponding (?) cell
                # next line idk how --> i tells which y cell, j which x cell
                # i.e x = 0.5, S = 13 --> int(S * x) = 6 --> 6th cell
                i, j = int(scale_y * y), int(scale_x * x)  # which cell
                # targets[scale_idx] --> shape (3, 13, 13, 6)
                # best group of anchors
                # targets[scale_idx][anchor_on_scale] --> shape (13,13,6)
                # i and j are needed to slice to the right cell
                # 0 is the idx corresponding to p_o
                # I guess [anchor_on_scale, i, j, 0] equals to
                # [anchor_on_scale][i][j][0]
                # check that the anchor hasn't been already taken by
                # another object (rare)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]
                # if not anchor_taken == if anchor_taken is still == 0
                # cause in the following
                # lines will be set to one
                # if not has_anchor[scale_idx] --> if this scale has
                # not been already taken
                # by another anchor which were ordered in descending
                # order by iou, hence
                # the previous ones are better
                if not anchor_taken and not has_anchor[scale_idx]:
                    # here below we are going to populate all the
                    # 6 elements of targets[scale_idx][anchor_on_scale, i, j]
                    # setting p_o of the chosen cell = 1 since there
                    # is an object there
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1
                    # setting the values of the coordinates x, y
                    # i.e (6.5 - 6) = 0.5 --> x_coord is in the middle
                    # of this particular cell
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
