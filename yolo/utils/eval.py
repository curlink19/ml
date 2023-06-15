import torch
from tqdm import tqdm
import time
from utils import cells_to_bboxes, non_max_suppression
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class YOLO_EVAL:
    def __init__(self, conf_threshold, nms_iou_thresh, map_iou_thresh, device):
        self.conf_threshold = conf_threshold
        # considers for nms only bboxes with obj_score > conf_thres
        self.nms_iou_thresh = nms_iou_thresh
        # during nms keeps all same clafss bboxes with iou < nms_iou_thresh
        # w.r.t. each others
        self.map_iou_threshold = map_iou_thresh
        # while computing map, a prediction to be classified as true
        # positive needs to have iou > map_theshold w.r.t. gt bbox

        self.device = device

        # map and accuracies are computed in 2 distinct methods,
        # to use csv.writerow and store
        # map metrics and accuracies all at once the
        # three below are set as class attributes
        # initialising them here for readability
        self.class_accuracy = None
        self.noobj_accuracy = None
        self.obj_accuracy = None

    def check_class_accuracy(self, model, loader):
        model.eval()
        print(".. Computing: class and obj accuracies ..")
        tot_class_preds, correct_class = 0, 0
        tot_obj, correct_obj = 0, 0

        for idx, (images, y) in enumerate(tqdm(loader)):
            images = images.to(self.device)
            images = images.float() / 255
            with torch.no_grad():
                out = model(images)

            for i in range(3):
                y[i] = y[i].to(self.device)
                obj = y[i][..., 4] == 1  # in paper this is Iobj_i

                correct_class += torch.sum(
                    torch.argmax(out[i][..., 5:][obj], dim=-1)
                    == y[i][..., 5][obj]
                )
                tot_class_preds += torch.sum(obj)

                obj_preds = torch.sigmoid(out[i][..., 0]) > self.conf_threshold
                correct_obj += torch.sum(obj_preds[obj] == y[i][..., 4][obj])
                tot_obj += torch.sum(obj)

        time.sleep(1)  # to not broke tqdm logger

        class_accuracy = correct_class / (tot_class_preds + 1e-16)
        obj_accuracy = correct_obj / (tot_obj + 1e-16)

        print("Class accuracy: {:.2f}%".format(class_accuracy * 100))
        print("Obj accuracy: {:.2f}%".format(obj_accuracy * 100))

        model.train()

    def map_pr_rec(self, model, loader, anchors, epoch):
        print(".. Getting Evaluation bboxes to compute MAP..")
        # make sure model is in eval before get bboxes

        model.eval()

        # 1) GET EVALUATION BBOXES
        preds = []
        targets = []

        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(self.device).float() / 255
            with torch.no_grad():
                predictions = model(images)

            pred_boxes = cells_to_bboxes(
                predictions,
                anchors,
                strides=model.head.stride,
                is_pred=True,
                to_list=False,
            )

            # we just want one bbox for each label, not one for each scale
            true_boxes = cells_to_bboxes(
                labels,
                anchors,
                strides=model.head.stride,
                is_pred=False,
                to_list=False,
            )

            pred_boxes = non_max_suppression(
                pred_boxes,
                iou_threshold=self.nms_iou_thresh,
                threshold=self.conf_threshold,
                tolist=False,
                max_detections=300,
            )

            true_boxes = non_max_suppression(
                true_boxes,
                iou_threshold=self.nms_iou_thresh,
                threshold=self.conf_threshold,
                tolist=False,
                max_detections=300,
            )

            preds.append(
                dict(
                    boxes=pred_boxes[..., 2:],
                    scores=pred_boxes[..., 1],
                    labels=pred_boxes[..., 0],
                )
            )

            targets.append(
                dict(
                    boxes=true_boxes[..., 2:],
                    labels=true_boxes[..., 0],
                )
            )

        print("...Computing MAP ...")
        metric = MeanAveragePrecision()
        metric.update(preds, targets)

        metrics = metric.compute()
        map50 = metrics["map_50"]
        map75 = metrics["map_75"]

        time.sleep(1)  # to not broke tqdm logger
        print(f"MAP50: {map50}, \nMAP75: {map75}")

        model.train()
