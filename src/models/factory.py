from functools import partial

import torch
import torchvision
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

MAX_DETS = 300


def build_model(name: str, num_classes: int = 2, img_size: int = 640):
    if name == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT",
            min_size=img_size,
            max_size=img_size,
            box_detections_per_img=MAX_DETS,
        )
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    elif name == "ssdlite":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights="DEFAULT",
            detections_per_img=MAX_DETS,
            topk_candidates=600,
        )
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer
        )

    elif name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights="DEFAULT",
            min_size=img_size,
            max_size=img_size,
            detections_per_img=MAX_DETS,
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            256, num_anchors, num_classes, norm_layer=partial(torch.nn.GroupNorm, 32)
        )

    else:
        raise ValueError(f"Неизвестная torchvision-модель: {name}")

    return model
