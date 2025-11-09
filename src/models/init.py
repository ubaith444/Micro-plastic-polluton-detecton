"""Model architectures package"""
from .yolo_detector import YOLODetector
from .faster_rcnn import FasterRCNNDetector
from .backbones import get_resnet_backbone

__all__ = ['YOLODetector', 'FasterRCNNDetector', 'get_resnet_backbone']
