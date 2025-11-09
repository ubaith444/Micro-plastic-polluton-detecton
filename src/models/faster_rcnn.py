"""Faster R-CNN Detection Model for UPD"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import List, Dict, Optional


class FasterRCNNDetector(nn.Module):
    """Faster R-CNN for underwater plastic detection"""
    
    def __init__(
        self,
        num_classes: int = 3,  # 2 debris + 1 background
        pretrained: bool = True,
        trainable_backbone_layers: int = 3
    ):
        """Initialize Faster R-CNN"""
        super().__init__()
        
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ):
        """Forward pass"""
        if self.training:
            assert targets is not None
            return self.model(images, targets)
        else:
            return self.model(images)
    
    def freeze_backbone(self):
        """Freeze backbone"""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
    
    @torch.no_grad()
    def predict(
        self,
        images: List[torch.Tensor],
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """Run inference"""
        self.eval()
        predictions = self.model(images)
        
        filtered_preds = []
        for pred in predictions:
            mask = pred['scores'] >= conf_threshold
            filtered_pred = {
                'boxes': pred['boxes'][mask],
                'labels': pred['labels'][mask],
                'scores': pred['scores'][mask]
            }
            filtered_preds.append(filtered_pred)
        
        return filtered_preds
    
    def load_weights(self, path: str):
        """Load weights"""
        self.load_state_dict(torch.load(path))
    
    def save_weights(self, path: str):
        """Save weights"""
        torch.save(self.state_dict(), path)
