"""Custom Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss"""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """Initialize Dice Loss"""
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()
        
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1.0 - dice_coeff


class CombinedLoss(nn.Module):
    """Combined loss"""
    
    def __init__(
        self,
        use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        l1_weight: float = 1.0
    ):
        """Initialize Combined Loss"""
        super().__init__()
        
        self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) if use_focal else nn.CrossEntropyLoss()
        self.l1_weight = l1_weight
    
    def forward(
        self,
        class_predictions: torch.Tensor,
        box_predictions: torch.Tensor,
        class_targets: torch.Tensor,
        box_targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate combined loss"""
        cls_loss = self.classification_loss(class_predictions, class_targets)
        loc_loss = F.smooth_l1_loss(box_predictions, box_targets, reduction='mean')
        
        return cls_loss + self.l1_weight * loc_loss
