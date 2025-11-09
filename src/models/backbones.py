"""Backbone Networks"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet_backbone(
    variant: str = 'resnet50',
    pretrained: bool = True,
    frozen_stages: int = 1
) -> nn.Module:
    """Get ResNet backbone"""
    
    if variant == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif variant == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif variant == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif variant == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown ResNet variant: {variant}")
    
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    
    if frozen_stages > 0:
        freeze_resnet_stages(backbone, frozen_stages)
    
    return backbone


def freeze_resnet_stages(backbone: nn.Module, num_stages: int):
    """Freeze ResNet stages"""
    stages = [[backbone], [backbone], [backbone], [backbone], [backbone]]
    
    for i in range(min(num_stages, len(stages))):
        for module in stages[i]:
            for param in module.parameters():
                param.requires_grad = False
