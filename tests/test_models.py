"""Model Tests"""

import pytest
import torch
from src.models import FasterRCNNDetector


class TestFasterRCNN:
    """Test Faster R-CNN"""
    
    def test_model_creation(self):
        """Test creation"""
        model = FasterRCNNDetector(num_classes=3, pretrained=False)
        assert model is not None

