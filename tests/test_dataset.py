"""Dataset Tests"""

import pytest
import torch
from src.data import UnderwaterPlasticDataset, get_training_augmentation


class TestUPDDataset:
    """Test UPD dataset"""
    
    def test_categories(self):
        """Test categories"""
        assert len(UnderwaterPlasticDataset.CATEGORIES) == 2
    
    def test_augmentation(self):
        """Test augmentation"""
        transform = get_training_augmentation(416)
        assert transform is not None
