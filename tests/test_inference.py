"""Inference Tests"""

import pytest
import numpy as np
from src.inference import PlasticVisualizer


class TestPlasticVisualizer:
    """Test visualization"""
    
    def test_draw_detections(self):
        """Test drawing"""
        image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
        detections = [{
            'bbox': [100, 100, 200, 200],
            'class_name': 'plastic',
            'confidence': 0.95
        }]
        
        viz = PlasticVisualizer()
        output = viz.draw_detections(image, detections)
        
        assert output.shape == image.shape
