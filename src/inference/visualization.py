"""Visualization Utilities"""

import cv2
import numpy as np
from typing import List, Dict


class PlasticVisualizer:
    """Visualization tools"""
    
    COLORS = {
        'plastic': (0, 255, 0),
        'trash': (255, 0, 0)
    }
    
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2,
        font_scale: float = 0.7
    ) -> np.ndarray:
        """Draw bounding boxes"""
        output_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = PlasticVisualizer.COLORS.get(class_name, (255, 255, 255))
            
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            cv2.rectangle(output_image, (x1, y1 - label_size - 5), (x1 + label_size, y1), color, -1)
            cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        return output_image
