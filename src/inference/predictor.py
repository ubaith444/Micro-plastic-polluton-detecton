"""Real-time Inference Pipeline"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
import time


class PlasticDetector:
    """Complete inference pipeline"""
    
    DEBRIS_CATEGORIES = ['plastic', 'trash']
    
    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str = 'faster_rcnn',
        device: str = 'cuda',
        conf_threshold: float = 0.5
    ):
        """Initialize detector"""
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = device
        self.conf_threshold = conf_threshold
        
        self.model = self.load_model()
        self.model.eval()
    
    def load_model(self):
        """Load model"""
        if self.model_type == 'yolov8':
            from ultralytics import YOLO
            return YOLO(str(self.model_path))
        else:
            return torch.load(self.model_path, map_location=self.device).to(self.device).eval()
    
    def predict_image(self, image_path: Union[str, Path], return_visualization: bool = False):
        """Run inference on single image"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        detections = self._run_inference(image_rgb)
        inference_time = time.time() - start_time
        
        for det in detections:
            det['inference_time'] = inference_time
            det['image_path'] = str(image_path)
        
        if return_visualization:
            from .visualization import PlasticVisualizer
            viz = PlasticVisualizer()
            vis_image = viz.draw_detections(image, detections)
            return detections, vis_image
        
        return detections
    
    def _run_inference(self, image: np.ndarray) -> List[Dict]:
        """Run model inference"""
        if self.model_type == 'yolov8':
            return self._yolo_inference(image)
        else:
            return self._rcnn_inference(image)
    
    def _yolo_inference(self, image: np.ndarray) -> List[Dict]:
        """YOLO inference"""
        results = self.model(image, conf=self.conf_threshold)
        
        detections = []
        for result in results:
            for box in result.boxes:
                det = {
                    'bbox': box.xyxy.cpu().numpy().tolist(),
                    'confidence': float(box.conf),
                    'class_id': int(box.cls),
                    'class_name': self.DEBRIS_CATEGORIES[int(box.cls)]
                }
                detections.append(det)
        
        return detections
    
    def get_statistics(self, detections: List[Dict]) -> Dict:
        """Calculate statistics"""
        stats = {
            'total_detections': len(detections),
            'confidence_mean': np.mean([d['confidence'] for d in detections]) if detections else 0,
            'confidence_min': np.min([d['confidence'] for d in detections]) if detections else 0,
            'confidence_max': np.max([d['confidence'] for d in detections]) if detections else 0,
            'class_distribution': {}
        }
        
        for det in detections:
            class_name = det['class_name']
            stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
        
        return stats
