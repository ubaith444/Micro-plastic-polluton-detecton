"""YOLOv8 Detection Model Wrapper for UPD"""

from ultralytics import YOLO
import torch
from typing import List, Dict, Optional


class YOLODetector:
    """YOLOv8 wrapper for underwater plastic detection"""
    
    DEBRIS_CATEGORIES = ['plastic', 'trash']
    
    def __init__(
        self,
        model_size: str = 'm',
        weights: Optional[str] = None,
        device: str = 'cuda'
    ):
        """Initialize YOLO detector"""
        self.model_size = model_size
        self.device = device
        
        if weights:
            self.model = YOLO(weights)
        else:
            self.model = YOLO(f'yolov8{model_size}.pt')
        
        self.model.to(device)
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 416,
        batch: int = 16,
        patience: int = 20,
        **kwargs
    ):
        """Train YOLO model"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=self.device,
            workers=4,
            project='runs/detect',
            name='upd_yolo',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            save=True,
            save_period=10,
            **kwargs
        )
        return results
    
    def predict(
        self,
        source,
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 416,
        save: bool = False,
        **kwargs
    ):
        """Run inference"""
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            save=save,
            **kwargs
        )
    
    def validate(self, data_yaml: str, batch: int = 16, imgsz: int = 416):
        """Validate model"""
        return self.model.val(
            data=data_yaml,
            batch=batch,
            imgsz=imgsz,
            device=self.device
        )
    
    def export(self, format: str = 'onnx', imgsz: int = 416, **kwargs):
        """Export model"""
        return self.model.export(format=format, imgsz=imgsz, **kwargs)
    
    @staticmethod
    def parse_results(results, conf_threshold: float = 0.5):
        """Parse YOLO results"""
        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                if conf >= conf_threshold:
                    detection = {
                        'bbox': box.xyxy.cpu().numpy().tolist(),
                        'confidence': conf,
                        'class_id': int(box.cls),
                        'class_name': result.names[int(box.cls)]
                    }
                    detections.append(detection)
        return detections
