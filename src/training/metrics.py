"""Detection Metrics"""

import numpy as np
import torch
from typing import List, Dict


class DetectionMetrics:
    """Calculate detection metrics"""
    
    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU for boxes in [x1, y1, x2, y2] format"""
        b1 = np.array(box1, dtype=float)
        b2 = np.array(box2, dtype=float)

        x1_inter = max(b1[0], b2[0])
        y1_inter = max(b1[1], b2[1])
        x2_inter = min(b1[2], b2[2])
        y2_inter = min(b1[3], b2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def calculate_ap(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
        """Calculate Average Precision"""
        # If there are no ground truths and no predictions, AP is 0
        if not predictions and not ground_truths:
            return 0.0
        
        sorted_preds = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))
        used_gt = set()
        
        for idx, pred in enumerate(sorted_preds):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in used_gt:
                    continue
                
                iou = DetectionMetrics.calculate_iou(pred['bbox'], gt['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp[idx] = 1
                used_gt.add(best_gt_idx)
            else:
                fp[idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths) if len(ground_truths) > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Build precision envelope and integrate area under PR curve
        mrec = np.concatenate([np.array([0.0]), np.asarray(recalls), np.array([1.0])])
        mpre = np.concatenate([np.array([0.0]), np.asarray(precisions), np.array([0.0])])
        
        # Make precision monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        
        # Calculate AP as the area under the precision-recall curve
        idxs = np.where(mrec[1:] != mrec[:-1])[0]
        ap = 0.0
        for i in idxs:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        
        return ap
    
    @staticmethod
    def calculate_map(predictions_list: List[List[Dict]], ground_truths_list: List[List[Dict]], num_classes: int = 2) -> float:
        """Calculate mAP"""
        aps = []
        
        for class_id in range(num_classes):
            class_preds = []
            class_gts = []
            
            for preds, gts in zip(predictions_list, ground_truths_list):
                class_preds.extend([p for p in preds if p.get('class_id') == class_id])
                class_gts.extend([g for g in gts if g.get('class_id') == class_id])
            
            if len(class_gts) > 0:
                ap = DetectionMetrics.calculate_ap(class_preds, class_gts)
                aps.append(ap)
        
        return float(np.mean(aps)) if aps else 0.0
