"""Underwater Plastic Dataset (UPD) Loader - YOLOv5 Format"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class UnderwaterPlasticDataset(Dataset):
    """
    Underwater Plastic Dataset (UPD) Loader
    
    Dataset: https://zenodo.org/records/6907230
    Format: YOLOv5 PyTorch compatible
    Categories: 2 (plastic, trash)
    Images: 1,220 total (1,100 train, 120 test)
    """
    
    CATEGORIES = {
        0: 'plastic',
        1: 'trash'
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transforms=None,
        image_size: int = 416
    ):
        """
        Initialize UPD Dataset
        
        Args:
            root_dir: Path to UPD dataset root
            split: 'train', 'val', or 'test'
            transforms: Albumentations transforms
            image_size: Target image size (default: 416)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.image_size = image_size
        
        # Setup paths
        self.images_dir = self.root_dir / split / 'images'
        self.labels_dir = self.root_dir / split / 'labels'
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Get all image files
        self.image_files = sorted([
            f for f in self.images_dir.glob('*.*') 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"✓ Loaded {len(self.image_files)} UPD images from '{split}' set")
        print(f"  Images dir: {self.images_dir}")
        print(f"  Labels dir: {self.labels_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get single item from dataset
        
        Returns:
            image: Tensor (C, H, W)
            target: Dict with 'boxes', 'labels', 'image_id'
        """
        img_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        
        # Load YOLO format labels
        label_path = self.labels_dir / f'{img_path.stem}.txt'
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        # YOLO format: class_id x_center y_center width height (normalized)
                        class_id = int(parts)
                        x_center = float(parts)
                        y_center = float(parts)
                        width = float(parts)
                        height = float(parts)
                        
                        # Convert normalized YOLO to pixel coordinates [x1, y1, x2, y2]
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h
                        
                        # Ensure boxes are within image bounds
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        x2 = max(0, min(x2, img_w))
                        y2 = max(0, min(y2, img_h))
                        
                        # Only add valid boxes
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id + 1)  # +1 for 1-indexed (0 = background)
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply augmentations
        if self.transforms:
            try:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes if len(boxes) > 0 else [],
                    labels=labels if len(labels) > 0 else []
                )
                image = transformed['image']
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['labels'], dtype=np.int64)
            except Exception as e:
                print(f"Warning: Transform failed for {img_path.name}: {e}")
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        boxes = torch.from_numpy(boxes) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.from_numpy(labels) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target
    
    def get_category_name(self, category_id: int) -> str:
        """Get category name from ID"""
        return self.CATEGORIES.get(category_id, 'unknown')
    
    def get_dataset_statistics(self) -> Dict:
        """Calculate dataset statistics"""
        total_boxes = 0
        category_counts = {0: 0, 1: 0}
        
        for idx in range(len(self)):
            _, target = self[idx]
            total_boxes += len(target['labels'])
            for label in target['labels']:
                category_counts[label.item() - 1] += 1
        
        stats = {
            'total_images': len(self),
            'total_boxes': total_boxes,
            'avg_boxes_per_image': total_boxes / len(self) if len(self) > 0 else 0,
            'category_distribution': {
                'plastic': category_counts,
                'trash': category_counts
            }
        }
        
        return stats


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles variable number of boxes per image
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets
