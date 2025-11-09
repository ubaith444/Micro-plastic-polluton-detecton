"""Data Augmentation Pipeline using Albumentations"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(image_size: int = 416):
    """
    Training augmentation pipeline optimized for underwater plastic detection
    
    Args:
        image_size: Target image size (default: 416 for UPD)
    
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        
        # Color augmentations (optimized for underwater)
        A.RandomBrightnessContrast(
            brightness_limit=0.22,
            contrast_limit=0.22,
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=25,
            sat_shift_limit=42,
            val_shift_limit=22,
            p=0.4
        ),
        
        # Convert to grayscale (47% as per UPD specs)
        A.ToGray(p=0.47),
        
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.25),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Cutout (8 boxes with 10% size as per UPD specs)
        A.CoarseDropout(
            max_holes=8,
            max_height=int(image_size * 0.1),
            max_width=int(image_size * 0.1),
            min_holes=1,
            min_height=int(image_size * 0.05),
            min_width=int(image_size * 0.05),
            fill_value=0,
            p=0.5
        ),
        
        # Normalization (ImageNet statistics)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,
        min_area=50
    ))


def get_validation_augmentation(image_size: int = 416):
    """
    Minimal validation augmentation
    
    Args:
        image_size: Target image size (default: 416)
    
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))


def get_test_augmentation(image_size: int = 416):
    """Test augmentation (same as validation)"""
    return get_validation_augmentation(image_size)
