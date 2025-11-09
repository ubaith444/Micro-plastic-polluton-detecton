"""Image Preprocessing Utilities"""

import cv2
import numpy as np
import torch
from typing import Tuple


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (416, 416),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: Input image (BGR or RGB)
        target_size: Target (width, height)
        normalize: Apply ImageNet normalization
    
    Returns:
        Preprocessed image
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to float
    image = image.astype(np.float32) / 255.0
    
    # Normalize
    if normalize:
        image = normalize_image(image)
    
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image * std) + mean
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)
