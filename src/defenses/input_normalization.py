"""
Input normalization defense layer with random transformations.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import logging
from albumentations import (
    RandomBrightnessContrast,
    GaussianBlur,
    Affine,
    GaussNoise,
    Compose
)

logger = logging.getLogger(__name__)


class InputNormalization:
    """
    Input normalization defense with random brightness, blur, affine transforms, and noise.
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        blur_prob: float = 0.3,
        blur_kernel_size: int = 5,
        affine_prob: float = 0.3,
        noise_std: float = 0.05,
        enabled: bool = True
    ):
        """
        Initialize input normalization defense.
        
        Args:
            brightness_range: Range for random brightness adjustment
            blur_prob: Probability of applying blur
            blur_kernel_size: Kernel size for Gaussian blur
            affine_prob: Probability of applying affine transformation
            noise_std: Standard deviation for Gaussian noise
            enabled: Whether defense is enabled
        """
        self.brightness_range = brightness_range
        self.blur_prob = blur_prob
        self.blur_kernel_size = blur_kernel_size
        self.affine_prob = affine_prob
        self.noise_std = noise_std
        self.enabled = enabled
        
        # Setup Albumentations pipeline
        self.transform = Compose([
            RandomBrightnessContrast(
                brightness_limit=(brightness_range[0] - 1.0, brightness_range[1] - 1.0),
                contrast_limit=0.2,
                p=0.5
            ),
            GaussianBlur(blur_limit=(3, blur_kernel_size), p=blur_prob),
            Affine(
                rotate=(-15, 15),
                translate_percent=(-0.1, 0.1),
                scale=(0.9, 1.1),
                p=affine_prob
            ),
            GaussNoise(var_limit=(0, noise_std * 255), p=0.3)
        ])
        
        logger.info(f"Initialized InputNormalization defense (enabled={enabled})")
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply input normalization to image.
        
        Args:
            image: Input image tensor (B, C, H, W) or (C, H, W)
            
        Returns:
            Normalized image tensor
        """
        if not self.enabled:
            return image
        
        # Convert to numpy for albumentations
        is_batch = len(image.shape) == 4
        if is_batch:
            batch_size = image.shape[0]
            processed_images = []
            for i in range(batch_size):
                img = self._process_single_image(image[i])
                processed_images.append(img)
            return torch.stack(processed_images)
        else:
            return self._process_single_image(image)
    
    def _process_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process a single image.
        
        Args:
            image: Image tensor (C, H, W)
            
        Returns:
            Processed image tensor
        """
        # Convert to numpy (H, W, C) format
        img_np = image.permute(1, 2, 0).detach().cpu().numpy()
        
        # Denormalize if needed (assuming ImageNet normalization)
        if img_np.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
        
        # Clip to [0, 1]
        img_np = np.clip(img_np, 0, 1)
        
        # Convert to uint8
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        # Apply transformations
        transformed = self.transform(image=img_uint8)['image']
        
        # Convert back to float tensor
        img_float = transformed.astype(np.float32) / 255.0
        
        # Renormalize if needed
        if image.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_float = (img_float - mean) / std
        
        # Convert back to tensor (C, H, W)
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).to(image.device)
        
        return img_tensor
    
    def apply_brightness(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random brightness adjustment."""
        if np.random.random() > 0.5:
            return image
        
        brightness_factor = np.random.uniform(*self.brightness_range)
        return image * brightness_factor
    
    def apply_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur."""
        if np.random.random() > self.blur_prob:
            return image
        
        # Convert to numpy, apply blur, convert back
        img_np = image.permute(1, 2, 0).detach().cpu().numpy()
        if img_np.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        blurred = cv2.GaussianBlur(img_uint8, (self.blur_kernel_size, self.blur_kernel_size), 0)
        blurred_float = blurred.astype(np.float32) / 255.0
        
        if image.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            blurred_float = (blurred_float - mean) / std
        
        return torch.from_numpy(blurred_float).permute(2, 0, 1).to(image.device)
    
    def apply_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise."""
        noise = torch.randn_like(image) * self.noise_std
        return image + noise
