"""
Utilities for applying adversarial patches to images.
"""
import torch
import numpy as np
import cv2
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PatchApplier:
    """
    Utility class for applying adversarial patches to images.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize patch applier.
        
        Args:
            device: Computing device
        """
        self.device = device
    
    def apply_patch(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        location: Tuple[int, int, int, int],
        blend_alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Apply patch to image at specified location.
        
        Args:
            image: Input image tensor (B, C, H, W) or (C, H, W)
            patch: Patch tensor (C, H, W) in [0, 1] range
            location: (x1, y1, x2, y2) patch location
            blend_alpha: Blending factor (1.0 = full replacement)
            
        Returns:
            Image with patch applied
        """
        is_batch = len(image.shape) == 4
        
        if is_batch:
            batch_size = image.shape[0]
            result = []
            for i in range(batch_size):
                result.append(self._apply_patch_single(image[i], patch, location, blend_alpha))
            return torch.stack(result)
        else:
            return self._apply_patch_single(image, patch, location, blend_alpha)
    
    def _apply_patch_single(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        location: Tuple[int, int, int, int],
        blend_alpha: float
    ) -> torch.Tensor:
        """
        Apply patch to a single image.
        
        Args:
            image: Input image tensor (C, H, W)
            patch: Patch tensor (C, H, W)
            location: (x1, y1, x2, y2) patch location
            blend_alpha: Blending factor
            
        Returns:
            Image with patch applied
        """
        x1, y1, x2, y2 = location
        
        # Ensure location is within image bounds
        _, img_h, img_w = image.shape
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(x1, min(x2, img_w))
        y2 = max(y1, min(y2, img_h))
        
        # Resize patch to fit location
        patch_h = y2 - y1
        patch_w = x2 - x1
        
        if patch_h <= 0 or patch_w <= 0:
            return image
        
        # Resize patch
        patch_resized = torch.nn.functional.interpolate(
            patch.unsqueeze(0),
            size=(patch_h, patch_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Clone image to avoid modifying original
        result = image.clone()
        
        # Apply patch with blending
        result[:, y1:y2, x1:x2] = (
            (1 - blend_alpha) * result[:, y1:y2, x1:x2] +
            blend_alpha * patch_resized
        )
        
        return result
    
    def apply_patch_random_location(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        min_size_ratio: float = 0.1,
        max_size_ratio: float = 0.3
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Apply patch at random location.
        
        Args:
            image: Input image tensor
            patch: Patch tensor
            min_size_ratio: Minimum patch size as ratio of image
            max_size_ratio: Maximum patch size as ratio of image
            
        Returns:
            Tuple of (patched_image, location)
        """
        is_batch = len(image.shape) == 4
        
        if is_batch:
            _, _, img_h, img_w = image.shape
        else:
            _, img_h, img_w = image.shape
        
        # Random patch size
        patch_size_ratio = np.random.uniform(min_size_ratio, max_size_ratio)
        patch_h = int(img_h * patch_size_ratio)
        patch_w = int(img_w * patch_size_ratio)
        
        # Random location
        max_x = img_w - patch_w
        max_y = img_h - patch_h
        
        if max_x <= 0 or max_y <= 0:
            # Patch too large, use center
            x1 = max(0, (img_w - patch_w) // 2)
            y1 = max(0, (img_h - patch_h) // 2)
        else:
            x1 = np.random.randint(0, max_x)
            y1 = np.random.randint(0, max_y)
        
        x2 = x1 + patch_w
        y2 = y1 + patch_h
        
        location = (x1, y1, x2, y2)
        patched_image = self.apply_patch(image, patch, location)
        
        return patched_image, location
    
    def apply_patch_with_transform(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        location: Tuple[int, int, int, int],
        rotation: float = 0.0,
        scale: float = 1.0,
        brightness: float = 1.0
    ) -> torch.Tensor:
        """
        Apply patch with geometric and photometric transformations.
        
        Args:
            image: Input image tensor
            patch: Patch tensor
            location: (x1, y1, x2, y2) patch location
            rotation: Rotation angle in degrees
            scale: Scale factor
            brightness: Brightness multiplier
            
        Returns:
            Transformed patch applied to image
        """
        x1, y1, x2, y2 = location
        
        # Apply transformations to patch
        patch_transformed = patch.clone()
        
        # Brightness
        patch_transformed = patch_transformed * brightness
        patch_transformed = torch.clamp(patch_transformed, 0, 1)
        
        # Rotation and scale (simplified - would use affine transform in practice)
        if rotation != 0.0 or scale != 1.0:
            # Convert to numpy for OpenCV transform
            patch_np = patch_transformed.permute(1, 2, 0).cpu().numpy()
            patch_uint8 = (patch_np * 255).astype(np.uint8)
            
            h, w = patch_uint8.shape[:2]
            center = (w // 2, h // 2)
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D(center, rotation, scale)
            patch_transformed_np = cv2.warpAffine(patch_uint8, M, (w, h))
            
            # Convert back to tensor
            patch_transformed = torch.from_numpy(
                patch_transformed_np.astype(np.float32) / 255.0
            ).permute(2, 0, 1).to(patch.device)
        
        return self.apply_patch(image, patch_transformed, location)


