"""
Entropy-based adversarial patch detection and localization.
Based on "Jedi: Entropy-based Localization and Removal of Adversarial Patches" (ICCV 2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
import logging
from scipy import ndimage
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class EntropyPatchDetector:
    """
    Detects adversarial patches using local entropy analysis.
    Adversarial patches typically have high entropy (random-looking patterns).
    """
    
    def __init__(
        self,
        window_size: int = 32,
        stride: int = 8,
        entropy_threshold: float = 7.0,
        patch_size_estimate: Tuple[int, int] = (100, 100),
        enabled: bool = True
    ):
        """
        Initialize entropy-based patch detector.
        
        Args:
            window_size: Size of sliding window for entropy computation
            stride: Stride for sliding window
            entropy_threshold: Threshold for flagging high-entropy regions
            patch_size_estimate: Estimated patch size (H, W)
            enabled: Whether detector is enabled
        """
        self.window_size = window_size
        self.stride = stride
        self.entropy_threshold = entropy_threshold
        self.patch_size_estimate = patch_size_estimate
        self.enabled = enabled
        
        logger.info(f"Initialized EntropyPatchDetector (window={window_size}, threshold={entropy_threshold}, enabled={enabled})")
    
    def compute_local_entropy(self, image: np.ndarray) -> np.ndarray:
        """
        Compute local entropy map using sliding window.
        
        Args:
            image: Input image (H, W, C) in uint8 format
            
        Returns:
            Entropy map (H, W)
        """
        if len(image.shape) == 3:
            # Convert to grayscale for entropy computation
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        entropy_map = np.zeros((h, w), dtype=np.float32)
        
        # Sliding window
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                window = gray[y:y+self.window_size, x:x+self.window_size]
                
                # Compute histogram
                hist, _ = np.histogram(window.flatten(), bins=256, range=(0, 256))
                hist = hist + 1e-10  # Avoid log(0)
                hist = hist / hist.sum()  # Normalize
                
                # Compute entropy
                ent = entropy(hist, base=2)
                
                # Assign entropy to all pixels in window
                entropy_map[y:y+self.window_size, x:x+self.window_size] = np.maximum(
                    entropy_map[y:y+self.window_size, x:x+self.window_size],
                    ent
                )
        
        return entropy_map
    
    def detect_patch(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Detect adversarial patch in image.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Detection result dictionary
        """
        if not self.enabled:
            return {
                'patch_detected': False,
                'confidence': 0.0,
                'patch_location': None,
                'entropy_map': None
            }
        
        # Handle batch dimension
        is_batch = len(image.shape) == 4
        if is_batch:
            image = image[0]  # Process first image
        
        # Convert to numpy
        img_np = image.permute(1, 2, 0).detach().cpu().numpy()
        
        # Denormalize if needed
        if img_np.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
        
        img_np = np.clip(img_np, 0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        # Compute entropy map
        entropy_map = self.compute_local_entropy(img_uint8)
        
        # Find high-entropy regions
        high_entropy_mask = entropy_map > self.entropy_threshold
        
        # Find largest connected component (likely the patch)
        if np.any(high_entropy_mask):
            labeled, num_features = ndimage.label(high_entropy_mask)
            if num_features > 0:
                # Find largest component
                component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
                largest_component = np.argmax(component_sizes) + 1
                patch_mask = (labeled == largest_component)
                
                # Get bounding box
                coords = np.argwhere(patch_mask)
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    patch_location = (x_min, y_min, x_max, y_max)
                    
                    # Compute confidence based on entropy and size
                    patch_entropy = entropy_map[patch_mask].mean()
                    patch_area = np.sum(patch_mask)
                    expected_area = self.patch_size_estimate[0] * self.patch_size_estimate[1]
                    size_ratio = min(patch_area / expected_area, 1.0)
                    
                    confidence = min(1.0, (patch_entropy / 8.0) * size_ratio)
                    
                    return {
                        'patch_detected': True,
                        'confidence': float(confidence),
                        'patch_location': patch_location,
                        'entropy_map': entropy_map,
                        'patch_mask': patch_mask,
                        'patch_entropy': float(patch_entropy)
                    }
        
        return {
            'patch_detected': False,
            'confidence': 0.0,
            'patch_location': None,
            'entropy_map': entropy_map,
            'patch_mask': None
        }
    
    def mask_patch(self, image: torch.Tensor, detection_result: Dict[str, Any]) -> torch.Tensor:
        """
        Mask detected patch region in image.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            detection_result: Result from detect_patch()
            
        Returns:
            Masked image tensor
        """
        if not detection_result['patch_detected']:
            return image
        
        patch_location = detection_result['patch_location']
        if patch_location is None:
            return image
        
        x_min, y_min, x_max, y_max = patch_location
        
        # Handle batch dimension
        is_batch = len(image.shape) == 4
        if is_batch:
            image = image[0]  # Process first image
        
        # Convert to numpy
        img_np = image.permute(1, 2, 0).detach().cpu().numpy()
        
        # Denormalize if needed
        if img_np.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
        
        img_np = np.clip(img_np, 0, 1)
        
        # Mask the patch region (set to mean color or black)
        img_np[y_min:y_max, x_min:x_max] = 0.5  # Gray mask
        
        # Renormalize if needed
        if image.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np - mean) / std
        
        # Convert back to tensor
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(image.device)
        
        # Restore batch dimension if needed
        if is_batch:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor


class EntropyPatchDefense:
    """
    Complete entropy-based defense that detects and mitigates patches.
    """
    
    def __init__(
        self,
        window_size: int = 32,
        stride: int = 8,
        entropy_threshold: float = 7.0,
        patch_size_estimate: Tuple[int, int] = (100, 100),
        mask_patches: bool = True,
        enabled: bool = True
    ):
        """
        Initialize entropy-based defense.
        
        Args:
            window_size: Size of sliding window for entropy computation
            stride: Stride for sliding window
            entropy_threshold: Threshold for flagging high-entropy regions
            patch_size_estimate: Estimated patch size (H, W)
            mask_patches: Whether to mask detected patches
            enabled: Whether defense is enabled
        """
        self.detector = EntropyPatchDetector(
            window_size=window_size,
            stride=stride,
            entropy_threshold=entropy_threshold,
            patch_size_estimate=patch_size_estimate,
            enabled=enabled
        )
        self.mask_patches = mask_patches
        self.enabled = enabled
        
        logger.info(f"Initialized EntropyPatchDefense (mask_patches={mask_patches}, enabled={enabled})")
    
    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process image through entropy-based defense.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Tuple of (processed_image, detection_result)
        """
        if not self.enabled:
            return image, {'patch_detected': False}
        
        # Detect patch
        detection_result = self.detector.detect_patch(image)
        
        # Mask if detected and masking enabled
        if detection_result['patch_detected'] and self.mask_patches:
            processed_image = self.detector.mask_patch(image, detection_result)
        else:
            processed_image = image
        
        return processed_image, detection_result

