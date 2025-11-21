"""
Gradient-based saliency analysis for adversarial patch detection.
Uses model gradients to identify regions that strongly influence predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class GradientSaliencyDefense:
    """
    Detects adversarial patches using gradient-based saliency maps.
    Patches create strong gradients in the model's decision boundary.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        saliency_threshold: float = 0.7,
        patch_size_estimate: Tuple[int, int] = (100, 100),
        mask_patches: bool = True,
        enabled: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize gradient saliency defense.
        
        Args:
            model: Model to compute gradients from (if None, must be provided in __call__)
            saliency_threshold: Threshold for flagging high-saliency regions
            patch_size_estimate: Estimated patch size (H, W)
            mask_patches: Whether to mask detected patches
            enabled: Whether defense is enabled
            device: Computing device
        """
        self.model = model
        self.saliency_threshold = saliency_threshold
        self.patch_size_estimate = patch_size_estimate
        self.mask_patches = mask_patches
        self.enabled = enabled
        self.device = device
        
        if model is not None:
            self.model.eval()
            self.model.to(device)
        
        logger.info(f"Initialized GradientSaliencyDefense (enabled={enabled})")
    
    def compute_gradcam_saliency(
        self,
        image: torch.Tensor,
        model: nn.Module,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM saliency map.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            model: Model to compute gradients from
            target_class: Target class (if None, uses predicted class)
            
        Returns:
            Saliency map (H, W)
        """
        # Handle batch dimension
        is_batch = len(image.shape) == 4
        if not is_batch:
            image = image.unsqueeze(0)
        
        image = image.to(self.device).float()  # Ensure float32
        image.requires_grad_(True)
        
        # Forward pass
        output = model(image)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward pass
        model.zero_grad()
        loss = output[0, target_class] if isinstance(target_class, int) else output[0, target_class[0]]
        loss.backward()
        
        # Get gradients
        gradients = image.grad.data
        
        # Get activations from last convolutional layer
        # This is a simplified version - in practice, you'd hook into specific layers
        # For now, we'll use the gradients directly
        
        # Compute saliency as magnitude of gradients
        saliency = torch.abs(gradients)
        saliency = saliency.squeeze(0)  # Remove batch dimension
        saliency = saliency.sum(dim=0)  # Sum across channels
        
        # Normalize
        saliency = saliency / (saliency.max() + 1e-10)
        
        # Convert to numpy
        saliency_np = saliency.detach().cpu().numpy()
        
        return saliency_np
    
    def compute_integrated_gradients_saliency(
        self,
        image: torch.Tensor,
        model: nn.Module,
        target_class: Optional[int] = None,
        steps: int = 50
    ) -> np.ndarray:
        """
        Compute Integrated Gradients saliency map.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            model: Model to compute gradients from
            target_class: Target class (if None, uses predicted class)
            steps: Number of integration steps
            
        Returns:
            Saliency map (H, W)
        """
        # Handle batch dimension
        is_batch = len(image.shape) == 4
        if not is_batch:
            image = image.unsqueeze(0)
        
        image = image.to(self.device).float()  # Ensure float32
        
        # Baseline (black image)
        baseline = torch.zeros_like(image)
        
        # Forward pass to get target class
        with torch.no_grad():
            output = model(image)
            if target_class is None:
                target_class = torch.argmax(output, dim=1)
        
        # Compute integrated gradients
        alphas = torch.linspace(0, 1, steps).to(self.device)
        integrated_gradients = torch.zeros_like(image)
        
        for alpha in alphas:
            # Interpolate between baseline and image
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = model(interpolated)
            
            # Backward pass
            model.zero_grad()
            loss = output[0, target_class] if isinstance(target_class, int) else output[0, target_class[0]]
            loss.backward()
            
            # Accumulate gradients
            integrated_gradients += interpolated.grad.data
        
        # Average and multiply by (image - baseline)
        integrated_gradients = integrated_gradients / steps
        integrated_gradients = integrated_gradients * (image - baseline)
        
        # Compute saliency as magnitude
        saliency = torch.abs(integrated_gradients)
        saliency = saliency.squeeze(0)  # Remove batch dimension
        saliency = saliency.sum(dim=0)  # Sum across channels
        
        # Normalize
        saliency = saliency / (saliency.max() + 1e-10)
        
        # Convert to numpy
        saliency_np = saliency.detach().cpu().numpy()
        
        return saliency_np
    
    def detect_patch(
        self,
        image: torch.Tensor,
        model: Optional[nn.Module] = None,
        method: str = 'gradcam'
    ) -> Dict[str, Any]:
        """
        Detect adversarial patch using saliency analysis.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            model: Model to compute gradients from (if None, uses self.model)
            method: Saliency method ('gradcam' or 'integrated_gradients')
            
        Returns:
            Detection result dictionary
        """
        if not self.enabled:
            return {
                'patch_detected': False,
                'confidence': 0.0,
                'patch_location': None
            }
        
        model = model or self.model
        if model is None:
            logger.warning("No model provided for saliency computation")
            return {
                'patch_detected': False,
                'confidence': 0.0,
                'patch_location': None
            }
        
        # Compute saliency map
        if method == 'gradcam':
            saliency_map = self.compute_gradcam_saliency(image, model)
        elif method == 'integrated_gradients':
            saliency_map = self.compute_integrated_gradients_saliency(image, model)
        else:
            raise ValueError(f"Unknown saliency method: {method}")
        
        # Find high-saliency regions
        high_saliency_mask = saliency_map > self.saliency_threshold
        
        # Find largest connected component
        from scipy import ndimage
        if np.any(high_saliency_mask):
            labeled, num_features = ndimage.label(high_saliency_mask)
            if num_features > 0:
                component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
                largest_component = np.argmax(component_sizes) + 1
                patch_mask = (labeled == largest_component)
                
                # Get bounding box
                coords = np.argwhere(patch_mask)
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    patch_location = (x_min, y_min, x_max, y_max)
                    
                    # Compute confidence based on saliency
                    patch_saliency = saliency_map[patch_mask].mean()
                    patch_area = np.sum(patch_mask)
                    expected_area = self.patch_size_estimate[0] * self.patch_size_estimate[1]
                    size_ratio = min(patch_area / expected_area, 1.0)
                    
                    confidence = min(1.0, patch_saliency * size_ratio)
                    
                    return {
                        'patch_detected': True,
                        'confidence': float(confidence),
                        'patch_location': patch_location,
                        'saliency_map': saliency_map,
                        'patch_saliency': float(patch_saliency)
                    }
        
        return {
            'patch_detected': False,
            'confidence': 0.0,
            'patch_location': None,
            'saliency_map': saliency_map
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
        
        # Mask the patch region
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
    
    def __call__(
        self,
        image: torch.Tensor,
        model: Optional[nn.Module] = None,
        method: str = 'gradcam'
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process image through gradient saliency defense.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            model: Model to compute gradients from (if None, uses self.model)
            method: Saliency method ('gradcam' or 'integrated_gradients')
            
        Returns:
            Tuple of (processed_image, detection_result)
        """
        if not self.enabled:
            return image, {'patch_detected': False}
        
        # Detect patch
        detection_result = self.detect_patch(image, model, method)
        
        # Mask if detected and masking enabled
        if detection_result['patch_detected'] and self.mask_patches:
            processed_image = self.mask_patch(image, detection_result)
        else:
            processed_image = image
        
        return processed_image, detection_result

