"""
Frequency domain analysis for adversarial patch detection.
Adversarial patches often have distinctive frequency signatures.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
import logging
from scipy import ndimage
from scipy.fft import fft2, fftshift

logger = logging.getLogger(__name__)


class FrequencyPatchDetector:
    """
    Detects adversarial patches using frequency domain analysis.
    """
    
    def __init__(
        self,
        high_freq_threshold: float = 0.3,
        anomaly_threshold: float = 2.0,
        patch_size_estimate: Tuple[int, int] = (100, 100),
        enabled: bool = True
    ):
        """
        Initialize frequency-based patch detector.
        
        Args:
            high_freq_threshold: Threshold for high-frequency content
            anomaly_threshold: Threshold for frequency anomalies (std deviations)
            patch_size_estimate: Estimated patch size (H, W)
            enabled: Whether detector is enabled
        """
        self.high_freq_threshold = high_freq_threshold
        self.anomaly_threshold = anomaly_threshold
        self.patch_size_estimate = patch_size_estimate
        self.enabled = enabled
        
        logger.info(f"Initialized FrequencyPatchDetector (enabled={enabled})")
    
    def compute_frequency_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 2D FFT spectrum of image.
        
        Args:
            image: Input image (H, W) in uint8 format
            
        Returns:
            Magnitude spectrum (H, W)
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Compute FFT
        fft = fft2(img_float)
        fft_shifted = fftshift(fft)
        
        # Compute magnitude spectrum
        magnitude = np.abs(fft_shifted)
        
        return magnitude
    
    def detect_high_frequency_anomalies(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Detect anomalous high-frequency regions.
        
        Args:
            magnitude: Magnitude spectrum (H, W)
            
        Returns:
            Anomaly map (H, W)
        """
        h, w = magnitude.shape
        
        # Get center (DC component)
        center_y, center_x = h // 2, w // 2
        
        # Create distance map from center
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance / max_distance
        
        # High frequency region (outer 30% of spectrum)
        high_freq_mask = normalized_distance > (1.0 - self.high_freq_threshold)
        
        # Compute statistics of high-frequency region
        high_freq_values = magnitude[high_freq_mask]
        if len(high_freq_values) > 0:
            mean_hf = np.mean(high_freq_values)
            std_hf = np.std(high_freq_values)
            
            # Find anomalies (values significantly above mean)
            anomaly_map = np.zeros_like(magnitude)
            anomaly_map[high_freq_mask] = (magnitude[high_freq_mask] - mean_hf) / (std_hf + 1e-10)
            
            return anomaly_map
        
        return np.zeros_like(magnitude)
    
    def detect_patch(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Detect adversarial patch using frequency analysis.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Detection result dictionary
        """
        if not self.enabled:
            return {
                'patch_detected': False,
                'confidence': 0.0,
                'patch_location': None
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
        
        # Convert to grayscale
        if len(img_uint8.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8
        
        # Compute frequency spectrum
        magnitude = self.compute_frequency_spectrum(gray)
        
        # Detect anomalies
        anomaly_map = self.detect_high_frequency_anomalies(magnitude)
        
        # Convert frequency anomalies back to spatial domain
        # (This is a simplified approach - in practice, you'd use inverse FFT)
        # For now, we'll use the spatial correlation of high-frequency content
        
        # Compute spatial high-frequency content using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(np.float32), (31, 31), 0)
        
        # Normalize
        edge_density = edge_density / (edge_density.max() + 1e-10)
        
        # Find regions with high edge density (potential patches)
        high_edge_mask = edge_density > 0.5
        
        if np.any(high_edge_mask):
            # Find largest connected component
            labeled, num_features = ndimage.label(high_edge_mask)
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
                    
                    # Compute confidence based on edge density and frequency anomalies
                    patch_edge_density = edge_density[patch_mask].mean()
                    patch_area = np.sum(patch_mask)
                    expected_area = self.patch_size_estimate[0] * self.patch_size_estimate[1]
                    size_ratio = min(patch_area / expected_area, 1.0)
                    
                    confidence = min(1.0, patch_edge_density * size_ratio)
                    
                    return {
                        'patch_detected': True,
                        'confidence': float(confidence),
                        'patch_location': patch_location,
                        'edge_density': float(patch_edge_density)
                    }
        
        return {
            'patch_detected': False,
            'confidence': 0.0,
            'patch_location': None
        }
    
    def filter_high_frequencies(self, image: torch.Tensor, cutoff: float = 0.3) -> torch.Tensor:
        """
        Apply low-pass filter to remove high-frequency content.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            cutoff: Frequency cutoff (0-1)
            
        Returns:
            Filtered image tensor
        """
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
        
        # Apply Gaussian blur (low-pass filter)
        kernel_size = int(5 + (1.0 - cutoff) * 20)  # Larger kernel for lower cutoff
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        filtered = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        filtered_float = filtered.astype(np.float32) / 255.0
        
        # Renormalize if needed
        if image.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            filtered_float = (filtered_float - mean) / std
        
        # Convert back to tensor
        img_tensor = torch.from_numpy(filtered_float).permute(2, 0, 1).to(image.device)
        
        # Restore batch dimension if needed
        if is_batch:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor


class FrequencyPatchDefense:
    """
    Complete frequency-based defense that detects and filters patches.
    """
    
    def __init__(
        self,
        high_freq_threshold: float = 0.3,
        anomaly_threshold: float = 2.0,
        patch_size_estimate: Tuple[int, int] = (100, 100),
        filter_patches: bool = True,
        filter_cutoff: float = 0.3,
        enabled: bool = True
    ):
        """
        Initialize frequency-based defense.
        
        Args:
            high_freq_threshold: Threshold for high-frequency content
            anomaly_threshold: Threshold for frequency anomalies
            patch_size_estimate: Estimated patch size (H, W)
            filter_patches: Whether to filter detected patches
            filter_cutoff: Frequency cutoff for filtering
            enabled: Whether defense is enabled
        """
        self.detector = FrequencyPatchDetector(
            high_freq_threshold=high_freq_threshold,
            anomaly_threshold=anomaly_threshold,
            patch_size_estimate=patch_size_estimate,
            enabled=enabled
        )
        self.filter_patches = filter_patches
        self.filter_cutoff = filter_cutoff
        self.enabled = enabled
        
        logger.info(f"Initialized FrequencyPatchDefense (filter_patches={filter_patches}, enabled={enabled})")
    
    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process image through frequency-based defense.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Tuple of (processed_image, detection_result)
        """
        if not self.enabled:
            return image, {'patch_detected': False}
        
        # Detect patch
        detection_result = self.detector.detect_patch(image)
        
        # Filter if detected and filtering enabled
        if detection_result['patch_detected'] and self.filter_patches:
            processed_image = self.detector.filter_high_frequencies(image, self.filter_cutoff)
        else:
            processed_image = image
        
        return processed_image, detection_result

