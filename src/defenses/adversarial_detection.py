"""
Adversarial example detection using norm-based and learned detectors.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NormBasedDetector:
    """
    Simple norm-based adversarial example detector.
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize norm-based detector.
        
        Args:
            threshold: Threshold for L2 norm to flag adversarial examples
        """
        self.threshold = threshold
    
    def detect(self, image: torch.Tensor, reference: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Detect adversarial examples based on norm.
        
        Args:
            image: Input image tensor
            reference: Optional reference image for comparison
            
        Returns:
            Detection result dictionary
        """
        # Compute L2 norm
        l2_norm = torch.norm(image).item()
        
        # If reference provided, compute distance
        if reference is not None:
            distance = torch.norm(image - reference).item()
            is_adversarial = distance > self.threshold
        else:
            # Use absolute threshold
            is_adversarial = l2_norm > self.threshold
            distance = l2_norm
        
        return {
            'is_adversarial': is_adversarial,
            'norm': l2_norm,
            'distance': distance if reference is not None else None,
            'confidence': min(1.0, distance / self.threshold) if reference is not None else min(1.0, l2_norm / self.threshold)
        }


class OODDetector(nn.Module):
    """
    Out-of-distribution detector using learned features.
    """
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        """
        Initialize OOD detector.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.detector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features (B, feature_dim)
            
        Returns:
            OOD probability (B, 1)
        """
        return self.detector(features)
    
    def detect(self, features: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect out-of-distribution samples.
        
        Args:
            features: Input features
            threshold: Detection threshold
            
        Returns:
            Detection result dictionary
        """
        with torch.no_grad():
            ood_prob = self.forward(features)
            is_ood = (ood_prob > threshold).cpu().numpy()
        
        return {
            'is_ood': is_ood,
            'ood_probability': ood_prob.cpu().numpy(),
            'confidence': ood_prob.item() if len(ood_prob) == 1 else ood_prob.cpu().numpy()
        }


class AdversarialDetection:
    """
    Combined adversarial detection system using norm-based and learned detectors.
    """
    
    def __init__(
        self,
        norm_threshold: float = 0.1,
        ood_detector: bool = True,
        ood_threshold: float = 0.5,
        enabled: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize adversarial detection defense.
        
        Args:
            norm_threshold: Threshold for norm-based detection
            ood_detector: Whether to use OOD detector
            ood_threshold: Threshold for OOD detection
            enabled: Whether defense is enabled
            device: Computing device
        """
        self.norm_detector = NormBasedDetector(threshold=norm_threshold)
        self.ood_detector_enabled = ood_detector
        self.ood_threshold = ood_threshold
        self.enabled = enabled
        self.device = device
        
        if ood_detector:
            # Initialize OOD detector (would be trained in practice)
            self.ood_detector = OODDetector().to(device)
            self.ood_detector.eval()
            logger.info("OOD detector initialized (untrained - would need training in production)")
        else:
            self.ood_detector = None
        
        logger.info(f"Initialized AdversarialDetection defense (enabled={enabled})")
    
    def detect(self, image: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Detect adversarial examples.
        
        Args:
            image: Input image tensor
            features: Optional feature vector for OOD detection
            
        Returns:
            Detection result dictionary
        """
        if not self.enabled:
            return {
                'is_adversarial': False,
                'confidence': 0.0,
                'method': 'disabled'
            }
        
        # Norm-based detection
        norm_result = self.norm_detector.detect(image)
        
        # OOD detection if enabled and features provided
        ood_result = None
        if self.ood_detector_enabled and self.ood_detector is not None and features is not None:
            ood_result = self.ood_detector.detect(features, threshold=self.ood_threshold)
        
        # Combine results
        is_adversarial = norm_result['is_adversarial']
        confidence = norm_result['confidence']
        
        if ood_result is not None:
            # Combine both detectors (OR logic - flag if either detects)
            is_adversarial = is_adversarial or ood_result['is_ood'].any()
            confidence = max(confidence, float(ood_result['confidence'].max() if isinstance(ood_result['confidence'], np.ndarray) else ood_result['confidence']))
        
        return {
            'is_adversarial': bool(is_adversarial),
            'confidence': float(confidence),
            'norm_result': norm_result,
            'ood_result': ood_result,
            'method': 'combined' if ood_result is not None else 'norm_only'
        }


