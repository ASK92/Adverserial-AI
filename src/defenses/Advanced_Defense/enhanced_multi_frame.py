"""
Enhanced multi-frame smoothing with temporal gradient checks and stricter consensus.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EnhancedMultiFrameSmoothing:
    """
    Enhanced multi-frame smoothing with:
    - Larger frame windows
    - Stricter consensus requirements
    - Temporal gradient checks
    - Confidence stability requirements
    """
    
    def __init__(
        self,
        frame_window: int = 15,
        consensus_threshold: float = 0.8,
        temporal_gradient_threshold: float = 0.3,
        confidence_stability_threshold: float = 0.1,
        enabled: bool = True
    ):
        """
        Initialize enhanced multi-frame smoothing defense.
        
        Args:
            frame_window: Number of consecutive frames to consider
            consensus_threshold: Minimum ratio of frames that must agree (0-1)
            temporal_gradient_threshold: Maximum allowed change in predictions between frames
            confidence_stability_threshold: Maximum allowed change in confidence
            enabled: Whether defense is enabled
        """
        self.frame_window = frame_window
        self.consensus_threshold = consensus_threshold
        self.temporal_gradient_threshold = temporal_gradient_threshold
        self.confidence_stability_threshold = confidence_stability_threshold
        self.enabled = enabled
        
        # Store recent frame predictions with timestamps
        self.frame_history = deque(maxlen=frame_window)
        
        logger.info(
            f"Initialized EnhancedMultiFrameSmoothing "
            f"(window={frame_window}, consensus={consensus_threshold}, "
            f"temporal_grad={temporal_gradient_threshold}, enabled={enabled})"
        )
    
    def add_frame(self, prediction: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        """
        Add a new frame prediction to the history.
        
        Args:
            prediction: Prediction dictionary with 'class', 'confidence', etc.
            timestamp: Optional timestamp (if None, uses current time)
        """
        if not self.enabled:
            return
        
        import time
        if timestamp is None:
            timestamp = time.time()
        
        frame_data = {
            **prediction,
            'timestamp': timestamp
        }
        
        self.frame_history.append(frame_data)
    
    def check_temporal_gradient(self) -> Dict[str, Any]:
        """
        Check if predictions change too rapidly (potential attack).
        
        Returns:
            Temporal gradient check result
        """
        if len(self.frame_history) < 2:
            return {
                'passed': True,
                'max_gradient': 0.0,
                'reason': 'insufficient_frames'
            }
        
        # Extract classes and confidences
        classes = [f.get('class') for f in self.frame_history]
        confidences = [f.get('confidence', 0.0) for f in self.frame_history]
        timestamps = [f.get('timestamp', 0.0) for f in self.frame_history]
        
        # Compute class changes
        class_changes = 0
        for i in range(1, len(classes)):
            if classes[i] != classes[i-1]:
                class_changes += 1
        
        class_change_rate = class_changes / (len(classes) - 1) if len(classes) > 1 else 0.0
        
        # Compute confidence gradient
        confidence_gradients = []
        for i in range(1, len(confidences)):
            if timestamps[i] != timestamps[i-1]:
                time_diff = timestamps[i] - timestamps[i-1]
                conf_diff = abs(confidences[i] - confidences[i-1])
                gradient = conf_diff / (time_diff + 1e-10)
                confidence_gradients.append(gradient)
        
        max_gradient = max(confidence_gradients) if confidence_gradients else 0.0
        
        # Check if gradient is too high
        passed = (
            class_change_rate <= self.temporal_gradient_threshold and
            max_gradient <= self.confidence_stability_threshold
        )
        
        return {
            'passed': passed,
            'class_change_rate': float(class_change_rate),
            'max_gradient': float(max_gradient),
            'reason': 'temporal_stable' if passed else 'temporal_unstable'
        }
    
    def check_confidence_stability(self) -> Dict[str, Any]:
        """
        Check if confidence is stable across frames.
        
        Returns:
            Confidence stability check result
        """
        if len(self.frame_history) < self.frame_window:
            return {
                'passed': True,
                'std': 0.0,
                'reason': 'insufficient_frames'
            }
        
        confidences = [f.get('confidence', 0.0) for f in self.frame_history]
        conf_std = np.std(confidences)
        conf_mean = np.mean(confidences)
        
        # Coefficient of variation
        cv = conf_std / (conf_mean + 1e-10)
        
        # Pass if coefficient of variation is low
        passed = cv <= self.confidence_stability_threshold
        
        return {
            'passed': passed,
            'std': float(conf_std),
            'mean': float(conf_mean),
            'coefficient_of_variation': float(cv),
            'reason': 'confidence_stable' if passed else 'confidence_unstable'
        }
    
    def get_consensus(self) -> Dict[str, Any]:
        """
        Get consensus prediction from frame history with enhanced checks.
        
        Returns:
            Consensus result dictionary
        """
        if not self.enabled or len(self.frame_history) < self.frame_window:
            return {
                'consensus_reached': False,
                'predicted_class': None,
                'confidence': 0.0,
                'frames_analyzed': len(self.frame_history),
                'temporal_check': {'passed': True},
                'confidence_check': {'passed': True}
            }
        
        # Extract classes and confidences
        classes = []
        confidences = []
        
        for frame_pred in self.frame_history:
            if 'class' in frame_pred:
                classes.append(frame_pred['class'])
            if 'confidence' in frame_pred:
                confidences.append(frame_pred['confidence'])
        
        if len(classes) == 0:
            return {
                'consensus_reached': False,
                'predicted_class': None,
                'confidence': 0.0,
                'frames_analyzed': len(self.frame_history),
                'temporal_check': {'passed': True},
                'confidence_check': {'passed': True}
            }
        
        # Find most common class
        unique_classes, counts = np.unique(classes, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_class = unique_classes[most_common_idx]
        agreement_ratio = counts[most_common_idx] / len(classes)
        
        # Check consensus threshold
        consensus_reached = agreement_ratio >= self.consensus_threshold
        
        # Average confidence
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        # Run additional checks
        temporal_check = self.check_temporal_gradient()
        confidence_check = self.check_confidence_stability()
        
        # All checks must pass
        final_consensus = (
            consensus_reached and
            temporal_check['passed'] and
            confidence_check['passed']
        )
        
        return {
            'consensus_reached': final_consensus,
            'predicted_class': int(most_common_class),
            'confidence': float(avg_confidence),
            'agreement_ratio': float(agreement_ratio),
            'frames_analyzed': len(self.frame_history),
            'temporal_check': temporal_check,
            'confidence_check': confidence_check
        }
    
    def reset(self) -> None:
        """Reset frame history."""
        self.frame_history.clear()
    
    def is_valid_detection(self, prediction: Dict[str, Any], timestamp: Optional[float] = None) -> bool:
        """
        Check if a detection is valid based on frame history.
        
        Args:
            prediction: Current frame prediction
            timestamp: Optional timestamp
            
        Returns:
            True if detection is valid (consensus reached and checks passed)
        """
        if not self.enabled:
            return True
        
        self.add_frame(prediction, timestamp)
        consensus = self.get_consensus()
        return consensus['consensus_reached']

