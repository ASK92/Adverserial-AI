"""
Multi-frame smoothing defense requiring consensus across consecutive frames.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MultiFrameSmoothing:
    """
    Multi-frame smoothing defense requiring detection consensus across frames.
    """
    
    def __init__(
        self,
        frame_window: int = 5,
        consensus_threshold: float = 0.6,
        enabled: bool = True
    ):
        """
        Initialize multi-frame smoothing defense.
        
        Args:
            frame_window: Number of consecutive frames to consider
            consensus_threshold: Minimum ratio of frames that must agree
            enabled: Whether defense is enabled
        """
        self.frame_window = frame_window
        self.consensus_threshold = consensus_threshold
        self.enabled = enabled
        
        # Store recent frame predictions
        self.frame_history = deque(maxlen=frame_window)
        
        logger.info(f"Initialized MultiFrameSmoothing defense (window={frame_window}, threshold={consensus_threshold}, enabled={enabled})")
    
    def add_frame(self, prediction: Dict[str, Any]) -> None:
        """
        Add a new frame prediction to the history.
        
        Args:
            prediction: Prediction dictionary with 'class', 'confidence', etc.
        """
        if not self.enabled:
            return
        
        self.frame_history.append(prediction)
    
    def get_consensus(self) -> Dict[str, Any]:
        """
        Get consensus prediction from frame history.
        
        Returns:
            Consensus result dictionary
        """
        if not self.enabled or len(self.frame_history) < self.frame_window:
            return {
                'consensus_reached': False,
                'predicted_class': None,
                'confidence': 0.0,
                'frames_analyzed': len(self.frame_history)
            }
        
        # Extract classes from history
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
                'frames_analyzed': len(self.frame_history)
            }
        
        # Find most common class
        unique_classes, counts = np.unique(classes, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_class = unique_classes[most_common_idx]
        agreement_ratio = counts[most_common_idx] / len(classes)
        
        # Check if consensus threshold is met
        consensus_reached = agreement_ratio >= self.consensus_threshold
        
        # Average confidence
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        return {
            'consensus_reached': consensus_reached,
            'predicted_class': int(most_common_class),
            'confidence': float(avg_confidence),
            'agreement_ratio': float(agreement_ratio),
            'frames_analyzed': len(self.frame_history)
        }
    
    def reset(self) -> None:
        """Reset frame history."""
        self.frame_history.clear()
    
    def is_valid_detection(self, prediction: Dict[str, Any]) -> bool:
        """
        Check if a detection is valid based on frame history.
        
        Args:
            prediction: Current frame prediction
            
        Returns:
            True if detection is valid (consensus reached)
        """
        if not self.enabled:
            return True
        
        self.add_frame(prediction)
        consensus = self.get_consensus()
        return consensus['consensus_reached']


