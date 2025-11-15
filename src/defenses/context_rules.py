"""
Contextual rule engine for environmental, temporal, and multi-signal validation.
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ContextRuleEngine:
    """
    Contextual rule engine for validating detections based on environmental and temporal cues.
    """
    
    def __init__(
        self,
        temporal_check: bool = True,
        spatial_check: bool = True,
        environmental_check: bool = True,
        enabled: bool = True
    ):
        """
        Initialize context rule engine.
        
        Args:
            temporal_check: Enable temporal consistency checks
            spatial_check: Enable spatial consistency checks
            environmental_check: Enable environmental cue checks
            enabled: Whether defense is enabled
        """
        self.temporal_check = temporal_check
        self.spatial_check = spatial_check
        self.environmental_check = environmental_check
        self.enabled = enabled
        
        # Store detection history for temporal checks
        self.detection_history = []
        self.max_history = 100
        
        # Environmental state
        self.environmental_state = {
            'lighting': 'normal',
            'time_of_day': None,
            'location': None
        }
        
        logger.info(f"Initialized ContextRuleEngine (temporal={temporal_check}, spatial={spatial_check}, environmental={environmental_check}, enabled={enabled})")
    
    def update_environmental_state(self, state: Dict[str, Any]) -> None:
        """
        Update environmental state information.
        
        Args:
            state: Dictionary with environmental information
        """
        self.environmental_state.update(state)
        if 'time_of_day' not in self.environmental_state:
            hour = datetime.now().hour
            if 6 <= hour < 12:
                self.environmental_state['time_of_day'] = 'morning'
            elif 12 <= hour < 18:
                self.environmental_state['time_of_day'] = 'afternoon'
            elif 18 <= hour < 22:
                self.environmental_state['time_of_day'] = 'evening'
            else:
                self.environmental_state['time_of_day'] = 'night'
    
    def check_temporal_consistency(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check temporal consistency of detection.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Temporal check result
        """
        if not self.temporal_check or len(self.detection_history) == 0:
            return {
                'passed': True,
                'reason': 'no_history' if len(self.detection_history) == 0 else 'disabled'
            }
        
        # Check if detection is consistent with recent history
        recent_detections = self.detection_history[-10:]  # Last 10 detections
        
        # Extract classes from recent history
        recent_classes = [d.get('class') for d in recent_detections if 'class' in d]
        
        if len(recent_classes) == 0:
            return {'passed': True, 'reason': 'no_class_history'}
        
        current_class = detection.get('class')
        
        # Check if current class appears in recent history
        class_consistency = current_class in recent_classes if current_class is not None else False
        
        # Check timing - detections should not be too frequent (potential attack)
        if len(self.detection_history) >= 2:
            time_diffs = []
            for i in range(1, len(self.detection_history)):
                if 'timestamp' in self.detection_history[i] and 'timestamp' in self.detection_history[i-1]:
                    diff = (self.detection_history[i]['timestamp'] - self.detection_history[i-1]['timestamp']).total_seconds()
                    time_diffs.append(diff)
            
            if time_diffs:
                avg_time_diff = np.mean(time_diffs)
                # Flag if detections are too frequent (< 0.1 seconds apart)
                too_frequent = avg_time_diff < 0.1
            else:
                too_frequent = False
        else:
            too_frequent = False
        
        passed = class_consistency and not too_frequent
        
        return {
            'passed': passed,
            'class_consistency': class_consistency,
            'too_frequent': too_frequent,
            'reason': 'temporal_consistent' if passed else 'temporal_inconsistent'
        }
    
    def check_spatial_consistency(self, detection: Dict[str, Any], previous_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check spatial consistency of detection.
        
        Args:
            detection: Current detection
            previous_detections: Previous detections in sequence
            
        Returns:
            Spatial check result
        """
        if not self.spatial_check:
            return {'passed': True, 'reason': 'disabled'}
        
        # Check if bounding box location is consistent
        if 'bbox' not in detection or len(previous_detections) == 0:
            return {'passed': True, 'reason': 'no_bbox_or_history'}
        
        current_bbox = detection['bbox']
        if len(current_bbox) != 4:
            return {'passed': True, 'reason': 'invalid_bbox'}
        
        # Get center of current bbox
        cx = (current_bbox[0] + current_bbox[2]) / 2
        cy = (current_bbox[1] + current_bbox[3]) / 2
        
        # Check consistency with previous detections
        previous_centers = []
        for prev_det in previous_detections[-5:]:  # Last 5 detections
            if 'bbox' in prev_det and len(prev_det['bbox']) == 4:
                prev_bbox = prev_det['bbox']
                prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                previous_centers.append((prev_cx, prev_cy))
        
        if len(previous_centers) == 0:
            return {'passed': True, 'reason': 'no_previous_centers'}
        
        # Check if current center is close to previous centers
        distances = [np.sqrt((cx - pcx)**2 + (cy - pcy)**2) for pcx, pcy in previous_centers]
        avg_distance = np.mean(distances)
        
        # Threshold: if average distance is too large, might be inconsistent
        max_distance = 100  # pixels
        passed = avg_distance < max_distance
        
        return {
            'passed': passed,
            'avg_distance': float(avg_distance),
            'reason': 'spatial_consistent' if passed else 'spatial_inconsistent'
        }
    
    def check_environmental_cues(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check environmental cues for detection validity.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Environmental check result
        """
        if not self.environmental_check:
            return {'passed': True, 'reason': 'disabled'}
        
        # Example: Check if detection makes sense given time of day
        time_of_day = self.environmental_state.get('time_of_day')
        lighting = self.environmental_state.get('lighting', 'normal')
        
        # Example rule: Some objects are less likely at night
        # This is a placeholder - real rules would be domain-specific
        passed = True
        reason = 'environmental_ok'
        
        # Example: If lighting is very poor, reduce confidence
        if lighting == 'dark' and detection.get('confidence', 1.0) > 0.9:
            # High confidence in dark conditions might be suspicious
            passed = False
            reason = 'high_confidence_dark_lighting'
        
        return {
            'passed': passed,
            'time_of_day': time_of_day,
            'lighting': lighting,
            'reason': reason
        }
    
    def validate_detection(self, detection: Dict[str, Any], previous_detections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Validate detection using all context rules.
        
        Args:
            detection: Current detection
            previous_detections: Previous detections for context
            
        Returns:
            Validation result
        """
        if not self.enabled:
            return {
                'valid': True,
                'passed_all': True,
                'checks': {}
            }
        
        if previous_detections is None:
            previous_detections = self.detection_history
        
        # Run all checks
        temporal_result = self.check_temporal_consistency(detection)
        spatial_result = self.check_spatial_consistency(detection, previous_detections)
        environmental_result = self.check_environmental_cues(detection)
        
        # All checks must pass
        passed_all = (
            temporal_result['passed'] and
            spatial_result['passed'] and
            environmental_result['passed']
        )
        
        # Add to history
        detection_with_timestamp = detection.copy()
        detection_with_timestamp['timestamp'] = datetime.now()
        self.detection_history.append(detection_with_timestamp)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return {
            'valid': passed_all,
            'passed_all': passed_all,
            'checks': {
                'temporal': temporal_result,
                'spatial': spatial_result,
                'environmental': environmental_result
            }
        }


