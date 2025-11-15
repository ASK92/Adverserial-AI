"""
Integrated defense pipeline combining all defense layers.
"""
import torch
from typing import Dict, Any, List, Optional
import logging

from .input_normalization import InputNormalization
from .adversarial_detection import AdversarialDetection
from .multi_frame_smoothing import MultiFrameSmoothing
from .context_rules import ContextRuleEngine

logger = logging.getLogger(__name__)


class DefensePipeline:
    """
    Integrated pipeline combining all defense layers.
    """
    
    def __init__(
        self,
        input_normalization: Optional[InputNormalization] = None,
        adversarial_detection: Optional[AdversarialDetection] = None,
        multi_frame_smoothing: Optional[MultiFrameSmoothing] = None,
        context_rules: Optional[ContextRuleEngine] = None,
        enabled: bool = True
    ):
        """
        Initialize defense pipeline.
        
        Args:
            input_normalization: Input normalization defense
            adversarial_detection: Adversarial detection defense
            multi_frame_smoothing: Multi-frame smoothing defense
            context_rules: Context rule engine
            enabled: Whether pipeline is enabled
        """
        self.input_normalization = input_normalization or InputNormalization(enabled=False)
        self.adversarial_detection = adversarial_detection or AdversarialDetection(enabled=False)
        self.multi_frame_smoothing = multi_frame_smoothing or MultiFrameSmoothing(enabled=False)
        self.context_rules = context_rules or ContextRuleEngine(enabled=False)
        self.enabled = enabled
        
        logger.info(f"Initialized DefensePipeline (enabled={enabled})")
    
    def process_input(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process input through normalization layer.
        
        Args:
            image: Input image tensor
            
        Returns:
            Processed image tensor
        """
        if not self.enabled:
            return image
        
        return self.input_normalization(image)
    
    def validate_detection(
        self,
        image: torch.Tensor,
        prediction: Dict[str, Any],
        features: Optional[torch.Tensor] = None,
        frame_number: int = 0
    ) -> Dict[str, Any]:
        """
        Validate detection through all defense layers.
        
        Args:
            image: Input image tensor
            prediction: Model prediction dictionary
            features: Optional feature vector for OOD detection
            frame_number: Current frame number
            
        Returns:
            Validation result with defense outcomes
        """
        if not self.enabled:
            return {
                'valid': True,
                'passed_all_defenses': True,
                'defense_results': {}
            }
        
        defense_results = {}
        
        # 1. Adversarial detection
        adv_detection_result = self.adversarial_detection.detect(image, features)
        defense_results['adversarial_detection'] = adv_detection_result
        
        if adv_detection_result['is_adversarial']:
            return {
                'valid': False,
                'passed_all_defenses': False,
                'defense_results': defense_results,
                'failed_defense': 'adversarial_detection'
            }
        
        # 2. Multi-frame smoothing
        self.multi_frame_smoothing.add_frame(prediction)
        frame_consensus = self.multi_frame_smoothing.get_consensus()
        defense_results['multi_frame_smoothing'] = frame_consensus
        
        if not frame_consensus['consensus_reached']:
            return {
                'valid': False,
                'passed_all_defenses': False,
                'defense_results': defense_results,
                'failed_defense': 'multi_frame_smoothing'
            }
        
        # 3. Context rules
        context_validation = self.context_rules.validate_detection(prediction)
        defense_results['context_rules'] = context_validation
        
        if not context_validation['valid']:
            return {
                'valid': False,
                'passed_all_defenses': False,
                'defense_results': defense_results,
                'failed_defense': 'context_rules'
            }
        
        # All defenses passed
        return {
            'valid': True,
            'passed_all_defenses': True,
            'defense_results': defense_results
        }
    
    def reset(self) -> None:
        """Reset stateful defenses."""
        self.multi_frame_smoothing.reset()
        self.context_rules.detection_history.clear()


