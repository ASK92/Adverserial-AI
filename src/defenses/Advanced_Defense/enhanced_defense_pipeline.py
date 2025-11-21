"""
Enhanced Defense Pipeline integrating all advanced defense mechanisms.
This pipeline can be used alongside or instead of the basic defense pipeline.
"""
import torch
from typing import Dict, Any, List, Optional, Tuple
import logging

from .entropy_detection import EntropyPatchDefense
from .frequency_analysis import FrequencyPatchDefense
from .gradient_saliency import GradientSaliencyDefense
from .enhanced_multi_frame import EnhancedMultiFrameSmoothing

logger = logging.getLogger(__name__)


class EnhancedDefensePipeline:
    """
    Enhanced defense pipeline combining multiple advanced defense mechanisms.
    """
    
    def __init__(
        self,
        entropy_defense: Optional[EntropyPatchDefense] = None,
        frequency_defense: Optional[FrequencyPatchDefense] = None,
        gradient_saliency_defense: Optional[GradientSaliencyDefense] = None,
        enhanced_multi_frame: Optional[EnhancedMultiFrameSmoothing] = None,
        defense_mode: str = 'cascade',  # 'cascade' or 'ensemble'
        enabled: bool = True
    ):
        """
        Initialize enhanced defense pipeline.
        
        Args:
            entropy_defense: Entropy-based patch detection
            frequency_defense: Frequency-based patch detection
            gradient_saliency_defense: Gradient saliency-based detection
            enhanced_multi_frame: Enhanced multi-frame smoothing
            defense_mode: 'cascade' (sequential) or 'ensemble' (vote)
            enabled: Whether pipeline is enabled
        """
        self.entropy_defense = entropy_defense or EntropyPatchDefense(enabled=False)
        self.frequency_defense = frequency_defense or FrequencyPatchDefense(enabled=False)
        self.gradient_saliency_defense = gradient_saliency_defense
        self.enhanced_multi_frame = enhanced_multi_frame or EnhancedMultiFrameSmoothing(enabled=False)
        self.defense_mode = defense_mode
        self.enabled = enabled
        
        logger.info(f"Initialized EnhancedDefensePipeline (mode={defense_mode}, enabled={enabled})")
    
    def process_input(self, image: torch.Tensor, model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """
        Process input through all enabled defenses.
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            model: Model for gradient saliency (if needed)
            
        Returns:
            Processed image tensor
        """
        if not self.enabled:
            return image
        
        processed_image = image
        
        # Apply defenses in sequence (cascade mode)
        if self.defense_mode == 'cascade':
            # 1. Entropy-based detection and masking
            if self.entropy_defense.enabled:
                processed_image, entropy_result = self.entropy_defense(processed_image)
                if entropy_result.get('patch_detected', False):
                    logger.warning(f"Entropy defense detected patch: {entropy_result.get('confidence', 0.0):.3f}")
            
            # 2. Frequency-based filtering
            if self.frequency_defense.enabled:
                # Handle batch dimension for frequency defense
                is_batch = len(processed_image.shape) == 4
                if is_batch:
                    processed_image_single = processed_image[0]
                    processed_image_single, freq_result = self.frequency_defense(processed_image_single)
                    processed_image = processed_image_single.unsqueeze(0)
                else:
                    processed_image, freq_result = self.frequency_defense(processed_image)
                if freq_result.get('patch_detected', False):
                    logger.warning(f"Frequency defense detected patch: {freq_result.get('confidence', 0.0):.3f}")
            
            # 3. Gradient saliency-based masking
            if self.gradient_saliency_defense is not None and self.gradient_saliency_defense.enabled:
                if model is not None:
                    processed_image, saliency_result = self.gradient_saliency_defense(processed_image, model)
                    if saliency_result.get('patch_detected', False):
                        logger.warning(f"Saliency defense detected patch: {saliency_result.get('confidence', 0.0):.3f}")
        
        elif self.defense_mode == 'ensemble':
            # Apply all defenses and vote
            # For now, use cascade as ensemble is more complex
            return self.process_input(image, model)  # Fallback to cascade
        
        return processed_image
    
    def validate_detection(
        self,
        image: torch.Tensor,
        prediction: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate detection through all defense layers.
        
        Args:
            image: Input image tensor
            prediction: Model prediction dictionary
            model: Model for gradient saliency (if needed)
            frame_number: Current frame number
            timestamp: Optional timestamp
            
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
        patch_detected = False
        detection_confidence = 0.0
        
        # 1. Entropy detection
        if self.entropy_defense.enabled:
            _, entropy_result = self.entropy_defense(image)
            defense_results['entropy'] = entropy_result
            if entropy_result.get('patch_detected', False):
                patch_detected = True
                detection_confidence = max(detection_confidence, entropy_result.get('confidence', 0.0))
        
        # 2. Frequency detection
        if self.frequency_defense.enabled:
            _, freq_result = self.frequency_defense(image)
            defense_results['frequency'] = freq_result
            if freq_result.get('patch_detected', False):
                patch_detected = True
                detection_confidence = max(detection_confidence, freq_result.get('confidence', 0.0))
        
        # 3. Gradient saliency detection
        if self.gradient_saliency_defense is not None and self.gradient_saliency_defense.enabled:
            if model is not None:
                _, saliency_result = self.gradient_saliency_defense(image, model)
                defense_results['gradient_saliency'] = saliency_result
                if saliency_result.get('patch_detected', False):
                    patch_detected = True
                    detection_confidence = max(detection_confidence, saliency_result.get('confidence', 0.0))
        
        # 4. Enhanced multi-frame smoothing
        if self.enhanced_multi_frame.enabled:
            self.enhanced_multi_frame.add_frame(prediction, timestamp)
            consensus = self.enhanced_multi_frame.get_consensus()
            defense_results['enhanced_multi_frame'] = consensus
            
            if not consensus['consensus_reached']:
                return {
                    'valid': False,
                    'passed_all_defenses': False,
                    'defense_results': defense_results,
                    'failed_defense': 'enhanced_multi_frame',
                    'patch_detected': patch_detected,
                    'detection_confidence': detection_confidence
                }
        
        # If patch detected by any method, reject
        if patch_detected:
            return {
                'valid': False,
                'passed_all_defenses': False,
                'defense_results': defense_results,
                'failed_defense': 'patch_detection',
                'patch_detected': True,
                'detection_confidence': detection_confidence
            }
        
        # All defenses passed
        return {
            'valid': True,
            'passed_all_defenses': True,
            'defense_results': defense_results,
            'patch_detected': False,
            'detection_confidence': 0.0
        }
    
    def reset(self) -> None:
        """Reset stateful defenses."""
        if self.enhanced_multi_frame.enabled:
            self.enhanced_multi_frame.reset()

