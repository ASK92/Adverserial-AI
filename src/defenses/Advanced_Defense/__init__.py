"""
Advanced Defense Mechanisms for Adversarial Patch Protection
These defenses are experimental and can be integrated alongside existing defenses.
"""
from .entropy_detection import EntropyPatchDefense, EntropyPatchDetector
from .frequency_analysis import FrequencyPatchDefense, FrequencyPatchDetector
from .gradient_saliency import GradientSaliencyDefense
from .enhanced_multi_frame import EnhancedMultiFrameSmoothing
from .enhanced_defense_pipeline import EnhancedDefensePipeline

__all__ = [
    'EntropyPatchDefense',
    'EntropyPatchDetector',
    'FrequencyPatchDefense',
    'FrequencyPatchDetector',
    'GradientSaliencyDefense',
    'EnhancedMultiFrameSmoothing',
    'EnhancedDefensePipeline'
]

