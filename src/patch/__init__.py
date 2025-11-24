"""
Adversarial patch generation and optimization.
"""
from .patch_generator import AdversarialPatchGenerator
from .patch_optimizer import PatchOptimizer
from .patch_applier import PatchApplier
from .patch_metadata import PatchMetadata, create_patch_metadata

__all__ = [
    'AdversarialPatchGenerator',
    'PatchOptimizer',
    'PatchApplier',
    'PatchMetadata',
    'create_patch_metadata'
]


