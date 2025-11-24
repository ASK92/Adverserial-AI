"""
Ablation Study: Individual Defense Layer Effectiveness
Demonstrates contribution of each defense layer.
"""
import torch
import numpy as np
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.models.model_loader import ModelLoader
from src.defenses.defense_pipeline import (
    DefensePipeline, InputNormalization, AdversarialDetection,
    MultiFrameSmoothing, ContextRuleEngine
)
from src.defenses.Advanced_Defense import (
    EntropyPatchDefense, FrequencyPatchDefense,
    GradientSaliencyDefense, EnhancedMultiFrameSmoothing
)
from src.patch.patch_applier import PatchApplier

logger = setup_logger()

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_test_images(num_images: int = 50, device: str = 'cuda') -> torch.Tensor:
    """Generate test images."""
    images = []
    for i in range(num_images):
        img = torch.randn(3, 224, 224)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img * std) + mean
        img = torch.clamp(img, 0, 1)
        images.append(img)
    return torch.stack(images).to(device)

def evaluate_defense_combination(
    patch: torch.Tensor,
    model: torch.nn.Module,
    test_images: torch.Tensor,
    defense_config: Dict[str, bool],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Evaluate a specific defense combination."""
    patch_applier = PatchApplier(device=device)
    
    defense_pipeline = DefensePipeline(
        input_normalization=InputNormalization(enabled=defense_config.get('input_norm', False)),
        adversarial_detection=AdversarialDetection(enabled=defense_config.get('adv_detection', False), device=device),
        multi_frame_smoothing=MultiFrameSmoothing(enabled=defense_config.get('multi_frame', False)),
        context_rules=ContextRuleEngine(enabled=defense_config.get('context_rules', False)),
        enabled=True
    )
    
    blocked = 0
    bypassed = 0
    
    with torch.no_grad():
        for i in range(len(test_images)):
            image = test_images[i:i+1]
            patched_image, loc = patch_applier.apply_patch_random_location(image, patch)
            
            output = model(patched_image)
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
            else:
                continue
            
            prediction = {'class': pred_class, 'confidence': confidence, 'bbox': list(loc)}
            validation = defense_pipeline.validate_detection(patched_image, prediction, frame_number=i)
            
            if validation.get('valid', False):
                bypassed += 1
            else:
                blocked += 1
    
    total = blocked + bypassed
    effectiveness = blocked / max(total, 1)
    
    return {
        'blocked': blocked,
        'bypassed': bypassed,
        'total': total,
        'effectiveness': effectiveness
    }

def main():
    """Run ablation study on defense layers."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("="*80)
    logger.info("DEFENSE LAYER ABLATION STUDY")
    logger.info("="*80)
    
    # Load model
    model_loader = ModelLoader(device=device)
    model = model_loader.load_resnet('resnet50', pretrained=True)
    model.eval()
    
    # Load patch
    patch_path = 'data/patches/final_deployment/malware_patch_final.pt'
    if not os.path.exists(patch_path):
        patch_path = 'data/patches/resnet_breaker_70pct.pt'
    
    patch_data = torch.load(patch_path, weights_only=False)
    if isinstance(patch_data, dict) and 'patch' in patch_data:
        patch = patch_data['patch']
    elif isinstance(patch_data, torch.Tensor):
        patch = patch_data
    else:
        patch = torch.from_numpy(np.array(patch_data))
    
    if not isinstance(patch, torch.Tensor):
        patch = torch.from_numpy(np.array(patch_data))
    
    patch = patch.to(device)
    if len(patch.shape) == 3 and patch.shape[0] != 3:
        patch = patch.permute(2, 0, 1)
    
    test_images = generate_test_images(num_images=50, device=device)
    
    # Test all combinations
    defense_combinations = {
        'No Defenses': {},
        'Input Normalization Only': {'input_norm': True},
        'Adversarial Detection Only': {'adv_detection': True},
        'Multi-Frame Smoothing Only': {'multi_frame': True},
        'Context Rules Only': {'context_rules': True},
        'Input Norm + Adv Detection': {'input_norm': True, 'adv_detection': True},
        'Input Norm + Multi-Frame': {'input_norm': True, 'multi_frame': True},
        'Adv Detection + Multi-Frame': {'adv_detection': True, 'multi_frame': True},
        'All Basic Defenses': {'input_norm': True, 'adv_detection': True, 'multi_frame': True, 'context_rules': True}
    }
    
    results = {}
    
    for name, config in defense_combinations.items():
        logger.info(f"\nEvaluating: {name}")
        result = evaluate_defense_combination(patch, model, test_images, config, device)
        results[name] = result
        logger.info(f"  Effectiveness: {result['effectiveness']:.2%} ({result['blocked']}/{result['total']} blocked)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'data/results/ablation_study_{timestamp}.json'
    os.makedirs('data/results', exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'random_seed': RANDOM_SEED,
            'patch_path': patch_path,
            'results': results
        }, f, indent=2, default=str)
    
    logger.info(f"\nAblation study results saved to: {report_path}")
    logger.info("="*80)

if __name__ == '__main__':
    main()

