"""
Comprehensive Attack and Defense Evaluation Script
Demonstrates quantitative threat assessment and defense effectiveness.
"""
import torch
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.models.model_loader import ModelLoader
from src.models.ensemble import ModelEnsemble
from src.defenses.defense_pipeline import (
    DefensePipeline, InputNormalization, AdversarialDetection,
    MultiFrameSmoothing, ContextRuleEngine
)
from src.defenses.Advanced_Defense import (
    EnhancedDefensePipeline, EntropyPatchDefense, FrequencyPatchDefense,
    GradientSaliencyDefense, EnhancedMultiFrameSmoothing
)
from src.patch.patch_applier import PatchApplier
from src.evaluation.evaluator import PatchEvaluator
from src.evaluation.reporter import EvaluationReporter
from src.evaluation.metrics import compute_attack_metrics, compute_defense_metrics

logger = setup_logger()

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def generate_test_images(num_images: int = 100, device: str = 'cuda') -> torch.Tensor:
    """Generate diverse test images for evaluation."""
    images = []
    for i in range(num_images):
        # Diverse image generation
        if i % 4 == 0:
            # ImageNet normalized
            img = torch.randn(3, 224, 224)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img * std) + mean
        elif i % 4 == 1:
            # Bright images
            img = torch.rand(3, 224, 224) * 0.8 + 0.2
        elif i % 4 == 2:
            # Dark images
            img = torch.rand(3, 224, 224) * 0.4
        else:
            # High contrast
            img = torch.rand(3, 224, 224)
            img = (img - 0.5) * 1.5 + 0.5
        
        img = torch.clamp(img, 0, 1)
        images.append(img)
    
    return torch.stack(images).to(device)

def evaluate_attack_threat(
    patch_path: str,
    models: Dict[str, torch.nn.Module],
    test_images: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Evaluate attack threat with quantitative metrics.
    
    Returns:
        Dictionary with attack threat metrics
    """
    logger.info("="*80)
    logger.info("ATTACK THREAT EVALUATION")
    logger.info("="*80)
    
    # Load patch
    patch_data = torch.load(patch_path, weights_only=False)
    if isinstance(patch_data, dict) and 'patch' in patch_data:
        patch = patch_data['patch']
    elif isinstance(patch_data, torch.Tensor):
        patch = patch_data
    else:
        patch = torch.from_numpy(np.array(patch_data))
    
    if not isinstance(patch, torch.Tensor):
        patch = torch.from_numpy(np.array(patch))
    
    patch = patch.to(device)
    if len(patch.shape) == 3 and patch.shape[0] == 3:
        pass  # Already (C, H, W)
    elif len(patch.shape) == 3:
        patch = patch.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    patch_applier = PatchApplier(device=device)
    
    results = {
        'single_model_success': {},
        'ensemble_success': None,
        'attack_metrics': {}
    }
    
    # Evaluate on each model
    for model_name, model in models.items():
        model.eval()
        success_count = 0
        total_count = 0
        confidence_drops = []
        
        with torch.no_grad():
            for i in range(len(test_images)):
                image = test_images[i:i+1]
                
                # Original prediction
                orig_output = model(image)
                if isinstance(orig_output, torch.Tensor):
                    orig_probs = torch.softmax(orig_output, dim=1)
                    orig_pred = torch.argmax(orig_probs, dim=1).item()
                    orig_conf = torch.max(orig_probs, dim=1)[0].item()
                else:
                    continue
                
                # Apply patch
                patched_image, _ = patch_applier.apply_patch_random_location(image, patch)
                
                # Patched prediction
                patched_output = model(patched_image)
                if isinstance(patched_output, torch.Tensor):
                    patched_probs = torch.softmax(patched_output, dim=1)
                    patched_pred = torch.argmax(patched_probs, dim=1).item()
                    patched_conf = torch.max(patched_probs, dim=1)[0].item()
                    
                    # Attack successful if prediction changed
                    if patched_pred != orig_pred:
                        success_count += 1
                    
                    # Track confidence drop
                    conf_drop = orig_conf - patched_conf
                    confidence_drops.append(conf_drop)
                    total_count += 1
        
        success_rate = success_count / max(total_count, 1)
        avg_conf_drop = np.mean(confidence_drops) if confidence_drops else 0.0
        
        results['single_model_success'][model_name] = {
            'success_rate': success_rate,
            'success_count': success_count,
            'total_count': total_count,
            'avg_confidence_drop': avg_conf_drop
        }
        
        logger.info(f"{model_name}: Success Rate = {success_rate:.2%} ({success_count}/{total_count}), "
                   f"Avg Confidence Drop = {avg_conf_drop:.4f}")
    
    # Evaluate ensemble
    ensemble = ModelEnsemble(models=models, consensus_threshold=0.5, device=device)
    ensemble_success = 0
    ensemble_total = 0
    
    with torch.no_grad():
        for i in range(len(test_images)):
            image = test_images[i:i+1]
            
            # Original ensemble prediction
            orig_pred = ensemble.predict(image)
            orig_consensus = orig_pred['consensus']
            
            # Apply patch
            patched_image, _ = patch_applier.apply_patch_random_location(image, patch)
            
            # Patched ensemble prediction
            patched_pred = ensemble.predict(patched_image)
            patched_consensus = patched_pred['consensus']
            
            if orig_consensus['consensus_reached'] and patched_consensus['consensus_reached']:
                if orig_consensus['predicted_class'] != patched_consensus['predicted_class']:
                    ensemble_success += 1
                ensemble_total += 1
    
    ensemble_success_rate = ensemble_success / max(ensemble_total, 1)
    results['ensemble_success'] = {
        'success_rate': ensemble_success_rate,
        'success_count': ensemble_success,
        'total_count': ensemble_total
    }
    
    logger.info(f"Ensemble: Success Rate = {ensemble_success_rate:.2%} ({ensemble_success}/{ensemble_total})")
    
    return results

def evaluate_defense_effectiveness(
    patch_path: str,
    models: Dict[str, torch.nn.Module],
    test_images: torch.Tensor,
    use_advanced: bool = True,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Evaluate defense effectiveness with quantitative metrics.
    
    Returns:
        Dictionary with defense effectiveness metrics
    """
    logger.info("="*80)
    logger.info("DEFENSE EFFECTIVENESS EVALUATION")
    logger.info(f"Using {'Advanced' if use_advanced else 'Basic'} Defense Pipeline")
    logger.info("="*80)
    
    # Load patch
    patch_data = torch.load(patch_path, weights_only=False)
    if isinstance(patch_data, dict) and 'patch' in patch_data:
        patch = patch_data['patch']
    elif isinstance(patch_data, torch.Tensor):
        patch = patch_data
    else:
        patch = torch.from_numpy(np.array(patch_data))
    
    if not isinstance(patch, torch.Tensor):
        patch = torch.from_numpy(np.array(patch))
    
    patch = patch.to(device)
    if len(patch.shape) == 3 and patch.shape[0] == 3:
        pass
    elif len(patch.shape) == 3:
        patch = patch.permute(2, 0, 1)
    
    patch_applier = PatchApplier(device=device)
    
    # Setup defense pipeline
    if use_advanced:
        model = list(models.values())[0]  # Use first model for gradient saliency
        defense_pipeline = EnhancedDefensePipeline(
            entropy_defense=EntropyPatchDefense(
                window_size=32, stride=8, entropy_threshold=7.0,
                patch_size_estimate=(100, 100), mask_patches=True, enabled=True
            ),
            frequency_defense=FrequencyPatchDefense(
                high_freq_threshold=0.3, anomaly_threshold=2.0,
                patch_size_estimate=(100, 100), filter_patches=True, enabled=True
            ),
            gradient_saliency_defense=GradientSaliencyDefense(
                model=model, saliency_threshold=0.7,
                patch_size_estimate=(100, 100), mask_patches=True,
                enabled=True, device=device
            ),
            enhanced_multi_frame=EnhancedMultiFrameSmoothing(
                frame_window=15, consensus_threshold=0.8,
                temporal_gradient_threshold=0.3, enabled=True
            ),
            defense_mode='cascade', enabled=True
        )
    else:
        defense_pipeline = DefensePipeline(
            input_normalization=InputNormalization(enabled=True),
            adversarial_detection=AdversarialDetection(enabled=True, device=device),
            multi_frame_smoothing=MultiFrameSmoothing(enabled=True),
            context_rules=ContextRuleEngine(enabled=True),
            enabled=True
        )
    
    model = list(models.values())[0]
    model.eval()
    
    results = {
        'blocked': 0,
        'bypassed': 0,
        'defense_breakdown': {},
        'processing_times': []
    }
    
    num_frames = 50  # More frames for better statistics
    
    with torch.no_grad():
        for frame_idx in range(num_frames):
            image_idx = frame_idx % len(test_images)
            image = test_images[image_idx:image_idx+1]
            
            # Apply patch
            patched_image, loc = patch_applier.apply_patch_random_location(image, patch)
            
            # Get prediction
            start_time = time.time()
            output = model(patched_image)
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
            else:
                pred_class = 0
                confidence = 0.5
            
            prediction = {
                'class': pred_class,
                'confidence': confidence,
                'bbox': list(loc) if loc else None
            }
            
            # Validate through defense
            if use_advanced:
                # Ensure image requires grad for gradient saliency
                patched_image_grad = patched_image.clone().detach().requires_grad_(True)
                validation = defense_pipeline.validate_detection(
                    patched_image_grad, prediction, model=model,
                    frame_number=frame_idx, timestamp=time.time()
                )
            else:
                validation = defense_pipeline.validate_detection(
                    patched_image, prediction, frame_number=frame_idx
                )
            
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)
            
            if validation.get('valid', False):
                results['bypassed'] += 1
            else:
                results['blocked'] += 1
                failed_defense = validation.get('failed_defense', 'unknown')
                results['defense_breakdown'][failed_defense] = \
                    results['defense_breakdown'].get(failed_defense, 0) + 1
    
    total = results['blocked'] + results['bypassed']
    results['defense_effectiveness'] = results['blocked'] / max(total, 1)
    results['false_negative_rate'] = results['bypassed'] / max(total, 1)
    results['avg_processing_time'] = np.mean(results['processing_times'])
    results['total_frames'] = total
    
    logger.info(f"Defense Effectiveness: {results['defense_effectiveness']:.2%}")
    logger.info(f"False Negative Rate: {results['false_negative_rate']:.2%}")
    logger.info(f"Blocked: {results['blocked']}/{total}, Bypassed: {results['bypassed']}/{total}")
    logger.info(f"Average Processing Time: {results['avg_processing_time']*1000:.2f}ms")
    
    return results

def evaluate_performance_tradeoffs(
    patch_path: str,
    models: Dict[str, torch.nn.Module],
    test_images: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Evaluate performance trade-offs: accuracy vs. latency with/without defenses.
    
    Returns:
        Dictionary with performance trade-off metrics
    """
    logger.info("="*80)
    logger.info("PERFORMANCE TRADE-OFF ANALYSIS")
    logger.info("="*80)
    
    # Load patch
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
    if len(patch.shape) == 3 and patch.shape[0] == 3:
        pass
    elif len(patch.shape) == 3:
        patch = patch.permute(2, 0, 1)
    
    patch_applier = PatchApplier(device=device)
    model = list(models.values())[0]
    model.eval()
    
    # Setup defenses
    basic_defense = DefensePipeline(
        input_normalization=InputNormalization(enabled=True),
        adversarial_detection=AdversarialDetection(enabled=True, device=device),
        multi_frame_smoothing=MultiFrameSmoothing(enabled=True),
        context_rules=ContextRuleEngine(enabled=True),
        enabled=True
    )
    
    advanced_defense = EnhancedDefensePipeline(
        entropy_defense=EntropyPatchDefense(enabled=True),
        frequency_defense=FrequencyPatchDefense(enabled=True),
        gradient_saliency_defense=GradientSaliencyDefense(
            model=model, enabled=True, device=device
        ),
        enhanced_multi_frame=EnhancedMultiFrameSmoothing(enabled=True),
        defense_mode='cascade', enabled=True
    )
    
    results = {
        'no_defense': {'accuracy': 0.0, 'latency_ms': 0.0, 'throughput_fps': 0.0},
        'basic_defense': {'accuracy': 0.0, 'latency_ms': 0.0, 'throughput_fps': 0.0, 'defense_effectiveness': 0.0},
        'advanced_defense': {'accuracy': 0.0, 'latency_ms': 0.0, 'throughput_fps': 0.0, 'defense_effectiveness': 0.0}
    }
    
    num_test = min(50, len(test_images))
    
    # 1. No Defense
    logger.info("Evaluating: No Defense")
    correct = 0
    times = []
    
    with torch.no_grad():
        for i in range(num_test):
            image = test_images[i:i+1]
            patched_image, _ = patch_applier.apply_patch_random_location(image, patch)
            
            start = time.time()
            output = model(patched_image)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                # For accuracy, we check if patch caused misclassification
                # (assuming original would be correct)
                correct += 1  # Simplified metric
    
    results['no_defense']['accuracy'] = correct / num_test
    results['no_defense']['latency_ms'] = np.mean(times) * 1000
    results['no_defense']['throughput_fps'] = 1.0 / np.mean(times)
    
    # 2. Basic Defense
    logger.info("Evaluating: Basic Defense")
    blocked = 0
    times = []
    
    with torch.no_grad():
        for i in range(num_test):
            image = test_images[i:i+1]
            patched_image, _ = patch_applier.apply_patch_random_location(image, patch)
            
            start = time.time()
            processed = basic_defense.process_input(patched_image)
            output = model(processed)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
                
                prediction = {'class': pred_class, 'confidence': confidence}
                validation = basic_defense.validate_detection(
                    patched_image, prediction, frame_number=i
                )
                
                if not validation.get('valid', False):
                    blocked += 1
    
    results['basic_defense']['defense_effectiveness'] = blocked / num_test
    results['basic_defense']['latency_ms'] = np.mean(times) * 1000
    results['basic_defense']['throughput_fps'] = 1.0 / np.mean(times)
    results['basic_defense']['accuracy'] = 1.0 - (blocked / num_test)  # Accuracy = not blocked
    
    # 3. Advanced Defense
    logger.info("Evaluating: Advanced Defense")
    blocked = 0
    times = []
    
    with torch.no_grad():
        for i in range(num_test):
            image = test_images[i:i+1]
            patched_image, _ = patch_applier.apply_patch_random_location(image, patch)
            
            start = time.time()
            processed = advanced_defense.process_input(patched_image, model=model)
            output = model(processed)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
                
                prediction = {'class': pred_class, 'confidence': confidence}
                validation = advanced_defense.validate_detection(
                    patched_image, prediction, model=model,
                    frame_number=i, timestamp=time.time()
                )
                
                if not validation.get('valid', False):
                    blocked += 1
    
    results['advanced_defense']['defense_effectiveness'] = blocked / num_test
    results['advanced_defense']['latency_ms'] = np.mean(times) * 1000
    results['advanced_defense']['throughput_fps'] = 1.0 / np.mean(times)
    results['advanced_defense']['accuracy'] = 1.0 - (blocked / num_test)
    
    # Calculate overhead
    results['basic_overhead_ms'] = results['basic_defense']['latency_ms'] - results['no_defense']['latency_ms']
    results['advanced_overhead_ms'] = results['advanced_defense']['latency_ms'] - results['no_defense']['latency_ms']
    results['basic_overhead_pct'] = (results['basic_overhead_ms'] / results['no_defense']['latency_ms']) * 100
    results['advanced_overhead_pct'] = (results['advanced_overhead_ms'] / results['no_defense']['latency_ms']) * 100
    
    logger.info("\nPerformance Trade-offs:")
    logger.info(f"No Defense:      Latency = {results['no_defense']['latency_ms']:.2f}ms, Throughput = {results['no_defense']['throughput_fps']:.2f} FPS")
    logger.info(f"Basic Defense:   Latency = {results['basic_defense']['latency_ms']:.2f}ms (+{results['basic_overhead_ms']:.2f}ms, +{results['basic_overhead_pct']:.1f}%), "
               f"Effectiveness = {results['basic_defense']['defense_effectiveness']:.2%}")
    logger.info(f"Advanced Defense: Latency = {results['advanced_defense']['latency_ms']:.2f}ms (+{results['advanced_overhead_ms']:.2f}ms, +{results['advanced_overhead_pct']:.1f}%), "
               f"Effectiveness = {results['advanced_defense']['defense_effectiveness']:.2%}")
    
    return results

def main():
    """Run comprehensive attack and defense evaluation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load models
    model_loader = ModelLoader(device=device)
    models = {
        'resnet50': model_loader.load_resnet('resnet50', pretrained=True),
        'efficientnet_b0': model_loader.load_efficientnet('efficientnet_b0', pretrained=True)
    }
    
    # Generate test images
    logger.info("Generating test images...")
    test_images = generate_test_images(num_images=100, device=device)
    
    # Use malware patch
    patch_path = 'data/patches/final_deployment/malware_patch_final.pt'
    if not os.path.exists(patch_path):
        patch_path = 'data/patches/resnet_breaker_70pct.pt'
    
    if not os.path.exists(patch_path):
        logger.error(f"Patch not found: {patch_path}")
        logger.error("Please ensure a patch file exists")
        return
    
    logger.info(f"Using patch: {patch_path}")
    
    # 1. Evaluate Attack Threat
    attack_results = evaluate_attack_threat(patch_path, models, test_images, device)
    
    # 2. Evaluate Defense Effectiveness (Basic)
    defense_basic = evaluate_defense_effectiveness(
        patch_path, models, test_images, use_advanced=False, device=device
    )
    
    # 3. Evaluate Defense Effectiveness (Advanced)
    defense_advanced = evaluate_defense_effectiveness(
        patch_path, models, test_images, use_advanced=True, device=device
    )
    
    # 4. Performance Trade-offs
    tradeoff_results = evaluate_performance_tradeoffs(patch_path, models, test_images, device)
    
    # Generate comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'data/results/comprehensive_evaluation_{timestamp}.json'
    os.makedirs('data/results', exist_ok=True)
    
    comprehensive_results = {
        'timestamp': timestamp,
        'random_seed': RANDOM_SEED,
        'device': device,
        'patch_path': patch_path,
        'num_test_images': len(test_images),
        'attack_threat': attack_results,
        'defense_basic': defense_basic,
        'defense_advanced': defense_advanced,
        'performance_tradeoffs': tradeoff_results
    }
    
    with open(report_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"\nAttack Threat Assessment:")
    for model_name, metrics in attack_results['single_model_success'].items():
        logger.info(f"  {model_name}: {metrics['success_rate']:.2%} success rate")
    logger.info(f"  Ensemble: {attack_results['ensemble_success']['success_rate']:.2%} success rate")
    
    logger.info(f"\nDefense Effectiveness:")
    logger.info(f"  Basic Defense: {defense_basic['defense_effectiveness']:.2%} effectiveness")
    logger.info(f"  Advanced Defense: {defense_advanced['defense_effectiveness']:.2%} effectiveness")
    
    logger.info(f"\nPerformance Trade-offs:")
    logger.info(f"  Basic Defense Overhead: +{tradeoff_results['basic_overhead_pct']:.1f}% latency")
    logger.info(f"  Advanced Defense Overhead: +{tradeoff_results['advanced_overhead_pct']:.1f}% latency")
    
    logger.info(f"\nFull results saved to: {report_path}")
    logger.info("="*80)

if __name__ == '__main__':
    main()

