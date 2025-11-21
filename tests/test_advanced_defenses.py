"""
Test script for advanced defense mechanisms.
Evaluates effectiveness of new defenses against adversarial patches.
"""
import torch
import numpy as np
import cv2
import os
import sys
from pathlib import Path
import argparse
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent))

from src.models.model_loader import ModelLoader
from src.defenses.Advanced_Defense.entropy_detection import EntropyPatchDefense
from src.defenses.Advanced_Defense.frequency_analysis import FrequencyPatchDefense
from src.defenses.Advanced_Defense.gradient_saliency import GradientSaliencyDefense
from src.defenses.Advanced_Defense.enhanced_multi_frame import EnhancedMultiFrameSmoothing
from src.defenses.Advanced_Defense.enhanced_defense_pipeline import EnhancedDefensePipeline
from src.utils.logger import setup_logger

logger = setup_logger()


def load_test_image(image_path: str, device: str = 'cuda') -> torch.Tensor:
    """Load and preprocess test image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_rgb.astype(np.float32) / 255.0 - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor


def test_entropy_defense(image: torch.Tensor, patch_image: torch.Tensor, device: str = 'cuda') -> Dict[str, Any]:
    """Test entropy-based defense."""
    print("\n" + "="*70)
    print("Testing Entropy-Based Defense")
    print("="*70)
    
    defense = EntropyPatchDefense(
        window_size=32,
        stride=8,
        entropy_threshold=7.0,
        patch_size_estimate=(100, 100),
        mask_patches=True,
        enabled=True
    )
    
    # Test on clean image
    clean_result = defense.detector.detect_patch(image)
    print(f"Clean image - Patch detected: {clean_result['patch_detected']}")
    print(f"Clean image - Confidence: {clean_result.get('confidence', 0.0):.3f}")
    
    # Test on patch image
    patch_result = defense.detector.detect_patch(patch_image)
    print(f"Patch image - Patch detected: {patch_result['patch_detected']}")
    print(f"Patch image - Confidence: {patch_result.get('confidence', 0.0):.3f}")
    if patch_result.get('patch_location'):
        print(f"Patch location: {patch_result['patch_location']}")
    
    return {
        'clean_detected': clean_result['patch_detected'],
        'patch_detected': patch_result['patch_detected'],
        'patch_confidence': patch_result.get('confidence', 0.0)
    }


def test_frequency_defense(image: torch.Tensor, patch_image: torch.Tensor, device: str = 'cuda') -> Dict[str, Any]:
    """Test frequency-based defense."""
    print("\n" + "="*70)
    print("Testing Frequency-Based Defense")
    print("="*70)
    
    defense = FrequencyPatchDefense(
        high_freq_threshold=0.3,
        anomaly_threshold=2.0,
        patch_size_estimate=(100, 100),
        filter_patches=True,
        filter_cutoff=0.3,
        enabled=True
    )
    
    # Test on clean image
    clean_result = defense.detector.detect_patch(image)
    print(f"Clean image - Patch detected: {clean_result['patch_detected']}")
    print(f"Clean image - Confidence: {clean_result.get('confidence', 0.0):.3f}")
    
    # Test on patch image
    patch_result = defense.detector.detect_patch(patch_image)
    print(f"Patch image - Patch detected: {patch_result['patch_detected']}")
    print(f"Patch image - Confidence: {patch_result.get('confidence', 0.0):.3f}")
    if patch_result.get('patch_location'):
        print(f"Patch location: {patch_result['patch_location']}")
    
    return {
        'clean_detected': clean_result['patch_detected'],
        'patch_detected': patch_result['patch_detected'],
        'patch_confidence': patch_result.get('confidence', 0.0)
    }


def test_gradient_saliency_defense(
    image: torch.Tensor,
    patch_image: torch.Tensor,
    model: torch.nn.Module,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Test gradient saliency-based defense."""
    print("\n" + "="*70)
    print("Testing Gradient Saliency-Based Defense")
    print("="*70)
    
    defense = GradientSaliencyDefense(
        model=model,
        saliency_threshold=0.7,
        patch_size_estimate=(100, 100),
        mask_patches=True,
        enabled=True,
        device=device
    )
    
    # Test on clean image
    clean_result = defense.detect_patch(image, model, method='gradcam')
    print(f"Clean image - Patch detected: {clean_result['patch_detected']}")
    print(f"Clean image - Confidence: {clean_result.get('confidence', 0.0):.3f}")
    
    # Test on patch image
    patch_result = defense.detect_patch(patch_image, model, method='gradcam')
    print(f"Patch image - Patch detected: {patch_result['patch_detected']}")
    print(f"Patch image - Confidence: {patch_result.get('confidence', 0.0):.3f}")
    if patch_result.get('patch_location'):
        print(f"Patch location: {patch_result['patch_location']}")
    
    return {
        'clean_detected': clean_result['patch_detected'],
        'patch_detected': patch_result['patch_detected'],
        'patch_confidence': patch_result.get('confidence', 0.0)
    }


def test_enhanced_pipeline(
    image: torch.Tensor,
    patch_image: torch.Tensor,
    model: torch.nn.Module,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Test enhanced defense pipeline."""
    print("\n" + "="*70)
    print("Testing Enhanced Defense Pipeline")
    print("="*70)
    
    # Create defenses
    entropy_defense = EntropyPatchDefense(enabled=True)
    frequency_defense = FrequencyPatchDefense(enabled=True)
    gradient_saliency_defense = GradientSaliencyDefense(
        model=model,
        enabled=True,
        device=device
    )
    enhanced_multi_frame = EnhancedMultiFrameSmoothing(
        frame_window=15,
        consensus_threshold=0.8,
        enabled=True
    )
    
    # Create pipeline
    pipeline = EnhancedDefensePipeline(
        entropy_defense=entropy_defense,
        frequency_defense=frequency_defense,
        gradient_saliency_defense=gradient_saliency_defense,
        enhanced_multi_frame=enhanced_multi_frame,
        defense_mode='cascade',
        enabled=True
    )
    
    # Test on clean image
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
    
    clean_prediction = {
        'class': pred_class.item(),
        'confidence': confidence.item()
    }
    
    clean_validation = pipeline.validate_detection(
        image,
        clean_prediction,
        model=model,
        frame_number=0
    )
    
    print(f"Clean image - Valid: {clean_validation['valid']}")
    print(f"Clean image - Passed all defenses: {clean_validation['passed_all_defenses']}")
    print(f"Clean image - Patch detected: {clean_validation.get('patch_detected', False)}")
    
    # Test on patch image
    with torch.no_grad():
        output = model(patch_image)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
    
    patch_prediction = {
        'class': pred_class.item(),
        'confidence': confidence.item()
    }
    
    patch_validation = pipeline.validate_detection(
        patch_image,
        patch_prediction,
        model=model,
        frame_number=0
    )
    
    print(f"Patch image - Valid: {patch_validation['valid']}")
    print(f"Patch image - Passed all defenses: {patch_validation['passed_all_defenses']}")
    print(f"Patch image - Patch detected: {patch_validation.get('patch_detected', False)}")
    print(f"Patch image - Detection confidence: {patch_validation.get('detection_confidence', 0.0):.3f}")
    
    return {
        'clean_valid': clean_validation['valid'],
        'patch_valid': patch_validation['valid'],
        'patch_detected': patch_validation.get('patch_detected', False),
        'detection_confidence': patch_validation.get('detection_confidence', 0.0)
    }


def main():
    parser = argparse.ArgumentParser(description='Test advanced defense mechanisms')
    parser.add_argument('--patch-image', type=str, default='data/patches/attack_patch.png',
                       help='Path to patch image')
    parser.add_argument('--clean-image', type=str, default=None,
                       help='Path to clean test image (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all defenses')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("ADVANCED DEFENSE TESTING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Patch image: {args.patch_image}")
    
    # Load patch image
    if not os.path.exists(args.patch_image):
        print(f"ERROR: Patch image not found: {args.patch_image}")
        return
    
    patch_image = load_test_image(args.patch_image, device)
    
    # Load clean image (or create dummy)
    if args.clean_image and os.path.exists(args.clean_image):
        clean_image = load_test_image(args.clean_image, device)
    else:
        # Create dummy clean image
        clean_image = torch.randn(1, 3, 224, 224).to(device)
        clean_image = torch.clamp(clean_image, -2.0, 2.0)
    
    # Load model
    print("\nLoading ResNet50 model...")
    model_loader = ModelLoader(device=device)
    model = model_loader.load_resnet('resnet50', pretrained=True)
    model.eval()
    
    results = {}
    
    # Test individual defenses
    if args.test_all:
        results['entropy'] = test_entropy_defense(clean_image, patch_image, device)
        results['frequency'] = test_frequency_defense(clean_image, patch_image, device)
        results['gradient_saliency'] = test_gradient_saliency_defense(
            clean_image, patch_image, model, device
        )
    
    # Test enhanced pipeline
    results['pipeline'] = test_enhanced_pipeline(clean_image, patch_image, model, device)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if 'pipeline' in results:
        r = results['pipeline']
        print(f"Enhanced Pipeline:")
        print(f"  - Clean image passed: {r['clean_valid']}")
        print(f"  - Patch detected: {r['patch_detected']}")
        print(f"  - Detection confidence: {r['detection_confidence']:.3f}")
        print(f"  - Patch blocked: {not r['patch_valid']}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

