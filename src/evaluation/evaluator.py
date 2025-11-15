"""
Comprehensive evaluator for adversarial patch attacks.
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from tqdm import tqdm

from ..models.model_loader import ModelLoader
from ..models.ensemble import ModelEnsemble
from ..defenses.defense_pipeline import DefensePipeline
from ..patch.patch_generator import AdversarialPatchGenerator
from ..patch.patch_applier import PatchApplier
from ..utils.visualization import visualize_detection, visualize_attack_results

logger = logging.getLogger(__name__)


class PatchEvaluator:
    """
    Comprehensive evaluator for adversarial patch effectiveness.
    """
    
    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        defense_pipeline: Optional[DefensePipeline] = None,
        model_ensemble: Optional[ModelEnsemble] = None,
        device: str = 'cuda'
    ):
        """
        Initialize patch evaluator.
        
        Args:
            models: Dictionary of models to evaluate
            defense_pipeline: Defense pipeline to test
            model_ensemble: Model ensemble for consensus evaluation
            device: Computing device
        """
        self.models = models
        self.defense_pipeline = defense_pipeline
        self.model_ensemble = model_ensemble
        self.device = device
        self.patch_applier = PatchApplier(device=device)
        
        logger.info(f"Initialized PatchEvaluator with {len(models)} models")
    
    def evaluate_patch(
        self,
        patch: torch.Tensor,
        test_images: torch.Tensor,
        patch_location: Optional[Tuple[int, int, int, int]] = None,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        num_frames: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate patch effectiveness across multiple scenarios.
        
        Args:
            patch: Adversarial patch tensor
            test_images: Test images (B, C, H, W)
            patch_location: Fixed patch location (None for random)
            scenarios: List of scenario dictionaries (lighting, angle, etc.)
            num_frames: Number of frames for temporal evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'single_model_results': {},
            'ensemble_results': {},
            'defense_results': {},
            'scenario_results': {},
            'frame_results': []
        }
        
        # Evaluate on single models
        for model_name, model in self.models.items():
            logger.info(f"Evaluating on {model_name}...")
            model_results = self._evaluate_single_model(
                model, model_name, patch, test_images, patch_location
            )
            results['single_model_results'][model_name] = model_results
        
        # Evaluate on ensemble
        if self.model_ensemble is not None:
            logger.info("Evaluating on model ensemble...")
            ensemble_results = self._evaluate_ensemble(
                patch, test_images, patch_location
            )
            results['ensemble_results'] = ensemble_results
        
        # Evaluate with defenses
        if self.defense_pipeline is not None:
            logger.info("Evaluating with defense pipeline...")
            defense_results = self._evaluate_with_defenses(
                patch, test_images, patch_location, num_frames
            )
            results['defense_results'] = defense_results
        
        # Evaluate across scenarios
        if scenarios is not None:
            logger.info("Evaluating across scenarios...")
            scenario_results = self._evaluate_scenarios(
                patch, test_images, scenarios, patch_location
            )
            results['scenario_results'] = scenario_results
        
        # Frame-by-frame evaluation
        logger.info("Evaluating frame-by-frame...")
        frame_results = self._evaluate_frames(
            patch, test_images, num_frames, patch_location
        )
        results['frame_results'] = frame_results
        
        # Compute overall metrics
        results['overall_metrics'] = self._compute_overall_metrics(results)
        
        return results
    
    def _evaluate_single_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        patch: torch.Tensor,
        images: torch.Tensor,
        patch_location: Optional[Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """Evaluate patch on a single model."""
        model.eval()
        success_count = 0
        total_count = 0
        
        with torch.no_grad():
            for i in range(len(images)):
                image = images[i:i+1]
                
                # Get original prediction
                original_output = model(image)
                if isinstance(original_output, torch.Tensor):
                    original_probs = torch.softmax(original_output, dim=1)
                    original_pred = torch.argmax(original_probs, dim=1).item()
                else:
                    # Handle YOLO or other formats
                    original_pred = 0
                
                # Apply patch
                if patch_location is None:
                    patched_image, loc = self.patch_applier.apply_patch_random_location(
                        image, patch
                    )
                else:
                    patched_image = self.patch_applier.apply_patch(
                        image, patch, patch_location
                    )
                    loc = patch_location
                
                # Get patched prediction
                patched_output = model(patched_image)
                if isinstance(patched_output, torch.Tensor):
                    patched_probs = torch.softmax(patched_output, dim=1)
                    patched_pred = torch.argmax(patched_probs, dim=1).item()
                    
                    # Attack successful if prediction changed
                    success = (patched_pred != original_pred)
                    success_count += int(success)
                    total_count += 1
        
        success_rate = success_count / max(total_count, 1)
        
        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'total_count': total_count
        }
    
    def _evaluate_ensemble(
        self,
        patch: torch.Tensor,
        images: torch.Tensor,
        patch_location: Optional[Tuple[int, int, int, int]]
    ) -> Dict[str, Any]:
        """Evaluate patch on model ensemble."""
        if self.model_ensemble is None:
            return {}
        
        success_count = 0
        total_count = 0
        
        with torch.no_grad():
            for i in range(len(images)):
                image = images[i:i+1]
                
                # Get original ensemble prediction
                original_pred = self.model_ensemble.predict(image)
                original_consensus = original_pred['consensus']
                
                # Apply patch
                if patch_location is None:
                    patched_image, _ = self.patch_applier.apply_patch_random_location(
                        image, patch
                    )
                else:
                    patched_image = self.patch_applier.apply_patch(
                        image, patch, patch_location
                    )
                
                # Get patched ensemble prediction
                patched_pred = self.model_ensemble.predict(patched_image)
                patched_consensus = patched_pred['consensus']
                
                # Attack successful if consensus changed
                if original_consensus['consensus_reached'] and patched_consensus['consensus_reached']:
                    success = (
                        original_consensus['predicted_class'] !=
                        patched_consensus['predicted_class']
                    )
                    success_count += int(success)
                    total_count += 1
        
        success_rate = success_count / max(total_count, 1)
        
        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'total_count': total_count
        }
    
    def _evaluate_with_defenses(
        self,
        patch: torch.Tensor,
        images: torch.Tensor,
        patch_location: Optional[Tuple[int, int, int, int]],
        num_frames: int
    ) -> Dict[str, Any]:
        """Evaluate patch with defense pipeline."""
        if self.defense_pipeline is None:
            return {}
        
        defense_results = {
            'bypassed': 0,
            'blocked': 0,
            'defense_breakdown': {}
        }
        
        # Reset defense pipeline
        self.defense_pipeline.reset()
        
        with torch.no_grad():
            for frame_idx in range(num_frames):
                # Cycle through images
                image_idx = frame_idx % len(images)
                image = images[image_idx:image_idx+1]
                
                # Apply patch
                if patch_location is None:
                    patched_image, loc = self.patch_applier.apply_patch_random_location(
                        image, patch
                    )
                else:
                    patched_image = self.patch_applier.apply_patch(
                        image, patch, patch_location
                    )
                    loc = patch_location
                
                # Get prediction (using first model as example)
                model_name = list(self.models.keys())[0]
                model = self.models[model_name]
                
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
                
                # Validate through defense pipeline
                validation = self.defense_pipeline.validate_detection(
                    patched_image, prediction, frame_number=frame_idx
                )
                
                if validation['valid']:
                    defense_results['bypassed'] += 1
                else:
                    defense_results['blocked'] += 1
                    failed_defense = validation.get('failed_defense', 'unknown')
                    defense_results['defense_breakdown'][failed_defense] = \
                        defense_results['defense_breakdown'].get(failed_defense, 0) + 1
        
        total = defense_results['bypassed'] + defense_results['blocked']
        bypass_rate = defense_results['bypassed'] / max(total, 1)
        
        defense_results['bypass_rate'] = bypass_rate
        defense_results['total_frames'] = total
        
        return defense_results
    
    def _evaluate_scenarios(
        self,
        patch: torch.Tensor,
        images: torch.Tensor,
        scenarios: List[Dict[str, Any]],
        patch_location: Optional[Tuple[int, int, int, int]]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate patch across different scenarios."""
        scenario_results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'unknown')
            logger.info(f"Evaluating scenario: {scenario_name}")
            
            # Apply scenario-specific transformations
            transformed_images = self._apply_scenario_transforms(images, scenario)
            
            # Evaluate on transformed images
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            model_results = self._evaluate_single_model(
                model, model_name, patch, transformed_images, patch_location
            )
            
            scenario_results[scenario_name] = model_results
        
        return scenario_results
    
    def _apply_scenario_transforms(
        self,
        images: torch.Tensor,
        scenario: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply scenario-specific transformations."""
        # Placeholder for scenario transformations
        # In practice, would apply lighting, angle, etc.
        return images
    
    def _evaluate_frames(
        self,
        patch: torch.Tensor,
        images: torch.Tensor,
        num_frames: int,
        patch_location: Optional[Tuple[int, int, int, int]]
    ) -> List[Dict[str, Any]]:
        """Evaluate patch across multiple frames."""
        frame_results = []
        
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            for frame_idx in range(num_frames):
                image_idx = frame_idx % len(images)
                image = images[image_idx:image_idx+1]
                
                # Apply patch
                if patch_location is None:
                    patched_image, _ = self.patch_applier.apply_patch_random_location(
                        image, patch
                    )
                else:
                    patched_image = self.patch_applier.apply_patch(
                        image, patch, patch_location
                    )
                
                # Get prediction
                output = model(patched_image)
                if isinstance(output, torch.Tensor):
                    probs = torch.softmax(output, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = torch.max(probs, dim=1)[0].item()
                else:
                    pred_class = 0
                    confidence = 0.5
                
                frame_results.append({
                    'frame': frame_idx,
                    'predicted_class': pred_class,
                    'confidence': confidence
                })
        
        return frame_results
    
    def _compute_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall attack metrics."""
        metrics = {}
        
        # Average success rate across single models
        if 'single_model_results' in results:
            success_rates = [
                r['success_rate']
                for r in results['single_model_results'].values()
            ]
            metrics['avg_single_model_success'] = np.mean(success_rates) if success_rates else 0.0
        
        # Ensemble success rate
        if 'ensemble_results' in results and results['ensemble_results']:
            metrics['ensemble_success'] = results['ensemble_results'].get('success_rate', 0.0)
        
        # Defense bypass rate
        if 'defense_results' in results and results['defense_results']:
            metrics['defense_bypass_rate'] = results['defense_results'].get('bypass_rate', 0.0)
        
        return metrics


