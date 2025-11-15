"""
Optimizer for adversarial patches with multi-defense bypass loss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging

from .patch_generator import AdversarialPatchGenerator
from .patch_applier import PatchApplier
from ..defenses.defense_pipeline import DefensePipeline

logger = logging.getLogger(__name__)


class PatchOptimizer:
    """
    Optimizer for adversarial patches that bypasses multiple defense layers.
    """
    
    def __init__(
        self,
        patch_generator: AdversarialPatchGenerator,
        models: Dict[str, nn.Module],
        defense_pipeline: Optional[DefensePipeline] = None,
        patch_applier: Optional[PatchApplier] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = 'cuda'
    ):
        """
        Initialize patch optimizer.
        
        Args:
            patch_generator: Patch generator instance
            models: Dictionary of models to attack
            defense_pipeline: Defense pipeline to bypass
            patch_applier: Patch applier utility
            loss_weights: Weights for different loss components
            device: Computing device
        """
        self.patch_generator = patch_generator
        self.models = models
        self.defense_pipeline = defense_pipeline
        self.patch_applier = patch_applier or PatchApplier(device=device)
        self.device = device
        
        # Default loss weights
        self.loss_weights = loss_weights or {
            'classification': 1.0,
            'detection': 1.0,
            'defense_evasion': 1.0,
            'temporal': 0.5,
            'context': 0.3,
            'tv': 0.1  # Total variation for smoothness
        }
        
        logger.info(f"Initialized PatchOptimizer with {len(models)} models")
    
    def compute_loss(
        self,
        patched_images: torch.Tensor,
        target_class: Optional[int] = None,
        original_predictions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component loss for patch optimization.
        
        Args:
            patched_images: Images with patch applied
            target_class: Target class for attack (None for untargeted)
            original_predictions: Original predictions before patch
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # 1. Classification loss (attack model predictions)
        classification_loss = self._compute_classification_loss(
            patched_images, target_class
        )
        losses['classification'] = classification_loss
        
        # 2. Detection loss (for object detection models)
        detection_loss = self._compute_detection_loss(patched_images)
        losses['detection'] = detection_loss
        
        # 3. Defense evasion loss
        if self.defense_pipeline is not None:
            defense_loss = self._compute_defense_evasion_loss(patched_images)
            losses['defense_evasion'] = defense_loss
        else:
            losses['defense_evasion'] = torch.tensor(0.0, device=self.device)
        
        # 4. Total variation loss (for patch smoothness)
        tv_loss = self._compute_tv_loss()
        losses['tv'] = tv_loss
        
        # Total weighted loss
        total_loss = sum(
            self.loss_weights.get(name, 1.0) * loss
            for name, loss in losses.items()
        )
        losses['total'] = total_loss
        
        return losses
    
    def _compute_classification_loss(
        self,
        images: torch.Tensor,
        target_class: Optional[int]
    ) -> torch.Tensor:
        """Compute classification loss across all models."""
        total_loss = 0.0
        num_models = 0
        
        for model_name, model in self.models.items():
            if 'yolo' in model_name.lower():
                # Skip YOLO for classification loss (handled in detection loss)
                continue
            
            try:
                with torch.no_grad():
                    # Get original prediction
                    original_output = model(images)
                    if isinstance(original_output, torch.Tensor):
                        original_probs = torch.softmax(original_output, dim=1)
                        original_pred = torch.argmax(original_probs, dim=1)
                
                # Get patched prediction
                output = model(images)
                if isinstance(output, torch.Tensor):
                    probs = torch.softmax(output, dim=1)
                    
                    if target_class is not None:
                        # Targeted attack: maximize target class probability
                        loss = -torch.log(probs[:, target_class] + 1e-8).mean()
                    else:
                        # Untargeted attack: minimize confidence of original prediction
                        loss = torch.log(probs.gather(1, original_pred.unsqueeze(1)) + 1e-8).mean()
                    
                    total_loss += loss
                    num_models += 1
            except Exception as e:
                logger.warning(f"Classification loss computation failed for {model_name}: {e}")
        
        return total_loss / max(num_models, 1)
    
    def _compute_detection_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Compute detection loss for object detection models."""
        total_loss = 0.0
        num_models = 0
        
        for model_name, model in self.models.items():
            if 'yolo' in model_name.lower():
                try:
                    # YOLO-specific loss (simplified)
                    # In practice, would compute loss on detection confidence
                    output = model(images)
                    # Placeholder: minimize detection confidence
                    # Actual implementation would parse YOLO output
                    loss = torch.tensor(0.0, device=self.device)
                    total_loss += loss
                    num_models += 1
                except Exception as e:
                    logger.warning(f"Detection loss computation failed for {model_name}: {e}")
        
        return total_loss / max(num_models, 1)
    
    def _compute_defense_evasion_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Compute loss to evade defense pipeline."""
        if self.defense_pipeline is None:
            return torch.tensor(0.0, device=self.device)
        
        # Process through defense pipeline
        processed_images = self.defense_pipeline.process_input(images)
        
        # Try to make patch undetectable by adversarial detector
        # This is a simplified version - in practice, would optimize against detector
        loss = torch.tensor(0.0, device=self.device)
        
        # Add penalty if adversarial detection would flag it
        # (This would require differentiable adversarial detector)
        
        return loss
    
    def _compute_tv_loss(self) -> torch.Tensor:
        """Compute total variation loss for patch smoothness."""
        patch = self.patch_generator.forward()
        
        # Compute TV loss
        diff_h = torch.abs(patch[:, 1:, :] - patch[:, :-1, :])
        diff_w = torch.abs(patch[:, :, 1:] - patch[:, :, :-1])
        
        tv_loss = diff_h.mean() + diff_w.mean()
        return tv_loss
    
    def optimize(
        self,
        images: torch.Tensor,
        iterations: int = 1000,
        learning_rate: float = 0.1,
        target_class: Optional[int] = None,
        patch_location: Optional[Tuple[int, int, int, int]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize patch to attack models and bypass defenses.
        
        Args:
            images: Training images (B, C, H, W)
            iterations: Number of optimization iterations
            learning_rate: Learning rate
            target_class: Target class for attack
            patch_location: Fixed patch location (None for random)
            verbose: Whether to print progress
            
        Returns:
            Optimization history and final patch
        """
        # Setup optimizer
        optimizer = optim.Adam(
            self.patch_generator.parameters(),
            lr=learning_rate
        )
        
        history = {
            'losses': [],
            'classification_losses': [],
            'defense_losses': [],
            'tv_losses': []
        }
        
        logger.info(f"Starting patch optimization for {iterations} iterations")
        
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            # Get current patch
            patch = self.patch_generator.forward()
            
            # Apply patch to images
            if patch_location is None:
                patched_images, _ = self.patch_applier.apply_patch_random_location(
                    images, patch
                )
            else:
                patched_images = self.patch_applier.apply_patch(
                    images, patch, patch_location
                )
            
            # Compute loss
            losses = self.compute_loss(patched_images, target_class)
            
            # Backward pass
            losses['total'].backward()
            optimizer.step()
            
            # Clamp patch to valid range
            with torch.no_grad():
                self.patch_generator.patch.data = torch.clamp(
                    self.patch_generator.patch.data, 0, 1
                )
            
            # Record history
            history['losses'].append(losses['total'].item())
            history['classification_losses'].append(losses['classification'].item())
            history['defense_losses'].append(losses['defense_evasion'].item())
            history['tv_losses'].append(losses['tv'].item())
            
            if verbose and (iteration + 1) % 100 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{iterations}: "
                    f"Total Loss = {losses['total'].item():.4f}, "
                    f"Class Loss = {losses['classification'].item():.4f}, "
                    f"Defense Loss = {losses['defense_evasion'].item():.4f}"
                )
        
        logger.info("Patch optimization completed")
        
        return {
            'history': history,
            'final_patch': self.patch_generator.get_patch()
        }
