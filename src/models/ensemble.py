"""
Model ensemble for consensus-based detection.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    Ensemble of multiple models for robust detection.
    """
    
    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        consensus_threshold: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize model ensemble.
        
        Args:
            models: Dictionary of model_name -> model_instance
            consensus_threshold: Minimum agreement ratio for consensus
            device: Computing device
        """
        self.models = models
        self.consensus_threshold = consensus_threshold
        self.device = device
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Get predictions from all models in ensemble.
        
        Args:
            image: Input image tensor (B, C, H, W)
            
        Returns:
            Dictionary with predictions from each model and consensus result
        """
        predictions = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                try:
                    if 'yolo' in name.lower():
                        # YOLO returns different format
                        pred = model(image)
                        predictions[name] = self._process_yolo_pred(pred)
                    else:
                        # Standard classification models
                        output = model(image)
                        if isinstance(output, torch.Tensor):
                            probs = torch.softmax(output, dim=1)
                            pred_class = torch.argmax(probs, dim=1)
                            predictions[name] = {
                                'class': pred_class.cpu().numpy(),
                                'confidence': torch.max(probs, dim=1)[0].cpu().numpy(),
                                'probabilities': probs.cpu().numpy()
                            }
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
                    predictions[name] = None
        
        # Compute consensus
        consensus = self._compute_consensus(predictions)
        
        return {
            'individual_predictions': predictions,
            'consensus': consensus
        }
    
    def _process_yolo_pred(self, yolo_output) -> Dict[str, Any]:
        """
        Process YOLO output to standard format.
        
        Args:
            yolo_output: Raw YOLO model output
            
        Returns:
            Standardized prediction dictionary
        """
        # YOLO output format varies, this is a simplified version
        # In practice, you'd parse the actual YOLO output structure
        return {
            'detections': [],
            'confidence': 0.0
        }
    
    def _compute_consensus(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute consensus from individual predictions.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Consensus result with agreement metrics
        """
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        
        if len(valid_preds) == 0:
            return {
                'agreement': 0.0,
                'consensus_reached': False,
                'predicted_class': None
            }
        
        # For classification tasks, check class agreement
        classes = []
        for pred in valid_preds.values():
            if 'class' in pred:
                classes.append(pred['class'][0] if isinstance(pred['class'], np.ndarray) else pred['class'])
        
        if len(classes) == 0:
            return {
                'agreement': 0.0,
                'consensus_reached': False,
                'predicted_class': None
            }
        
        # Find most common class
        unique_classes, counts = np.unique(classes, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_class = unique_classes[most_common_idx]
        agreement = counts[most_common_idx] / len(classes)
        
        consensus_reached = agreement >= self.consensus_threshold
        
        return {
            'agreement': float(agreement),
            'consensus_reached': consensus_reached,
            'predicted_class': int(most_common_class),
            'num_models': len(valid_preds)
        }


