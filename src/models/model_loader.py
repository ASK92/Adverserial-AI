"""
Model loading utilities for YOLOv5, ResNet, and EfficientNet.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Unified interface for loading and using different computer vision models.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize model loader.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device
        self.models = {}
    
    def load_yolov5(self, model_name: str = 'yolov5s', pretrained: bool = True):
        """
        Load YOLOv5 model.
        
        Args:
            model_name: YOLOv5 model variant ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            pretrained: Whether to load pretrained weights
            
        Returns:
            YOLOv5 model instance
        """
        try:
            from ultralytics import YOLO
            model = YOLO(f'{model_name}.pt' if pretrained else model_name)
            model.to(self.device)
            model.eval()
            self.models['yolov5'] = model
            logger.info(f"Loaded YOLOv5 model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLOv5: {e}")
            raise
    
    def load_resnet(self, model_name: str = 'resnet50', pretrained: bool = True, num_classes: int = 1000):
        """
        Load ResNet model.
        
        Args:
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', etc.)
            pretrained: Whether to load pretrained weights
            num_classes: Number of output classes
            
        Returns:
            ResNet model instance
        """
        try:
            if hasattr(models, model_name):
                model = getattr(models, model_name)(pretrained=pretrained)
                if num_classes != 1000:
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.to(self.device)
                model.eval()
                self.models['resnet'] = model
                logger.info(f"Loaded ResNet model: {model_name}")
                return model
            else:
                raise ValueError(f"Unknown ResNet variant: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load ResNet: {e}")
            raise
    
    def load_efficientnet(self, model_name: str = 'efficientnet_b0', pretrained: bool = True, num_classes: int = 1000):
        """
        Load EfficientNet model.
        
        Args:
            model_name: EfficientNet variant (e.g., 'efficientnet_b0', 'efficientnet_b1', etc.)
            pretrained: Whether to load pretrained weights
            num_classes: Number of output classes
            
        Returns:
            EfficientNet model instance
        """
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            model.to(self.device)
            model.eval()
            self.models['efficientnet'] = model
            logger.info(f"Loaded EfficientNet model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load EfficientNet: {e}")
            raise


def load_yolov5(model_name: str = 'yolov5s', pretrained: bool = True, device: str = 'cuda'):
    """Convenience function to load YOLOv5."""
    loader = ModelLoader(device=device)
    return loader.load_yolov5(model_name, pretrained)


def load_resnet(model_name: str = 'resnet50', pretrained: bool = True, num_classes: int = 1000, device: str = 'cuda'):
    """Convenience function to load ResNet."""
    loader = ModelLoader(device=device)
    return loader.load_resnet(model_name, pretrained, num_classes)


def load_efficientnet(model_name: str = 'efficientnet_b0', pretrained: bool = True, num_classes: int = 1000, device: str = 'cuda'):
    """Convenience function to load EfficientNet."""
    loader = ModelLoader(device=device)
    return loader.load_efficientnet(model_name, pretrained, num_classes)


