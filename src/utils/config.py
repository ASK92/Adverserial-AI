"""
Configuration management for the adversarial patch pipeline.
"""
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Central configuration class for the adversarial patch system.
    
    Attributes:
        device: Computing device ('cuda' or 'cpu')
        random_seed: Random seed for reproducibility
        data_dir: Directory for datasets
        models_dir: Directory for model checkpoints
        patches_dir: Directory for saved patches
        results_dir: Directory for evaluation results
        logs_dir: Directory for log files
    """
    device: str = 'cuda'
    random_seed: int = 42
    data_dir: str = 'data'
    models_dir: str = 'data/models'
    patches_dir: str = 'data/patches'
    results_dir: str = 'data/results'
    logs_dir: str = 'logs'
    
    # Model configurations
    models: Dict[str, Any] = field(default_factory=lambda: {
        'yolov5': {
            'name': 'yolov5s',
            'pretrained': True,
            'confidence_threshold': 0.5
        },
        'resnet': {
            'name': 'resnet50',
            'pretrained': True,
            'num_classes': 1000
        },
        'efficientnet': {
            'name': 'efficientnet_b0',
            'pretrained': True,
            'num_classes': 1000
        }
    })
    
    # Defense configurations
    defenses: Dict[str, Any] = field(default_factory=lambda: {
        'input_normalization': {
            'enabled': True,
            'brightness_range': (0.7, 1.3),
            'blur_prob': 0.3,
            'blur_kernel_size': 5,
            'affine_prob': 0.3,
            'noise_std': 0.05
        },
        'adversarial_detection': {
            'enabled': True,
            'norm_threshold': 0.1,
            'ood_detector': True
        },
        'multi_frame_smoothing': {
            'enabled': True,
            'frame_window': 5,
            'consensus_threshold': 0.6
        },
        'context_rules': {
            'enabled': True,
            'temporal_check': True,
            'spatial_check': True
        },
        'model_ensemble': {
            'enabled': True,
            'models': ['yolov5', 'resnet', 'efficientnet'],
            'consensus_threshold': 0.5
        }
    })
    
    # Patch generation configurations
    patch: Dict[str, Any] = field(default_factory=lambda: {
        'size': (100, 100),
        'learning_rate': 0.1,
        'iterations': 1000,
        'patch_type': 'universal',
        'target_class': None,
        'optimizer': 'adam',
        'loss_weights': {
            'classification': 1.0,
            'detection': 1.0,
            'defense_evasion': 1.0,
            'temporal': 0.5,
            'context': 0.3
        }
    })
    
    # Evaluation configurations
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        'num_samples': 100,
        'lighting_scenarios': ['normal', 'bright', 'dark', 'harsh'],
        'angle_scenarios': [0, 15, 30, 45],
        'num_frames': 10,
        'save_visualizations': True
    })
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        dir_path = os.path.dirname(yaml_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        config_dict = {
            'device': self.device,
            'random_seed': self.random_seed,
            'data_dir': self.data_dir,
            'models_dir': self.models_dir,
            'patches_dir': self.patches_dir,
            'results_dir': self.results_dir,
            'logs_dir': self.logs_dir,
            'models': self.models,
            'defenses': self.defenses,
            'patch': self.patch,
            'evaluation': self.evaluation
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        dirs = [
            self.data_dir, self.models_dir, self.patches_dir,
            self.results_dir, self.logs_dir
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
