"""
Main entry point for the adversarial patch pipeline.
"""
import click
import torch
import numpy as np
import yaml
import os
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.model_loader import ModelLoader
from src.models.ensemble import ModelEnsemble
from src.defenses.defense_pipeline import (
    DefensePipeline,
    InputNormalization,
    AdversarialDetection,
    MultiFrameSmoothing,
    ContextRuleEngine
)
from src.patch.patch_generator import AdversarialPatchGenerator
from src.patch.patch_optimizer import PatchOptimizer
from src.patch.patch_applier import PatchApplier
from src.evaluation.evaluator import PatchEvaluator
from src.evaluation.reporter import EvaluationReporter

logger = setup_logger()


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to configuration YAML file')
@click.option('--device', default='cuda', help='Computing device (cuda/cpu)')
@click.pass_context
def cli(ctx, config, device):
    """Adversarial Patch Pipeline - Attack and Defense Framework"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config()
        cfg.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    
    cfg.create_directories()
    ctx.obj['config'] = cfg
    ctx.obj['device'] = cfg.device
    
    logger.info(f"Initialized pipeline with device: {cfg.device}")


@cli.command()
@click.option('--patch-size', nargs=2, type=int, default=(100, 100), help='Patch size (height width)')
@click.option('--iterations', default=1000, help='Number of optimization iterations')
@click.option('--learning-rate', default=0.1, help='Learning rate')
@click.option('--output', default='data/patches/patch.pt', help='Output path for patch')
@click.option('--data-path', help='Path to training images')
@click.pass_context
def train_patch(ctx, patch_size, iterations, learning_rate, output, data_path):
    """Train an adversarial patch"""
    config = ctx.obj['config']
    device = ctx.obj['device']
    
    logger.info("Starting patch training...")
    
    # Load models
    model_loader = ModelLoader(device=device)
    models = {}
    
    for model_name, model_cfg in config.models.items():
        if model_cfg.get('pretrained', True):
            try:
                if 'yolo' in model_name:
                    models[model_name] = model_loader.load_yolov5(
                        model_cfg.get('name', 'yolov5s'),
                        model_cfg.get('pretrained', True)
                    )
                elif 'resnet' in model_name:
                    models[model_name] = model_loader.load_resnet(
                        model_cfg.get('name', 'resnet50'),
                        model_cfg.get('pretrained', True),
                        model_cfg.get('num_classes', 1000)
                    )
                elif 'efficientnet' in model_name:
                    models[model_name] = model_loader.load_efficientnet(
                        model_cfg.get('name', 'efficientnet_b0'),
                        model_cfg.get('pretrained', True),
                        model_cfg.get('num_classes', 1000)
                    )
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
    
    if len(models) == 0:
        raise RuntimeError("No models could be loaded. Please install required dependencies.")
    
    # Setup defense pipeline
    defense_cfg = config.defenses
    input_norm_cfg = defense_cfg.get('input_normalization', {})
    adv_det_cfg = defense_cfg.get('adversarial_detection', {})
    mf_smooth_cfg = defense_cfg.get('multi_frame_smoothing', {})
    context_cfg = defense_cfg.get('context_rules', {})
    
    defense_pipeline = DefensePipeline(
        input_normalization=InputNormalization(
            **{k: v for k, v in input_norm_cfg.items() if k != 'enabled'},
            enabled=input_norm_cfg.get('enabled', True)
        ),
        adversarial_detection=AdversarialDetection(
            **{k: v for k, v in adv_det_cfg.items() if k != 'enabled'},
            enabled=adv_det_cfg.get('enabled', True),
            device=device
        ),
        multi_frame_smoothing=MultiFrameSmoothing(
            **{k: v for k, v in mf_smooth_cfg.items() if k != 'enabled'},
            enabled=mf_smooth_cfg.get('enabled', True)
        ),
        context_rules=ContextRuleEngine(
            **{k: v for k, v in context_cfg.items() if k != 'enabled'},
            enabled=context_cfg.get('enabled', True)
        ),
        enabled=True
    )
    
    # Create patch generator
    patch_generator = AdversarialPatchGenerator(
        patch_size=tuple(patch_size),
        device=device
    )
    
    # Create optimizer
    optimizer = PatchOptimizer(
        patch_generator=patch_generator,
        models=models,
        defense_pipeline=defense_pipeline,
        loss_weights=config.patch.get('loss_weights', {}),
        device=device
    )
    
    # Load training data (placeholder - would load actual images)
    # For now, create dummy data
    dummy_images = torch.randn(10, 3, 224, 224).to(device)
    
    # Optimize patch
    results = optimizer.optimize(
        images=dummy_images,
        iterations=iterations,
        learning_rate=learning_rate
    )
    
    # Save patch
    os.makedirs(os.path.dirname(output), exist_ok=True)
    patch_np = results['final_patch']
    torch.save(patch_np, output)
    
    logger.info(f"Patch training completed. Saved to {output}")


@cli.command()
@click.option('--patch-path', required=True, help='Path to trained patch')
@click.option('--test-data', help='Path to test images')
@click.option('--output-dir', default='data/results', help='Output directory for results')
@click.pass_context
def evaluate(ctx, patch_path, test_data, output_dir):
    """Evaluate an adversarial patch"""
    config = ctx.obj['config']
    device = ctx.obj['device']
    
    logger.info("Starting patch evaluation...")
    
    # Load models
    model_loader = ModelLoader(device=device)
    models = {}
    
    for model_name, model_cfg in config.models.items():
        if model_cfg.get('pretrained', True):
            try:
                if 'yolo' in model_name:
                    models[model_name] = model_loader.load_yolov5(
                        model_cfg.get('name', 'yolov5s'),
                        model_cfg.get('pretrained', True)
                    )
                elif 'resnet' in model_name:
                    models[model_name] = model_loader.load_resnet(
                        model_cfg.get('name', 'resnet50'),
                        model_cfg.get('pretrained', True),
                        model_cfg.get('num_classes', 1000)
                    )
                elif 'efficientnet' in model_name:
                    models[model_name] = model_loader.load_efficientnet(
                        model_cfg.get('name', 'efficientnet_b0'),
                        model_cfg.get('pretrained', True),
                        model_cfg.get('num_classes', 1000)
                    )
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
    
    if len(models) == 0:
        raise RuntimeError("No models could be loaded. Please install required dependencies.")
    
    # Setup ensemble
    ensemble = ModelEnsemble(
        models=models,
        consensus_threshold=config.defenses.get('model_ensemble', {}).get('consensus_threshold', 0.5),
        device=device
    )
    
    # Setup defense pipeline
    defense_cfg = config.defenses
    input_norm_cfg = defense_cfg.get('input_normalization', {})
    adv_det_cfg = defense_cfg.get('adversarial_detection', {})
    mf_smooth_cfg = defense_cfg.get('multi_frame_smoothing', {})
    context_cfg = defense_cfg.get('context_rules', {})
    
    defense_pipeline = DefensePipeline(
        input_normalization=InputNormalization(
            **{k: v for k, v in input_norm_cfg.items() if k != 'enabled'},
            enabled=input_norm_cfg.get('enabled', True)
        ),
        adversarial_detection=AdversarialDetection(
            **{k: v for k, v in adv_det_cfg.items() if k != 'enabled'},
            enabled=adv_det_cfg.get('enabled', True),
            device=device
        ),
        multi_frame_smoothing=MultiFrameSmoothing(
            **{k: v for k, v in mf_smooth_cfg.items() if k != 'enabled'},
            enabled=mf_smooth_cfg.get('enabled', True)
        ),
        context_rules=ContextRuleEngine(
            **{k: v for k, v in context_cfg.items() if k != 'enabled'},
            enabled=context_cfg.get('enabled', True)
        ),
        enabled=True
    )
    
    # Load patch
    patch_tensor = torch.load(patch_path)
    if isinstance(patch_tensor, np.ndarray):
        patch_tensor = torch.from_numpy(patch_tensor)
    patch_tensor = patch_tensor.to(device)
    
    # Create evaluator
    evaluator = PatchEvaluator(
        models=models,
        defense_pipeline=defense_pipeline,
        model_ensemble=ensemble,
        device=device
    )
    
    # Load test data (placeholder)
    test_images = torch.randn(20, 3, 224, 224).to(device)
    
    # Evaluate
    results = evaluator.evaluate_patch(
        patch=patch_tensor,
        test_images=test_images,
        num_frames=config.evaluation.get('num_frames', 10)
    )
    
    # Generate report
    reporter = EvaluationReporter(output_dir=output_dir)
    report = reporter.generate_report(results, include_visualizations=True)
    
    print("\n" + report)
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


@cli.command()
@click.option('--output', default='config.yaml', help='Output path for config file')
def init_config(output):
    """Generate a default configuration file"""
    config = Config()
    config.to_yaml(output)
    click.echo(f"Configuration file created at {output}")


if __name__ == '__main__':
    cli()
