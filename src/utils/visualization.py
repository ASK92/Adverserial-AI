"""
Visualization utilities for patches, detections, and results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict, Any
import cv2


def visualize_patch(
    patch: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Adversarial Patch'
) -> None:
    """
    Visualize an adversarial patch.
    
    Args:
        patch: Patch array (H, W, C) in range [0, 1] or [0, 255]
        save_path: Optional path to save the visualization
        title: Plot title
    """
    # Normalize to [0, 1] if needed
    if patch.max() > 1.0:
        patch = patch / 255.0
    
    plt.figure(figsize=(8, 8))
    plt.imshow(patch)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_detection(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    patch_location: Optional[Tuple[int, int, int, int]] = None,
    save_path: Optional[str] = None,
    title: str = 'Detection Results'
) -> None:
    """
    Visualize detection results with bounding boxes and labels.
    
    Args:
        image: Input image (H, W, C)
        detections: List of detection dictionaries with 'bbox', 'class', 'confidence'
        patch_location: Optional (x1, y1, x2, y2) patch location
        save_path: Optional path to save the visualization
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Normalize image if needed
    if image.max() > 1.0:
        display_image = image / 255.0
    else:
        display_image = image
    
    ax.imshow(display_image)
    
    # Draw patch location if provided
    if patch_location:
        x1, y1, x2, y2 = patch_location
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, 'ADVERSARIAL PATCH', 
                color='red', fontsize=12, fontweight='bold')
    
    # Draw detections
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f"{det.get('class', 'Unknown')}: {det.get('confidence', 0):.2f}"
            ax.text(x1, y1 - 5, label, 
                   color='green', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_attack_results(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize comprehensive attack results with multiple metrics.
    
    Args:
        results: Dictionary containing attack results and metrics
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success rate by defense
    if 'defense_success' in results:
        defenses = list(results['defense_success'].keys())
        success_rates = [results['defense_success'][d] for d in defenses]
        axes[0, 0].bar(defenses, success_rates, color='steelblue')
        axes[0, 0].set_title('Success Rate by Defense Layer', fontweight='bold')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Success rate by scenario
    if 'scenario_success' in results:
        scenarios = list(results['scenario_success'].keys())
        success_rates = [results['scenario_success'][s] for s in scenarios]
        axes[0, 1].bar(scenarios, success_rates, color='coral')
        axes[0, 1].set_title('Success Rate by Scenario', fontweight='bold')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Frame-by-frame success
    if 'frame_success' in results:
        frames = range(len(results['frame_success']))
        axes[1, 0].plot(frames, results['frame_success'], marker='o', color='green')
        axes[1, 0].set_title('Success Rate Across Frames', fontweight='bold')
        axes[1, 0].set_xlabel('Frame Number')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].grid(alpha=0.3)
    
    # Overall metrics
    if 'overall_metrics' in results:
        metrics = results['overall_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        axes[1, 1].barh(metric_names, metric_values, color='purple')
        axes[1, 1].set_title('Overall Attack Metrics', fontweight='bold')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


