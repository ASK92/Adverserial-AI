"""
Metrics computation for attack and defense evaluation.
"""
import numpy as np
from typing import Dict, Any, List


def compute_attack_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute comprehensive attack metrics.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Dictionary of attack metrics
    """
    metrics = {}
    
    # Single model success rates
    if 'single_model_results' in results:
        for model_name, model_results in results['single_model_results'].items():
            metrics[f'{model_name}_success_rate'] = model_results.get('success_rate', 0.0)
    
    # Ensemble metrics
    if 'ensemble_results' in results:
        metrics['ensemble_success_rate'] = results['ensemble_results'].get('success_rate', 0.0)
    
    # Defense bypass metrics
    if 'defense_results' in results:
        defense_results = results['defense_results']
        metrics['defense_bypass_rate'] = defense_results.get('bypass_rate', 0.0)
        metrics['defense_blocked_count'] = defense_results.get('blocked', 0)
        metrics['defense_bypassed_count'] = defense_results.get('bypassed', 0)
    
    # Scenario metrics
    if 'scenario_results' in results:
        scenario_rates = [
            r.get('success_rate', 0.0)
            for r in results['scenario_results'].values()
        ]
        metrics['avg_scenario_success'] = np.mean(scenario_rates) if scenario_rates else 0.0
        metrics['min_scenario_success'] = np.min(scenario_rates) if scenario_rates else 0.0
        metrics['max_scenario_success'] = np.max(scenario_rates) if scenario_rates else 0.0
    
    # Frame consistency
    if 'frame_results' in results:
        frame_results = results['frame_results']
        if frame_results:
            confidences = [f.get('confidence', 0.0) for f in frame_results]
            metrics['avg_frame_confidence'] = np.mean(confidences)
            metrics['std_frame_confidence'] = np.std(confidences)
    
    return metrics


def compute_defense_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute defense effectiveness metrics.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Dictionary of defense metrics
    """
    metrics = {}
    
    if 'defense_results' not in results:
        return metrics
    
    defense_results = results['defense_results']
    
    # Overall defense effectiveness
    total = defense_results.get('bypassed', 0) + defense_results.get('blocked', 0)
    if total > 0:
        metrics['defense_effectiveness'] = defense_results.get('blocked', 0) / total
        metrics['false_negative_rate'] = defense_results.get('bypassed', 0) / total
    else:
        metrics['defense_effectiveness'] = 0.0
        metrics['false_negative_rate'] = 0.0
    
    # Defense breakdown
    if 'defense_breakdown' in defense_results:
        breakdown = defense_results['defense_breakdown']
        for defense_name, count in breakdown.items():
            metrics[f'{defense_name}_blocks'] = count
    
    return metrics


