"""
Report generation for evaluation results.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import json
import os
from datetime import datetime

from .metrics import compute_attack_metrics, compute_defense_metrics
from ..utils.visualization import visualize_attack_results


class EvaluationReporter:
    """
    Generate comprehensive evaluation reports.
    """
    
    def __init__(self, output_dir: str = 'data/results'):
        """
        Initialize reporter.
        
        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        include_visualizations: bool = True
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            save_path: Optional path to save report
            include_visualizations: Whether to include visualizations
            
        Returns:
            Report text
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'evaluation_report_{timestamp}.txt')
        
        # Compute metrics
        attack_metrics = compute_attack_metrics(results)
        defense_metrics = compute_defense_metrics(results)
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ADVERSARIAL PATCH EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Attack metrics
        report_lines.append("ATTACK METRICS")
        report_lines.append("-" * 80)
        for metric_name, value in attack_metrics.items():
            report_lines.append(f"{metric_name:40s}: {value:.4f}")
        report_lines.append("")
        
        # Defense metrics
        report_lines.append("DEFENSE METRICS")
        report_lines.append("-" * 80)
        for metric_name, value in defense_metrics.items():
            report_lines.append(f"{metric_name:40s}: {value:.4f}")
        report_lines.append("")
        
        # Single model results
        if 'single_model_results' in results:
            report_lines.append("SINGLE MODEL RESULTS")
            report_lines.append("-" * 80)
            for model_name, model_results in results['single_model_results'].items():
                success_rate = model_results.get('success_rate', 0.0)
                report_lines.append(
                    f"{model_name:40s}: "
                    f"Success Rate = {success_rate:.4f} "
                    f"({model_results.get('success_count', 0)}/{model_results.get('total_count', 0)})"
                )
            report_lines.append("")
        
        # Ensemble results
        if 'ensemble_results' in results and results['ensemble_results']:
            report_lines.append("ENSEMBLE RESULTS")
            report_lines.append("-" * 80)
            ensemble = results['ensemble_results']
            report_lines.append(
                f"Success Rate: {ensemble.get('success_rate', 0.0):.4f} "
                f"({ensemble.get('success_count', 0)}/{ensemble.get('total_count', 0)})"
            )
            report_lines.append("")
        
        # Defense results
        if 'defense_results' in results and results['defense_results']:
            report_lines.append("DEFENSE PIPELINE RESULTS")
            report_lines.append("-" * 80)
            defense = results['defense_results']
            report_lines.append(f"Bypass Rate: {defense.get('bypass_rate', 0.0):.4f}")
            report_lines.append(f"Bypassed: {defense.get('bypassed', 0)}")
            report_lines.append(f"Blocked: {defense.get('blocked', 0)}")
            
            if 'defense_breakdown' in defense:
                report_lines.append("\nDefense Breakdown:")
                for defense_name, count in defense['defense_breakdown'].items():
                    report_lines.append(f"  {defense_name}: {count} blocks")
            report_lines.append("")
        
        # Scenario results
        if 'scenario_results' in results:
            report_lines.append("SCENARIO RESULTS")
            report_lines.append("-" * 80)
            for scenario_name, scenario_results in results['scenario_results'].items():
                success_rate = scenario_results.get('success_rate', 0.0)
                report_lines.append(
                    f"{scenario_name:40s}: Success Rate = {success_rate:.4f}"
                )
            report_lines.append("")
        
        # Frame results summary
        if 'frame_results' in results and results['frame_results']:
            report_lines.append("FRAME-BY-FRAME SUMMARY")
            report_lines.append("-" * 80)
            frame_results = results['frame_results']
            confidences = [f.get('confidence', 0.0) for f in frame_results]
            classes = [f.get('predicted_class', 0) for f in frame_results]
            report_lines.append(f"Average Confidence: {np.mean(confidences):.4f}")
            report_lines.append(f"Std Confidence: {np.std(confidences):.4f}")
            report_lines.append(f"Unique Classes: {len(set(classes))}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        # Generate CSV summary
        csv_path = save_path.replace('.txt', '_summary.csv')
        self._generate_csv_summary(attack_metrics, defense_metrics, csv_path)
        
        # Generate visualizations
        if include_visualizations:
            viz_path = save_path.replace('.txt', '_visualization.png')
            visualize_attack_results(results, save_path=viz_path)
        
        return report_text
    
    def _generate_csv_summary(
        self,
        attack_metrics: Dict[str, float],
        defense_metrics: Dict[str, float],
        csv_path: str
    ) -> None:
        """Generate CSV summary of metrics."""
        all_metrics = {**attack_metrics, **defense_metrics}
        df = pd.DataFrame([all_metrics])
        df.to_csv(csv_path, index=False)


