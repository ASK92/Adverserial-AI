"""
Evaluation framework for adversarial patch attacks and defenses.
"""
from .evaluator import PatchEvaluator
from .metrics import compute_attack_metrics, compute_defense_metrics
from .reporter import EvaluationReporter

__all__ = [
    'PatchEvaluator',
    'compute_attack_metrics',
    'compute_defense_metrics',
    'EvaluationReporter'
]


