"""
Evaluation framework for OCI Coordinator.

This module provides evaluation capabilities for testing
intent classification, agent routing, and response quality.
"""

from src.evaluation.dataset import EvaluationCase, EvaluationDataset, load_dataset
from src.evaluation.judge import JudgmentCriteria, JudgmentResult, LLMJudge
from src.evaluation.metrics import EvaluationMetrics, MetricsReport
from src.evaluation.runner import EvaluationResult, EvaluationRunner

__all__ = [
    # Dataset
    "EvaluationCase",
    "EvaluationDataset",
    "load_dataset",
    # Judge
    "LLMJudge",
    "JudgmentResult",
    "JudgmentCriteria",
    # Metrics
    "EvaluationMetrics",
    "MetricsReport",
    # Runner
    "EvaluationRunner",
    "EvaluationResult",
]
