"""
Self-Healing Agent Capabilities.

Provides LLM-powered self-healing, error correction, and logic validation
for OCI Coordinator agents.

Components:
- SelfHealingMixin: Mixin class adding self-healing to any agent
- ErrorAnalyzer: LLM-based error analysis and diagnosis
- ParameterCorrector: Automatic parameter fixing based on errors
- LogicValidator: Pre-execution validation of agent decisions
- RetryStrategy: Smart retry with learning from failures
"""

from src.agents.self_healing.analyzer import ErrorAnalyzer, ErrorAnalysis
from src.agents.self_healing.corrector import ParameterCorrector, CorrectionResult
from src.agents.self_healing.validator import LogicValidator, ValidationResult
from src.agents.self_healing.retry import RetryStrategy, RetryDecision
from src.agents.self_healing.mixin import SelfHealingMixin

__all__ = [
    "SelfHealingMixin",
    "ErrorAnalyzer",
    "ErrorAnalysis",
    "ParameterCorrector",
    "CorrectionResult",
    "LogicValidator",
    "ValidationResult",
    "RetryStrategy",
    "RetryDecision",
]
