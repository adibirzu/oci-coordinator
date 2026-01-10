"""
Model Tier Strategy for intelligent LLM selection.

Implements a three-tier model strategy:
- OPUS: Critical decisions, complex reasoning, multi-step planning
- SONNET: Standard queries, moderate complexity
- HAIKU: Simple operations, validation, formatting

Selection is based on:
1. Task complexity indicators
2. Token budget constraints
3. Response quality requirements
4. Latency requirements

Example usage:
    strategy = ModelTierStrategy()
    model = strategy.select_model(
        task_type="planning",
        complexity=TaskComplexity.HIGH,
        token_budget=10000
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import re
import structlog

logger = structlog.get_logger()


class ModelTier(Enum):
    """Model tier levels."""
    OPUS = "opus"       # Best quality, highest cost
    SONNET = "sonnet"   # Balanced quality/cost
    HAIKU = "haiku"     # Fast, low cost


class TaskComplexity(Enum):
    """Task complexity levels."""
    CRITICAL = "critical"  # Requires best model, no compromise
    HIGH = "high"          # Complex, benefits from better model
    MEDIUM = "medium"      # Standard tasks
    LOW = "low"            # Simple operations


@dataclass
class ModelConfig:
    """Configuration for a model tier."""

    tier: ModelTier
    model_id: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float

    # Capabilities
    supports_tools: bool = True
    supports_vision: bool = False
    supports_long_context: bool = False

    # Performance characteristics
    avg_latency_ms: int = 1000
    quality_score: float = 1.0  # Relative quality (0-1)


# Default model configurations (can be overridden)
DEFAULT_MODEL_CONFIGS = {
    ModelTier.OPUS: ModelConfig(
        tier=ModelTier.OPUS,
        model_id="claude-opus-4-5-20251101",
        max_tokens=8192,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        supports_vision=True,
        supports_long_context=True,
        avg_latency_ms=3000,
        quality_score=1.0,
    ),
    ModelTier.SONNET: ModelConfig(
        tier=ModelTier.SONNET,
        model_id="claude-sonnet-4-20250514",
        max_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_vision=True,
        supports_long_context=True,
        avg_latency_ms=1500,
        quality_score=0.85,
    ),
    ModelTier.HAIKU: ModelConfig(
        tier=ModelTier.HAIKU,
        model_id="claude-3-5-haiku-20241022",
        max_tokens=4096,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        supports_vision=False,
        supports_long_context=False,
        avg_latency_ms=500,
        quality_score=0.7,
    ),
}


# Task type to default complexity mapping
TASK_TYPE_COMPLEXITY = {
    # Critical tasks - always use best model
    "security_audit": TaskComplexity.CRITICAL,
    "incident_response": TaskComplexity.CRITICAL,
    "architecture_decision": TaskComplexity.CRITICAL,

    # High complexity tasks
    "planning": TaskComplexity.HIGH,
    "multi_step_reasoning": TaskComplexity.HIGH,
    "code_review": TaskComplexity.HIGH,
    "root_cause_analysis": TaskComplexity.HIGH,
    "error_analysis": TaskComplexity.HIGH,

    # Medium complexity tasks
    "query_execution": TaskComplexity.MEDIUM,
    "data_analysis": TaskComplexity.MEDIUM,
    "report_generation": TaskComplexity.MEDIUM,
    "intent_classification": TaskComplexity.MEDIUM,

    # Low complexity tasks
    "validation": TaskComplexity.LOW,
    "formatting": TaskComplexity.LOW,
    "summarization": TaskComplexity.LOW,
    "simple_lookup": TaskComplexity.LOW,
}

# Complexity indicators in user queries
COMPLEXITY_INDICATORS = {
    TaskComplexity.CRITICAL: [
        r'\b(security|breach|incident|critical|urgent|emergency)\b',
        r'\b(audit|compliance|vulnerability)\b',
    ],
    TaskComplexity.HIGH: [
        r'\b(analyze|investigate|diagnose|troubleshoot)\b',
        r'\b(why|how come|root cause|explain)\b',
        r'\b(plan|design|architect|strategy)\b',
        r'\b(compare|evaluate|assess|recommend)\b',
    ],
    TaskComplexity.MEDIUM: [
        r'\b(show|display|list|get|retrieve)\b',
        r'\b(check|verify|validate|confirm)\b',
        r'\b(report|summary|status)\b',
    ],
    TaskComplexity.LOW: [
        r'\b(format|convert|transform)\b',
        r'\b(count|sum|total|average)\b',
    ],
}


@dataclass
class ModelSelection:
    """Result of model selection."""

    tier: ModelTier
    config: ModelConfig
    reason: str

    # Selection factors
    task_type: Optional[str] = None
    complexity: Optional[TaskComplexity] = None
    estimated_tokens: int = 0
    estimated_cost: float = 0.0


class ModelTierStrategy:
    """
    Strategy for selecting the appropriate model tier.

    The strategy considers:
    1. Task complexity (critical, high, medium, low)
    2. Token budget (if limited)
    3. Latency requirements
    4. Quality requirements

    Usage:
        strategy = ModelTierStrategy()

        # Simple selection
        model = strategy.select_model(task_type="planning")

        # With constraints
        model = strategy.select_model(
            task_type="data_analysis",
            max_cost=0.10,
            max_latency_ms=2000
        )

        # With query analysis
        model = strategy.select_for_query(
            "Why is my database slow?",
            context={"domain": "database"}
        )
    """

    def __init__(
        self,
        model_configs: Optional[Dict[ModelTier, ModelConfig]] = None,
        default_tier: ModelTier = ModelTier.SONNET
    ):
        self.model_configs = model_configs or DEFAULT_MODEL_CONFIGS
        self.default_tier = default_tier

    def select_model(
        self,
        task_type: Optional[str] = None,
        complexity: Optional[TaskComplexity] = None,
        estimated_tokens: int = 1000,
        max_cost: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
        require_vision: bool = False,
        require_long_context: bool = False,
    ) -> ModelSelection:
        """
        Select the appropriate model for a task.

        Args:
            task_type: Type of task (e.g., "planning", "validation")
            complexity: Override complexity level
            estimated_tokens: Estimated input + output tokens
            max_cost: Maximum cost budget for this call
            max_latency_ms: Maximum acceptable latency
            require_vision: Whether vision capability is needed
            require_long_context: Whether long context is needed

        Returns:
            ModelSelection with chosen tier and reasoning
        """
        # Determine complexity
        if complexity is None and task_type:
            complexity = TASK_TYPE_COMPLEXITY.get(task_type, TaskComplexity.MEDIUM)
        elif complexity is None:
            complexity = TaskComplexity.MEDIUM

        # Get candidate tiers based on complexity
        if complexity == TaskComplexity.CRITICAL:
            candidates = [ModelTier.OPUS]
        elif complexity == TaskComplexity.HIGH:
            candidates = [ModelTier.OPUS, ModelTier.SONNET]
        elif complexity == TaskComplexity.MEDIUM:
            candidates = [ModelTier.SONNET, ModelTier.HAIKU]
        else:  # LOW
            candidates = [ModelTier.HAIKU, ModelTier.SONNET]

        # Filter by capability requirements
        if require_vision:
            candidates = [t for t in candidates if self.model_configs[t].supports_vision]
        if require_long_context:
            candidates = [t for t in candidates if self.model_configs[t].supports_long_context]

        # Filter by constraints
        if max_latency_ms:
            candidates = [
                t for t in candidates
                if self.model_configs[t].avg_latency_ms <= max_latency_ms
            ]

        if max_cost:
            candidates = [
                t for t in candidates
                if self._estimate_cost(t, estimated_tokens) <= max_cost
            ]

        # Select best remaining candidate
        if not candidates:
            # No candidates meet constraints, use default with warning
            logger.warning(
                "no_model_meets_constraints",
                task_type=task_type,
                complexity=complexity.value if complexity else None,
                using_default=self.default_tier.value
            )
            tier = self.default_tier
            reason = "No model meets all constraints, using default"
        else:
            tier = candidates[0]
            reason = self._build_reason(tier, task_type, complexity)

        config = self.model_configs[tier]
        estimated_cost = self._estimate_cost(tier, estimated_tokens)

        return ModelSelection(
            tier=tier,
            config=config,
            reason=reason,
            task_type=task_type,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
        )

    def select_for_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelSelection:
        """
        Select model based on query analysis.

        Analyzes the query text to determine complexity,
        then delegates to select_model().
        """
        # Analyze query for complexity
        complexity = self._analyze_query_complexity(query)

        # Check context for additional hints
        if context:
            # Domain-specific overrides
            domain = context.get("domain", "")
            if domain in ["security", "compliance"]:
                complexity = max(complexity, TaskComplexity.HIGH,
                               key=lambda c: list(TaskComplexity).index(c))

            # Urgent flag
            if context.get("urgent", False):
                complexity = TaskComplexity.CRITICAL

        # Estimate tokens from query length
        estimated_tokens = self._estimate_tokens(query)

        return self.select_model(
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            **kwargs
        )

    def _analyze_query_complexity(self, query: str) -> TaskComplexity:
        """Analyze query text to determine complexity."""
        query_lower = query.lower()

        # Check complexity indicators in order (critical -> low)
        for complexity in [TaskComplexity.CRITICAL, TaskComplexity.HIGH,
                          TaskComplexity.MEDIUM, TaskComplexity.LOW]:
            patterns = COMPLEXITY_INDICATORS.get(complexity, [])
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return complexity

        return TaskComplexity.MEDIUM  # Default

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        # Rough approximation: ~4 chars per token
        input_tokens = len(text) // 4 + 100  # Add buffer
        output_tokens = input_tokens * 2  # Assume output is 2x input
        return input_tokens + output_tokens

    def _estimate_cost(self, tier: ModelTier, tokens: int) -> float:
        """Estimate cost for token count."""
        config = self.model_configs[tier]
        # Assume 1/3 input, 2/3 output
        input_tokens = tokens // 3
        output_tokens = tokens - input_tokens
        return (
            (input_tokens / 1000) * config.cost_per_1k_input +
            (output_tokens / 1000) * config.cost_per_1k_output
        )

    def _build_reason(
        self,
        tier: ModelTier,
        task_type: Optional[str],
        complexity: Optional[TaskComplexity]
    ) -> str:
        """Build human-readable selection reason."""
        reasons = []

        if complexity:
            reasons.append(f"complexity={complexity.value}")
        if task_type:
            reasons.append(f"task_type={task_type}")

        tier_reasons = {
            ModelTier.OPUS: "best quality for critical/complex tasks",
            ModelTier.SONNET: "balanced quality and cost",
            ModelTier.HAIKU: "fast and efficient for simple tasks",
        }
        reasons.append(tier_reasons[tier])

        return f"Selected {tier.value}: {', '.join(reasons)}"

    def get_tier_info(self, tier: ModelTier) -> Dict[str, Any]:
        """Get information about a model tier."""
        config = self.model_configs[tier]
        return {
            "tier": tier.value,
            "model_id": config.model_id,
            "max_tokens": config.max_tokens,
            "supports_vision": config.supports_vision,
            "supports_long_context": config.supports_long_context,
            "avg_latency_ms": config.avg_latency_ms,
            "quality_score": config.quality_score,
            "cost_per_1k_input": config.cost_per_1k_input,
            "cost_per_1k_output": config.cost_per_1k_output,
        }

    def get_all_tiers_info(self) -> List[Dict[str, Any]]:
        """Get information about all model tiers."""
        return [self.get_tier_info(tier) for tier in ModelTier]


# Convenience function for quick model selection
def get_model_for_task(
    task_type: str,
    **kwargs
) -> ModelConfig:
    """
    Quick model selection for a task type.

    Args:
        task_type: Type of task (see TASK_TYPE_COMPLEXITY)
        **kwargs: Additional arguments for select_model()

    Returns:
        ModelConfig for the selected model
    """
    strategy = ModelTierStrategy()
    selection = strategy.select_model(task_type=task_type, **kwargs)
    return selection.config


# Decorator for automatic model selection
def with_model_tier(
    task_type: Optional[str] = None,
    complexity: Optional[TaskComplexity] = None,
    **strategy_kwargs
):
    """
    Decorator to automatically select model tier for a function.

    The decorated function receives 'model_config' as first argument.

    Example:
        @with_model_tier(task_type="planning")
        async def plan_workflow(model_config: ModelConfig, query: str):
            # Use model_config.model_id for LLM calls
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            strategy = ModelTierStrategy()
            selection = strategy.select_model(
                task_type=task_type,
                complexity=complexity,
                **strategy_kwargs
            )

            logger.debug(
                "model_tier_selected",
                function=func.__name__,
                tier=selection.tier.value,
                reason=selection.reason
            )

            return await func(selection.config, *args, **kwargs)
        return wrapper
    return decorator
