"""LLM-specific OpenTelemetry instrumentation.

Implements the OpenTelemetry GenAI Semantic Conventions for LLM observability.
Tracks token usage, latency, model parameters, and agent operations.

Based on:
- OpenTelemetry GenAI Semantic Convention v1.37+
- https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenAI, Anthropic, and Oracle Code Assist integration

Supported by:
- OCI APM (via Zipkin/OTLP exporter)
- OCI Monitoring (custom metrics for dashboards)
- viewapp LLM Observability dashboards
- DataDog LLM Observability (compatible attributes)

Metrics Published to OCI Monitoring:
- llm_input_tokens, llm_output_tokens, llm_total_tokens
- llm_latency_ms, llm_request_count, llm_error_count
- llm_cost_estimate_usd

Usage:
    from src.observability.llm_tracing import LLMInstrumentor, llm_span

    # Decorate LLM calls
    @llm_span(operation="chat", model="oca/gpt5")
    async def call_llm(prompt: str) -> str:
        ...

    # Or use context manager
    with LLMInstrumentor.span("tool_call", model="oca/gpt5") as span:
        span.set_tokens(input=100, output=50)
        result = await call_tool(...)

    # For Oracle Code Assist
    with OracleCodeAssistInstrumentor.chat_span(model="oca/gpt5") as span:
        span.set_prompt(prompt)
        response = await oca_client.chat(prompt)
        span.set_completion(response)
        span.set_tokens(input=100, output=50)
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

import structlog
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

logger = structlog.get_logger(__name__)

# Configuration
CAPTURE_CONTENT = os.getenv("OTEL_GENAI_CAPTURE_CONTENT", "false").lower() == "true"
MAX_CONTENT_LENGTH = int(os.getenv("OTEL_GENAI_MAX_CONTENT_LENGTH", "1000"))

# Get tracer for LLM operations
_tracer = trace.get_tracer("oci-llm-observability", "1.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# GenAI Semantic Convention Attributes
# Based on https://opentelemetry.io/docs/specs/semconv/gen-ai/
# ─────────────────────────────────────────────────────────────────────────────

class GenAIAttributes:
    """OpenTelemetry GenAI semantic convention attribute names.

    Based on https://opentelemetry.io/docs/specs/semconv/gen-ai/
    Compatible with OCI APM, viewapp, and DataDog LLM Observability.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Core GenAI Attributes (OpenTelemetry Semantic Conventions)
    # ─────────────────────────────────────────────────────────────────────────

    # System identification
    SYSTEM = "gen_ai.system"  # e.g., "openai", "anthropic", "oci", "oracle_code_assist"

    # Request attributes
    REQUEST_MODEL = "gen_ai.request.model"  # e.g., "oca/gpt5", "claude-3"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"

    # Response attributes
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Token usage attributes
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Operation attributes
    OPERATION_NAME = "gen_ai.operation.name"  # e.g., "chat", "tool_call", "agent_run"

    # ─────────────────────────────────────────────────────────────────────────
    # Content Attributes (optional, for debugging)
    # Enable with OTEL_GENAI_CAPTURE_CONTENT=true
    # ─────────────────────────────────────────────────────────────────────────

    PROMPT = "gen_ai.prompt"  # User/system prompt content
    COMPLETION = "gen_ai.completion"  # Model response content

    # ─────────────────────────────────────────────────────────────────────────
    # Span Event Names (for chat messages)
    # ─────────────────────────────────────────────────────────────────────────

    EVENT_PROMPT = "gen_ai.content.prompt"  # Span event for prompt
    EVENT_COMPLETION = "gen_ai.content.completion"  # Span event for completion

    # ─────────────────────────────────────────────────────────────────────────
    # Agent-specific Attributes (OCI Coordinator Extensions)
    # ─────────────────────────────────────────────────────────────────────────

    AGENT_NAME = "gen_ai.agent.name"
    AGENT_WORKFLOW = "gen_ai.agent.workflow"
    AGENT_TOOL_NAME = "gen_ai.agent.tool_name"
    AGENT_TOOL_DURATION_MS = "gen_ai.agent.tool_duration_ms"
    AGENT_ITERATION = "gen_ai.agent.iteration"
    AGENT_SUCCESS = "gen_ai.agent.success"

    # ─────────────────────────────────────────────────────────────────────────
    # Cost Estimation Attributes (Custom)
    # ─────────────────────────────────────────────────────────────────────────

    ESTIMATED_COST_USD = "gen_ai.cost.estimated_usd"

    # ─────────────────────────────────────────────────────────────────────────
    # Oracle Code Assist Specific Attributes
    # ─────────────────────────────────────────────────────────────────────────

    OCA_ENDPOINT = "gen_ai.oca.endpoint"  # OCA service endpoint
    OCA_REGION = "gen_ai.oca.region"  # OCI region
    OCA_TENANCY = "gen_ai.oca.tenancy"  # Tenancy OCID
    OCA_MODEL_ID = "gen_ai.oca.model_id"  # OCA model identifier
    OCA_REQUEST_TYPE = "gen_ai.oca.request_type"  # chat, completion, embedding

    # ─────────────────────────────────────────────────────────────────────────
    # Error Attributes
    # ─────────────────────────────────────────────────────────────────────────

    ERROR_TYPE = "gen_ai.error.type"  # rate_limit, timeout, api_error, etc.
    ERROR_MESSAGE = "gen_ai.error.message"


# ─────────────────────────────────────────────────────────────────────────────
# LLM Span Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _truncate_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Truncate content to max length with ellipsis."""
    if len(content) <= max_length:
        return content
    return content[:max_length - 3] + "..."


@dataclass
class LLMSpanContext:
    """Context for tracking LLM span metrics.

    Provides methods to record:
    - Token usage (input/output)
    - Prompt and completion content (optional)
    - Tool calls
    - Errors and finish reasons
    - Cost estimates

    Automatically publishes metrics to OCI Monitoring when span ends.
    """

    span: Any
    start_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    operation: str = ""
    system: str = "oci"
    agent: str = "coordinator"
    success: bool = True
    tool_calls: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    def set_tokens(self, input: int = 0, output: int = 0) -> None:
        """Set token counts for the span."""
        self.input_tokens = input
        self.output_tokens = output
        self.span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, input)
        self.span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, output)
        self.span.set_attribute(GenAIAttributes.USAGE_TOTAL_TOKENS, input + output)

    def set_prompt(self, prompt: str, role: str = "user") -> None:
        """Record prompt content (if content capture is enabled).

        Args:
            prompt: The prompt text
            role: Message role (user, system, assistant)
        """
        self.messages.append({"role": role, "content": prompt})

        if CAPTURE_CONTENT:
            truncated = _truncate_content(prompt)
            self.span.set_attribute(GenAIAttributes.PROMPT, truncated)

        # Always add as span event (content may be truncated)
        self.span.add_event(
            GenAIAttributes.EVENT_PROMPT,
            attributes={
                "gen_ai.prompt.role": role,
                "gen_ai.prompt.content": _truncate_content(prompt) if CAPTURE_CONTENT else "[content capture disabled]",
            },
        )

    def set_completion(self, completion: str, role: str = "assistant") -> None:
        """Record completion/response content (if content capture is enabled).

        Args:
            completion: The model response text
            role: Message role (usually assistant)
        """
        self.messages.append({"role": role, "content": completion})

        if CAPTURE_CONTENT:
            truncated = _truncate_content(completion)
            self.span.set_attribute(GenAIAttributes.COMPLETION, truncated)

        # Always add as span event
        self.span.add_event(
            GenAIAttributes.EVENT_COMPLETION,
            attributes={
                "gen_ai.completion.role": role,
                "gen_ai.completion.content": _truncate_content(completion) if CAPTURE_CONTENT else "[content capture disabled]",
            },
        )

    def set_response_id(self, response_id: str) -> None:
        """Set the response ID from the LLM."""
        self.span.set_attribute(GenAIAttributes.RESPONSE_ID, response_id)

    def set_response_model(self, model: str) -> None:
        """Set the actual model used in the response (may differ from request)."""
        self.span.set_attribute(GenAIAttributes.RESPONSE_MODEL, model)

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool = True,
        arguments: dict | None = None,
        result: str | None = None,
    ) -> None:
        """Record a tool call within the LLM span.

        Args:
            tool_name: Name of the tool called
            duration_ms: Duration of tool execution in milliseconds
            success: Whether the tool call succeeded
            arguments: Tool arguments (optional)
            result: Tool result summary (optional)
        """
        self.tool_calls.append({
            "name": tool_name,
            "duration_ms": duration_ms,
            "success": success,
        })

        # Add as span event with additional context
        event_attrs = {
            GenAIAttributes.AGENT_TOOL_NAME: tool_name,
            GenAIAttributes.AGENT_TOOL_DURATION_MS: duration_ms,
            GenAIAttributes.AGENT_SUCCESS: success,
        }
        if arguments and CAPTURE_CONTENT:
            event_attrs["gen_ai.tool.arguments"] = str(arguments)[:500]
        if result and CAPTURE_CONTENT:
            event_attrs["gen_ai.tool.result"] = _truncate_content(result, 500)

        self.span.add_event("tool_call", attributes=event_attrs)

    def set_finish_reason(self, reason: str) -> None:
        """Set the finish reason (e.g., 'stop', 'length', 'tool_calls')."""
        self.span.set_attribute(GenAIAttributes.RESPONSE_FINISH_REASONS, [reason])

    def set_error(self, error: Exception, error_type: str | None = None) -> None:
        """Record an error on the span.

        Args:
            error: The exception that occurred
            error_type: Type of error (rate_limit, timeout, api_error, etc.)
        """
        self.success = False
        self.span.set_status(Status(StatusCode.ERROR, str(error)))
        self.span.record_exception(error)

        if error_type:
            self.span.set_attribute(GenAIAttributes.ERROR_TYPE, error_type)
        self.span.set_attribute(GenAIAttributes.ERROR_MESSAGE, str(error))

    def set_cost_estimate(self, cost_usd: float) -> None:
        """Set estimated cost for this LLM call."""
        self.span.set_attribute(GenAIAttributes.ESTIMATED_COST_USD, cost_usd)

    def set_request_params(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        """Set LLM request parameters.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        if max_tokens is not None:
            self.span.set_attribute(GenAIAttributes.REQUEST_MAX_TOKENS, max_tokens)
        if temperature is not None:
            self.span.set_attribute(GenAIAttributes.REQUEST_TEMPERATURE, temperature)
        if top_p is not None:
            self.span.set_attribute(GenAIAttributes.REQUEST_TOP_P, top_p)
        if top_k is not None:
            self.span.set_attribute(GenAIAttributes.REQUEST_TOP_K, top_k)

    def set_agent(self, agent_name: str) -> None:
        """Set the agent name for this span."""
        self.agent = agent_name
        self.span.set_attribute(GenAIAttributes.AGENT_NAME, agent_name)

    def end(self) -> None:
        """End the span and calculate final metrics.

        Publishes metrics to:
        1. In-memory aggregator (_llm_metrics) for API endpoints
        2. OCI Monitoring for dashboard visualization (if enabled)
        """
        duration_ms = (time.time() - self.start_time) * 1000
        self.span.set_attribute("duration_ms", duration_ms)

        # Update in-memory metrics (fast, always available)
        _llm_metrics.record(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=duration_ms,
            tool_calls=len(self.tool_calls),
            error=not self.success,
        )

        # Publish to OCI Monitoring (for dashboards)
        try:
            from src.observability.metrics import record_llm_usage

            record_llm_usage(
                model=self.model,
                operation=self.operation,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                latency_ms=duration_ms,
                success=self.success,
                agent=self.agent,
                system=self.system,
            )
        except ImportError:
            pass  # Metrics module not available
        except Exception as e:
            logger.debug("Failed to publish LLM metrics to OCI", error=str(e))

        # Log summary
        logger.debug(
            "LLM span completed",
            operation=self.operation,
            model=self.model,
            system=self.system,
            agent=self.agent,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            duration_ms=round(duration_ms, 2),
            tool_calls=len(self.tool_calls),
            success=self.success,
        )


class LLMInstrumentor:
    """Instrumentor for LLM operations with GenAI semantic conventions."""

    # Token cost estimates per 1K tokens (approximate)
    TOKEN_COSTS = {
        "oca/gpt5": {"input": 0.001, "output": 0.002},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }

    @classmethod
    @contextmanager
    def span(
        cls,
        operation: str,
        model: str = "unknown",
        system: str = "oci",
        agent: str = "coordinator",
        **attributes: Any,
    ):
        """Create an LLM span with GenAI semantic conventions.

        Args:
            operation: Operation name (chat, tool_call, agent_run, etc.)
            model: Model identifier
            system: LLM provider system
            agent: Agent name for metric grouping
            **attributes: Additional span attributes

        Yields:
            LLMSpanContext for recording metrics
        """
        span_name = f"gen_ai.{operation}"

        with _tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            # Set GenAI semantic attributes
            span.set_attribute(GenAIAttributes.SYSTEM, system)
            span.set_attribute(GenAIAttributes.OPERATION_NAME, operation)
            span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
            span.set_attribute(GenAIAttributes.AGENT_NAME, agent)

            # Set any additional attributes
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)

            ctx = LLMSpanContext(
                span=span,
                start_time=time.time(),
                model=model,
                operation=operation,
                system=system,
                agent=agent,
            )

            try:
                yield ctx
            finally:
                ctx.end()

    @classmethod
    def estimate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD for token usage.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        costs = cls.TOKEN_COSTS.get(model, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return round(input_cost + output_cost, 6)


def llm_span(
    operation: str = "chat",
    model: str | None = None,
    system: str = "oci",
) -> Callable:
    """Decorator for instrumenting LLM function calls.

    Args:
        operation: Operation name (chat, tool_call, agent_run, etc.)
        model: Model identifier (can be extracted from function args)
        system: LLM provider system

    Example:
        @llm_span(operation="agent_run", model="oca/gpt5")
        async def run_agent(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to extract model from kwargs
            actual_model = model or kwargs.get("model") or "unknown"

            with LLMInstrumentor.span(
                operation=operation,
                model=actual_model,
                system=system,
            ) as ctx:
                try:
                    result = await func(*args, **kwargs)

                    # Try to extract token counts from result if available
                    if hasattr(result, "usage"):
                        usage = result.usage
                        ctx.set_tokens(
                            input=getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0),
                            output=getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0),
                        )
                        # Estimate cost
                        cost = LLMInstrumentor.estimate_cost(
                            actual_model,
                            ctx.input_tokens,
                            ctx.output_tokens,
                        )
                        ctx.set_cost_estimate(cost)

                    return result

                except Exception as e:
                    ctx.set_error(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            actual_model = model or kwargs.get("model") or "unknown"

            with LLMInstrumentor.span(
                operation=operation,
                model=actual_model,
                system=system,
            ) as ctx:
                try:
                    result = func(*args, **kwargs)

                    if hasattr(result, "usage"):
                        usage = result.usage
                        ctx.set_tokens(
                            input=getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0),
                            output=getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0),
                        )

                    return result

                except Exception as e:
                    ctx.set_error(e)
                    raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Agent Instrumentation
# ─────────────────────────────────────────────────────────────────────────────


class AgentInstrumentor:
    """Instrumentor for agent workflow operations."""

    @classmethod
    @contextmanager
    def workflow_span(
        cls,
        workflow_name: str,
        agent_name: str = "coordinator",
        **attributes: Any,
    ):
        """Create a span for workflow execution.

        Args:
            workflow_name: Name of the workflow
            agent_name: Name of the agent executing the workflow
            **attributes: Additional attributes

        Yields:
            LLMSpanContext for recording metrics
        """
        with LLMInstrumentor.span(
            operation="workflow",
            model="workflow",
            system="oci",
            **{
                GenAIAttributes.AGENT_NAME: agent_name,
                GenAIAttributes.AGENT_WORKFLOW: workflow_name,
                **attributes,
            },
        ) as ctx:
            yield ctx

    @classmethod
    @contextmanager
    def tool_span(
        cls,
        tool_name: str,
        agent_name: str = "coordinator",
        **attributes: Any,
    ):
        """Create a span for tool execution.

        Args:
            tool_name: Name of the tool being called
            agent_name: Name of the calling agent
            **attributes: Additional attributes

        Yields:
            LLMSpanContext
        """
        with LLMInstrumentor.span(
            operation="tool_call",
            model="mcp",
            system="oci",
            **{
                GenAIAttributes.AGENT_NAME: agent_name,
                GenAIAttributes.AGENT_TOOL_NAME: tool_name,
                **attributes,
            },
        ) as ctx:
            yield ctx

    @classmethod
    @contextmanager
    def classification_span(
        cls,
        query: str,
        model: str = "oca/gpt5",
        **attributes: Any,
    ):
        """Create a span for intent classification.

        Args:
            query: User query being classified
            model: Model used for classification
            **attributes: Additional attributes

        Yields:
            LLMSpanContext
        """
        with LLMInstrumentor.span(
            operation="classification",
            model=model,
            system="oci",
            **{
                "query_length": len(query),
                **attributes,
            },
        ) as ctx:
            yield ctx


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Collection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMMetrics:
    """Aggregated LLM metrics for monitoring."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_errors: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        error: bool = False,
        tool_calls: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record metrics from an LLM call."""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms
        self.total_tool_calls += tool_calls
        self.estimated_cost_usd += cost_usd
        if error:
            self.total_errors += 1

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for reporting."""
        avg_latency = self.total_latency_ms / max(self.total_requests, 1)
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_requests, 1),
            "total_tool_calls": self.total_tool_calls,
            "avg_latency_ms": round(avg_latency, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }


# Global metrics instance
_llm_metrics = LLMMetrics()


def get_llm_metrics() -> LLMMetrics:
    """Get the global LLM metrics instance."""
    return _llm_metrics


def reset_llm_metrics() -> None:
    """Reset the global LLM metrics (useful for testing)."""
    global _llm_metrics
    _llm_metrics = LLMMetrics()


# ─────────────────────────────────────────────────────────────────────────────
# Oracle Code Assist Instrumentor
# ─────────────────────────────────────────────────────────────────────────────


class OracleCodeAssistInstrumentor:
    """Instrumentor for Oracle Code Assist (OCA) LLM operations.

    Provides specialized tracing for OCA with OCI-specific attributes.
    Compatible with OCI APM, viewapp LLM Observability, and DataDog.

    Usage:
        with OracleCodeAssistInstrumentor.chat_span(model="oca/gpt5") as span:
            span.set_prompt(prompt)
            span.set_request_params(temperature=0.7, max_tokens=1000)
            response = await oca_client.chat(prompt)
            span.set_completion(response.content)
            span.set_tokens(
                input=response.usage.input_tokens,
                output=response.usage.output_tokens
            )
    """

    SYSTEM = "oracle_code_assist"

    @classmethod
    @contextmanager
    def chat_span(
        cls,
        model: str = "oca/gpt5",
        endpoint: str | None = None,
        region: str | None = None,
        tenancy: str | None = None,
        **attributes: Any,
    ):
        """Create a span for OCA chat completion.

        Args:
            model: OCA model identifier
            endpoint: OCA service endpoint (optional)
            region: OCI region (optional)
            tenancy: Tenancy OCID (optional)
            **attributes: Additional span attributes

        Yields:
            LLMSpanContext for recording metrics
        """
        span_attrs = {
            GenAIAttributes.OCA_REQUEST_TYPE: "chat",
        }
        if endpoint:
            span_attrs[GenAIAttributes.OCA_ENDPOINT] = endpoint
        if region:
            span_attrs[GenAIAttributes.OCA_REGION] = region
        if tenancy:
            span_attrs[GenAIAttributes.OCA_TENANCY] = tenancy
        span_attrs.update(attributes)

        with LLMInstrumentor.span(
            operation="chat",
            model=model,
            system=cls.SYSTEM,
            **span_attrs,
        ) as ctx:
            yield ctx

    @classmethod
    @contextmanager
    def completion_span(
        cls,
        model: str = "oca/gpt5",
        endpoint: str | None = None,
        **attributes: Any,
    ):
        """Create a span for OCA text completion.

        Args:
            model: OCA model identifier
            endpoint: OCA service endpoint
            **attributes: Additional span attributes

        Yields:
            LLMSpanContext
        """
        span_attrs = {
            GenAIAttributes.OCA_REQUEST_TYPE: "completion",
        }
        if endpoint:
            span_attrs[GenAIAttributes.OCA_ENDPOINT] = endpoint
        span_attrs.update(attributes)

        with LLMInstrumentor.span(
            operation="text_completion",
            model=model,
            system=cls.SYSTEM,
            **span_attrs,
        ) as ctx:
            yield ctx

    @classmethod
    @contextmanager
    def embedding_span(
        cls,
        model: str = "oca/embed",
        input_count: int = 1,
        **attributes: Any,
    ):
        """Create a span for OCA embedding generation.

        Args:
            model: OCA embedding model identifier
            input_count: Number of inputs being embedded
            **attributes: Additional span attributes

        Yields:
            LLMSpanContext
        """
        with LLMInstrumentor.span(
            operation="embeddings",
            model=model,
            system=cls.SYSTEM,
            **{
                GenAIAttributes.OCA_REQUEST_TYPE: "embedding",
                "gen_ai.embedding.input_count": input_count,
                **attributes,
            },
        ) as ctx:
            yield ctx

    @classmethod
    def instrument_oca_client(cls, oca_client: Any) -> Any:
        """Instrument an OCA client for automatic tracing.

        This wraps the OCA client's methods to automatically create spans.
        Note: This is a placeholder for future implementation.

        Args:
            oca_client: The OCA client instance to instrument

        Returns:
            Instrumented client
        """
        # Future: Wrap client methods with automatic tracing
        logger.info("OCA client instrumentation requested (placeholder)")
        return oca_client


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def create_llm_span(
    operation: str,
    model: str,
    system: str = "oci",
    **attributes: Any,
) -> LLMSpanContext:
    """Create an LLM span context (convenience function).

    Args:
        operation: Operation name (chat, completion, embedding, etc.)
        model: Model identifier
        system: LLM provider system
        **attributes: Additional span attributes

    Returns:
        LLMSpanContext (must be used as context manager)
    """
    return LLMInstrumentor.span(operation, model, system, **attributes)
