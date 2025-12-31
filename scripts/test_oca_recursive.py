#!/usr/bin/env python
"""
Recursive OCA LLM Integration and APM Tracing Test.

This script performs comprehensive testing of:
1. OCA Authentication (token validity, refresh capability)
2. OCA Endpoint connectivity and health
3. OCA LLM inference with nested spans
4. APM trace export verification
5. Multi-level span hierarchy simulation

Usage:
    poetry run python scripts/test_oca_recursive.py
    poetry run python scripts/test_oca_recursive.py --depth 5  # Deeper nesting
    poetry run python scripts/test_oca_recursive.py --skip-llm  # Skip actual LLM calls
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    duration_ms: float
    trace_id: str | None = None
    details: dict = field(default_factory=dict)
    error: str | None = None


class OCARecursiveTest:
    """Comprehensive OCA integration test suite."""

    def __init__(self, depth: int = 3, skip_llm: bool = False):
        self.depth = depth
        self.skip_llm = skip_llm
        self.results: list[TestResult] = []
        self.trace_ids: list[tuple[str, str]] = []

    def _print_header(self, title: str, char: str = "=") -> None:
        """Print a formatted header."""
        line = char * 70
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}")

    def _print_section(self, title: str) -> None:
        """Print a section divider."""
        print(f"\n  {'-' * 40}")
        print(f"  {title}")
        print(f"  {'-' * 40}")

    async def test_environment_check(self) -> TestResult:
        """Check all required environment variables."""
        self._print_header("TEST: Environment Configuration")
        start = time.time()

        required_vars = {
            "OCI_APM_ENDPOINT": "APM endpoint for tracing",
            "OCI_APM_PRIVATE_DATA_KEY": "APM data key for authentication",
        }

        optional_vars = {
            "OCA_ENDPOINT": "OCA API endpoint",
            "OCA_MODEL": "Default OCA model",
            "OCI_APM_DATA_UPLOAD_ENDPOINT": "Alternative APM upload endpoint",
            "OCI_APM_PUBLIC_DATA_KEY": "Public APM key (alternative)",
            "OTEL_TRACING_ENABLED": "Tracing enable flag",
        }

        details = {"required": {}, "optional": {}, "missing": []}

        print("\n  Required Variables:")
        for var, desc in required_vars.items():
            value = os.getenv(var)
            masked = f"{value[:20]}..." if value and len(value) > 20 else value
            status = "SET" if value else "MISSING"
            details["required"][var] = bool(value)
            if not value:
                details["missing"].append(var)
            print(f"    {var}: {status}")
            if value:
                print(f"      -> {masked}")

        print("\n  Optional Variables:")
        for var, desc in optional_vars.items():
            value = os.getenv(var)
            masked = f"{value[:30]}..." if value and len(value) > 30 else value
            status = "SET" if value else "not set"
            details["optional"][var] = bool(value)
            print(f"    {var}: {status}")
            if value:
                print(f"      -> {masked}")

        passed = len(details["missing"]) == 0
        duration = (time.time() - start) * 1000

        if not passed:
            print(f"\n  ERROR: Missing required variables: {details['missing']}")

        return TestResult(
            name="Environment Configuration",
            passed=passed,
            duration_ms=duration,
            details=details,
            error=f"Missing: {details['missing']}" if not passed else None,
        )

    async def test_oca_authentication(self) -> TestResult:
        """Test OCA authentication status and token validity."""
        self._print_header("TEST: OCA Authentication")
        start = time.time()

        from src.llm.oca import OCA_CONFIG, is_oca_authenticated, oca_token_manager

        details = {
            "endpoint": OCA_CONFIG.OCA_ENDPOINT,
            "token_endpoint": OCA_CONFIG.token_endpoint,
            "default_model": OCA_CONFIG.DEFAULT_MODEL,
            "litellm_url": OCA_CONFIG.litellm_url,
        }

        print(f"\n  OCA Configuration:")
        print(f"    Endpoint: {OCA_CONFIG.OCA_ENDPOINT}")
        print(f"    Token Endpoint: {OCA_CONFIG.token_endpoint}")
        print(f"    LiteLLM URL: {OCA_CONFIG.litellm_url}")
        print(f"    Default Model: {OCA_CONFIG.DEFAULT_MODEL}")

        self._print_section("Token Status")
        info = oca_token_manager.get_token_info()
        details["token_info"] = info

        print(f"    has_token: {info['has_token']}")
        print(f"    is_valid: {info['is_valid']}")
        print(f"    can_refresh: {info['can_refresh']}")
        print(
            f"    expires_in: {info['expires_in_seconds']:.0f}s ({info['expires_in_seconds']/60:.1f} min)"
        )
        print(
            f"    refresh_expires_in: {info['refresh_expires_in_seconds']:.0f}s ({info['refresh_expires_in_seconds']/3600:.1f} hrs)"
        )

        authenticated = is_oca_authenticated()
        details["authenticated"] = authenticated

        self._print_section("Authentication Result")
        print(f"    is_oca_authenticated(): {authenticated}")

        if not authenticated:
            print("\n  ACTION REQUIRED: OCA authentication is invalid!")
            print("  Please complete the OAuth login flow:")
            print("    1. Open the IDCS OAuth URL in a browser")
            print("    2. Complete the PKCE flow")
            print("    3. Token will be cached in ~/.oca/token.json")

        duration = (time.time() - start) * 1000

        return TestResult(
            name="OCA Authentication",
            passed=authenticated,
            duration_ms=duration,
            details=details,
            error="Authentication invalid - please login" if not authenticated else None,
        )

    async def test_oca_endpoint_health(self) -> TestResult:
        """Test OCA endpoint connectivity and response."""
        self._print_header("TEST: OCA Endpoint Health")
        start = time.time()

        from src.llm.oca import OCA_CONFIG, oca_token_manager

        details = {"endpoints_tested": [], "connectivity": {}}

        # Test base endpoint connectivity
        endpoints_to_test = [
            (OCA_CONFIG.OCA_ENDPOINT, "OCA Base"),
            (f"{OCA_CONFIG.litellm_url}/models", "LiteLLM Models"),
        ]

        token = oca_token_manager.get_access_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        all_reachable = True

        async with httpx.AsyncClient(timeout=30.0) as client:
            for url, name in endpoints_to_test:
                self._print_section(f"Testing: {name}")
                print(f"    URL: {url}")

                try:
                    response = await client.get(url, headers=headers)
                    status = response.status_code

                    details["connectivity"][name] = {
                        "url": url,
                        "status": status,
                        "reachable": True,
                    }

                    print(f"    Status: {status}")

                    # For models endpoint, try to parse and show available models
                    if "models" in url and status == 200:
                        try:
                            models_data = response.json()
                            if "data" in models_data:
                                print(f"    Available Models:")
                                for model in models_data.get("data", [])[:5]:
                                    print(f"      - {model.get('id', 'unknown')}")
                                details["connectivity"][name]["models"] = [
                                    m.get("id") for m in models_data.get("data", [])
                                ]
                        except Exception:
                            pass

                    # 404 on base URL is expected - only API paths work
                    if status >= 400 and "models" in url:
                        all_reachable = False
                        print(f"    Response: {response.text[:200]}")
                    elif status >= 400:
                        print(f"    Note: Base URL returns 404 (expected - no root handler)")

                except httpx.ConnectError as e:
                    all_reachable = False
                    details["connectivity"][name] = {
                        "url": url,
                        "status": None,
                        "reachable": False,
                        "error": str(e),
                    }
                    print(f"    ERROR: Connection failed - {e}")

                except Exception as e:
                    all_reachable = False
                    details["connectivity"][name] = {
                        "url": url,
                        "status": None,
                        "reachable": False,
                        "error": str(e),
                    }
                    print(f"    ERROR: {type(e).__name__} - {e}")

        duration = (time.time() - start) * 1000

        return TestResult(
            name="OCA Endpoint Health",
            passed=all_reachable,
            duration_ms=duration,
            details=details,
            error="Some endpoints unreachable" if not all_reachable else None,
        )

    async def test_tracing_configuration(self) -> TestResult:
        """Test OpenTelemetry tracing configuration."""
        self._print_header("TEST: Tracing Configuration")
        start = time.time()

        from src.observability import (
            SERVICE_NAMES,
            force_flush_traces,
            get_tracer,
            init_observability,
            is_otel_enabled,
        )
        from src.observability.tracing import _get_apm_config, _tracer_provider

        details = {"apm_config": {}, "service_names": SERVICE_NAMES, "exporter": None}

        # Check APM configuration
        apm_config = _get_apm_config()
        details["apm_config"] = {
            k: f"{v[:30]}..." if v and len(v) > 30 else v for k, v in apm_config.items()
        }

        self._print_section("APM Configuration")
        for key, value in details["apm_config"].items():
            print(f"    {key}: {value}")

        # Initialize tracing
        self._print_section("Initializing Tracing")
        init_observability(agent_name="test-recursive")

        enabled = is_otel_enabled()
        details["otel_enabled"] = enabled
        print(f"    OTEL Enabled: {enabled}")

        # Check tracer provider
        if _tracer_provider:
            print(f"    Tracer Provider: {type(_tracer_provider).__name__}")
            details["tracer_provider"] = type(_tracer_provider).__name__

            # Try to get exporter info
            if hasattr(_tracer_provider, "_active_span_processor"):
                proc = _tracer_provider._active_span_processor
                print(f"    Span Processor: {type(proc).__name__}")
                details["span_processor"] = type(proc).__name__

                # Try to find exporter
                if hasattr(proc, "_span_exporter"):
                    exp = proc._span_exporter
                    print(f"    Exporter: {type(exp).__name__}")
                    details["exporter"] = type(exp).__name__
                elif hasattr(proc, "_exporters"):
                    for exp in proc._exporters:
                        print(f"    Exporter: {type(exp).__name__}")
                        details["exporter"] = type(exp).__name__
        else:
            print("    Tracer Provider: None (tracing disabled)")
            details["tracer_provider"] = None

        # Create a test span
        self._print_section("Test Span Creation")
        tracer = get_tracer("coordinator")

        with tracer.start_as_current_span("test.tracing_config_verification") as span:
            ctx = span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")
            span_id = format(ctx.span_id, "016x")

            span.set_attribute("test.type", "configuration_check")
            span.set_attribute("test.timestamp", time.time())

            print(f"    Trace ID: {trace_id}")
            print(f"    Span ID: {span_id}")
            print(f"    Is Recording: {span.is_recording()}")

            details["test_span"] = {
                "trace_id": trace_id,
                "span_id": span_id,
                "is_recording": span.is_recording(),
            }

            self.trace_ids.append(("Tracing Config", trace_id))

        # Force flush
        self._print_section("Flushing Traces")
        flushed = force_flush_traces(timeout_ms=5000)
        print(f"    Flush Result: {'SUCCESS' if flushed else 'FAILED'}")
        details["flush_success"] = flushed

        duration = (time.time() - start) * 1000
        passed = enabled and flushed

        return TestResult(
            name="Tracing Configuration",
            passed=passed,
            duration_ms=duration,
            trace_id=trace_id,
            details=details,
            error="Tracing not properly configured" if not passed else None,
        )

    async def test_oca_llm_inference(self) -> TestResult:
        """Test OCA LLM inference with tracing."""
        self._print_header("TEST: OCA LLM Inference")
        start = time.time()

        if self.skip_llm:
            print("\n  SKIPPED: --skip-llm flag set")
            return TestResult(
                name="OCA LLM Inference",
                passed=True,
                duration_ms=0,
                details={"skipped": True},
            )

        from langchain_core.messages import HumanMessage

        from src.llm.oca import get_oca_llm, is_oca_authenticated
        from src.observability import force_flush_traces, get_tracer

        details = {"model": None, "response": None, "token_usage": None}

        if not is_oca_authenticated():
            print("\n  SKIPPED: OCA not authenticated")
            return TestResult(
                name="OCA LLM Inference",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                details=details,
                error="OCA not authenticated",
            )

        tracer = get_tracer("coordinator")

        # Create root span for inference test
        with tracer.start_as_current_span("test.oca_inference") as root_span:
            ctx = root_span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")

            root_span.set_attribute("test.type", "oca_inference")
            root_span.set_attribute("test.depth", self.depth)

            self._print_section("Creating LLM Instance")
            llm = get_oca_llm(model="oca/gpt5", temperature=0.1, max_tokens=100)
            details["model"] = llm.model

            print(f"    Model: {llm.model}")
            print(f"    Temperature: {llm.temperature}")
            print(f"    Max Tokens: {llm.max_tokens}")

            self._print_section("Sending Test Message")
            try:
                messages = [
                    HumanMessage(
                        content="Respond with exactly: 'OCA recursive test successful'"
                    )
                ]

                inference_start = time.time()
                response = await llm.ainvoke(messages)
                inference_duration = (time.time() - inference_start) * 1000

                content = response.content
                details["response"] = content[:200]
                details["response_length"] = len(content)
                details["inference_duration_ms"] = inference_duration

                # Get token usage from generation info
                if hasattr(response, "response_metadata"):
                    details["token_usage"] = response.response_metadata.get("token_usage")

                print(f"    Response: {content[:100]}...")
                print(f"    Length: {len(content)} chars")
                print(f"    Duration: {inference_duration:.2f}ms")

                root_span.set_attribute("test.response_length", len(content))
                root_span.set_attribute("test.inference_duration_ms", inference_duration)
                root_span.set_attribute("test.success", True)

                self.trace_ids.append(("OCA Inference", trace_id))

            except Exception as e:
                root_span.set_attribute("test.success", False)
                root_span.set_attribute("test.error", str(e))
                root_span.record_exception(e)

                print(f"\n  ERROR: {type(e).__name__}: {e}")
                return TestResult(
                    name="OCA LLM Inference",
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    trace_id=trace_id,
                    details=details,
                    error=str(e),
                )

        # Flush traces
        self._print_section("Flushing Traces")
        flushed = force_flush_traces(timeout_ms=10000)
        print(f"    Flush Result: {'SUCCESS' if flushed else 'FAILED'}")

        duration = (time.time() - start) * 1000

        return TestResult(
            name="OCA LLM Inference",
            passed=True,
            duration_ms=duration,
            trace_id=trace_id,
            details=details,
        )

    async def test_recursive_nested_spans(self) -> TestResult:
        """Test deeply nested span hierarchy (recursive pattern)."""
        self._print_header(f"TEST: Recursive Nested Spans (depth={self.depth})")
        start = time.time()

        from src.observability import force_flush_traces, get_tracer

        tracer = get_tracer("coordinator")
        details = {"depth": self.depth, "spans_created": 0, "levels": []}

        async def create_nested_span(level: int, parent_name: str) -> None:
            """Recursively create nested spans."""
            span_name = f"test.recursive.level_{level}"

            with tracer.start_as_current_span(span_name) as span:
                ctx = span.get_span_context()
                span_id = format(ctx.span_id, "016x")

                span.set_attribute("test.level", level)
                span.set_attribute("test.parent", parent_name)
                span.set_attribute("test.recursion_depth", self.depth)

                details["spans_created"] += 1
                details["levels"].append(
                    {"level": level, "span_id": span_id, "name": span_name}
                )

                print(f"    {'  ' * level}Level {level}: {span_name} ({span_id[:8]}...)")

                # Simulate some work
                await asyncio.sleep(0.01 * level)

                # Recurse if not at max depth
                if level < self.depth:
                    await create_nested_span(level + 1, span_name)

        # Create root span
        with tracer.start_as_current_span("test.recursive_nested_root") as root_span:
            ctx = root_span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")

            root_span.set_attribute("test.type", "recursive_nested")
            root_span.set_attribute("test.max_depth", self.depth)

            print(f"\n    Trace ID: {trace_id}")
            print(f"\n    Creating nested spans:")
            details["spans_created"] += 1

            # Start recursion
            await create_nested_span(1, "root")

            root_span.set_attribute("test.total_spans", details["spans_created"])

            self.trace_ids.append(("Recursive Nested", trace_id))

        # Flush
        self._print_section("Flushing Traces")
        flushed = force_flush_traces(timeout_ms=10000)
        print(f"    Flush Result: {'SUCCESS' if flushed else 'FAILED'}")
        print(f"    Total Spans Created: {details['spans_created']}")

        details["flush_success"] = flushed
        duration = (time.time() - start) * 1000

        return TestResult(
            name="Recursive Nested Spans",
            passed=flushed,
            duration_ms=duration,
            trace_id=trace_id,
            details=details,
        )

    async def test_agent_workflow_simulation(self) -> TestResult:
        """Simulate a full agent workflow with OCA and tools."""
        self._print_header("TEST: Agent Workflow Simulation")
        start = time.time()

        from langchain_core.messages import HumanMessage, SystemMessage

        from src.llm.oca import get_oca_llm, is_oca_authenticated
        from src.observability import force_flush_traces, get_tracer

        tracer = get_tracer("coordinator")
        details = {"stages": [], "llm_calls": 0, "tool_calls": 0}

        # Root span: Slack message received
        with tracer.start_as_current_span("slack.message_received") as slack_span:
            ctx = slack_span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")

            slack_span.set_attribute("slack.user_id", "U123456")
            slack_span.set_attribute("slack.channel_id", "C789012")
            slack_span.set_attribute("slack.message", "Check OCI database health")

            print(f"\n    Trace ID: {trace_id}")
            details["stages"].append("slack.message_received")

            # Stage 1: Coordinator routing
            with tracer.start_as_current_span("coordinator.route_request") as route_span:
                route_span.set_attribute("coordinator.action", "classify_intent")
                print("    1. Coordinator: Classifying intent...")

                # Simulate intent classification
                await asyncio.sleep(0.05)

                route_span.set_attribute("coordinator.intent", "database_health_check")
                route_span.set_attribute("coordinator.confidence", 0.95)
                route_span.set_attribute("coordinator.selected_agent", "db-troubleshoot-agent")
                details["stages"].append("coordinator.route_request")

                # Stage 2: Agent execution
                with tracer.start_as_current_span("agent.db_troubleshoot.execute") as agent_span:
                    agent_span.set_attribute("agent.name", "db-troubleshoot-agent")
                    agent_span.set_attribute("agent.skill", "health_check")

                    print("    2. Agent: Starting health check workflow...")
                    details["stages"].append("agent.execute")

                    # Stage 3: LLM reasoning (if authenticated)
                    if is_oca_authenticated() and not self.skip_llm:
                        with tracer.start_as_current_span("agent.llm_reasoning") as llm_span:
                            llm_span.set_attribute("llm.purpose", "plan_tool_calls")

                            print("    3. Agent -> LLM: Planning tool calls...")

                            try:
                                llm = get_oca_llm(model="oca/gpt5", max_tokens=150)
                                messages = [
                                    SystemMessage(
                                        content="You are a DB troubleshooting agent. Be brief."
                                    ),
                                    HumanMessage(
                                        content="What tools would you use to check database health? List 3."
                                    ),
                                ]

                                response = await llm.ainvoke(messages)
                                details["llm_calls"] += 1

                                llm_span.set_attribute("llm.success", True)
                                llm_span.set_attribute(
                                    "llm.response_preview", response.content[:100]
                                )
                                print(f"       LLM Response: {response.content[:80]}...")

                            except Exception as e:
                                llm_span.set_attribute("llm.success", False)
                                llm_span.set_attribute("llm.error", str(e))
                                llm_span.record_exception(e)
                                print(f"       LLM Error: {e}")
                    else:
                        print("    3. Agent -> LLM: Skipped (not authenticated or --skip-llm)")

                    # Stage 4: Tool calls (simulated)
                    tool_names = [
                        "get_fleet_summary",
                        "analyze_cpu_usage",
                        "get_performance_summary",
                    ]

                    for tool_name in tool_names:
                        with tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
                            tool_span.set_attribute("tool.name", tool_name)
                            tool_span.set_attribute("tool.tier", 1 if "fleet" in tool_name else 2)

                            print(f"    4. Agent -> Tool: {tool_name}")
                            await asyncio.sleep(0.02)

                            tool_span.set_attribute("tool.success", True)
                            tool_span.set_attribute("tool.duration_ms", 20)
                            details["tool_calls"] += 1

                    agent_span.set_attribute("agent.success", True)
                    agent_span.set_attribute("agent.tool_calls", len(tool_names))

                route_span.set_attribute("coordinator.success", True)

            slack_span.set_attribute("slack.response_sent", True)
            self.trace_ids.append(("Agent Workflow", trace_id))

        # Flush
        self._print_section("Flushing Traces")
        flushed = force_flush_traces(timeout_ms=10000)
        print(f"    Flush Result: {'SUCCESS' if flushed else 'FAILED'}")
        print(f"    Stages Completed: {len(details['stages'])}")
        print(f"    LLM Calls: {details['llm_calls']}")
        print(f"    Tool Calls: {details['tool_calls']}")

        details["flush_success"] = flushed
        duration = (time.time() - start) * 1000

        return TestResult(
            name="Agent Workflow Simulation",
            passed=flushed,
            duration_ms=duration,
            trace_id=trace_id,
            details=details,
        )

    async def run_all_tests(self) -> bool:
        """Run all tests and report results."""
        self._print_header("OCA RECURSIVE INTEGRATION TEST SUITE", "=")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Recursion Depth: {self.depth}")
        print(f"  Skip LLM Calls: {self.skip_llm}")

        # Run tests in order
        test_methods = [
            self.test_environment_check,
            self.test_oca_authentication,
            self.test_oca_endpoint_health,
            self.test_tracing_configuration,
            self.test_oca_llm_inference,
            self.test_recursive_nested_spans,
            self.test_agent_workflow_simulation,
        ]

        for test_method in test_methods:
            result = await test_method()
            self.results.append(result)

            # For auth failures, continue with tracing tests but skip LLM tests
            if result.name == "OCA Authentication" and not result.passed:
                print("\n  WARNING: OCA auth failed - skipping LLM tests, continuing with tracing tests")
                self.skip_llm = True  # Force skip LLM for remaining tests

        # Print summary
        self._print_header("TEST SUMMARY")

        all_passed = True
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            emoji = "✓" if result.passed else "✗"
            print(f"  {emoji} {result.name}: {status} ({result.duration_ms:.1f}ms)")
            if result.error:
                print(f"      Error: {result.error}")
            if not result.passed:
                all_passed = False

        # Print trace IDs for APM lookup
        if self.trace_ids:
            print("\n  Trace IDs for APM Lookup:")
            for name, tid in self.trace_ids:
                print(f"    {name}: {tid}")

        # Print APM link hint
        apm_endpoint = os.getenv("OCI_APM_ENDPOINT", "")
        if apm_endpoint:
            region = "us-phoenix-1"  # Extract from endpoint if needed
            print("\n  To view traces in OCI APM:")
            print(f"    1. Go to OCI Console -> Application Performance Monitoring")
            print(f"    2. Select your APM domain")
            print(f"    3. Go to Trace Explorer")
            print(f"    4. Search by trace ID from above")

        self._print_header("TEST COMPLETE")
        print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"  Total Tests: {len(self.results)}")
        print(f"  Passed: {sum(1 for r in self.results if r.passed)}")
        print(f"  Failed: {sum(1 for r in self.results if not r.passed)}")

        return all_passed


def main():
    """Run the test suite."""
    parser = argparse.ArgumentParser(
        description="Recursive OCA Integration Test Suite"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Recursion depth for nested span tests (default: 3)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip actual LLM inference calls",
    )

    args = parser.parse_args()

    test_suite = OCARecursiveTest(depth=args.depth, skip_llm=args.skip_llm)
    success = asyncio.run(test_suite.run_all_tests())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
