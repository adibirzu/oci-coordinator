#!/usr/bin/env python
"""
Comprehensive test for OCA LLM integration and APM tracing.

Tests:
1. OCA authentication status
2. OCA LLM call with proper tracing
3. Nested span hierarchy (Agent -> LLM -> Tool)
4. Trace context propagation
5. Force flush to ensure traces are exported

Usage:
    poetry run python tests/test_oca_tracing.py
"""

import asyncio
import time
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.local")


async def test_oca_authentication():
    """Test OCA authentication status."""
    from src.llm.oca import oca_token_manager, is_oca_authenticated, OCA_CONFIG

    print("\n" + "=" * 60)
    print("TEST 1: OCA Authentication")
    print("=" * 60)

    info = oca_token_manager.get_token_info()

    print(f"\n  OCA Endpoint: {OCA_CONFIG.OCA_ENDPOINT}")
    print(f"  Token Endpoint: {OCA_CONFIG.token_endpoint}")
    print(f"  Default Model: {OCA_CONFIG.DEFAULT_MODEL}")
    print(f"\n  Token Status:")
    print(f"    - Has Token: {info['has_token']}")
    print(f"    - Is Valid: {info['is_valid']}")
    print(f"    - Can Refresh: {info['can_refresh']}")
    print(f"    - Expires In: {info['expires_in_seconds']:.0f}s ({info['expires_in_seconds']/60:.1f} min)")
    print(f"    - Refresh Expires In: {info['refresh_expires_in_seconds']:.0f}s ({info['refresh_expires_in_seconds']/3600:.1f} hrs)")

    authenticated = is_oca_authenticated()
    print(f"\n  is_oca_authenticated(): {authenticated}")

    if not authenticated:
        print("\n  ERROR: OCA not authenticated!")
        print("  Please run the OCA login flow first.")
        return False

    print("\n  SUCCESS: OCA authentication valid")
    return True


async def test_oca_llm_call():
    """Test OCA LLM call with tracing."""
    from opentelemetry import trace
    from src.observability import init_observability, get_tracer, force_flush_traces
    from src.llm.oca import get_oca_llm
    from langchain_core.messages import HumanMessage

    print("\n" + "=" * 60)
    print("TEST 2: OCA LLM Call with Tracing")
    print("=" * 60)

    # Initialize observability
    init_observability(agent_name="test-oca")
    tracer = get_tracer("coordinator")

    # Get current trace context
    current_span = trace.get_current_span()
    print(f"\n  Current Span: {current_span}")

    # Create root span for test
    with tracer.start_as_current_span("test.oca_llm_call") as root_span:
        ctx = root_span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")

        print(f"\n  Root Span Created:")
        print(f"    - Trace ID: {trace_id}")
        print(f"    - Span ID: {span_id}")

        root_span.set_attribute("test.name", "oca_llm_call")
        root_span.set_attribute("test.timestamp", time.time())

        try:
            # Create LLM instance
            llm = get_oca_llm(model="oca/gpt5", temperature=0.1, max_tokens=100)
            print(f"\n  LLM Instance: {llm._llm_type}")
            print(f"    - Model: {llm.model}")
            print(f"    - Temperature: {llm.temperature}")
            print(f"    - Max Tokens: {llm.max_tokens}")

            # Make test call
            print("\n  Sending test message to OCA...")
            start_time = time.time()

            messages = [
                HumanMessage(content="Say 'OCA test successful' in exactly 5 words.")
            ]

            response = await llm.ainvoke(messages)

            duration = time.time() - start_time
            content = response.content

            print(f"\n  Response received in {duration:.2f}s:")
            print(f"    - Content: {content[:100]}...")
            print(f"    - Length: {len(content)} chars")

            root_span.set_attribute("test.success", True)
            root_span.set_attribute("test.duration_ms", duration * 1000)
            root_span.set_attribute("test.response_length", len(content))

            print("\n  SUCCESS: OCA LLM call completed")

        except Exception as e:
            root_span.set_attribute("test.success", False)
            root_span.set_attribute("test.error", str(e))
            print(f"\n  ERROR: {e}")
            return False, trace_id

    # Force flush traces
    print("\n  Flushing traces to APM...")
    flushed = force_flush_traces(timeout_ms=10000)
    print(f"    - Flush result: {'SUCCESS' if flushed else 'FAILED'}")

    return True, trace_id


async def test_nested_agent_workflow():
    """Test nested span hierarchy in a simulated agent workflow."""
    from opentelemetry import trace
    from src.observability import init_observability, get_tracer, force_flush_traces
    from src.llm.oca import get_oca_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    print("\n" + "=" * 60)
    print("TEST 3: Nested Agent Workflow with Tracing")
    print("=" * 60)

    init_observability(agent_name="test-agent")
    tracer = get_tracer("coordinator")

    # Create root span simulating a Slack message
    with tracer.start_as_current_span("slack.message_received") as slack_span:
        ctx = slack_span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")

        print(f"\n  Trace ID: {trace_id}")
        slack_span.set_attribute("slack.user", "test_user")
        slack_span.set_attribute("slack.channel", "test_channel")

        # Simulate coordinator routing
        with tracer.start_as_current_span("coordinator.route_request") as route_span:
            route_span.set_attribute("coordinator.intent", "test_query")
            route_span.set_attribute("coordinator.selected_agent", "infrastructure-agent")

            print("  1. Coordinator: Routing request...")
            await asyncio.sleep(0.1)

            # Simulate agent execution
            with tracer.start_as_current_span("agent.execute") as agent_span:
                agent_span.set_attribute("agent.name", "infrastructure-agent")
                agent_span.set_attribute("agent.iteration", 1)

                print("  2. Agent: Processing request...")

                # LLM call within agent context
                llm = get_oca_llm(model="oca/gpt5", temperature=0.1, max_tokens=150)

                messages = [
                    SystemMessage(content="You are an OCI infrastructure agent. Respond briefly."),
                    HumanMessage(content="List 3 OCI compute shapes for testing."),
                ]

                print("  3. Agent -> LLM: Making inference call...")
                start = time.time()

                try:
                    response = await llm.ainvoke(messages)
                    duration = time.time() - start

                    agent_span.set_attribute("agent.llm_duration_ms", duration * 1000)
                    agent_span.set_attribute("agent.response_length", len(response.content))

                    print(f"     LLM responded in {duration:.2f}s")
                    print(f"     Response: {response.content[:100]}...")

                    # Simulate tool call
                    with tracer.start_as_current_span("tool.oci_list_compartments") as tool_span:
                        tool_span.set_attribute("tool.name", "oci_list_compartments")
                        tool_span.set_attribute("tool.input", "{}")

                        print("  4. Agent -> Tool: Calling MCP tool...")
                        await asyncio.sleep(0.05)

                        tool_span.set_attribute("tool.success", True)
                        tool_span.set_attribute("tool.output_length", 500)
                        print("     Tool call simulated")

                    agent_span.set_attribute("agent.success", True)

                except Exception as e:
                    agent_span.set_attribute("agent.success", False)
                    agent_span.set_attribute("agent.error", str(e))
                    print(f"     ERROR: {e}")
                    return False, trace_id

            route_span.set_attribute("coordinator.success", True)

        slack_span.set_attribute("slack.response_sent", True)

    # Flush traces
    print("\n  Flushing traces...")
    flushed = force_flush_traces(timeout_ms=10000)
    print(f"    - Flush result: {'SUCCESS' if flushed else 'FAILED'}")

    print("\n  SUCCESS: Nested workflow completed")
    print(f"\n  To view trace in OCI APM:")
    print(f"    Search for trace_id: {trace_id}")

    return True, trace_id


async def test_tracing_export_verification():
    """Verify traces are being exported to APM."""
    import os
    from src.observability import init_observability, get_tracer, force_flush_traces
    from src.observability.tracing import _tracer_provider, is_otel_enabled

    print("\n" + "=" * 60)
    print("TEST 4: Tracing Export Verification")
    print("=" * 60)

    init_observability(agent_name="test-export")

    print(f"\n  OTEL Enabled: {is_otel_enabled()}")
    print(f"  APM Endpoint: {os.getenv('OCI_APM_ENDPOINT', 'NOT SET')[:50]}...")
    print(f"  Private Key: {'SET' if os.getenv('OCI_APM_PRIVATE_DATA_KEY') else 'NOT SET'}")

    if _tracer_provider:
        print(f"  Tracer Provider: {type(_tracer_provider).__name__}")

        # Check span processors
        if hasattr(_tracer_provider, '_active_span_processor'):
            proc = _tracer_provider._active_span_processor
            print(f"  Span Processor: {type(proc).__name__}")
    else:
        print("  Tracer Provider: None")

    # Create a test span and flush
    tracer = get_tracer("coordinator")

    with tracer.start_as_current_span("test.export_verification") as span:
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")

        span.set_attribute("test.purpose", "export_verification")
        span.set_attribute("test.timestamp", time.time())

        print(f"\n  Created test span with trace_id: {trace_id}")

    # Force flush
    print("\n  Force flushing traces...")
    flushed = force_flush_traces(timeout_ms=10000)
    print(f"    - Result: {'SUCCESS' if flushed else 'FAILED'}")

    return flushed, trace_id


async def run_all_tests():
    """Run all OCA and tracing tests."""
    print("\n" + "=" * 60)
    print("OCA + APM TRACING COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    trace_ids = []

    # Test 1: Authentication
    auth_ok = await test_oca_authentication()
    results.append(("OCA Authentication", auth_ok))

    if not auth_ok:
        print("\n\nABORTING: OCA authentication required for remaining tests")
        return

    # Test 2: LLM Call
    llm_ok, trace_id = await test_oca_llm_call()
    results.append(("OCA LLM Call", llm_ok))
    trace_ids.append(("LLM Call", trace_id))

    # Test 3: Nested Workflow
    nested_ok, trace_id = await test_nested_agent_workflow()
    results.append(("Nested Workflow", nested_ok))
    trace_ids.append(("Nested Workflow", trace_id))

    # Test 4: Export Verification
    export_ok, trace_id = await test_tracing_export_verification()
    results.append(("Export Verification", export_ok))
    trace_ids.append(("Export Verification", trace_id))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n  Trace IDs for APM lookup:")
    for name, tid in trace_ids:
        print(f"    {name}: {tid}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - check output above")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    asyncio.run(run_all_tests())
