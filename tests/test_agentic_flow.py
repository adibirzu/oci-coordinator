#!/usr/bin/env python
"""
Test script for full agentic flow with tracing verification.

This script tests:
1. Agent invocation with MCP tools
2. OTel trace generation with proper spans
3. Log-trace correlation
4. End-to-end response quality

Run with: poetry run python tests/test_agentic_flow.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
from dotenv import load_dotenv
env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)


async def test_full_agentic_flow():
    """Test the complete agentic flow with tracing."""
    from opentelemetry import trace

    # Initialize observability first
    from src.observability import init_observability, get_tracer
    init_observability()

    # Initialize MCP
    from src.mcp.config import initialize_mcp_from_config, load_mcp_config
    from src.mcp.catalog import ToolCatalog

    print("\n" + "="*60)
    print("OCI AGENTIC FLOW TEST")
    print("="*60)

    # Load MCP config
    print("\n[1/6] Loading MCP configuration...")
    config = load_mcp_config()
    print(f"   ✓ Loaded {len(config.servers)} MCP servers")

    # Initialize MCP servers using the standard method
    print("\n[2/6] Connecting to MCP servers...")
    registry = await initialize_mcp_from_config(config)

    # Get tool catalog
    catalog = ToolCatalog.get_instance()
    await catalog.refresh()
    tools = catalog.list_tools()
    print(f"   ✓ Connected to MCP servers with {len(tools)} tools")

    # Test LLM
    print("\n[3/6] Testing LLM connection...")
    from src.llm import get_llm
    from src.llm.oca import is_oca_authenticated

    if not is_oca_authenticated():
        print("   ✗ OCA not authenticated - skipping LLM test")
        print("     Please authenticate via Slack first")
        llm = None
    else:
        llm = get_llm()
        print(f"   ✓ LLM ready: {type(llm).__name__}")

    # Test ReAct agent
    print("\n[4/6] Testing ReAct agent with MCP tools...")
    tracer = get_tracer("test-agentic-flow")

    with tracer.start_as_current_span("test.agentic_flow") as test_span:
        test_span.set_attribute("test.name", "full_agentic_flow")

        # Get current trace info
        ctx = trace.get_current_span().get_span_context()
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")

        print(f"   Trace ID: {trace_id}")
        print(f"   Span ID:  {span_id}")

        if llm:
            from src.agents.react_agent import SpecializedReActAgent
            from src.cache.oci_resource_cache import OCIResourceCache

            # Create agent
            agent = SpecializedReActAgent(
                domain="infrastructure",
                llm=llm,
                tool_catalog=catalog,
                resource_cache=None,  # Skip cache for now
                max_iterations=3,
            )

            # Test query - use one that triggers tool usage
            test_query = "Use the list_compartments tool to show me all compartments in my tenancy"
            print(f"\n   Query: {test_query}")

            try:
                result = await agent.run(test_query, user_id="test-user")

                print(f"\n   Result:")
                print(f"   - Success: {result.success}")
                print(f"   - Steps: {len(result.steps)}")
                print(f"   - Tool calls: {len(result.tool_calls)}")

                if result.tool_calls:
                    print(f"\n   Tool Calls Made:")
                    for tc in result.tool_calls:
                        print(f"   - {tc['tool']}: {tc.get('output', '')[:100]}...")

                print(f"\n   Response (first 500 chars):")
                print(f"   {result.response[:500]}...")

                test_span.set_attribute("test.success", result.success)
                test_span.set_attribute("test.steps", len(result.steps))
                test_span.set_attribute("test.tool_calls", len(result.tool_calls))

            except Exception as e:
                print(f"\n   ✗ Agent error: {e}")
                test_span.set_attribute("test.error", str(e))
        else:
            print("   Skipping agent test (LLM not available)")

    # Test MCP tool directly
    print("\n[5/6] Testing MCP tool execution with tracing...")
    with tracer.start_as_current_span("test.mcp_tool") as tool_span:
        # Find a list tool - tools are ToolDefinition objects
        list_tools = [t for t in tools if "list" in t.name.lower()]
        if list_tools:
            test_tool = list_tools[0].name
            print(f"   Testing tool: {test_tool}")

            try:
                result = await catalog.execute(test_tool, {})
                print(f"   ✓ Tool executed successfully")
                print(f"   - Success: {result.success}")
                print(f"   - Duration: {result.duration_ms}ms")
                if result.error:
                    print(f"   - Error: {result.error}")
                tool_span.set_attribute("tool.name", test_tool)
                tool_span.set_attribute("tool.success", result.success)
                tool_span.set_attribute("tool.duration_ms", result.duration_ms or 0)
            except Exception as e:
                print(f"   ✗ Tool error: {e}")
                tool_span.set_attribute("tool.error", str(e))
        else:
            print("   No list tools found for testing")

    # Summary
    print("\n[6/6] Test Summary")
    print("="*60)
    print(f"   Trace ID for APM lookup: {trace_id}")
    print(f"\n   Expected spans in trace:")
    print(f"   - test.agentic_flow (root)")
    print(f"   - react_agent.run (if LLM available)")
    print(f"   - llm.oca.chat_completion (LLM calls)")
    print(f"   - mcp.tool.<name> (MCP tool calls)")
    print(f"   - test.mcp_tool (direct tool test)")
    print(f"\n   Check OCI APM Trace Explorer with trace ID above")
    print(f"   Check OCI Logging for logs with trace_id={trace_id}")
    print("="*60)

    # Cleanup
    print("\n   Cleaning up...")
    # Registry returned from initialize_mcp_from_config
    if registry and hasattr(registry, 'disconnect_all'):
        await registry.disconnect_all()
    else:
        print("   (skipping disconnect - registry not available)")

    # Give traces time to export
    print("   Waiting for trace export...")
    await asyncio.sleep(3)

    print("\n   ✓ Test complete!")
    return trace_id


if __name__ == "__main__":
    trace_id = asyncio.run(test_full_agentic_flow())
    print(f"\n\nTrace ID: {trace_id}")
