#!/usr/bin/env python3
"""End-to-End Workflow Test for OCI Coordinator.

This script tests the complete message processing pipeline without
requiring live Slack events. It validates that when Slack events ARE
received, the coordinator will process them correctly.

Tests:
1. MCP server connectivity
2. Tool catalog availability
3. LangGraph coordinator initialization
4. Intent classification and routing
5. Workflow execution with real MCP tools

Usage:
    poetry run python scripts/test_e2e_workflow.py
    poetry run python scripts/test_e2e_workflow.py --query "show fleet health"
    poetry run python scripts/test_e2e_workflow.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env.local FIRST
from dotenv import load_dotenv

env_file = project_root / ".env.local"
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"✅ Loaded environment from {env_file}")


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print()


def print_step(step: str, status: str = "running") -> None:
    """Print a step indicator."""
    icons = {
        "running": "🔄",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️",
    }
    icon = icons.get(status, "•")
    print(f"   {icon} {step}")


def print_result(success: bool, message: str) -> None:
    """Print a result with color coding."""
    if success:
        print(f"   {Colors.GREEN}✅ {message}{Colors.RESET}")
    else:
        print(f"   {Colors.RED}❌ {message}{Colors.RESET}")


async def test_mcp_connectivity(verbose: bool = False) -> tuple[bool, dict]:
    """Test MCP server connectivity."""
    print_header("1. MCP Server Connectivity")

    results = {"servers": {}, "total_tools": 0}

    try:
        from src.mcp.config import initialize_mcp_from_config, load_mcp_config

        print_step("Loading MCP configuration...")
        config = load_mcp_config()
        enabled_servers = config.get_enabled_servers()
        print_result(True, f"Found {len(enabled_servers)} enabled servers")

        print_step("Initializing MCP connections...")
        registry, catalog = await initialize_mcp_from_config(config)

        # Check each server
        for server_id in registry.list_servers():
            status = registry.get_status(server_id)
            is_connected = status == "connected"
            results["servers"][server_id] = is_connected

            if verbose or not is_connected:
                print_step(f"{server_id}: {status}", "success" if is_connected else "error")

        # Count tools
        tools = catalog.list_tools()
        results["total_tools"] = len(tools)
        print_result(True, f"Tool catalog has {len(tools)} tools registered")

        if verbose:
            # Show tools by domain
            from collections import Counter
            domains = Counter()
            for tool in tools:
                # Handle both dict and ToolDefinition objects
                name = tool.get("name", "") if isinstance(tool, dict) else getattr(tool, "name", "")
                if "_" in name:
                    domain = name.split("_")[1] if name.startswith("oci_") else name.split("_")[0]
                    domains[domain] += 1

            print()
            print("   Tools by domain:")
            for domain, count in domains.most_common(10):
                print(f"      {domain}: {count}")

        connected = sum(1 for v in results["servers"].values() if v)
        results["success"] = connected > 0
        return results["success"], results

    except Exception as e:
        print_result(False, f"MCP initialization failed: {e}")
        results["error"] = str(e)
        results["success"] = False
        return False, results


async def test_coordinator_init(verbose: bool = False) -> tuple[bool, dict]:
    """Test LangGraph coordinator initialization."""
    print_header("2. LangGraph Coordinator Initialization")

    results = {}

    try:
        from src.agents.catalog import AgentCatalog
        from src.agents.coordinator.graph import LangGraphCoordinator
        from src.agents.coordinator.workflows import get_workflow_registry
        from src.llm import get_llm
        from src.mcp.catalog import ToolCatalog
        from src.mcp.config import load_mcp_config, initialize_mcp_from_config
        from src.memory.manager import SharedMemoryManager

        print_step("Getting LLM client...")
        llm = get_llm()
        results["llm_provider"] = llm.__class__.__name__
        print_result(True, f"LLM: {results['llm_provider']}")

        print_step("Loading MCP tool catalog...")
        config = load_mcp_config()
        _, tool_catalog = await initialize_mcp_from_config(config)
        results["tool_count"] = len(tool_catalog.list_tools())
        print_result(True, f"Tools: {results['tool_count']}")

        print_step("Initializing agent catalog...")
        agent_catalog = AgentCatalog.get_instance(tool_catalog=tool_catalog)
        agent_catalog.auto_discover()
        agent_catalog.sync_mcp_tools(tool_catalog)
        agents = agent_catalog.list_all()
        results["agent_count"] = len(agents)
        print_result(True, f"Agents: {results['agent_count']}")

        if verbose:
            print()
            print("   Registered agents:")
            for agent in agents:
                print(f"      • {agent.role}")

        print_step("Loading workflow registry...")
        workflow_registry = get_workflow_registry()
        results["workflow_count"] = len(workflow_registry)
        print_result(True, f"Workflows: {results['workflow_count']}")

        print_step("Creating memory manager...")
        # Use in-memory mode for testing
        memory = SharedMemoryManager(use_in_memory=True)
        print_result(True, "Memory manager initialized (in-memory mode)")

        print_step("Creating LangGraph coordinator...")
        coordinator = LangGraphCoordinator(
            llm=llm,
            tool_catalog=tool_catalog,
            agent_catalog=agent_catalog,
            memory=memory,
            workflow_registry=workflow_registry,
        )

        print_step("Initializing coordinator graph...")
        await coordinator.initialize()
        print_result(True, "Coordinator graph compiled successfully")

        results["success"] = True
        results["coordinator"] = coordinator
        return True, results

    except Exception as e:
        import traceback
        print_result(False, f"Coordinator initialization failed: {e}")
        if verbose:
            traceback.print_exc()
        results["error"] = str(e)
        results["success"] = False
        return False, results


async def test_intent_classification(verbose: bool = False) -> tuple[bool, dict]:
    """Test intent classification for common queries."""
    print_header("3. Intent Classification")

    # Test queries that have pre-classifiers (keyword-based, no LLM needed)
    test_queries = [
        ("show fleet health", "db_fleet_health"),  # database pre-classifier
        ("list databases", "list_databases"),  # database pre-classifier
        ("how much am I spending", "cost_summary"),  # cost pre-classifier
        ("show database costs", "database_costs"),  # cost pre-classifier with domain
        ("get AWR report", "awr_report"),  # dbmgmt pre-classifier
    ]

    results = {"tests": [], "passed": 0, "failed": 0}

    try:
        from src.agents.coordinator.nodes import CoordinatorNodes
        from src.agents.coordinator.workflows import get_workflow_registry

        workflow_registry = get_workflow_registry()
        nodes = CoordinatorNodes(
            llm=None,  # Not needed for pre-classification
            tool_catalog=None,
            agent_catalog=None,
            memory=None,
            workflow_registry=workflow_registry,
        )

        for query, expected_workflow in test_queries:
            # Try pre-classification first (no LLM needed)
            intent = None

            # Check each pre-classifier (in priority order)
            for classifier_name in [
                "_pre_classify_dbmgmt_query",  # DB Management (AWR, Top SQL)
                "_pre_classify_opsi_query",  # OPSI (fleet health, ADDM)
                "_pre_classify_database_query",  # General database
                "_pre_classify_cost_query",  # Cost analysis
                "_pre_classify_resource_cost_query",  # Resource + cost
            ]:
                if hasattr(nodes, classifier_name):
                    result = getattr(nodes, classifier_name)(query)
                    if result:
                        intent = result
                        break

            test_result = {
                "query": query,
                "expected": expected_workflow,
                "actual": intent.suggested_workflow if intent else None,
                "confidence": intent.confidence if intent else 0,
            }

            passed = intent and intent.suggested_workflow == expected_workflow
            test_result["passed"] = passed
            results["tests"].append(test_result)

            if passed:
                results["passed"] += 1
                print_step(f'"{query}" → {expected_workflow} (conf: {intent.confidence:.2f})', "success")
            else:
                results["failed"] += 1
                actual = intent.suggested_workflow if intent else "None"
                print_step(f'"{query}" → expected {expected_workflow}, got {actual}', "error")

        results["success"] = results["failed"] == 0
        return results["success"], results

    except Exception as e:
        import traceback
        print_result(False, f"Classification test failed: {e}")
        if verbose:
            traceback.print_exc()
        results["error"] = str(e)
        results["success"] = False
        return False, results


async def test_workflow_execution(query: str, verbose: bool = False) -> tuple[bool, dict]:
    """Test end-to-end workflow execution."""
    print_header("4. Workflow Execution")

    results = {"query": query, "steps": []}

    try:
        from src.agents.catalog import AgentCatalog
        from src.agents.coordinator.graph import LangGraphCoordinator
        from src.agents.coordinator.workflows import get_workflow_registry
        from src.llm import get_llm
        from src.mcp.config import load_mcp_config, initialize_mcp_from_config
        from src.memory.manager import SharedMemoryManager

        print_step(f'Executing query: "{query}"')
        start_time = time.time()

        # Initialize components
        print_step("Setting up coordinator...")
        llm = get_llm()
        config = load_mcp_config()
        _, tool_catalog = await initialize_mcp_from_config(config)

        agent_catalog = AgentCatalog.get_instance(tool_catalog=tool_catalog)
        agent_catalog.auto_discover()
        agent_catalog.sync_mcp_tools(tool_catalog)

        workflow_registry = get_workflow_registry()
        memory = SharedMemoryManager(use_in_memory=True)

        coordinator = LangGraphCoordinator(
            llm=llm,
            tool_catalog=tool_catalog,
            agent_catalog=agent_catalog,
            memory=memory,
            workflow_registry=workflow_registry,
        )
        await coordinator.initialize()
        results["steps"].append({"step": "coordinator_init", "success": True})

        # Thinking callback for verbose mode
        thinking_updates = []

        def on_thinking(update: str) -> None:
            thinking_updates.append(update)
            if verbose:
                print(f"      💭 {update[:80]}...")

        # Execute query
        print_step("Invoking coordinator...")
        result = await coordinator.invoke(
            query=query,
            thread_id="test-thread",
            metadata={"user_id": "test-user", "channel": "test-channel"},
            on_thinking_update=on_thinking if verbose else None,
        )

        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        results["thinking_steps"] = len(thinking_updates)

        # Check result
        if result.get("success"):
            response = result.get("response", "")
            results["response_length"] = len(response)
            results["route_taken"] = result.get("metadata", {}).get("route_taken", "unknown")

            print_result(True, f"Query completed in {elapsed:.2f}s")
            print_step(f"Route: {results['route_taken']}", "info")
            print_step(f"Response length: {len(response)} chars", "info")

            if verbose:
                print()
                print("   Response preview:")
                preview = response[:500] + "..." if len(response) > 500 else response
                for line in preview.split("\n")[:10]:
                    print(f"      {line}")

            results["success"] = True
        else:
            error = result.get("error", "Unknown error")
            results["error"] = error
            print_result(False, f"Query failed: {error}")
            results["success"] = False

        return results["success"], results

    except Exception as e:
        import traceback
        print_result(False, f"Workflow execution failed: {e}")
        if verbose:
            traceback.print_exc()
        results["error"] = str(e)
        results["success"] = False
        return False, results


async def test_slack_handler_integration(verbose: bool = False) -> tuple[bool, dict]:
    """Test SlackHandler can invoke coordinator."""
    print_header("5. Slack Handler Integration")

    results = {}

    try:
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        app_token = os.getenv("SLACK_APP_TOKEN")

        if not bot_token:
            print_result(False, "SLACK_BOT_TOKEN not configured")
            results["success"] = False
            return False, results

        if not app_token:
            print_step("SLACK_APP_TOKEN not configured (Socket Mode disabled)", "warning")

        from src.channels.slack import SlackHandler

        print_step("Creating SlackHandler...")
        handler = SlackHandler(
            bot_token=bot_token,
            app_token=app_token,
        )
        print_result(True, "SlackHandler created")

        print_step("Verifying bot token...")
        from slack_sdk.web.async_client import AsyncWebClient
        client = AsyncWebClient(token=bot_token)
        auth = await client.auth_test()

        results["bot_user"] = auth.get("user")
        results["bot_id"] = auth.get("bot_id")
        results["team"] = auth.get("team")

        print_result(True, f"Bot: {results['bot_user']} (ID: {results['bot_id']})")
        print_result(True, f"Team: {results['team']}")

        # Check if coordinator can be invoked
        print_step("Testing coordinator invocation path...")

        # Check if _invoke_langgraph_coordinator exists
        if hasattr(handler, "_invoke_langgraph_coordinator"):
            print_result(True, "_invoke_langgraph_coordinator method available")
        else:
            print_step("_invoke_langgraph_coordinator not found", "warning")

        results["success"] = True
        return True, results

    except Exception as e:
        import traceback
        print_result(False, f"Slack handler test failed: {e}")
        if verbose:
            traceback.print_exc()
        results["error"] = str(e)
        results["success"] = False
        return False, results


def print_summary(all_results: dict) -> None:
    """Print test summary."""
    print_header("Test Summary")

    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r.get("success", False))
    failed = total - passed

    for test_name, result in all_results.items():
        status = "success" if result.get("success") else "error"
        print_step(test_name, status)

    print()
    if failed == 0:
        print(f"   {Colors.GREEN}{Colors.BOLD}All {total} tests passed!{Colors.RESET}")
        print()
        print(f"   {Colors.GREEN}✅ The coordinator is ready to process messages.{Colors.RESET}")
        print(f"   {Colors.YELLOW}⚠️  Make sure Slack App has event subscriptions enabled:{Colors.RESET}")
        print("      → app_mention, message.im, message.channels")
        print()
        print(f"   Run the Slack bot: poetry run python -m src.main")
    else:
        print(f"   {Colors.RED}{Colors.BOLD}{failed}/{total} tests failed{Colors.RESET}")
        print()
        print("   Fix the failing tests before running the Slack bot.")


async def main():
    """Run all end-to-end tests."""
    parser = argparse.ArgumentParser(description="OCI Coordinator E2E Test")
    parser.add_argument("--query", default="show fleet health", help="Test query to execute")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-workflow", action="store_true", help="Skip workflow execution test")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}OCI AI Agent Coordinator - End-to-End Test{Colors.RESET}")
    print(f"Testing query: \"{args.query}\"")

    all_results = {}

    # Run tests
    success, result = await test_mcp_connectivity(args.verbose)
    all_results["MCP Connectivity"] = result

    if success:
        success, result = await test_coordinator_init(args.verbose)
        all_results["Coordinator Init"] = result

    success, result = await test_intent_classification(args.verbose)
    all_results["Intent Classification"] = result

    if not args.skip_workflow:
        success, result = await test_workflow_execution(args.query, args.verbose)
        all_results["Workflow Execution"] = result

    success, result = await test_slack_handler_integration(args.verbose)
    all_results["Slack Handler"] = result

    # Print summary
    print_summary(all_results)

    # Exit with appropriate code
    all_passed = all(r.get("success", False) for r in all_results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
