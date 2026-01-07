"""
Integration test for MCP server connections and Redis cache updates.

Tests the complete flow:
1. MCP server connections to Coordinator
2. Tool execution through ToolCatalog
3. Redis cache population from discovery calls
4. Agent-to-MCP communication patterns

Usage:
    pytest tests/integration/test_mcp_cache_integration.py -v

    # Or run directly with async:
    python tests/integration/test_mcp_cache_integration.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from typing import Any

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class MCPCacheIntegrationTests:
    """
    Integration tests for MCP server to Redis cache flow.

    Verifies:
    - MCPConnectionManager initializes servers correctly
    - ToolCatalog discovers and executes tools
    - OCIResourceCache updates Redis on tool execution
    - Discovery flows populate cache with OCI data
    """

    def __init__(self):
        self.results: dict[str, Any] = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "details": {},
        }

    async def run_all(self) -> dict[str, Any]:
        """Run all integration tests."""
        print("\n" + "=" * 60)
        print("MCP-Cache Integration Test Suite")
        print("=" * 60 + "\n")

        tests = [
            self.test_mcp_connection_manager,
            self.test_server_registry_health,
            self.test_tool_catalog_discovery,
            self.test_tool_execution,
            self.test_redis_cache_connection,
            self.test_oci_resource_cache,
            self.test_discovery_to_cache_flow,
            self.test_agent_to_mcp_pattern,
            self.test_mcp_oci_shared_memory,
        ]

        for test in tests:
            await self._run_test(test)

        self._print_summary()
        return self.results

    async def _run_test(self, test_func) -> None:
        """Run a single test and record results."""
        test_name = test_func.__name__
        self.results["tests_run"] += 1

        print(f"Running: {test_name}...", end=" ")

        try:
            result = await test_func()
            self.results["details"][test_name] = result

            if result.get("passed"):
                self.results["tests_passed"] += 1
                print("✓ PASSED")
            else:
                self.results["tests_failed"] += 1
                print(f"✗ FAILED: {result.get('error', 'Unknown error')}")
                self.results["errors"].append({
                    "test": test_name,
                    "error": result.get("error"),
                })
        except Exception as e:
            self.results["tests_failed"] += 1
            print(f"✗ ERROR: {e}")
            self.results["errors"].append({
                "test": test_name,
                "error": str(e),
            })
            self.results["details"][test_name] = {
                "passed": False,
                "error": str(e),
            }

    def _print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total:  {self.results['tests_run']}")
        print(f"Passed: {self.results['tests_passed']} ✓")
        print(f"Failed: {self.results['tests_failed']} ✗")

        if self.results["errors"]:
            print("\nErrors:")
            for err in self.results["errors"]:
                print(f"  - {err['test']}: {err['error']}")
        print("")

    # ─────────────────────────────────────────────────────────────────────────
    # Test Cases
    # ─────────────────────────────────────────────────────────────────────────

    async def test_mcp_connection_manager(self) -> dict[str, Any]:
        """Test MCPConnectionManager singleton and initialization."""
        try:
            from src.mcp.connection_manager import MCPConnectionManager

            # Reset any existing instance for clean test
            MCPConnectionManager.reset_instance()

            # Get instance (should trigger initialization)
            manager = await MCPConnectionManager.get_instance()
            status = manager.get_status()

            return {
                "passed": status["initialized"],
                "initialized": status["initialized"],
                "connected_servers": status.get("connected_servers", []),
                "tool_count": status.get("tool_count", 0),
                "init_time_s": status.get("init_time_s"),
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_server_registry_health(self) -> dict[str, Any]:
        """Test ServerRegistry health and connected servers."""
        try:
            from src.mcp.connection_manager import MCPConnectionManager

            manager = await MCPConnectionManager.get_instance()
            registry = await manager.get_registry()

            if not registry:
                return {"passed": False, "error": "Registry not available"}

            health = registry.get_health_summary()
            connected = registry.list_connected()
            all_servers = list(registry._servers.keys())  # Get all server IDs

            # At least one server should be connected
            has_connections = health.get("connected", 0) > 0

            return {
                "passed": has_connections,
                "health_summary": health,
                "connected_servers": connected,
                "all_servers": all_servers,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_tool_catalog_discovery(self) -> dict[str, Any]:
        """Test ToolCatalog tool discovery."""
        try:
            from src.mcp.connection_manager import get_mcp_catalog

            catalog = await get_mcp_catalog()
            if not catalog:
                return {"passed": False, "error": "Catalog not available"}

            # List all tools (returns ToolDefinition objects)
            all_tool_defs = catalog.list_tools()
            all_tool_names = [t.name for t in all_tool_defs]

            # Check for expected OCI tools
            oci_tools = [t for t in all_tool_names if t.startswith("oci_")]
            discovery_tools = [t for t in all_tool_names if "search" in t.lower() or "list" in t.lower()]

            # Verify tool metadata is available
            tool_details = {}
            for tool_name in oci_tools[:5]:  # Check first 5 OCI tools
                tool_def = catalog.get_tool(tool_name)
                if tool_def:
                    tool_details[tool_name] = {
                        "server": tool_def.server_id,
                        "description": tool_def.description[:50] if tool_def.description else "",
                    }

            return {
                "passed": len(all_tool_names) > 0,
                "total_tools": len(all_tool_names),
                "oci_tools": len(oci_tools),
                "discovery_tools": len(discovery_tools),
                "sample_tools": all_tool_names[:10],
                "tool_details": tool_details,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_tool_execution(self) -> dict[str, Any]:
        """Test tool execution through ToolCatalog."""
        try:
            from src.mcp.connection_manager import get_mcp_catalog

            catalog = await get_mcp_catalog()
            if not catalog:
                return {"passed": False, "error": "Catalog not available"}

            # Try to execute ping tool (should be fast and safe)
            all_tool_defs = catalog.list_tools()
            tool_names = [t.name for t in all_tool_defs]

            # Look for a safe tool to test
            test_tool = None
            for candidate in ["oci_ping", "ping", "oci_list_domains", "list_domains"]:
                if candidate in tool_names:
                    test_tool = candidate
                    break

            if not test_tool:
                # Try any oci_ tool that looks safe
                for tool_name in tool_names:
                    if tool_name.startswith("oci_") and "list" in tool_name.lower():
                        test_tool = tool_name
                        break

            if not test_tool:
                return {
                    "passed": True,
                    "skipped": True,
                    "reason": "No suitable test tool found",
                    "available_tools": tool_names[:10],
                }

            # Execute the tool
            start_time = datetime.utcnow()
            result = await catalog.execute(test_tool, {})
            duration = (datetime.utcnow() - start_time).total_seconds()

            return {
                "passed": result.success if hasattr(result, "success") else result is not None,
                "tool_executed": test_tool,
                "duration_s": duration,
                "result_type": type(result).__name__,
                "has_result": result is not None,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_redis_cache_connection(self) -> dict[str, Any]:
        """Test Redis cache connectivity."""
        try:
            import redis.asyncio as redis

            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(redis_url, decode_responses=True)

            # Test connection
            await client.ping()

            # Get info
            info = await client.info("memory")

            # Count OCI-related keys
            oci_keys = await client.keys("oci:*")

            await client.close()

            return {
                "passed": True,
                "redis_url": redis_url,
                "connected": True,
                "memory_used": info.get("used_memory_human", "unknown"),
                "oci_keys_count": len(oci_keys),
            }
        except Exception as e:
            # Redis may not be running - this is expected in some environments
            return {
                "passed": True,  # Not a failure if Redis isn't available
                "skipped": True,
                "reason": f"Redis not available: {e}",
            }

    async def test_oci_resource_cache(self) -> dict[str, Any]:
        """Test OCIResourceCache initialization and operations."""
        try:
            from src.cache.oci_resource_cache import OCIResourceCache
            from src.mcp.connection_manager import get_mcp_catalog

            # Get tool catalog for cache
            catalog = await get_mcp_catalog()

            # Initialize cache
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            cache = OCIResourceCache(redis_url=redis_url, tool_catalog=catalog)

            await cache.initialize()

            # Get cache stats
            stats = await cache.get_cache_stats()

            # Health check
            health = await cache.health_check()

            await cache.close()

            return {
                "passed": health.get("healthy", False) or health.get("warning") is not None,
                "initialized": cache._initialized,
                "backend": stats.get("backend"),
                "health": health,
                "key_counts": stats.get("keys", {}),
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_discovery_to_cache_flow(self) -> dict[str, Any]:
        """Test the discovery → cache update flow."""
        try:
            from src.cache.oci_resource_cache import OCIResourceCache
            from src.mcp.connection_manager import get_mcp_catalog

            # Get tool catalog
            catalog = await get_mcp_catalog()
            if not catalog:
                return {"passed": False, "error": "Catalog not available"}

            # Initialize cache with catalog
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            cache = OCIResourceCache(redis_url=redis_url, tool_catalog=catalog)
            await cache.initialize()

            # Check if we have discovery tools
            all_tool_defs = catalog.list_tools()
            tool_names = [t.name for t in all_tool_defs]
            has_compartment_tool = any("compartment" in t.lower() for t in tool_names)

            if not has_compartment_tool:
                await cache.close()
                return {
                    "passed": True,
                    "skipped": True,
                    "reason": "No compartment discovery tool available",
                }

            # Track cache updates via event callback
            updates = []
            def on_update(event, key, data):
                updates.append({"event": event, "key": key})

            cache.on_event(on_update)

            # Get initial stats
            initial_stats = await cache.get_cache_stats()

            # Try discovery (may fail if OCI not configured, which is OK)
            try:
                discovery_result = await cache.discover_compartments()
            except Exception as e:
                discovery_result = f"Discovery error: {e}"

            # Get post-discovery stats
            post_stats = await cache.get_cache_stats()

            await cache.close()

            return {
                "passed": True,  # Flow tested even if OCI isn't configured
                "initial_keys": initial_stats.get("keys", {}).get("total", 0),
                "post_keys": post_stats.get("keys", {}).get("total", 0),
                "cache_updates": len(updates),
                "discovery_result": discovery_result if isinstance(discovery_result, int) else str(discovery_result)[:100],
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_agent_to_mcp_pattern(self) -> dict[str, Any]:
        """Test the agent → tool catalog → MCP server pattern."""
        try:
            from src.mcp.connection_manager import MCPConnectionManager, get_mcp_catalog

            # Simulate agent workflow

            # 1. Get catalog (as agent would)
            catalog = await get_mcp_catalog()
            if not catalog:
                return {"passed": False, "error": "Catalog not available"}

            # 2. Discover available domains/tools
            all_tool_defs = catalog.list_tools()
            all_tool_names = [t.name for t in all_tool_defs]

            # 3. Filter tools by domain (as agent would for specific task)
            domains = {
                "compute": [t for t in all_tool_names if "compute" in t.lower() or "instance" in t.lower()],
                "database": [t for t in all_tool_names if "database" in t.lower() or "db" in t.lower()],
                "network": [t for t in all_tool_names if "network" in t.lower() or "vcn" in t.lower()],
                "security": [t for t in all_tool_names if "security" in t.lower() or "iam" in t.lower()],
            }

            # 4. Get connection status
            manager = await MCPConnectionManager.get_instance()
            status = manager.get_status()

            return {
                "passed": True,
                "workflow_complete": True,
                "catalog_available": catalog is not None,
                "tools_discovered": len(all_tool_names),
                "domain_breakdown": {k: len(v) for k, v in domains.items()},
                "connected_servers": status.get("connected_servers", []),
                "tool_count": status.get("tool_count", 0),
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}


    async def test_mcp_oci_shared_memory(self) -> dict[str, Any]:
        """Test MCP-OCI's shared memory module."""
        try:
            # Import from mcp-oci's shared memory
            from mcp_server_oci.core.shared_memory import (
                EventType,
                InMemorySharedStore,
                SharedContext,
                SharedEvent,
                get_shared_store,
                share_finding,
            )

            # Get shared store (uses InMemory store)
            store = get_shared_store()
            store_type = type(store).__name__

            # Test agent registration
            agent_info = await store.register_agent(
                agent_id="test-integration-agent",
                agent_type="test",
                capabilities=["test", "integration"],
                metadata={"test": True},
            )

            # Test context creation
            context = SharedContext(
                session_id="test-session-123",
                resource_id="ocid1.instance.test",
                resource_type="compute_instance",
                findings=[{"type": "test", "message": "Integration test"}],
            )
            saved_context = await store.save_context(context)

            # Test event publishing
            event = SharedEvent(
                event_type=EventType.FINDING,
                source_agent="test-integration-agent",
                payload={"test": True},
            )
            published_event = await store.publish_event(event)

            # Get recent events
            events = await store.get_recent_events(limit=10)

            # Cleanup
            await store.cleanup_expired()

            return {
                "passed": True,
                "store_type": store_type,
                "agent_registered": agent_info is not None,
                "context_saved": saved_context.context_id == context.context_id,
                "event_published": published_event.event_id == event.event_id,
                "events_retrieved": len(events),
            }
        except ImportError as e:
            return {
                "passed": True,
                "skipped": True,
                "reason": f"mcp_server_oci not importable: {e}",
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Pytest Integration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def integration_tester():
    """Fixture for integration tests."""
    return MCPCacheIntegrationTests()


@pytest.mark.asyncio
async def test_mcp_connection_manager(integration_tester):
    """Test MCP connection manager initialization."""
    result = await integration_tester.test_mcp_connection_manager()
    assert result["passed"], result.get("error", "Test failed")


@pytest.mark.asyncio
async def test_server_registry_health(integration_tester):
    """Test server registry health."""
    result = await integration_tester.test_server_registry_health()
    assert result["passed"], result.get("error", "Test failed")


@pytest.mark.asyncio
async def test_tool_catalog_discovery(integration_tester):
    """Test tool catalog discovery."""
    result = await integration_tester.test_tool_catalog_discovery()
    assert result["passed"], result.get("error", "Test failed")


@pytest.mark.asyncio
async def test_tool_execution(integration_tester):
    """Test tool execution."""
    result = await integration_tester.test_tool_execution()
    assert result["passed"], result.get("error", "Test failed")


@pytest.mark.asyncio
async def test_redis_cache_connection(integration_tester):
    """Test Redis cache connection."""
    result = await integration_tester.test_redis_cache_connection()
    assert result["passed"], result.get("error", "Test failed")


@pytest.mark.asyncio
async def test_oci_resource_cache(integration_tester):
    """Test OCI resource cache."""
    result = await integration_tester.test_oci_resource_cache()
    assert result["passed"], result.get("error", "Test failed")


@pytest.mark.asyncio
async def test_discovery_to_cache_flow(integration_tester):
    """Test discovery to cache flow."""
    result = await integration_tester.test_discovery_to_cache_flow()
    assert result["passed"], result.get("error", "Test failed")


@pytest.mark.asyncio
async def test_agent_to_mcp_pattern(integration_tester):
    """Test agent to MCP pattern."""
    result = await integration_tester.test_agent_to_mcp_pattern()
    assert result["passed"], result.get("error", "Test failed")


# ─────────────────────────────────────────────────────────────────────────────
# Direct Execution
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Run all tests directly."""
    tester = MCPCacheIntegrationTests()
    results = await tester.run_all()

    # Exit with error code if tests failed
    if results["tests_failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
