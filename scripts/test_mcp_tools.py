#!/usr/bin/env python3
"""
Comprehensive MCP Tool Testing Script.

Tests all MCP server tools with proper compartment OCID handling.
Target compartment: adrian_birzu (ocid1.compartment.oc1..aaaaaaaagy3yddkkampnhj3cqm5ar7w2p7tuq5twbojyycvol6wugfav3ckq)

Usage:
    poetry run python scripts/test_mcp_tools.py
    poetry run python scripts/test_mcp_tools.py --server oci-unified
    poetry run python scripts/test_mcp_tools.py --domain identity
    poetry run python scripts/test_mcp_tools.py --verbose
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test compartment OCID
TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaagy3yddkkampnhj3cqm5ar7w2p7tuq5twbojyycvol6wugfav3ckq"
TEST_COMPARTMENT_NAME = "adrian_birzu"


@dataclass
class TestResult:
    """Result of a single tool test."""
    tool_name: str
    server: str
    success: bool
    duration_ms: int = 0
    result_preview: str = ""
    error: str = ""
    parameters_used: dict = field(default_factory=dict)


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: list[TestResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def total(self) -> int:
        return len(self.results)


class MCPToolTester:
    """Tests MCP server tools."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.catalog = None
        self.registry = None
        self.suite = TestSuite(name=f"MCP Tool Test - {datetime.now().isoformat()}")

    async def setup(self) -> bool:
        """Initialize MCP infrastructure."""
        print("\n🔧 Setting up MCP infrastructure...")

        try:
            # Initialize MCP using the proper config loader
            from src.mcp.config import initialize_mcp_from_config, load_mcp_config

            config = load_mcp_config()
            enabled = config.get_enabled_servers()
            print(f"   Found {len(enabled)} enabled MCP servers: {list(enabled.keys())}")

            self.registry, self.catalog = await initialize_mcp_from_config(config)

            print(f"✅ MCP initialized: {len(self.catalog.list_tools())} tools available")

            # Print connected servers
            connected = self.registry.list_connected()
            print(f"   Connected servers: {connected}")

            return True

        except Exception as e:
            import traceback
            print(f"❌ Setup failed: {e}")
            traceback.print_exc()
            return False

    async def test_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        expected_contains: list[str] | None = None,
    ) -> TestResult:
        """Test a single tool."""
        import time
        start = time.time()

        result = TestResult(
            tool_name=tool_name,
            server="unknown",
            success=False,
            parameters_used=parameters,
        )

        try:
            # Get tool info
            tool_def = self.catalog.get_tool(tool_name)
            if not tool_def:
                result.error = f"Tool not found: {tool_name}"
                return result

            result.server = tool_def.server_id

            # Execute the tool
            call_result = await self.catalog.execute(tool_name, parameters)
            result.duration_ms = int((time.time() - start) * 1000)

            if call_result.success:
                result.success = True
                result_str = str(call_result.result)
                result.result_preview = result_str[:500] + "..." if len(result_str) > 500 else result_str

                # Check expected content
                if expected_contains:
                    for expected in expected_contains:
                        if expected.lower() not in result_str.lower():
                            result.success = False
                            result.error = f"Expected '{expected}' not found in result"
                            break
            else:
                result.error = call_result.error or "Unknown error"

        except Exception as e:
            result.error = str(e)
            result.duration_ms = int((time.time() - start) * 1000)

        return result

    def print_result(self, result: TestResult):
        """Print a single test result."""
        status = "✅" if result.success else "❌"
        duration = f"({result.duration_ms}ms)"

        print(f"\n{status} {result.tool_name} {duration}")
        print(f"   Server: {result.server}")
        print(f"   Params: {json.dumps(result.parameters_used, default=str)[:100]}")

        if result.success and self.verbose:
            print(f"   Preview: {result.result_preview[:200]}")
        elif not result.success:
            print(f"   Error: {result.error}")

    async def test_identity_tools(self):
        """Test identity domain tools."""
        print("\n" + "=" * 60)
        print("📋 Testing IDENTITY Tools")
        print("=" * 60)

        tests = [
            # List compartments (should default to tenancy root)
            ("oci_list_compartments", {"limit": 10, "format": "json"}, ["name", "id"]),

            # List compartments with specific compartment
            ("oci_list_compartments", {"compartment_id": TEST_COMPARTMENT_ID, "limit": 5}, ["name"]),

            # Get compartment details
            ("oci_get_compartment", {"compartment_id": TEST_COMPARTMENT_ID}, [TEST_COMPARTMENT_NAME]),

            # Search compartments
            ("oci_search_compartments", {"query": "adrian", "limit": 5}, ["adrian"]),

            # Get tenancy info
            ("oci_get_tenancy", {"format": "json"}, ["name", "id"]),

            # List regions
            ("oci_list_regions", {}, ["region"]),
        ]

        for tool_name, params, expected in tests:
            result = await self.test_tool(tool_name, params, expected)
            self.suite.results.append(result)
            self.print_result(result)

    async def test_compute_tools(self):
        """Test compute domain tools."""
        print("\n" + "=" * 60)
        print("💻 Testing COMPUTE Tools")
        print("=" * 60)

        tests = [
            # List instances in compartment
            ("oci_compute_list_instances", {
                "compartment_id": TEST_COMPARTMENT_ID,
                "limit": 10,
                "format": "json"
            }, None),

            # Find instance by name (search)
            ("oci_compute_find_instance", {
                "instance_name": "test",
                "compartment_id": TEST_COMPARTMENT_ID
            }, None),
        ]

        for tool_name, params, expected in tests:
            result = await self.test_tool(tool_name, params, expected)
            self.suite.results.append(result)
            self.print_result(result)

    async def test_network_tools(self):
        """Test network domain tools."""
        print("\n" + "=" * 60)
        print("🌐 Testing NETWORK Tools")
        print("=" * 60)

        tests = [
            # List VCNs
            ("oci_network_list_vcns", {
                "compartment_id": TEST_COMPARTMENT_ID,
                "limit": 10,
                "format": "json"
            }, None),

            # List subnets
            ("oci_network_list_subnets", {
                "compartment_id": TEST_COMPARTMENT_ID,
                "limit": 10
            }, None),

            # List security lists
            ("oci_network_list_security_lists", {
                "compartment_id": TEST_COMPARTMENT_ID,
                "limit": 10
            }, None),
        ]

        for tool_name, params, expected in tests:
            result = await self.test_tool(tool_name, params, expected)
            self.suite.results.append(result)
            self.print_result(result)

    async def test_cost_tools(self):
        """Test cost domain tools."""
        print("\n" + "=" * 60)
        print("💰 Testing COST Tools")
        print("=" * 60)

        tests = [
            # Get cost summary (defaults to tenancy)
            ("oci_cost_get_summary", {"days": 7}, ["cost", "summary"]),

            # Get cost summary with compartment filter
            ("oci_cost_get_summary", {
                "compartment_id": TEST_COMPARTMENT_ID,
                "days": 30
            }, None),

            # Get cost with service filter
            ("oci_cost_get_summary", {
                "days": 30,
                "service_filter": "database"
            }, None),
        ]

        for tool_name, params, expected in tests:
            result = await self.test_tool(tool_name, params, expected)
            self.suite.results.append(result)
            self.print_result(result)

    async def test_security_tools(self):
        """Test security domain tools."""
        print("\n" + "=" * 60)
        print("🔒 Testing SECURITY Tools")
        print("=" * 60)

        tests = [
            # List users (requires tenancy compartment)
            ("oci_security_list_users", {
                "compartment_id": TEST_COMPARTMENT_ID,
                "limit": 10
            }, None),
        ]

        for tool_name, params, expected in tests:
            result = await self.test_tool(tool_name, params, expected)
            self.suite.results.append(result)
            self.print_result(result)

    async def test_discovery_tools(self):
        """Test discovery domain tools."""
        print("\n" + "=" * 60)
        print("🔍 Testing DISCOVERY Tools")
        print("=" * 60)

        tests = [
            # Get cache status
            ("oci_discovery_cache_status", {}, None),

            # Get resource summary
            ("oci_discovery_summary", {}, None),

            # Search resources
            ("oci_discovery_search", {
                "query": "test",
                "resource_type": "instance"
            }, None),
        ]

        for tool_name, params, expected in tests:
            result = await self.test_tool(tool_name, params, expected)
            self.suite.results.append(result)
            self.print_result(result)

    async def test_search_capabilities(self):
        """Test the search_capabilities meta-tool."""
        print("\n" + "=" * 60)
        print("🔎 Testing SEARCH CAPABILITIES")
        print("=" * 60)

        tests = [
            ("search_capabilities", {"query": "compute"}, ["compute"]),
            ("search_capabilities", {"query": "cost"}, ["cost"]),
            ("search_capabilities", {"query": "database"}, ["database", "db"]),
        ]

        for tool_name, params, expected in tests:
            result = await self.test_tool(tool_name, params, expected)
            self.suite.results.append(result)
            self.print_result(result)

    def print_summary(self):
        """Print test summary."""
        self.suite.finished_at = datetime.now()
        duration = (self.suite.finished_at - self.suite.started_at).total_seconds()

        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {self.suite.total}")
        print(f"Passed: {self.suite.passed} ✅")
        print(f"Failed: {self.suite.failed} ❌")
        print(f"Duration: {duration:.1f}s")
        print(f"Success rate: {self.suite.passed / self.suite.total * 100:.1f}%")

        if self.suite.failed > 0:
            print("\n❌ FAILED TESTS:")
            for r in self.suite.results:
                if not r.success:
                    print(f"  - {r.tool_name}: {r.error}")

        # Save results to file
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "suite": self.suite.name,
                "started": self.suite.started_at.isoformat(),
                "finished": self.suite.finished_at.isoformat() if self.suite.finished_at else None,
                "total": self.suite.total,
                "passed": self.suite.passed,
                "failed": self.suite.failed,
                "results": [
                    {
                        "tool": r.tool_name,
                        "server": r.server,
                        "success": r.success,
                        "duration_ms": r.duration_ms,
                        "error": r.error,
                        "parameters": r.parameters_used,
                    }
                    for r in self.suite.results
                ]
            }, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {results_file}")

    async def run_all_tests(self, domains: list[str] | None = None):
        """Run all tests or specific domains."""
        if not await self.setup():
            return

        all_domains = ["identity", "compute", "network", "cost", "security", "discovery", "search"]
        domains_to_test = domains or all_domains

        test_methods = {
            "identity": self.test_identity_tools,
            "compute": self.test_compute_tools,
            "network": self.test_network_tools,
            "cost": self.test_cost_tools,
            "security": self.test_security_tools,
            "discovery": self.test_discovery_tools,
            "search": self.test_search_capabilities,
        }

        for domain in domains_to_test:
            if domain in test_methods:
                await test_methods[domain]()

        self.print_summary()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test MCP server tools")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--domain", "-d", help="Test specific domain(s)", action="append")
    parser.add_argument("--server", "-s", help="Test specific server")
    args = parser.parse_args()

    print(f"\n🧪 MCP Tool Test Suite")
    print(f"   Compartment: {TEST_COMPARTMENT_NAME}")
    print(f"   OCID: {TEST_COMPARTMENT_ID[:50]}...")

    tester = MCPToolTester(verbose=args.verbose)
    await tester.run_all_tests(domains=args.domain)


if __name__ == "__main__":
    asyncio.run(main())
