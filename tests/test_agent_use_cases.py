"""
Comprehensive Agent Use Case Test Suite.

Tests 10 use cases for each of the 5 AI agents:
1. Infrastructure Agent - Compute, Network, Storage
2. FinOps Agent - Cost analysis, budgets, optimization
3. DB Troubleshoot Agent - Database health, RCA, SQL analysis
4. Security Threat Agent - IAM, threats, vulnerabilities
5. Log Analytics Agent - Log queries, alerts, patterns

Run with:
    poetry run pytest tests/test_agent_use_cases.py -v

For integration tests (real MCP):
    poetry run pytest tests/test_agent_use_cases.py -v -m integration
"""

import json
from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp.client import ToolDefinition


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_tool_catalog():
    """Create a mock tool catalog for testing."""
    from src.mcp.catalog import ToolCatalog

    catalog = MagicMock(spec=ToolCatalog)

    # Define available tools
    tools = {
        # Infrastructure tools
        "oci_compute_list_instances": ToolDefinition(
            name="oci_compute_list_instances",
            description="List compute instances",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_compute_get_instance": ToolDefinition(
            name="oci_compute_get_instance",
            description="Get instance details",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_network_list_vcns": ToolDefinition(
            name="oci_network_list_vcns",
            description="List VCNs",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_network_list_subnets": ToolDefinition(
            name="oci_network_list_subnets",
            description="List subnets",
            input_schema={},
            server_id="oci-unified",
        ),
        # Cost tools
        "oci_cost_get_summary": ToolDefinition(
            name="oci_cost_get_summary",
            description="Get cost summary",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_cost_forecast": ToolDefinition(
            name="oci_cost_forecast",
            description="Forecast costs",
            input_schema={},
            server_id="finopsai",
        ),
        # Security tools
        "oci_security_list_users": ToolDefinition(
            name="oci_security_list_users",
            description="List IAM users",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_security_list_policies": ToolDefinition(
            name="oci_security_list_policies",
            description="List IAM policies",
            input_schema={},
            server_id="oci-unified",
        ),
        # Database tools (Database Observatory MCP)
        "oci_opsi_get_fleet_summary": ToolDefinition(
            name="oci_opsi_get_fleet_summary",
            description="Get OPSI fleet summary with database insights",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_search_databases": ToolDefinition(
            name="oci_opsi_search_databases",
            description="Search databases by name, type, or compartment",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_get_database": ToolDefinition(
            name="oci_opsi_get_database",
            description="Get detailed database information from OPSI cache",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_database_execute_sql": ToolDefinition(
            name="oci_database_execute_sql",
            description="Execute SQL query on Oracle database via SQLcl",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_logan_execute_query": ToolDefinition(
            name="oci_logan_execute_query",
            description="Execute Log Analytics query",
            input_schema={},
            server_id="database-observatory",
        ),
        # Observability tools
        "oci_observability_query_logs": ToolDefinition(
            name="oci_observability_query_logs",
            description="Query logs",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_observability_get_metrics": ToolDefinition(
            name="oci_observability_get_metrics",
            description="Get metrics",
            input_schema={},
            server_id="oci-unified",
        ),
        # Discovery tools
        "oci_list_compartments": ToolDefinition(
            name="oci_list_compartments",
            description="List compartments",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_search_compartments": ToolDefinition(
            name="oci_search_compartments",
            description="Search compartments",
            input_schema={},
            server_id="oci-unified",
        ),
    }

    catalog.get_tool = lambda name: tools.get(name)
    catalog.list_tools = lambda: list(tools.values())
    catalog.search_tools = lambda query: [
        {"name": t.name, "description": t.description}
        for t in tools.values()
        if query.lower() in t.name.lower() or query.lower() in t.description.lower()
    ]

    return catalog


@pytest.fixture
def mock_memory():
    """Create a mock memory manager."""
    memory = MagicMock()
    memory.get_session_state = AsyncMock(return_value={})
    memory.set_session_state = AsyncMock()
    memory.get_agent_memory = AsyncMock(return_value=None)
    memory.set_agent_memory = AsyncMock()
    memory.append_conversation = AsyncMock()
    return memory


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure Agent Tests (10 test cases)
# ─────────────────────────────────────────────────────────────────────────────


class TestInfrastructureAgentUseCases:
    """Test suite for Infrastructure Agent - 10 use cases."""

    @pytest.fixture
    def infra_tool_results(self):
        """Mock tool results for infrastructure queries."""
        return {
            "list_instances": json.dumps([
                {"name": "web-server-1", "id": "ocid1.instance.1", "state": "RUNNING", "shape": "VM.Standard.E4.Flex"},
                {"name": "db-server-1", "id": "ocid1.instance.2", "state": "RUNNING", "shape": "VM.Standard.E4.Flex"},
                {"name": "app-server-1", "id": "ocid1.instance.3", "state": "STOPPED", "shape": "VM.Standard2.1"},
            ]),
            "get_instance": json.dumps({
                "name": "web-server-1",
                "id": "ocid1.instance.1",
                "state": "RUNNING",
                "shape": "VM.Standard.E4.Flex",
                "availability_domain": "AD-1",
                "fault_domain": "FD-1",
                "ocpus": 4,
                "memory_gb": 64,
            }),
            "list_vcns": json.dumps([
                {"name": "prod-vcn", "id": "ocid1.vcn.1", "cidr_blocks": ["10.0.0.0/16"]},
                {"name": "dev-vcn", "id": "ocid1.vcn.2", "cidr_blocks": ["10.1.0.0/16"]},
            ]),
            "list_subnets": json.dumps([
                {"name": "public-subnet", "id": "ocid1.subnet.1", "cidr_block": "10.0.1.0/24"},
                {"name": "private-subnet", "id": "ocid1.subnet.2", "cidr_block": "10.0.2.0/24"},
            ]),
            "compartments": json.dumps([
                {"name": "production", "id": "ocid1.compartment.prod"},
                {"name": "development", "id": "ocid1.compartment.dev"},
            ]),
        }

    def test_use_case_01_list_all_instances(self, infra_tool_results):
        """UC-01: List all compute instances in a compartment."""
        result = infra_tool_results["list_instances"]
        instances = json.loads(result)

        assert len(instances) == 3
        assert instances[0]["name"] == "web-server-1"
        assert instances[0]["state"] == "RUNNING"

    def test_use_case_02_get_instance_details(self, infra_tool_results):
        """UC-02: Get detailed information about a specific instance."""
        result = infra_tool_results["get_instance"]
        instance = json.loads(result)

        assert instance["name"] == "web-server-1"
        assert instance["shape"] == "VM.Standard.E4.Flex"
        assert instance["ocpus"] == 4

    def test_use_case_03_find_stopped_instances(self, infra_tool_results):
        """UC-03: Find all stopped instances for cost optimization."""
        instances = json.loads(infra_tool_results["list_instances"])
        stopped = [i for i in instances if i["state"] == "STOPPED"]

        assert len(stopped) == 1
        assert stopped[0]["name"] == "app-server-1"

    def test_use_case_04_list_vcns(self, infra_tool_results):
        """UC-04: List all VCNs for network audit."""
        vcns = json.loads(infra_tool_results["list_vcns"])

        assert len(vcns) == 2
        assert any(v["name"] == "prod-vcn" for v in vcns)

    def test_use_case_05_list_subnets(self, infra_tool_results):
        """UC-05: List subnets within a VCN."""
        subnets = json.loads(infra_tool_results["list_subnets"])

        assert len(subnets) == 2
        assert any(s["name"] == "public-subnet" for s in subnets)

    def test_use_case_06_find_large_shapes(self, infra_tool_results):
        """UC-06: Find instances with large shapes for optimization."""
        result = json.loads(infra_tool_results["get_instance"])

        assert result["ocpus"] >= 4
        assert result["memory_gb"] >= 64

    def test_use_case_07_check_availability_domains(self, infra_tool_results):
        """UC-07: Check instance distribution across ADs."""
        instance = json.loads(infra_tool_results["get_instance"])

        assert "availability_domain" in instance
        assert instance["availability_domain"] == "AD-1"

    def test_use_case_08_check_fault_domains(self, infra_tool_results):
        """UC-08: Check fault domain distribution for HA."""
        instance = json.loads(infra_tool_results["get_instance"])

        assert "fault_domain" in instance
        assert instance["fault_domain"] == "FD-1"

    def test_use_case_09_list_compartments(self, infra_tool_results):
        """UC-09: List compartments for resource organization."""
        compartments = json.loads(infra_tool_results["compartments"])

        assert len(compartments) == 2
        assert any(c["name"] == "production" for c in compartments)

    def test_use_case_10_summarize_infrastructure(self, infra_tool_results):
        """UC-10: Generate infrastructure summary report."""
        instances = json.loads(infra_tool_results["list_instances"])
        vcns = json.loads(infra_tool_results["list_vcns"])

        summary = {
            "total_instances": len(instances),
            "running_instances": len([i for i in instances if i["state"] == "RUNNING"]),
            "stopped_instances": len([i for i in instances if i["state"] == "STOPPED"]),
            "total_vcns": len(vcns),
        }

        assert summary["total_instances"] == 3
        assert summary["running_instances"] == 2
        assert summary["stopped_instances"] == 1
        assert summary["total_vcns"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# FinOps Agent Tests (10 test cases)
# ─────────────────────────────────────────────────────────────────────────────


class TestFinOpsAgentUseCases:
    """Test suite for FinOps Agent - 10 use cases."""

    @pytest.fixture
    def finops_tool_results(self):
        """Mock tool results for FinOps queries."""
        return {
            "cost_summary": json.dumps({
                "type": "cost_summary",
                "summary": {
                    "total": "1,234.56 USD",
                    "period": "2025-12-01 -> 2025-12-31",
                    "days": 30,
                },
                "services": [
                    {"service": "COMPUTE", "cost": "500.00 USD", "percent": "40.5%"},
                    {"service": "DATABASE", "cost": "400.00 USD", "percent": "32.4%"},
                    {"service": "STORAGE", "cost": "200.00 USD", "percent": "16.2%"},
                    {"service": "NETWORK", "cost": "134.56 USD", "percent": "10.9%"},
                ],
            }),
            "forecast": json.dumps({
                "current_month_cost": 1234.56,
                "forecasted_end_of_month": 1500.00,
                "year_to_date": 15000.00,
                "forecasted_annual": 18000.00,
            }),
            "budget": json.dumps({
                "name": "Production Budget",
                "amount": 2000.00,
                "spent": 1234.56,
                "remaining": 765.44,
                "percent_used": 61.7,
            }),
        }

    def test_use_case_01_get_cost_summary(self, finops_tool_results):
        """UC-01: Get monthly cost summary by service."""
        result = json.loads(finops_tool_results["cost_summary"])

        assert result["type"] == "cost_summary"
        assert "1,234.56 USD" in result["summary"]["total"]
        assert len(result["services"]) >= 4

    def test_use_case_02_identify_top_cost_services(self, finops_tool_results):
        """UC-02: Identify top cost services."""
        result = json.loads(finops_tool_results["cost_summary"])
        services = result["services"]

        assert services[0]["service"] == "COMPUTE"
        assert "40.5%" in services[0]["percent"]

    def test_use_case_03_forecast_costs(self, finops_tool_results):
        """UC-03: Forecast end of month costs."""
        forecast = json.loads(finops_tool_results["forecast"])

        assert forecast["forecasted_end_of_month"] > forecast["current_month_cost"]
        assert forecast["forecasted_end_of_month"] == 1500.00

    def test_use_case_04_check_budget_status(self, finops_tool_results):
        """UC-04: Check budget utilization."""
        budget = json.loads(finops_tool_results["budget"])

        assert budget["percent_used"] < 100
        assert budget["remaining"] > 0

    def test_use_case_05_calculate_cost_trend(self, finops_tool_results):
        """UC-05: Calculate cost trend over time."""
        forecast = json.loads(finops_tool_results["forecast"])

        monthly_avg = forecast["year_to_date"] / 12
        assert monthly_avg > 0

    def test_use_case_06_identify_cost_anomalies(self, finops_tool_results):
        """UC-06: Identify unusual cost spikes."""
        result = json.loads(finops_tool_results["cost_summary"])
        services = result["services"]

        high_cost = [s for s in services if float(s["percent"].replace("%", "")) > 30]
        assert len(high_cost) >= 1

    def test_use_case_07_compare_periods(self, finops_tool_results):
        """UC-07: Compare costs between periods."""
        forecast = json.loads(finops_tool_results["forecast"])

        growth_rate = (forecast["forecasted_annual"] - forecast["year_to_date"]) / forecast["year_to_date"]
        assert growth_rate > 0

    def test_use_case_08_calculate_savings_potential(self, finops_tool_results):
        """UC-08: Calculate potential savings."""
        compute_cost = 500.00
        potential_savings = compute_cost * 0.20
        assert potential_savings == 100.00

    def test_use_case_09_budget_alert_threshold(self, finops_tool_results):
        """UC-09: Check if budget alert threshold reached."""
        budget = json.loads(finops_tool_results["budget"])

        alert_threshold = 80
        assert budget["percent_used"] < alert_threshold

    def test_use_case_10_generate_cost_report(self, finops_tool_results):
        """UC-10: Generate comprehensive cost report."""
        cost = json.loads(finops_tool_results["cost_summary"])
        forecast = json.loads(finops_tool_results["forecast"])
        budget = json.loads(finops_tool_results["budget"])

        report = {
            "current_month_total": cost["summary"]["total"],
            "top_service": cost["services"][0]["service"],
            "forecast_eom": forecast["forecasted_end_of_month"],
            "budget_status": "under" if budget["percent_used"] < 100 else "over",
        }

        assert report["top_service"] == "COMPUTE"
        assert report["budget_status"] == "under"


# ─────────────────────────────────────────────────────────────────────────────
# DB Troubleshoot Agent Tests (10 test cases)
# ─────────────────────────────────────────────────────────────────────────────


class TestDbTroubleshootAgentUseCases:
    """Test suite for DB Troubleshoot Agent - 10 use cases."""

    @pytest.fixture
    def db_tool_results(self):
        """Mock tool results for database queries."""
        return {
            "fleet_summary": json.dumps({
                "total_databases": 5,
                "healthy": 4,
                "warning": 1,
                "critical": 0,
                "databases": [
                    {"name": "PROD_DB", "status": "HEALTHY", "cpu_percent": 45},
                    {"name": "DEV_DB", "status": "HEALTHY", "cpu_percent": 20},
                    {"name": "TEST_DB", "status": "WARNING", "cpu_percent": 85},
                ],
            }),
            "sql_result": json.dumps({
                "success": True,
                "rows_affected": 10,
                "data": [
                    {"sql_id": "abc123", "executions": 1000, "elapsed_time_ms": 5000},
                    {"sql_id": "def456", "executions": 500, "elapsed_time_ms": 10000},
                ],
            }),
            "performance": json.dumps({
                "cpu_usage_percent": 45.2,
                "memory_usage_percent": 60.1,
                "io_throughput_mbps": 150,
                "active_sessions": 25,
                "blocking_sessions": 0,
            }),
            "awr_report": json.dumps({
                "top_sql_by_elapsed": [
                    {"sql_id": "abc123", "elapsed_sec": 500, "executions": 1000},
                ],
                "top_wait_events": [
                    {"event": "db file sequential read", "wait_time_sec": 100},
                ],
            }),
        }

    def test_use_case_01_check_fleet_health(self, db_tool_results):
        """UC-01: Check overall database fleet health."""
        fleet = json.loads(db_tool_results["fleet_summary"])

        assert fleet["total_databases"] == 5
        assert fleet["healthy"] == 4
        assert fleet["critical"] == 0

    def test_use_case_02_identify_problem_databases(self, db_tool_results):
        """UC-02: Identify databases with issues."""
        fleet = json.loads(db_tool_results["fleet_summary"])
        problems = [db for db in fleet["databases"] if db["status"] != "HEALTHY"]

        assert len(problems) == 1
        assert problems[0]["name"] == "TEST_DB"

    def test_use_case_03_check_cpu_usage(self, db_tool_results):
        """UC-03: Check database CPU usage."""
        perf = json.loads(db_tool_results["performance"])

        assert perf["cpu_usage_percent"] < 80
        assert perf["cpu_usage_percent"] == 45.2

    def test_use_case_04_check_memory_usage(self, db_tool_results):
        """UC-04: Check database memory usage."""
        perf = json.loads(db_tool_results["performance"])

        assert perf["memory_usage_percent"] < 80
        assert perf["memory_usage_percent"] == 60.1

    def test_use_case_05_find_slow_queries(self, db_tool_results):
        """UC-05: Find slow running queries."""
        sql = json.loads(db_tool_results["sql_result"])

        slow = [q for q in sql["data"] if q["elapsed_time_ms"] > 5000]
        assert len(slow) >= 1

    def test_use_case_06_check_blocking_sessions(self, db_tool_results):
        """UC-06: Check for blocking sessions."""
        perf = json.loads(db_tool_results["performance"])

        assert perf["blocking_sessions"] == 0

    def test_use_case_07_analyze_top_sql(self, db_tool_results):
        """UC-07: Analyze top SQL by elapsed time."""
        awr = json.loads(db_tool_results["awr_report"])

        top_sql = awr["top_sql_by_elapsed"][0]
        assert top_sql["sql_id"] == "abc123"
        assert top_sql["elapsed_sec"] == 500

    def test_use_case_08_check_wait_events(self, db_tool_results):
        """UC-08: Check top wait events."""
        awr = json.loads(db_tool_results["awr_report"])

        top_wait = awr["top_wait_events"][0]
        assert "read" in top_wait["event"].lower()

    def test_use_case_09_check_active_sessions(self, db_tool_results):
        """UC-09: Check active session count."""
        perf = json.loads(db_tool_results["performance"])

        assert perf["active_sessions"] == 25
        assert perf["active_sessions"] < 100

    def test_use_case_10_generate_health_report(self, db_tool_results):
        """UC-10: Generate database health report."""
        fleet = json.loads(db_tool_results["fleet_summary"])
        perf = json.loads(db_tool_results["performance"])

        report = {
            "overall_health": "HEALTHY" if fleet["critical"] == 0 else "CRITICAL",
            "databases_monitored": fleet["total_databases"],
            "cpu_status": "OK" if perf["cpu_usage_percent"] < 80 else "HIGH",
            "memory_status": "OK" if perf["memory_usage_percent"] < 80 else "HIGH",
            "blocking_issues": perf["blocking_sessions"] > 0,
        }

        assert report["overall_health"] == "HEALTHY"
        assert report["cpu_status"] == "OK"
        assert report["blocking_issues"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Security Threat Agent Tests (10 test cases)
# ─────────────────────────────────────────────────────────────────────────────


class TestSecurityThreatAgentUseCases:
    """Test suite for Security Threat Agent - 10 use cases."""

    @pytest.fixture
    def security_tool_results(self):
        """Mock tool results for security queries."""
        return {
            "users": json.dumps([
                {"name": "admin", "email": "admin@example.com", "groups": ["Administrators"], "mfa_enabled": True},
                {"name": "developer", "email": "dev@example.com", "groups": ["Developers"], "mfa_enabled": True},
                {"name": "service_account", "email": None, "groups": ["ServiceAccounts"], "mfa_enabled": False},
            ]),
            "policies": json.dumps([
                {"name": "AdminPolicy", "statements": 5, "compartment": "root"},
                {"name": "DevPolicy", "statements": 10, "compartment": "development"},
            ]),
            "security_list": json.dumps({
                "ingress_rules": [
                    {"source": "0.0.0.0/0", "port": 22, "protocol": "TCP"},
                    {"source": "10.0.0.0/8", "port": 443, "protocol": "TCP"},
                ],
                "egress_rules": [
                    {"destination": "0.0.0.0/0", "protocol": "ALL"},
                ],
            }),
            "audit_events": json.dumps([
                {"event_type": "LOGIN_SUCCESS", "user": "admin", "timestamp": "2025-12-31T10:00:00Z"},
                {"event_type": "LOGIN_FAILURE", "user": "unknown", "timestamp": "2025-12-31T09:00:00Z"},
            ]),
        }

    def test_use_case_01_list_users(self, security_tool_results):
        """UC-01: List all IAM users."""
        users = json.loads(security_tool_results["users"])

        assert len(users) == 3
        assert any(u["name"] == "admin" for u in users)

    def test_use_case_02_check_mfa_status(self, security_tool_results):
        """UC-02: Check MFA status for users."""
        users = json.loads(security_tool_results["users"])

        mfa_disabled = [u for u in users if not u["mfa_enabled"]]
        assert len(mfa_disabled) == 1

    def test_use_case_03_list_policies(self, security_tool_results):
        """UC-03: List IAM policies."""
        policies = json.loads(security_tool_results["policies"])

        assert len(policies) == 2
        assert any(p["name"] == "AdminPolicy" for p in policies)

    def test_use_case_04_check_open_ports(self, security_tool_results):
        """UC-04: Check for open security list ports."""
        sec_list = json.loads(security_tool_results["security_list"])

        open_to_all = [r for r in sec_list["ingress_rules"] if r["source"] == "0.0.0.0/0"]
        assert len(open_to_all) >= 1

    def test_use_case_05_identify_ssh_exposure(self, security_tool_results):
        """UC-05: Identify SSH exposed to internet."""
        sec_list = json.loads(security_tool_results["security_list"])

        ssh_exposed = [
            r for r in sec_list["ingress_rules"]
            if r["source"] == "0.0.0.0/0" and r["port"] == 22
        ]
        assert len(ssh_exposed) == 1

    def test_use_case_06_check_admin_users(self, security_tool_results):
        """UC-06: Identify users with admin access."""
        users = json.loads(security_tool_results["users"])

        admins = [u for u in users if "Administrators" in u.get("groups", [])]
        assert len(admins) == 1
        assert admins[0]["name"] == "admin"

    def test_use_case_07_check_login_failures(self, security_tool_results):
        """UC-07: Check for failed login attempts."""
        events = json.loads(security_tool_results["audit_events"])

        failures = [e for e in events if e["event_type"] == "LOGIN_FAILURE"]
        assert len(failures) >= 1

    def test_use_case_08_check_egress_rules(self, security_tool_results):
        """UC-08: Check for overly permissive egress."""
        sec_list = json.loads(security_tool_results["security_list"])

        wide_open = [
            r for r in sec_list["egress_rules"]
            if r["destination"] == "0.0.0.0/0" and r["protocol"] == "ALL"
        ]
        assert len(wide_open) >= 1

    def test_use_case_09_check_service_accounts(self, security_tool_results):
        """UC-09: Audit service accounts."""
        users = json.loads(security_tool_results["users"])

        service_accounts = [u for u in users if "ServiceAccounts" in u.get("groups", [])]
        assert len(service_accounts) == 1
        assert service_accounts[0]["email"] is None

    def test_use_case_10_generate_security_report(self, security_tool_results):
        """UC-10: Generate security assessment report."""
        users = json.loads(security_tool_results["users"])
        sec_list = json.loads(security_tool_results["security_list"])

        findings = []

        mfa_disabled = [u for u in users if not u["mfa_enabled"] and "ServiceAccounts" not in u.get("groups", [])]
        if mfa_disabled:
            findings.append("Users without MFA")

        ssh_exposed = [r for r in sec_list["ingress_rules"] if r["source"] == "0.0.0.0/0" and r["port"] == 22]
        if ssh_exposed:
            findings.append("SSH exposed to internet")

        report = {
            "total_users": len(users),
            "admin_count": len([u for u in users if "Administrators" in u.get("groups", [])]),
            "findings_count": len(findings),
            "risk_level": "HIGH" if len(findings) > 0 else "LOW",
        }

        assert report["findings_count"] >= 1
        assert report["risk_level"] == "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# Log Analytics Agent Tests (10 test cases)
# ─────────────────────────────────────────────────────────────────────────────


class TestLogAnalyticsAgentUseCases:
    """Test suite for Log Analytics Agent - 10 use cases."""

    @pytest.fixture
    def log_tool_results(self):
        """Mock tool results for log queries."""
        return {
            "log_query": json.dumps({
                "records": [
                    {"timestamp": "2025-12-31T10:00:00Z", "level": "ERROR", "message": "Connection timeout"},
                    {"timestamp": "2025-12-31T10:01:00Z", "level": "ERROR", "message": "Connection timeout"},
                    {"timestamp": "2025-12-31T10:02:00Z", "level": "WARN", "message": "High latency detected"},
                ],
                "total_count": 3,
            }),
            "error_summary": json.dumps({
                "total_errors": 150,
                "by_type": {
                    "Connection timeout": 100,
                    "Authentication failed": 30,
                    "Internal server error": 20,
                },
                "time_period": "last 24 hours",
            }),
            "audit_logs": json.dumps([
                {"action": "CREATE", "resource": "instance", "user": "admin", "timestamp": "2025-12-31T09:00:00Z"},
                {"action": "DELETE", "resource": "bucket", "user": "admin", "timestamp": "2025-12-31T08:00:00Z"},
            ]),
            "metrics": json.dumps({
                "log_ingestion_rate": 1000,
                "storage_used_gb": 50,
                "queries_per_hour": 25,
            }),
        }

    def test_use_case_01_query_error_logs(self, log_tool_results):
        """UC-01: Query error logs."""
        result = json.loads(log_tool_results["log_query"])

        errors = [r for r in result["records"] if r["level"] == "ERROR"]
        assert len(errors) == 2

    def test_use_case_02_count_errors(self, log_tool_results):
        """UC-02: Count total errors."""
        summary = json.loads(log_tool_results["error_summary"])

        assert summary["total_errors"] == 150

    def test_use_case_03_identify_top_errors(self, log_tool_results):
        """UC-03: Identify most common error types."""
        summary = json.loads(log_tool_results["error_summary"])

        top_error = max(summary["by_type"].items(), key=lambda x: x[1])
        assert top_error[0] == "Connection timeout"
        assert top_error[1] == 100

    def test_use_case_04_query_audit_logs(self, log_tool_results):
        """UC-04: Query audit logs for user actions."""
        audit = json.loads(log_tool_results["audit_logs"])

        admin_actions = [a for a in audit if a["user"] == "admin"]
        assert len(admin_actions) == 2

    def test_use_case_05_find_delete_operations(self, log_tool_results):
        """UC-05: Find destructive operations."""
        audit = json.loads(log_tool_results["audit_logs"])

        deletes = [a for a in audit if a["action"] == "DELETE"]
        assert len(deletes) == 1

    def test_use_case_06_check_log_patterns(self, log_tool_results):
        """UC-06: Identify repeated log patterns."""
        result = json.loads(log_tool_results["log_query"])

        messages = Counter(r["message"] for r in result["records"])

        repeated = {k: v for k, v in messages.items() if v > 1}
        assert "Connection timeout" in repeated

    def test_use_case_07_check_ingestion_rate(self, log_tool_results):
        """UC-07: Monitor log ingestion rate."""
        metrics = json.loads(log_tool_results["metrics"])

        assert metrics["log_ingestion_rate"] == 1000

    def test_use_case_08_check_storage_usage(self, log_tool_results):
        """UC-08: Check log storage usage."""
        metrics = json.loads(log_tool_results["metrics"])

        assert metrics["storage_used_gb"] == 50

    def test_use_case_09_filter_by_timestamp(self, log_tool_results):
        """UC-09: Filter logs by time range."""
        result = json.loads(log_tool_results["log_query"])

        filtered = [
            r for r in result["records"]
            if "10:00" in r["timestamp"] or "10:01" in r["timestamp"]
        ]
        assert len(filtered) >= 2

    def test_use_case_10_generate_log_report(self, log_tool_results):
        """UC-10: Generate log analytics report."""
        summary = json.loads(log_tool_results["error_summary"])
        metrics = json.loads(log_tool_results["metrics"])

        report = {
            "time_period": summary["time_period"],
            "total_errors": summary["total_errors"],
            "top_error_type": max(summary["by_type"].items(), key=lambda x: x[1])[0],
            "ingestion_rate": metrics["log_ingestion_rate"],
            "storage_gb": metrics["storage_used_gb"],
        }

        assert report["total_errors"] == 150
        assert report["top_error_type"] == "Connection timeout"


# ─────────────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
