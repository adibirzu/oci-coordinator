"""
MCP Tool Catalog.

Provides unified tool discovery, lookup, and execution across all MCP servers.
Supports progressive disclosure and LangChain tool conversion.

Features:
- Automatic tool discovery from all MCP servers
- Progressive disclosure via search
- Tool tier classification for risk management
- LangChain tool conversion
- Dynamic tool/skill registration at runtime
- Per-interaction refresh for hot-reload support
- Usage tracking and statistics
- Resilience: deadletter queue, bulkhead isolation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog
import yaml
from langchain_core.tools import StructuredTool

from src.mcp.client import ToolCallResult, ToolDefinition
from src.mcp.registry import ServerRegistry
from src.resilience.deadletter import FailureType

if TYPE_CHECKING:
    from src.memory.manager import SharedMemoryManager
    from src.resilience.bulkhead import Bulkhead
    from src.resilience.deadletter import DeadLetterQueue

logger = structlog.get_logger(__name__)

# Refresh configuration
DEFAULT_REFRESH_INTERVAL_SECONDS = 30  # Minimum time between refreshes
STALE_THRESHOLD_SECONDS = 60  # Consider catalog stale after this time
CACHE_CONTEXT_ENV_KEYS = (
    "OCI_PROFILE",
    "OCI_CLI_PROFILE",
    "OCI_CONFIG_FILE",
    "OCI_REGION",
    "OCI_TENANCY_OCID",
    "COMPARTMENT_OCID",
)


@dataclass
class ToolTier:
    """Tool tier classification for progressive disclosure."""

    tier: int  # 1=instant, 2=fast, 3=moderate, 4=slow
    latency_estimate_ms: int
    risk_level: str  # none, low, medium, high
    requires_confirmation: bool = False


@dataclass
class ToolMetadata:
    """Supplemental tool metadata sourced from server manifests."""

    server_id: str
    domain: str | None = None
    read_only: bool | None = None
    idempotent: bool | None = None
    mutates: bool | None = None
    requires_confirmation: bool | None = None
    cache_ttl_seconds: int | None = None
    timeouts: dict[str, Any] | None = None
    source: str = "manifest"


# Tool tier classifications
# Tier 1: Instant (<100ms), Tier 2: Fast (100ms-1s), Tier 3: Moderate (1s-30s), Tier 4: Mutations
# ═══════════════════════════════════════════════════════════════════════════════
# Tool Aliases for Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════════════
# Maps legacy tool names to standardized oci_{domain}_{action} format
TOOL_ALIASES: dict[str, str] = {
    # database-observatory legacy names → standardized names
    "execute_sql": "oci_database_execute_sql",
    "get_schema_info": "oci_database_get_schema",
    "list_connections": "oci_database_list_connections",
    "database_status": "oci_database_get_status",
    "get_fleet_summary": "oci_opsi_get_fleet_summary",
    "search_databases": "oci_opsi_search_databases",
    "list_database_insights": "oci_opsi_list_insights",
    "analyze_cpu_usage": "oci_opsi_analyze_cpu",
    "analyze_memory_usage": "oci_opsi_analyze_memory",
    "analyze_io_usage": "oci_opsi_analyze_io",
    "get_blocking_sessions": "oci_opsi_get_blocking_sessions",
    "analyze_wait_events": "oci_opsi_analyze_wait_events",
    "get_sql_statistics": "oci_opsi_get_sql_statistics",
    "compare_awr_periods": "oci_opsi_compare_awr",
    "list_tablespaces": "oci_database_list_tablespaces",
    "list_users": "oci_database_list_users",
    "get_sql_plan": "oci_database_get_sql_plan",
    "list_awr_snapshots": "oci_database_list_awr_snapshots",
    "sqlwatch_get_plan_history": "oci_sqlwatch_get_plan_history",
    # DB Management legacy names → standardized names
    "search_managed_databases": "oci_dbmgmt_search_databases",
    "list_managed_databases": "oci_dbmgmt_list_databases",
    "get_managed_database": "oci_dbmgmt_get_database",
    "get_awr_report": "oci_dbmgmt_get_awr_report",
    "get_awr_db_report": "oci_dbmgmt_get_awr_report",
    "get_awr_db_report_auto": "oci_dbmgmt_get_awr_report_auto",
    "list_awr_db_snapshots": "oci_dbmgmt_list_awr_snapshots",
    "get_database_fleet_health": "oci_dbmgmt_get_fleet_health",
    "get_database_metrics": "oci_dbmgmt_get_metrics",
    "get_db_metrics": "oci_dbmgmt_get_metrics",
    "sqlwatch_analyze_regression": "oci_sqlwatch_analyze_regression",
    "query_warehouse_standard": "oci_opsi_query_warehouse",
    # Legacy short names
    "list_instances": "oci_compute_list_instances",
    "start_instance": "oci_compute_start_instance",
    "stop_instance": "oci_compute_stop_instance",
    "restart_instance": "oci_compute_restart_instance",
    "get_instance_metrics": "oci_observability_get_instance_metrics",
    # Agent-referenced names (from DOMAIN_PROMPTS)
    "list_autonomous_databases": "oci_database_list_autonomous",
    "analyze_performance": "oci_opsi_get_performance_summary",
    # finopsai MCP server tool aliases
    # finopsai exposes `get_cost_summary` but agents may look for `oci_cost_get_summary`
    "oci_cost_get_summary": "get_cost_summary",
    "oci_cost_summary": "get_cost_summary",
    "cost_summary": "get_cost_summary",
    "finops_get_cost_summary": "get_cost_summary",
    # finopsai multicloud tools
    "finops_cost_summary": "finops_cost_summary",
    "finops_anomalies": "finops_detect_anomalies",
    "cost_anomalies": "finops_detect_anomalies",
    "finops_commitments": "finops_list_commitments",
    "finops_optimization": "finops_rightsizing",
}

# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server to Domain Mapping
# ═══════════════════════════════════════════════════════════════════════════════
# Maps MCP servers to their supported domains for agent routing
# Server names must match config/mcp_servers.yaml
MCP_SERVER_DOMAINS: dict[str, list[str]] = {
    "oci-unified": ["identity", "compute", "network", "cost", "security", "observability", "discovery"],
    "database-observatory": ["database", "opsi", "logan", "sql", "performance", "awr", "schema", "sqlwatch"],
    "oci-infrastructure": ["compute", "network", "security", "cost", "database", "observability"],
    "finopsai": ["cost", "budget", "finops", "anomaly", "forecasting", "rightsizing"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# Domain-to-Tool Prefix Mapping
# ═══════════════════════════════════════════════════════════════════════════════
# Used for dynamic tool discovery by domain
DOMAIN_PREFIXES: dict[str, list[str]] = {
    "database": ["oci_database_", "oci_opsi_", "oci_dbmgmt_", "execute_sql", "database_status"],
    "infrastructure": ["oci_compute_", "oci_network_", "oci_list_", "oci_get_", "oci_search_"],
    "finops": ["oci_cost_", "finops_"],
    "cost": ["oci_cost_"],
    "security": ["oci_security_"],
    "cloudguard": ["oci_security_cloudguard_"],
    "observability": ["oci_observability_", "oci_logan_"],
    "identity": ["oci_list_compartments", "oci_get_compartment", "oci_search_compartments", "oci_get_tenancy"],
}

TOOL_TIERS: dict[str, ToolTier] = {
    # ═══════════════════════════════════════════════════════════════════════════
    # oci-unified & oci-infrastructure Tools
    # ═══════════════════════════════════════════════════════════════════════════
    # Tier 1: Discovery (Instant, no risk)
    "oci_ping": ToolTier(1, 50, "none"),
    "oci_list_domains": ToolTier(1, 50, "none"),
    "oci_search_tools": ToolTier(1, 100, "none"),
    "oci_get_capabilities": ToolTier(1, 50, "none"),
    "oci_get_cache_stats": ToolTier(1, 50, "none"),
    "set_feedback": ToolTier(1, 50, "none"),
    "append_feedback": ToolTier(1, 50, "none"),
    "get_feedback": ToolTier(1, 50, "none"),
    # Tier 2: Compute reads
    "oci_compute_list_instances": ToolTier(2, 500, "none"),
    "oci_compute_get_instance": ToolTier(2, 400, "none"),
    # Tier 2: Network reads
    "oci_network_list_vcns": ToolTier(2, 500, "none"),
    "oci_network_get_vcn": ToolTier(2, 400, "none"),
    "oci_network_list_subnets": ToolTier(2, 500, "none"),
    "oci_network_list_security_lists": ToolTier(2, 500, "none"),
    # Tier 2: Database reads
    "oci_database_list_autonomous": ToolTier(2, 500, "none"),
    "oci_database_get_autonomous": ToolTier(2, 400, "none"),
    "oci_database_list_db_systems": ToolTier(2, 500, "none"),
    "oci_database_list_mysql": ToolTier(2, 500, "none"),
    # Tier 2: Security reads
    "oci_security_list_users": ToolTier(2, 500, "none"),
    "oci_security_get_user": ToolTier(2, 400, "none"),
    "oci_security_list_groups": ToolTier(2, 500, "none"),
    "oci_security_list_policies": ToolTier(2, 500, "none"),
    "oci_security_list_cloud_guard_problems": ToolTier(2, 600, "none"),
    # Tier 2: Cost reads
    "oci_cost_get_summary": ToolTier(2, 800, "none"),
    "oci_cost_by_service": ToolTier(2, 800, "none"),
    # Tier 2: Observability reads
    "oci_observability_list_alarms": ToolTier(2, 500, "none"),
    "oci_observability_get_alarm_history": ToolTier(2, 600, "none"),
    "oci_observability_list_log_sources": ToolTier(2, 500, "none"),
    # Tier 3: Moderate operations (analysis, heavy queries)
    "oci_observability_get_instance_metrics": ToolTier(3, 2000, "low"),
    "oci_observability_execute_log_query": ToolTier(3, 3000, "low"),
    "oci_observability_overview": ToolTier(3, 3000, "low"),
    "oci_network_analyze_security": ToolTier(3, 2000, "low"),
    "oci_security_audit": ToolTier(3, 5000, "low"),
    "oci_cost_detect_anomalies": ToolTier(3, 3000, "low"),
    "oci_database_get_awr_report": ToolTier(3, 5000, "low"),
    "oci_skill_troubleshoot_instance": ToolTier(3, 10000, "low"),
    # Tier 4: Mutations (require confirmation)
    "oci_compute_start_instance": ToolTier(4, 10000, "medium", True),
    "oci_compute_stop_instance": ToolTier(4, 10000, "high", True),
    "oci_compute_restart_instance": ToolTier(4, 10000, "medium", True),
    "oci_database_scale": ToolTier(4, 15000, "high", True),
    # Tool aliases (backward compatibility)
    "list_instances": ToolTier(2, 500, "none"),
    "start_instance": ToolTier(4, 10000, "medium", True),
    "stop_instance": ToolTier(4, 10000, "high", True),
    "restart_instance": ToolTier(4, 10000, "medium", True),
    "get_instance_metrics": ToolTier(3, 2000, "low"),
    # ═══════════════════════════════════════════════════════════════════════════
    # Security Tools (oci_security_ prefix)
    # ═══════════════════════════════════════════════════════════════════════════
    # Tier 1: Health & Discovery
    "oci_security_ping": ToolTier(1, 50, "none"),
    "oci_security_health": ToolTier(1, 100, "none"),
    # Tier 2: Cloud Guard
    "oci_security_cloudguard_list_problems": ToolTier(2, 600, "none"),
    "oci_security_cloudguard_get_problem": ToolTier(2, 400, "none"),
    "oci_security_cloudguard_list_detectors": ToolTier(2, 500, "none"),
    "oci_security_cloudguard_list_responders": ToolTier(2, 500, "none"),
    "oci_security_cloudguard_get_security_score": ToolTier(2, 600, "none"),
    "oci_security_cloudguard_list_recommendations": ToolTier(2, 600, "none"),
    # Tier 2: Vulnerability Scanning
    "oci_security_vss_list_host_scans": ToolTier(2, 600, "none"),
    "oci_security_vss_get_host_scan": ToolTier(2, 400, "none"),
    "oci_security_vss_list_container_scans": ToolTier(2, 600, "none"),
    "oci_security_vss_list_vulnerabilities": ToolTier(2, 800, "none"),
    # Tier 2: Security Zones
    "oci_security_zones_list": ToolTier(2, 500, "none"),
    "oci_security_zones_get": ToolTier(2, 400, "none"),
    "oci_security_zones_list_policies": ToolTier(2, 500, "none"),
    # Tier 2: Bastion
    "oci_security_bastion_list": ToolTier(2, 500, "none"),
    "oci_security_bastion_get": ToolTier(2, 400, "none"),
    "oci_security_bastion_list_sessions": ToolTier(2, 500, "none"),
    # Tier 2: Data Safe
    "oci_security_datasafe_list_targets": ToolTier(2, 500, "none"),
    "oci_security_datasafe_list_assessments": ToolTier(2, 600, "none"),
    "oci_security_datasafe_get_assessment": ToolTier(2, 400, "none"),
    "oci_security_datasafe_list_findings": ToolTier(2, 600, "none"),
    # Tier 2: WAF
    "oci_security_waf_list_firewalls": ToolTier(2, 500, "none"),
    "oci_security_waf_get_firewall": ToolTier(2, 400, "none"),
    "oci_security_waf_list_policies": ToolTier(2, 500, "none"),
    "oci_security_waf_get_policy": ToolTier(2, 400, "none"),
    # Tier 2: Audit
    "oci_security_audit_list_events": ToolTier(2, 800, "none"),
    "oci_security_audit_get_configuration": ToolTier(2, 400, "none"),
    # Tier 2: Access Governance
    "oci_security_accessgov_list_instances": ToolTier(2, 500, "none"),
    "oci_security_accessgov_get_instance": ToolTier(2, 400, "none"),
    # Tier 2: KMS
    "oci_security_kms_list_vaults": ToolTier(2, 500, "none"),
    "oci_security_kms_get_vault": ToolTier(2, 400, "none"),
    "oci_security_kms_list_keys": ToolTier(2, 500, "none"),
    "oci_security_kms_get_key": ToolTier(2, 400, "none"),
    # Tier 3: Security Skills
    "oci_security_skill_posture_summary": ToolTier(3, 5000, "low"),
    "oci_security_skill_vulnerability_overview": ToolTier(3, 5000, "low"),
    "oci_security_skill_audit_digest": ToolTier(3, 3000, "low"),
    # Tier 4: Security Mutations
    "oci_security_cloudguard_remediate_problem": ToolTier(4, 5000, "medium", True),
    "oci_security_bastion_terminate_session": ToolTier(4, 3000, "high", True),
    # ═══════════════════════════════════════════════════════════════════════════
    # finopsai Tools (oci_cost_ prefix)
    # ═══════════════════════════════════════════════════════════════════════════
    # Tier 1: Health & Discovery
    "oci_cost_ping": ToolTier(1, 50, "none"),
    "oci_cost_templates": ToolTier(1, 50, "none"),
    # Tier 2: Cost Analysis
    "oci_cost_by_compartment": ToolTier(2, 800, "none"),
    "oci_cost_service_drilldown": ToolTier(2, 1000, "none"),
    "oci_cost_by_tag": ToolTier(2, 800, "none"),
    "oci_cost_monthly_trend": ToolTier(2, 800, "none"),
    "oci_cost_budget_status": ToolTier(2, 600, "none"),
    "oci_cost_object_storage": ToolTier(2, 800, "none"),
    "oci_cost_unit_cost": ToolTier(2, 800, "none"),
    "oci_cost_forecast_credits": ToolTier(2, 800, "none"),
    # Tier 3: Heavy Analysis
    "oci_cost_focus_health": ToolTier(3, 3000, "low"),
    "oci_cost_spikes": ToolTier(3, 2000, "low"),
    # Tier 4: Schedule Management (CREATE requires confirmation)
    "oci_cost_schedules": ToolTier(2, 600, "none"),  # LIST is safe, CREATE checks action param
    # ═══════════════════════════════════════════════════════════════════════════
    # database-observatory Tools
    # ═══════════════════════════════════════════════════════════════════════════
    # Tier 2: SQLcl & Database
    "execute_sql": ToolTier(3, 5000, "medium"),  # SQL execution is moderate risk
    "get_schema_info": ToolTier(2, 1000, "none"),
    "list_connections": ToolTier(2, 500, "none"),
    "database_status": ToolTier(2, 500, "none"),
    # Tier 2: OPSI Discovery
    "get_fleet_summary": ToolTier(2, 800, "none"),
    "search_databases": ToolTier(2, 600, "none"),
    "list_database_insights": ToolTier(2, 600, "none"),
    # Tier 3: OPSI Analysis
    "analyze_cpu_usage": ToolTier(3, 2000, "low"),
    "analyze_memory_usage": ToolTier(3, 2000, "low"),
    "analyze_io_usage": ToolTier(3, 2000, "low"),
    "query_warehouse_standard": ToolTier(3, 5000, "low"),
    # ═══════════════════════════════════════════════════════════════════════════
    # OCI Database Management (DB Mgmt) Tools - Standardized Names
    # ═══════════════════════════════════════════════════════════════════════════
    # Tier 2: Database Discovery & Metadata
    "oci_dbmgmt_list_databases": ToolTier(2, 1000, "none"),
    "oci_dbmgmt_get_database": ToolTier(2, 600, "none"),
    "oci_dbmgmt_search_databases": ToolTier(2, 800, "none"),
    "oci_dbmgmt_get_fleet_health": ToolTier(2, 1500, "none"),
    # Tier 3: AWR Analysis (can take longer)
    "oci_dbmgmt_list_awr_snapshots": ToolTier(2, 800, "none"),
    "oci_dbmgmt_get_awr_report": ToolTier(3, 5000, "low"),  # AWR reports can take time
    "oci_dbmgmt_get_awr_report_auto": ToolTier(3, 5000, "low"),  # Auto-finds snapshots
    "oci_dbmgmt_get_metrics": ToolTier(2, 1000, "none"),
    "oci_dbmgmt_get_top_sql": ToolTier(3, 2000, "low"),
    "oci_dbmgmt_get_wait_events": ToolTier(3, 2000, "low"),
    # ═══════════════════════════════════════════════════════════════════════════
    # OPSI Tools (database-observatory)
    # ═══════════════════════════════════════════════════════════════════════════
    # Tier 2: Database Management
    "list_tablespaces": ToolTier(2, 500, "none"),
    "list_users": ToolTier(2, 500, "none"),
    "get_sql_plan": ToolTier(2, 600, "none"),
    "list_awr_snapshots": ToolTier(2, 600, "none"),
    # Tier 3: OPSI Diagnostics
    "get_sql_statistics": ToolTier(3, 2000, "low"),
    "analyze_wait_events": ToolTier(3, 2000, "low"),
    "get_blocking_sessions": ToolTier(3, 2000, "low"),
    "compare_awr_periods": ToolTier(3, 5000, "low"),
    # Tier 3: SQLWatch
    "sqlwatch_get_plan_history": ToolTier(3, 2000, "low"),
    "sqlwatch_analyze_regression": ToolTier(3, 3000, "low"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Catalog Config Overrides (config/catalog/*)
# ═══════════════════════════════════════════════════════════════════════════════

_CATALOG_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "catalog"


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return cast("dict[str, Any]", data)
        return None
    except Exception as exc:
        logger.warning("Failed to load JSON config", path=str(path), error=str(exc))
        return None


def _load_yaml_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Failed to load YAML config", path=str(path), error=str(exc))
        return None


def _coerce_tool_tiers(raw: dict[str, Any]) -> dict[str, ToolTier]:
    converted: dict[str, ToolTier] = {}
    for name, entry in raw.items():
        if isinstance(entry, ToolTier):
            converted[name] = entry
            continue
        if not isinstance(entry, dict):
            continue
        tier = int(entry.get("tier", 2))
        latency = int(entry.get("latency_estimate_ms", 1000))
        risk = str(entry.get("risk_level", "low"))
        requires = bool(entry.get("requires_confirmation", False))
        converted[name] = ToolTier(
            tier=tier,
            latency_estimate_ms=latency,
            risk_level=risk,
            requires_confirmation=requires,
        )
    return converted


def _load_catalog_config() -> None:
    """Load catalog overrides from config/catalog if present."""
    global TOOL_ALIASES, DOMAIN_PREFIXES, MCP_SERVER_DOMAINS, TOOL_TIERS

    aliases = _load_json_file(_CATALOG_CONFIG_DIR / "tool_aliases.json")
    if aliases:
        TOOL_ALIASES = aliases

    prefixes = _load_json_file(_CATALOG_CONFIG_DIR / "domain_prefixes.json")
    if prefixes:
        DOMAIN_PREFIXES = prefixes

    server_domains = _load_json_file(_CATALOG_CONFIG_DIR / "server_domains.json")
    if server_domains:
        MCP_SERVER_DOMAINS = server_domains

    tiers_raw = _load_yaml_file(_CATALOG_CONFIG_DIR / "tool_tiers.yaml")
    if tiers_raw:
        TOOL_TIERS = _coerce_tool_tiers(tiers_raw)


_load_catalog_config()


def _manifest_overrides_enabled() -> bool:
    return os.getenv("COORDINATOR_USE_MANIFEST_OVERRIDES", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


def _tier_key(name: str) -> str:
    """Normalize tool names for tier lookups."""
    if ":" in name:
        name = name.split(":", 1)[1]
    if "__" in name:
        name = name.split("__", 1)[1]
    return name


class ToolCatalog:
    """
    Unified catalog of tools from all MCP servers.

    Features:
    - Automatic tool discovery from all servers
    - Progressive disclosure via search
    - Tool tier classification
    - LangChain tool conversion
    - Tool execution routing
    - Dynamic tool registration at runtime
    - Tool lifecycle management with events
    - Hot-reload support for MCP servers
    """

    _instance: ToolCatalog | None = None

    def __init__(
        self,
        registry: ServerRegistry | None = None,
        deadletter_queue: DeadLetterQueue | None = None,
        bulkhead: Bulkhead | None = None,
        memory_manager: SharedMemoryManager | None = None,
    ):
        self._registry = registry or ServerRegistry.get_instance()
        self._tools: dict[str, ToolDefinition] = {}
        self._tool_to_server: dict[str, str] = {}
        self._custom_tools: dict[str, ToolDefinition] = {}  # Dynamically registered
        self._tool_callbacks: dict[str, list[Callable[..., Any]]] = {}  # Event callbacks
        self._tool_usage_stats: dict[str, dict[str, Any]] = {}  # Usage tracking
        self._tool_metadata: dict[str, ToolMetadata] = {}  # Manifest metadata
        self._refresh_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        self._last_refresh: datetime | None = None
        self._refresh_interval = timedelta(seconds=DEFAULT_REFRESH_INTERVAL_SECONDS)
        self._stale_threshold = timedelta(seconds=STALE_THRESHOLD_SECONDS)
        self._logger = logger.bind(component="ToolCatalog")

        # Resilience components (optional)
        self._deadletter_queue = deadletter_queue
        self._bulkhead = bulkhead
        self._memory_manager = memory_manager

    @classmethod
    def get_instance(
        cls,
        registry: ServerRegistry | None = None,
        memory_manager: SharedMemoryManager | None = None,
    ) -> ToolCatalog:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(registry, memory_manager=memory_manager)
        elif memory_manager is not None:
            cls._instance.set_memory_manager(memory_manager)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def set_memory_manager(self, memory_manager: SharedMemoryManager | None) -> None:
        """Attach a shared memory manager for tool result caching."""
        self._memory_manager = memory_manager
        if memory_manager:
            self._logger.debug("Tool catalog memory manager attached")

    async def refresh(self) -> int:
        """
        Refresh tools from all connected servers.

        Returns:
            Number of tools discovered
        """
        # Use lock if available to prevent concurrent refreshes
        if self._refresh_lock:
            async with self._refresh_lock:
                return await self._do_refresh()
        return await self._do_refresh()

    async def _do_refresh(self) -> int:
        """Internal refresh implementation."""
        self._tools.clear()
        self._tool_to_server.clear()
        self._tool_metadata.clear()

        all_tools = self._registry.get_all_tools()

        for name, tool_def in all_tools.items():
            self._tools[name] = tool_def
            self._tool_to_server[name] = tool_def.server_id

        # Re-add custom tools
        for name, tool_def in self._custom_tools.items():
            self._tools[name] = tool_def
            self._tool_to_server[name] = tool_def.server_id

        # Update refresh timestamp
        self._last_refresh = datetime.utcnow()

        # Apply manifest overrides (tier/risk/TTL)
        await self._refresh_manifest_overrides()

        # Trigger refresh callbacks
        await self._trigger_event("refresh", {"tool_count": len(self._tools)})

        self._logger.info("Tool catalog refreshed", tool_count=len(self._tools))
        return len(self._tools)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-Interaction Refresh (Hot-Reload Support)
    # ─────────────────────────────────────────────────────────────────────────

    async def ensure_fresh(self, force: bool = False) -> bool:
        """
        Ensure the catalog is fresh before an interaction.

        Call this at the start of each agent interaction to ensure tools
        are up-to-date. Uses smart caching to avoid unnecessary refreshes.

        Args:
            force: Force refresh regardless of cache state

        Returns:
            True if catalog was refreshed, False if cache was used

        Example:
            # At the start of each agent invocation
            await catalog.ensure_fresh()
            tools = catalog.list_tools()
        """
        now = datetime.utcnow()

        # Always refresh if forced or never refreshed
        if force or self._last_refresh is None:
            await self.refresh()
            return True

        # Check if catalog is stale
        age = now - self._last_refresh
        if age > self._stale_threshold:
            self._logger.debug(
                "Catalog stale, refreshing",
                age_seconds=age.total_seconds(),
                threshold=self._stale_threshold.total_seconds(),
            )
            await self.refresh()
            return True

        # Check if registry has newer tools (invalidated cache)
        if not self._registry._tool_cache_time:
            self._logger.debug("Registry cache invalidated, refreshing catalog")
            await self.refresh()
            return True

        return False

    def is_stale(self) -> bool:
        """
        Check if the catalog is stale and needs refresh.

        Returns:
            True if catalog should be refreshed
        """
        if self._last_refresh is None:
            return True

        age = datetime.utcnow() - self._last_refresh
        return age > self._stale_threshold

    def set_refresh_interval(self, seconds: int) -> None:
        """
        Set the minimum interval between refreshes.

        Args:
            seconds: Minimum seconds between refreshes
        """
        self._refresh_interval = timedelta(seconds=seconds)
        self._logger.debug("Refresh interval updated", seconds=seconds)

    def set_stale_threshold(self, seconds: int) -> None:
        """
        Set when the catalog is considered stale.

        Args:
            seconds: Seconds after which catalog is stale
        """
        self._stale_threshold = timedelta(seconds=seconds)
        self._logger.debug("Stale threshold updated", seconds=seconds)

    async def refresh_if_needed(self) -> bool:
        """
        Refresh only if enough time has passed since last refresh.

        This is a more aggressive caching strategy than ensure_fresh().

        Returns:
            True if refreshed, False if skipped
        """
        now = datetime.utcnow()

        if self._last_refresh is None:
            await self.refresh()
            return True

        if now - self._last_refresh < self._refresh_interval:
            return False

        await self.refresh()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamic Tool Registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
        server_id: str = "custom",
        tier: int = 2,
        risk_level: str = "low",
    ) -> ToolDefinition:
        """
        Register a custom tool at runtime.

        Args:
            name: Tool name (must be unique)
            description: Tool description
            input_schema: JSON schema for tool inputs
            handler: Async function to handle tool calls
            server_id: Virtual server ID for the tool
            tier: Tool tier (1-4)
            risk_level: Risk level (none, low, medium, high)

        Returns:
            Created ToolDefinition

        Example:
            async def my_handler(args: dict) -> str:
                return f"Processed: {args}"

            catalog.register_tool(
                name="my_custom_tool",
                description="Does something custom",
                input_schema={"type": "object", "properties": {...}},
                handler=my_handler,
            )
        """
        if name in self._tools:
            self._logger.warning("Overwriting existing tool", name=name)

        tool_def = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            server_id=server_id,
        )

        # Store handler reference
        tool_def._handler = handler  # type: ignore[attr-defined]

        # Add to custom tools and main catalog
        self._custom_tools[name] = tool_def
        self._tools[name] = tool_def
        self._tool_to_server[name] = server_id

        # Add tier info
        tier_key = _tier_key(name)
        TOOL_TIERS[tier_key] = ToolTier(tier, 1000, risk_level)
        if tier_key != name:
            TOOL_TIERS[name] = TOOL_TIERS[tier_key]

        self._logger.info(
            "Custom tool registered",
            name=name,
            server_id=server_id,
            tier=tier,
        )

        return tool_def

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a custom tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._custom_tools:
            self._logger.warning("Tool not found in custom tools", name=name)
            return False

        del self._custom_tools[name]
        if name in self._tools:
            del self._tools[name]
        if name in self._tool_to_server:
            del self._tool_to_server[name]
        TOOL_TIERS.pop(name, None)
        TOOL_TIERS.pop(_tier_key(name), None)

        self._logger.info("Custom tool unregistered", name=name)
        return True

    def update_tool_tier(
        self,
        name: str,
        tier: int,
        latency_estimate_ms: int = 1000,
        risk_level: str = "low",
        requires_confirmation: bool = False,
    ) -> None:
        """
        Update the tier classification for a tool.

        Args:
            name: Tool name
            tier: New tier (1-4)
            latency_estimate_ms: Expected latency
            risk_level: Risk level
            requires_confirmation: Whether tool needs user confirmation
        """
        tier_key = _tier_key(name)
        TOOL_TIERS[tier_key] = ToolTier(
            tier=tier,
            latency_estimate_ms=latency_estimate_ms,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
        )
        if tier_key != name:
            TOOL_TIERS[name] = TOOL_TIERS[tier_key]
        self._logger.debug("Tool tier updated", name=name, tier=tier)

    # ─────────────────────────────────────────────────────────────────────────
    # Event System
    # ─────────────────────────────────────────────────────────────────────────

    def on_event(self, event: str, callback: Callable[..., Any]) -> None:
        """
        Register a callback for catalog events.

        Events:
        - refresh: Tool catalog was refreshed
        - tool_called: A tool was executed
        - tool_registered: New tool registered
        - tool_error: Tool execution failed

        Args:
            event: Event name
            callback: Async or sync callback function
        """
        if event not in self._tool_callbacks:
            self._tool_callbacks[event] = []
        self._tool_callbacks[event].append(callback)

    async def _refresh_manifest_overrides(self) -> None:
        """Load tier/risk/TTL overrides from server manifests."""
        if not _manifest_overrides_enabled():
            self._logger.debug(
                "Manifest overrides disabled",
                env="COORDINATOR_USE_MANIFEST_OVERRIDES",
            )
            return

        for server_id in self._registry.list_connected():
            client = self._registry.get_client(server_id)
            if not client or not client.resources:
                continue

            if "server://manifest" not in client.resources:
                continue

            try:
                raw = await client.read_resource("server://manifest")
                if not raw:
                    continue
                manifest = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as exc:
                self._logger.warning(
                    "Failed to read manifest",
                    server_id=server_id,
                    error=str(exc),
                )
                continue

            tools = manifest.get("tools", [])
            if not isinstance(tools, list):
                continue

            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                name = tool.get("name")
                if not name:
                    continue

                cache_ttl = tool.get("cache_ttl_seconds")
                if cache_ttl is not None:
                    try:
                        cache_ttl = int(cache_ttl)
                    except (TypeError, ValueError):
                        cache_ttl = None

                timeouts = tool.get("timeouts") if isinstance(tool.get("timeouts"), dict) else None

                metadata = ToolMetadata(
                    server_id=server_id,
                    domain=tool.get("domain"),
                    read_only=tool.get("read_only"),
                    idempotent=tool.get("idempotent"),
                    mutates=tool.get("mutates"),
                    requires_confirmation=tool.get("requires_confirmation"),
                    cache_ttl_seconds=cache_ttl,
                    timeouts=timeouts,
                )

                self._tool_metadata[name] = metadata
                self._tool_metadata[f"{server_id}:{name}"] = metadata

                tier_value = tool.get("tier")
                if tier_value is not None:
                    tier_info = ToolTier(
                        tier=int(tier_value),
                        latency_estimate_ms=int(tool.get("latency_ms", 1000)),
                        risk_level=str(tool.get("risk", "low")),
                        requires_confirmation=bool(tool.get("requires_confirmation", False)),
                    )
                    TOOL_TIERS[name] = tier_info
                    TOOL_TIERS[f"{server_id}:{name}"] = tier_info

                aliases = tool.get("aliases") or []
                if isinstance(aliases, list):
                    for alias in aliases:
                        if alias and alias not in TOOL_ALIASES:
                            TOOL_ALIASES[alias] = name

                if timeouts:
                    timeout_seconds = timeouts.get("default_seconds") or timeouts.get("timeout_seconds")
                    if timeout_seconds is not None:
                        try:
                            client.config.tool_timeouts[name] = int(timeout_seconds)
                        except (TypeError, ValueError):
                            self._logger.debug(
                                "Invalid manifest timeout value",
                                tool=name,
                                server_id=server_id,
                                timeout_value=timeout_seconds,
                            )

    async def _trigger_event(self, event: str, data: dict[str, Any]) -> None:
        """Trigger callbacks for an event."""
        callbacks = self._tool_callbacks.get(event, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                self._logger.error(
                    "Event callback failed",
                    event=event,
                    error=str(e),
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Usage Tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _track_usage(
        self,
        tool_name: str,
        success: bool,
        duration_ms: int,
    ) -> None:
        """Track tool usage statistics."""
        if tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[tool_name] = {
                "call_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_duration_ms": 0,
                "last_called": None,
            }

        stats = self._tool_usage_stats[tool_name]
        stats["call_count"] += 1
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        stats["total_duration_ms"] += duration_ms
        stats["last_called"] = time.time()

    def get_usage_stats(self, tool_name: str | None = None) -> dict[str, Any]:
        """
        Get tool usage statistics.

        Args:
            tool_name: Specific tool or None for all tools

        Returns:
            Usage statistics dictionary
        """
        if tool_name:
            stats = self._tool_usage_stats.get(tool_name, {})
            if stats and stats["call_count"] > 0:
                stats["avg_duration_ms"] = (
                    stats["total_duration_ms"] / stats["call_count"]
                )
                stats["success_rate"] = (
                    stats["success_count"] / stats["call_count"]
                )
            return stats

        # Return all stats with calculated fields
        all_stats = {}
        for name, stats in self._tool_usage_stats.items():
            stat_copy = dict(stats)
            if stat_copy["call_count"] > 0:
                stat_copy["avg_duration_ms"] = (
                    stat_copy["total_duration_ms"] / stat_copy["call_count"]
                )
                stat_copy["success_rate"] = (
                    stat_copy["success_count"] / stat_copy["call_count"]
                )
            all_stats[name] = stat_copy

        return all_stats

    def get_tool_metadata(self, tool_name: str) -> ToolMetadata | None:
        """Get tool metadata sourced from manifests."""
        if tool_name in self._tool_metadata:
            return self._tool_metadata[tool_name]
        return self._tool_metadata.get(_tier_key(tool_name))

    def _tool_matches_domain(
        self,
        tool_name: str,
        tool_def: ToolDefinition,
        domain: str,
    ) -> bool:
        domain_lower = domain.lower()
        metadata = self._resolve_metadata_for_tool(tool_name, tool_def)
        if metadata and metadata.domain:
            return metadata.domain.lower() == domain_lower

        prefixes = DOMAIN_PREFIXES.get(domain_lower, [])
        if prefixes:
            return any(
                tool_name.startswith(prefix) or tool_def.name.startswith(prefix)
                for prefix in prefixes
            )

        return tool_def.name.startswith(f"oci_{domain_lower}_")

    def _schema_supports_param(self, schema: dict[str, Any] | None, param: str) -> bool:
        """Return True if JSON schema defines the given parameter."""
        if not isinstance(schema, dict):
            return False

        props = schema.get("properties")
        if isinstance(props, dict) and param in props:
            return True

        for key in ("anyOf", "oneOf", "allOf"):
            variants = schema.get(key)
            if isinstance(variants, list):
                for variant in variants:
                    if self._schema_supports_param(variant, param):
                        return True

        return False

    def _resolve_metadata_for_tool(
        self,
        tool_name: str,
        tool_def: ToolDefinition | None = None,
    ) -> ToolMetadata | None:
        candidates: list[str] = []
        if tool_def:
            candidates.append(f"{tool_def.server_id}:{tool_def.name}")
            candidates.append(tool_def.name)
        candidates.append(tool_name)

        for candidate in candidates:
            metadata = self.get_tool_metadata(candidate)
            if metadata:
                return metadata
        return None

    def _hash_arguments(self, arguments: dict[str, Any]) -> str:
        payload = json.dumps(
            arguments or {},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _tool_cache_context(self, tool_def: ToolDefinition) -> dict[str, Any]:
        server_info = self._registry._servers.get(tool_def.server_id)
        if not server_info:
            return {}

        env = server_info.config.env or {}
        context = {
            key: value
            for key in CACHE_CONTEXT_ENV_KEYS
            if (value := env.get(key))
        }

        if server_info.config.url:
            context["server_url"] = server_info.config.url

        return context

    def _tool_cache_key(self, tool_def: ToolDefinition) -> str:
        context = self._tool_cache_context(tool_def)
        if not context:
            return f"{tool_def.server_id}:{tool_def.name}"
        return f"{tool_def.server_id}:{tool_def.name}:{self._hash_arguments(context)}"

    def _cache_ttl_seconds(self, metadata: ToolMetadata | None) -> int | None:
        if not metadata or metadata.cache_ttl_seconds is None:
            return None

        ttl = int(metadata.cache_ttl_seconds)
        if ttl <= 0:
            return None

        if metadata.mutates is True or metadata.requires_confirmation is True:
            return None

        if metadata.mutates is False:
            return ttl

        if metadata.read_only is True or metadata.idempotent is True:
            return ttl

        return None

    def get_tool(self, tool_name: str) -> ToolDefinition | None:
        """Get a tool definition by name.

        Supports:
        - Exact match
        - Alias resolution (legacy names → standard names)
        - Namespace prefix stripping
        """
        # Try exact match first
        if tool_name in self._tools:
            return self._tools[tool_name]

        # Try alias resolution (legacy → standard name)
        if tool_name in TOOL_ALIASES:
            canonical_name = TOOL_ALIASES[tool_name]
            if canonical_name in self._tools:
                return self._tools[canonical_name]
            # Also try finding canonical name with different casing/prefixes
            for name, tool_def in self._tools.items():
                if name.endswith(canonical_name.split("_")[-1]) or canonical_name in name:
                    return tool_def

        # Try without namespace prefix
        for name, tool_def in self._tools.items():
            if name.endswith(f":{tool_name}") or name == tool_name:
                return tool_def

        # Try partial match for flexibility
        tool_lower = tool_name.lower()
        for name, tool_def in self._tools.items():
            if tool_lower in name.lower():
                return tool_def

        return None

    def get_tools_for_domain(self, domain: str) -> list[ToolDefinition]:
        """Get all tools for a specific domain.

        Args:
            domain: Domain name (database, infrastructure, finops, etc.)

        Returns:
            List of tools matching the domain
        """
        matching_tools = []
        for name, tool_def in self._tools.items():
            if ":" in name:
                continue
            if self._tool_matches_domain(name, tool_def, domain):
                matching_tools.append(tool_def)

        self._logger.debug(
            "Tools for domain",
            domain=domain,
            count=len(matching_tools),
        )
        return matching_tools

    def get_tool_domain(self, tool_name: str) -> str | None:
        """Resolve a tool's domain using manifest metadata or naming."""
        tool_def = self.get_tool(tool_name)
        if not tool_def:
            return None

        metadata = self._resolve_metadata_for_tool(tool_def.name, tool_def)
        if metadata and metadata.domain:
            return metadata.domain

        for domain, prefixes in DOMAIN_PREFIXES.items():
            if any(
                tool_def.name.startswith(prefix) or tool_name.startswith(prefix)
                for prefix in prefixes
            ):
                return domain

        if tool_def.name.startswith("oci_"):
            parts = tool_def.name.split("_")
            if len(parts) >= 3:
                return parts[1]

        return None

    def resolve_alias(self, tool_name: str) -> str:
        """Resolve a tool alias to its canonical name.

        Args:
            tool_name: Possibly aliased tool name

        Returns:
            Canonical tool name (or original if not aliased)
        """
        return TOOL_ALIASES.get(tool_name, tool_name)

    def get_server_for_tool(self, tool_name: str) -> str | None:
        """Get the server ID that provides a tool."""
        tool = self.get_tool(tool_name)
        return tool.server_id if tool else None

    def list_tools(self) -> list[ToolDefinition]:
        """List all available tools."""
        return list(self._tools.values())

    def search_tools(
        self,
        query: str | None = None,
        domain: str | None = None,
        max_tier: int = 4,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search for tools with progressive disclosure.

        This is the primary tool discovery mechanism, implementing
        progressive disclosure to avoid overwhelming the LLM.

        Args:
            query: Search query (matches name, description)
            domain: Filter by domain (compute, database, etc.)
            max_tier: Maximum tier to include (1-4)
            limit: Maximum results to return

        Returns:
            List of matching tools with metadata
        """
        results = []

        for name, tool_def in self._tools.items():
            # Skip namespaced duplicates
            if ":" in name and name.split(":")[1] in self._tools:
                continue

            # Apply filters
            if query:
                query_lower = query.lower()
                if (
                    query_lower not in name.lower()
                    and query_lower not in tool_def.description.lower()
                ):
                    continue

            if domain:
                if not self._tool_matches_domain(name, tool_def, domain):
                    continue

            # Get tier info
            tier_info = TOOL_TIERS.get(_tier_key(name), ToolTier(2, 1000, "low"))
            if tier_info.tier > max_tier:
                continue

            results.append({
                "name": name,
                "description": tool_def.description,
                "server": tool_def.server_id,
                "tier": tier_info.tier,
                "latency_estimate": f"{tier_info.latency_estimate_ms}ms",
                "risk_level": tier_info.risk_level,
                "requires_confirmation": tier_info.requires_confirmation,
            })

            if len(results) >= limit:
                break

        # Sort by tier, then name
        results.sort(key=lambda x: (x["tier"], x["name"]))

        self._logger.debug(
            "Tool search",
            query=query,
            domain=domain,
            results=len(results),
        )

        return results

    def get_domain_summary(self) -> dict[str, list[str]]:
        """
        Get summary of tools by domain.

        Returns:
            Dictionary of domain -> list of tool names
        """
        domains: dict[str, list[str]] = {}

        for name, tool_def in self._tools.items():
            if ":" in name:
                continue
            domain = self.get_tool_domain(name)
            if not domain:
                continue
            domains.setdefault(domain, []).append(name)

        return domains

    async def execute(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """
        Execute a tool on its server.

        Includes resilience patterns:
        - Circuit breaker to prevent cascading failures
        - Bulkhead for resource isolation
        - Deadletter queue for failed operations

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolCallResult with execution result
        """
        start_time = time.time()

        # Debug logging for tool execution
        self._logger.debug(
            "Tool execute called",
            tool_name=tool_name,
            arguments=str(arguments)[:500],
            argument_keys=list(arguments.keys()) if arguments else [],
        )

        tool_def = self.get_tool(tool_name)
        if tool_def:
            self._logger.debug(
                "Tool resolved",
                tool_name=tool_name,
                server_id=tool_def.server_id,
            )
        if not tool_def:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
            )

        cache_ttl_seconds = None
        cache_key = None
        args_hash = None

        if self._memory_manager:
            metadata = self._resolve_metadata_for_tool(tool_name, tool_def)
            cache_ttl_seconds = self._cache_ttl_seconds(metadata)
            if cache_ttl_seconds:
                cache_key = self._tool_cache_key(tool_def)
                args_hash = self._hash_arguments(arguments or {})
                try:
                    cached_result = await self._memory_manager.get_tool_result(
                        cache_key, args_hash
                    )
                except Exception as exc:
                    self._logger.warning(
                        "Tool cache read failed",
                        tool=tool_name,
                        error=str(exc),
                    )
                else:
                    if cached_result is not None:
                        duration_ms = int((time.time() - start_time) * 1000)
                        self._track_usage(tool_name, True, duration_ms)
                        await self._trigger_event("tool_called", {
                            "tool": tool_name,
                            "success": True,
                            "duration_ms": duration_ms,
                            "cached": True,
                        })
                        return ToolCallResult(
                            tool_name=tool_name,
                            success=True,
                            result=cached_result,
                            duration_ms=duration_ms,
                        )

        # ─────────────────────────────────────────────────────────────────────
        # Bulkhead Acquisition
        # ─────────────────────────────────────────────────────────────────────
        # Acquire bulkhead slot for resource isolation
        # Timeout reduced from 30s to 10s - fail fast, don't cascade delays
        bulkhead_handle = None
        if self._bulkhead:
            partition = self._bulkhead.get_partition_for_tool(tool_name)
            try:
                bulkhead_handle = await self._bulkhead.acquire(partition, timeout=10.0)
            except TimeoutError:
                # Check if this is a read-only operation (list, get, search, etc.)
                # Read-only ops can proceed without isolation as a graceful degradation
                read_only_prefixes = ("list_", "get_", "search_", "analyze_", "describe_")
                is_read_only = any(
                    tool_name.split("_", 2)[-1].startswith(prefix)
                    for prefix in read_only_prefixes
                ) or tool_name.endswith("_list") or "_get_" in tool_name

                if is_read_only:
                    self._logger.warning(
                        "Bulkhead timeout, proceeding without isolation (read-only)",
                        tool=tool_name,
                        partition=partition.value,
                    )
                else:
                    self._logger.warning(
                        "Bulkhead timeout",
                        tool=tool_name,
                        partition=partition.value,
                    )
                    return ToolCallResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"Resource pool {partition.value} exhausted (bulkhead timeout)",
                    )

        # ─────────────────────────────────────────────────────────────────────
        # Circuit Breaker Check
        # ─────────────────────────────────────────────────────────────────────
        # Check if server circuit is open (too many recent failures)
        server_id = tool_def.server_id
        server_info = self._registry._servers.get(server_id)
        if server_info and server_info.is_circuit_open:
            self._logger.warning(
                "Circuit breaker open, rejecting tool call",
                tool=tool_name,
                server=server_id,
                circuit_open_until=server_info.circuit_open_until,
            )
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Server {server_id} temporarily unavailable (circuit breaker open). "
                      f"Will retry after {server_info.circuit_open_until}",
            )

        # Check if confirmation required
        tier_info = TOOL_TIERS.get(_tier_key(tool_name), ToolTier(2, 1000, "low"))
        if tier_info.requires_confirmation:
            self._logger.info(
                "Tool requires confirmation",
                tool=tool_name,
                risk=tier_info.risk_level,
            )

        # Check if this is a custom tool with a handler
        if tool_name in self._custom_tools and hasattr(tool_def, "_handler"):
            try:
                handler = tool_def._handler  # type: ignore[attr-defined]
                if asyncio.iscoroutinefunction(handler):
                    result_data = await handler(arguments)
                else:
                    result_data = handler(arguments)

                duration_ms = int((time.time() - start_time) * 1000)

                result = ToolCallResult(
                    tool_name=tool_name,
                    success=True,
                    result=result_data,
                    duration_ms=duration_ms,
                )

                # Track usage
                self._track_usage(tool_name, True, duration_ms)

                # Trigger event
                await self._trigger_event("tool_called", {
                    "tool": tool_name,
                    "success": True,
                    "duration_ms": duration_ms,
                })

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                self._track_usage(tool_name, False, duration_ms)
                await self._trigger_event("tool_error", {
                    "tool": tool_name,
                    "error": str(e),
                })
                return ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                )

        # Get the server client for MCP tools
        client = self._registry.get_client(tool_def.server_id)
        if not client:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Server not connected: {tool_def.server_id}",
            )

        # Auto-enrich arguments for FinOps tools (tenancy_ocid from session/profile)
        try:
            needs_tenancy = (
                "tenancy_ocid" not in (arguments or {})
                and self._schema_supports_param(tool_def.input_schema, "tenancy_ocid")
            )
            is_finops_tool = (
                tool_def.server_id == "finopsai"
                or tool_name.startswith("oci_cost_")
                or (self.get_tool_domain(tool_name) in ("finops", "cost"))
            )
            if needs_tenancy and is_finops_tool:
                finops_client = self._registry.get_client("finopsai")
                if finops_client:
                    # Try session context first
                    tenancy = None
                    try:
                        ctx_res = await finops_client.call_tool(
                            "finops_session_context",
                            {"action": "get", "key": "tenancy_ocid"},
                        )
                        if getattr(ctx_res, "success", False):
                            ctx_payload = ctx_res.result
                            if isinstance(ctx_payload, dict):
                                tenancy = ctx_payload.get("value") or ctx_payload.get("tenancy_ocid")
                            else:
                                try:
                                    parsed = json.loads(ctx_payload)
                                    tenancy = parsed.get("value") or parsed.get("tenancy_ocid")
                                except Exception:
                                    tenancy = None
                    except Exception:
                        tenancy = None

                    # If not in session, resolve from profile via get_tenancy_info
                    if not tenancy:
                        try:
                            info_res = await finops_client.call_tool(
                                "get_tenancy_info",
                                {"response_format": "json"},
                            )
                            if getattr(info_res, "success", False):
                                info_payload = info_res.result
                                if isinstance(info_payload, dict):
                                    data = info_payload.get("data") or info_payload
                                    if isinstance(data, dict):
                                        tenancy = data.get("tenancy_ocid")
                                else:
                                    try:
                                        parsed = json.loads(info_payload)
                                        data = parsed.get("data") or parsed
                                        if isinstance(data, dict):
                                            tenancy = data.get("tenancy_ocid")
                                    except Exception:
                                        tenancy = None
                        except Exception:
                            tenancy = None

                    # Persist to session for subsequent calls
                    if tenancy:
                        with suppress(Exception):
                            await finops_client.call_tool(
                                "finops_session_context",
                                {"action": "set", "key": "tenancy_ocid", "value": tenancy},
                            )
                        if arguments is None:
                            arguments = {}
                        arguments["tenancy_ocid"] = tenancy
        except Exception as exc:
            self._logger.debug("FinOps arg enrichment skipped", error=str(exc))

        # Execute the tool via MCP
        result = await client.call_tool(tool_name, arguments)

        duration_ms = result.duration_ms or int((time.time() - start_time) * 1000)

        # ─────────────────────────────────────────────────────────────────────
        # Circuit Breaker Recording
        # ─────────────────────────────────────────────────────────────────────
        # Record success/failure for circuit breaker state management
        if server_info:
            if result.success:
                server_info.record_success()
            else:
                server_info.record_failure()
                if server_info.is_circuit_open:
                    self._logger.warning(
                        "Circuit breaker opened due to failures",
                        server=server_id,
                        failures=server_info.consecutive_failures,
                        open_until=server_info.circuit_open_until,
                    )

        # Track usage
        self._track_usage(tool_name, result.success, duration_ms)

        # Trigger events
        if result.success:
            await self._trigger_event("tool_called", {
                "tool": tool_name,
                "success": True,
                "duration_ms": duration_ms,
            })
            if (
                cache_ttl_seconds
                and self._memory_manager
                and cache_key
                and args_hash
            ):
                try:
                    await self._memory_manager.set_tool_result(
                        cache_key,
                        args_hash,
                        result.result,
                        ttl=timedelta(seconds=cache_ttl_seconds),
                    )
                except Exception as exc:
                    self._logger.warning(
                        "Tool cache write failed",
                        tool=tool_name,
                        error=str(exc),
                    )
        else:
            await self._trigger_event("tool_error", {
                "tool": tool_name,
                "error": result.error,
            })

            # ─────────────────────────────────────────────────────────────────
            # Deadletter Queue
            # ─────────────────────────────────────────────────────────────────
            # Enqueue failed operations for analysis and retry
            if self._deadletter_queue:
                try:
                    await self._deadletter_queue.enqueue(
                        failure_type=FailureType.TOOL_CALL,
                        operation=tool_name,
                        error=result.error or "Unknown error",
                        params=arguments,
                        context={
                            "server_id": server_id,
                            "duration_ms": duration_ms,
                            "circuit_open": server_info.is_circuit_open if server_info else False,
                        },
                    )
                except Exception as dlq_error:
                    self._logger.warning(
                        "Failed to enqueue to deadletter",
                        error=str(dlq_error),
                    )

        # ─────────────────────────────────────────────────────────────────────
        # Bulkhead Release
        # ─────────────────────────────────────────────────────────────────────
        if bulkhead_handle:
            await bulkhead_handle.__aexit__(None, None, None)

        self._logger.info(
            "Tool executed",
            tool=tool_name,
            success=result.success,
            duration_ms=duration_ms,
        )

        return result

    def to_langchain_tools(
        self,
        tool_names: list[str] | None = None,
        max_tier: int = 3,
    ) -> list[StructuredTool]:
        """
        Convert MCP tools to LangChain StructuredTools.

        Args:
            tool_names: Specific tools to convert (None = all)
            max_tier: Maximum tier to include

        Returns:
            List of LangChain StructuredTool instances
        """
        langchain_tools = []

        for name, tool_def in self._tools.items():
            # Skip if not in requested list
            if tool_names and name not in tool_names:
                continue

            # Skip high tier tools by default
            tier_info = TOOL_TIERS.get(_tier_key(name), ToolTier(2, 1000, "low"))
            if tier_info.tier > max_tier:
                continue

            # Create async function for this tool
            async def tool_func(
                tool_name: str = name, **kwargs: Any
            ) -> str:
                result = await self.execute(tool_name, kwargs)
                if result.success:
                    return str(result.result)
                return f"Error: {result.error}"

            # Build input schema from tool definition

            structured_tool = StructuredTool.from_function(
                func=tool_func,
                name=name,
                description=tool_def.description,
                args_schema=None,  # Will use **kwargs
                coroutine=tool_func,
            )

            langchain_tools.append(structured_tool)

        self._logger.debug(
            "Converted to LangChain tools",
            count=len(langchain_tools),
        )

        return langchain_tools

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive catalog statistics."""
        domains = self.get_domain_summary()

        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for name in self._tools:
            tier = TOOL_TIERS.get(_tier_key(name), ToolTier(2, 1000, "low")).tier
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Calculate usage summary
        total_calls = sum(
            s.get("call_count", 0) for s in self._tool_usage_stats.values()
        )
        total_errors = sum(
            s.get("failure_count", 0) for s in self._tool_usage_stats.values()
        )

        return {
            "total_tools": len(self._tools),
            "custom_tools": len(self._custom_tools),
            "domains": {k: len(v) for k, v in domains.items()},
            "by_tier": tier_counts,
            "servers": list(set(self._tool_to_server.values())),
            "refresh": {
                "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                "is_stale": self.is_stale(),
                "refresh_interval_seconds": self._refresh_interval.total_seconds(),
                "stale_threshold_seconds": self._stale_threshold.total_seconds(),
            },
            "usage": {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate": total_errors / total_calls if total_calls > 0 else 0,
                "tools_used": len(self._tool_usage_stats),
            },
        }
