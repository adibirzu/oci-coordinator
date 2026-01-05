"""
Agent Catalog for OCI Coordinator.

Provides automatic registration, discovery, and lifecycle management for all agents.

Enhanced Features:
- Domain-based agent lookup
- Priority-based agent selection
- Performance metrics tracking
- Capability scoring for intelligent routing
- Agent dependency management
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
import json

from src.agents.base import AgentDefinition, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Domain to capabilities mapping
# Maps domains to their capabilities and associated MCP servers
DOMAIN_CAPABILITIES = {
    "database": [
        "database-analysis",
        "performance-diagnostics",
        "sql-tuning",
        "blocking-analysis",
        "wait-event-analysis",
        "awr-analysis",
        "ash-analysis",
        "opsi-diagnostics",
        "autonomous-db-management",
        "db-system-management",
        "mysql-management",
    ],
    "security": [
        "threat-detection",
        "compliance-monitoring",
        "security-posture",
        "mitre-mapping",
        "cloud-guard-analysis",
        "vulnerability-scanning",
        "bastion-management",
        "waf-analysis",
        "kms-management",
        "data-safe-analysis",
        "iam-analysis",
        "policy-analysis",
        "security-audit",
    ],
    "cloudguard": [
        "cloud-guard-analysis",
        "security-posture",
        "compliance-monitoring",
    ],
    "finops": [
        "cost-analysis",
        "budget-tracking",
        "optimization",
        "anomaly-detection",
        "usage-forecasting",
        "focus-data-analysis",
        "tag-based-costing",
        "unit-cost-analysis",
        "cost-by-service",
        "cost-by-compartment",
        "monthly-trend-analysis",
    ],
    "observability": [
        "log-analysis",
        "metric-analysis",
        "alerting",
        "pattern-detection",
        "trace-correlation",
        "alarm-management",
        "log-query-execution",
        "instance-metrics",
        "observability-overview",
    ],
    "infrastructure": [
        "compute-management",
        "network-management",
        "storage-management",
        "resource-scaling",
        "vcn-analysis",
        "instance-troubleshooting",
        "security-list-analysis",
        "subnet-management",
        "instance-lifecycle",
    ],
}

# MCP Server to domain mapping
# Maps each MCP server to the domains it handles for intelligent agent routing
MCP_SERVER_DOMAINS = {
    # mcp-oci: Reference implementation with core OCI services
    "mcp-oci": [
        "discovery",
        "infrastructure",
        "compute",
        "network",
        "database",
        "observability",
        "identity",
        "feedback",
    ],
    # oci-mcp-security: Comprehensive security services
    "oci-mcp-security": [
        "security",
        "cloudguard",
        "vss",
        "bastion",
        "audit",
        "kms",
        "waf",
        "datasafe",
        "accessgov",
        "zones",
    ],
    # finopsai-mcp: Cost management and FinOps
    "finopsai-mcp": [
        "finops",
        "cost",
        "budget",
        "forecast",
        "optimization",
    ],
    # database-observatory: Advanced database analytics
    "database-observatory": [
        "database",
        "sql",
        "performance",
        "awr",
        "schema",
        "opsi",
        "logan",
        "tenancy",
    ],
    # Legacy server names for backward compatibility
    "oci-unified": ["discovery", "infrastructure", "observability", "security", "finops"],
    "oci-infrastructure": ["infrastructure", "observability", "database", "security", "finops"],
    "finopsai": ["finops", "cost"],
    "opsi": ["database", "opsi"],
    "logan": ["observability"],
}

_CATALOG_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "catalog"


def _load_server_domains_config() -> None:
    """Load MCP server domain mappings from config/catalog if present."""
    global MCP_SERVER_DOMAINS
    config_path = _CATALOG_CONFIG_DIR / "server_domains.json"
    if not config_path.exists():
        return
    try:
        data = json.loads(config_path.read_text())
        if isinstance(data, dict) and data:
            MCP_SERVER_DOMAINS = data
    except Exception as exc:
        logger.warning(
            "Failed to load server domains config",
            path=str(config_path),
            error=str(exc),
        )


_load_server_domains_config()

# Tool lists are derived dynamically from ToolCatalog and manifests.
MCP_DOMAIN_TOOLS: dict[str, list[str]] = {}

# Tool aliases for backward compatibility
TOOL_ALIASES = {
    # Compute
    "list_instances": "oci_compute_list_instances",
    "start_instance": "oci_compute_start_instance",
    "stop_instance": "oci_compute_stop_instance",
    "restart_instance": "oci_compute_restart_instance",
    "get_instance_metrics": "oci_observability_get_instance_metrics",
    # Database Observatory - SQLcl (legacy names)
    "execute_sql": "oci_database_execute_sql",
    "get_schema_info": "oci_database_get_schema",
    "list_connections": "oci_database_list_connections",
    "database_status": "oci_database_get_status",
    # Database Observatory - OPSI (legacy names)
    "list_database_insights": "oci_opsi_list_insights",
    "search_databases": "oci_opsi_search_databases",
    "get_fleet_summary": "oci_opsi_get_fleet_summary",
    "analyze_cpu_usage": "oci_opsi_analyze_cpu",
    "analyze_memory_usage": "oci_opsi_analyze_memory",
    "analyze_io_usage": "oci_opsi_analyze_io",
    # Database Observatory - Logan (legacy names)
    "execute_logan_query": "oci_logan_execute_query",
    "list_log_sources": "oci_logan_list_sources",
    "list_active_log_sources": "oci_logan_list_active_sources",
}


def _normalize_tool_name(tool_name: str) -> str:
    """Normalize tool names for catalog lookups."""
    if ":" in tool_name:
        tool_name = tool_name.split(":", 1)[1]
    if "__" in tool_name:
        tool_name = tool_name.split("__", 1)[1]
    return tool_name

# Default priority scores by domain (higher = preferred for that domain)
DOMAIN_PRIORITY = {
    "database": {
        "db-troubleshoot-agent": 100,
        "log-analytics-agent": 30,
    },
    "security": {
        "security-threat-agent": 100,
        "log-analytics-agent": 40,
    },
    "cloudguard": {
        "security-threat-agent": 100,
    },
    "finops": {
        "finops-agent": 100,
    },
    "observability": {
        "log-analytics-agent": 100,
        "db-troubleshoot-agent": 20,
    },
    "infrastructure": {
        "infrastructure-agent": 100,
    },
}


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""

    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_duration_ms: int = 0
    last_invocation: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_invocations == 0:
            return 100.0
        return (self.successful_invocations / self.total_invocations) * 100

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration in milliseconds."""
        if self.total_invocations == 0:
            return 0.0
        return self.total_duration_ms / self.total_invocations

    def record_invocation(self, duration_ms: int, success: bool) -> None:
        """Record an invocation."""
        self.total_invocations += 1
        self.total_duration_ms += duration_ms
        self.last_invocation = datetime.utcnow()
        if success:
            self.successful_invocations += 1
        else:
            self.failed_invocations += 1


@dataclass
class AgentScore:
    """Scoring result for agent selection."""

    agent: AgentDefinition
    score: float
    capability_match: float
    domain_priority: float
    health_score: float
    performance_score: float
    reasoning: str = ""


class AgentCatalog:
    """
    Central registry for all agents with auto-discovery.

    Features:
    - Auto-registration from agents directory
    - Health monitoring and status tracking
    - Capability-based agent lookup
    - Skill-based agent lookup
    - Domain-based agent lookup
    - Priority-based agent selection
    - Performance metrics tracking
    - Best match selection for routing

    Usage:
        catalog = AgentCatalog.get_instance()
        catalog.auto_discover()

        # Get by capability
        agents = catalog.get_by_capability("database-analysis")

        # Get by domain
        db_agents = catalog.get_by_domain("database")

        # Get best agent for a query
        best = catalog.find_best_agent(
            domains=["database", "observability"],
            capabilities=["performance-diagnostics"],
        )
    """

    _instance: AgentCatalog | None = None

    @classmethod
    def get_instance(cls, tool_catalog: "ToolCatalog | None" = None) -> AgentCatalog:
        """
        Get singleton instance of AgentCatalog.

        Returns:
            The shared AgentCatalog instance
        """
        if cls._instance is None:
            cls._instance = cls(tool_catalog=tool_catalog)
        elif tool_catalog is not None:
            cls._instance.set_tool_catalog(tool_catalog)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None

    def __init__(self, tool_catalog: "ToolCatalog | None" = None) -> None:
        """Initialize empty catalog."""
        self._agents: dict[str, AgentDefinition] = {}
        self._agent_classes: dict[str, type[BaseAgent]] = {}
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._domain_index: dict[str, set[str]] = {}  # domain -> set of agent roles
        self._tool_catalog = tool_catalog
        self._logger = logger.bind(component="AgentCatalog")

    def set_tool_catalog(self, tool_catalog: "ToolCatalog | None") -> None:
        """Attach a tool catalog for dynamic tool resolution."""
        self._tool_catalog = tool_catalog

    def _resolve_tool_catalog(self) -> "ToolCatalog | None":
        if self._tool_catalog:
            return self._tool_catalog
        try:
            from src.mcp.catalog import ToolCatalog
        except Exception:
            return None
        self._tool_catalog = ToolCatalog.get_instance()
        return self._tool_catalog

    def auto_discover(self, agents_path: str = "src/agents") -> int:
        """
        Auto-discover and register agents from the agents directory.

        Scans all Python modules in the agents directory and registers
        any class that inherits from BaseAgent.

        Args:
            agents_path: Path to agents directory

        Returns:
            Number of agents discovered
        """
        agents_dir = Path(agents_path)
        discovered_count = 0

        if not agents_dir.exists():
            self._logger.warning("Agents directory not found", path=agents_path)
            return 0

        self._logger.info("Starting agent auto-discovery", path=agents_path)

        # Walk through all subdirectories
        for module_path in agents_dir.rglob("*.py"):
            # Skip __init__.py and private modules
            if module_path.name.startswith("_"):
                continue

            # Skip base.py and catalog.py
            if module_path.name in ("base.py", "catalog.py"):
                continue

            # Convert path to module name
            # Handle both absolute and relative paths
            try:
                if agents_dir.is_absolute():
                    relative_path = module_path.relative_to(agents_dir.parent.parent)
                else:
                    relative_path = module_path
                module_name = str(relative_path).replace("/", ".").replace("\\", ".")[:-3]
            except ValueError:
                # Fallback: construct module name from agents_path
                relative_to_agents = module_path.relative_to(agents_dir)
                module_name = f"src.agents.{str(relative_to_agents).replace('/', '.').replace(chr(92), '.')[:-3]}"

            try:
                module = importlib.import_module(module_name)

                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue

                    attr = getattr(module, attr_name)

                    # Check if it's a BaseAgent subclass
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseAgent)
                        and attr is not BaseAgent
                    ):
                        self.register_agent_class(attr)
                        discovered_count += 1
                        self._logger.info(
                            "Discovered agent",
                            agent_class=attr.__name__,
                            module=module_name,
                        )

            except ImportError as e:
                self._logger.warning(
                    "Failed to import agent module",
                    module=module_name,
                    error=str(e),
                )
            except Exception as e:
                self._logger.error(
                    "Error during agent discovery",
                    module=module_name,
                    error=str(e),
                )

        self._logger.info(
            "Agent auto-discovery complete",
            discovered=discovered_count,
            total_registered=len(self._agents),
        )

        return discovered_count

    def register(self, agent: AgentDefinition) -> None:
        """
        Register an agent definition.

        Args:
            agent: Agent definition to register
        """
        agent.registered_at = datetime.utcnow()
        self._agents[agent.agent_id] = agent

        self._logger.info(
            "Agent registered",
            agent_id=agent.agent_id,
            role=agent.role,
            capabilities=agent.capabilities,
            skills=agent.skills,
        )

    def register_agent_class(self, agent_class: type[BaseAgent]) -> None:
        """
        Register an agent class for instantiation.

        The agent's definition is extracted via get_definition() and stored.
        Also builds the domain index for fast domain-based lookups.
        Warns about duplicate capabilities across agents.

        Args:
            agent_class: Agent class inheriting from BaseAgent
        """
        try:
            agent_def = agent_class.get_definition()
            agent_def.registered_at = datetime.utcnow()

            # Check for duplicate capabilities across existing agents
            duplicate_caps = self._check_duplicate_capabilities(agent_def)
            if duplicate_caps:
                self._logger.warning(
                    "Duplicate capabilities detected during agent registration",
                    agent=agent_def.role,
                    duplicates=duplicate_caps,
                )

            self._agents[agent_def.role] = agent_def
            self._agent_classes[agent_def.role] = agent_class
            self._agent_metrics[agent_def.role] = AgentMetrics()

            # Build domain index based on capabilities
            for domain, capabilities in DOMAIN_CAPABILITIES.items():
                if any(cap in agent_def.capabilities for cap in capabilities):
                    if domain not in self._domain_index:
                        self._domain_index[domain] = set()
                    self._domain_index[domain].add(agent_def.role)

            self._logger.info(
                "Agent class registered",
                role=agent_def.role,
                agent_class=agent_class.__name__,
                domains=self._get_agent_domains(agent_def),
                capabilities_count=len(agent_def.capabilities),
            )
        except Exception as e:
            self._logger.error(
                "Failed to register agent class",
                agent_class=agent_class.__name__,
                error=str(e),
            )

    def _check_duplicate_capabilities(
        self, new_agent: AgentDefinition
    ) -> dict[str, list[str]]:
        """
        Check for duplicate capabilities with existing agents.

        Args:
            new_agent: Agent definition being registered

        Returns:
            Dict of {capability: [existing_agents_with_cap]}
        """
        duplicates: dict[str, list[str]] = {}

        for cap in new_agent.capabilities:
            existing_agents = []
            for role, agent_def in self._agents.items():
                if cap in agent_def.capabilities:
                    existing_agents.append(role)

            if existing_agents:
                duplicates[cap] = existing_agents

        return duplicates

    def _get_agent_domains(self, agent: AgentDefinition) -> list[str]:
        """Get domains for an agent based on its capabilities."""
        domains = []
        for domain, capabilities in DOMAIN_CAPABILITIES.items():
            if any(cap in agent.capabilities for cap in capabilities):
                domains.append(domain)
        return domains

    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent ID or role to unregister

        Returns:
            True if agent was unregistered
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._agent_classes.pop(agent_id, None)
            self._logger.info("Agent unregistered", agent_id=agent_id)
            return True
        return False

    def get(self, agent_id: str) -> AgentDefinition | None:
        """
        Get agent by ID or role.

        Args:
            agent_id: Agent ID or role name

        Returns:
            AgentDefinition or None if not found
        """
        return self._agents.get(agent_id)

    def get_agent_class(self, role: str) -> type[BaseAgent] | None:
        """
        Get agent class by role.

        Args:
            role: Agent role name

        Returns:
            Agent class or None if not found
        """
        return self._agent_classes.get(role)

    def instantiate(
        self,
        role: str,
        memory_manager: Any = None,
        tool_catalog: Any = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ) -> BaseAgent | None:
        """
        Create an instance of an agent by role.

        Args:
            role: Agent role name
            memory_manager: SharedMemoryManager instance
            tool_catalog: ToolCatalog instance
            config: Agent configuration
            llm: LangChain LLM for agent reasoning and analysis

        Returns:
            Instantiated agent or None if not found
        """
        agent_class = self._agent_classes.get(role)
        if not agent_class:
            self._logger.warning("Agent class not found for role", role=role)
            return None

        return agent_class(
            memory_manager=memory_manager,
            tool_catalog=tool_catalog,
            config=config,
            llm=llm,
        )

    def get_by_capability(self, capability: str) -> list[AgentDefinition]:
        """
        Find agents that have a specific capability.

        Args:
            capability: Capability to search for (e.g., "database-analysis")

        Returns:
            List of agents with the capability
        """
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    def get_by_skill(self, skill: str) -> list[AgentDefinition]:
        """
        Find agents that can execute a specific skill.

        Args:
            skill: Skill/workflow name (e.g., "rca_workflow")

        Returns:
            List of agents with the skill
        """
        return [agent for agent in self._agents.values() if skill in agent.skills]

    def get_by_mcp_tool(self, tool_name: str) -> list[AgentDefinition]:
        """
        Find agents that use a specific MCP tool.

        Args:
            tool_name: MCP tool name (e.g., "oci_database_list_autonomous")

        Returns:
            List of agents using the tool
        """
        domain = self.find_tool_domain(tool_name)
        if not domain:
            return []
        return self.get_by_domain(domain)

    def list_all(self) -> list[AgentDefinition]:
        """
        List all registered agents.

        Returns:
            List of all agent definitions
        """
        return list(self._agents.values())

    def list_healthy(self) -> list[AgentDefinition]:
        """
        List only healthy agents.

        Returns:
            List of agents with HEALTHY status
        """
        return [
            agent
            for agent in self._agents.values()
            if agent.status == AgentStatus.HEALTHY
        ]

    def list_by_status(self, status: AgentStatus) -> list[AgentDefinition]:
        """
        List agents by status.

        Args:
            status: Agent status to filter by

        Returns:
            List of agents with the specified status
        """
        return [agent for agent in self._agents.values() if agent.status == status]

    async def health_check_all(self) -> dict[str, AgentStatus]:
        """
        Run health checks on all agents.

        Returns:
            Dictionary of agent_id -> AgentStatus
        """
        results: dict[str, AgentStatus] = {}

        for agent_id, agent in self._agents.items():
            try:
                status = await self._check_agent_health(agent)
                agent.status = status
                agent.last_heartbeat = datetime.utcnow()
                results[agent_id] = status
            except Exception as e:
                self._logger.error(
                    "Health check failed",
                    agent_id=agent_id,
                    error=str(e),
                )
                agent.status = AgentStatus.UNHEALTHY
                results[agent_id] = AgentStatus.UNHEALTHY

        return results

    async def _check_agent_health(self, agent: AgentDefinition) -> AgentStatus:
        """
        Check individual agent health via health endpoint.

        Args:
            agent: Agent definition with health_endpoint

        Returns:
            AgentStatus based on health check result
        """
        if not agent.health_endpoint:
            # No health endpoint configured, assume healthy
            return AgentStatus.HEALTHY

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(agent.health_endpoint)

                if response.status_code == 200:
                    return AgentStatus.HEALTHY
                elif response.status_code == 503:
                    return AgentStatus.DEGRADED
                else:
                    return AgentStatus.UNHEALTHY

        except httpx.TimeoutException:
            self._logger.warning(
                "Health check timeout",
                agent_id=agent.agent_id,
                endpoint=agent.health_endpoint,
            )
            return AgentStatus.UNHEALTHY
        except httpx.ConnectError:
            self._logger.warning(
                "Health check connection failed",
                agent_id=agent.agent_id,
                endpoint=agent.health_endpoint,
            )
            return AgentStatus.OFFLINE

    def get_summary(self) -> dict[str, Any]:
        """
        Get catalog summary statistics.

        Returns:
            Dictionary with catalog statistics
        """
        status_counts = {}
        for status in AgentStatus:
            count = len(self.list_by_status(status))
            if count > 0:
                status_counts[status.value] = count

        all_capabilities = set()
        all_skills = set()
        for agent in self._agents.values():
            all_capabilities.update(agent.capabilities)
            all_skills.update(agent.skills)

        return {
            "total_agents": len(self._agents),
            "status_counts": status_counts,
            "unique_capabilities": len(all_capabilities),
            "unique_skills": len(all_skills),
            "capabilities": sorted(all_capabilities),
            "skills": sorted(all_skills),
        }

    def to_json_schema(self) -> list[dict[str, Any]]:
        """
        Export all agent definitions as JSON-serializable list.

        Returns:
            List of agent definitions as dictionaries
        """
        return [agent.to_dict() for agent in self._agents.values()]

    # ─────────────────────────────────────────────────────────────────────────
    # Domain-Based Lookup
    # ─────────────────────────────────────────────────────────────────────────

    def get_by_domain(self, domain: str) -> list[AgentDefinition]:
        """
        Find agents that can handle a specific domain.

        Args:
            domain: Domain name (e.g., "database", "security", "finops")

        Returns:
            List of agents that handle the domain
        """
        agent_roles = self._domain_index.get(domain, set())
        return [self._agents[role] for role in agent_roles if role in self._agents]

    def get_domains(self) -> list[str]:
        """
        Get all available domains.

        Returns:
            List of domain names
        """
        return list(self._domain_index.keys())

    def get_agent_domains(self, role: str) -> list[str]:
        """
        Get domains for a specific agent.

        Args:
            role: Agent role

        Returns:
            List of domains the agent handles
        """
        agent = self._agents.get(role)
        if not agent:
            return []
        return self._get_agent_domains(agent)

    # ─────────────────────────────────────────────────────────────────────────
    # Priority-Based Selection
    # ─────────────────────────────────────────────────────────────────────────

    def find_best_agent(
        self,
        domains: list[str] | None = None,
        capabilities: list[str] | None = None,
        skills: list[str] | None = None,
        prefer_healthy: bool = True,
    ) -> AgentScore | None:
        """
        Find the best agent for a given set of requirements.

        Scores agents based on:
        - Capability match (40%)
        - Domain priority (30%)
        - Health status (15%)
        - Performance history (15%)

        Args:
            domains: Required domains
            capabilities: Required capabilities
            skills: Required skills
            prefer_healthy: Prefer healthy agents

        Returns:
            AgentScore with best matching agent, or None if no match
        """
        candidates = self._get_candidate_agents(domains, capabilities, skills)

        if not candidates:
            return None

        scores = []
        for agent in candidates:
            score = self._score_agent(agent, domains, capabilities, skills, prefer_healthy)
            scores.append(score)

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)

        best = scores[0]
        self._logger.debug(
            "Best agent selected",
            agent=best.agent.role,
            score=best.score,
            reasoning=best.reasoning,
        )

        return best

    def find_agents_ranked(
        self,
        domains: list[str] | None = None,
        capabilities: list[str] | None = None,
        skills: list[str] | None = None,
        prefer_healthy: bool = True,
        limit: int = 5,
    ) -> list[AgentScore]:
        """
        Find and rank agents for given requirements.

        Args:
            domains: Required domains
            capabilities: Required capabilities
            skills: Required skills
            prefer_healthy: Prefer healthy agents
            limit: Maximum number of agents to return

        Returns:
            List of AgentScore sorted by score descending
        """
        candidates = self._get_candidate_agents(domains, capabilities, skills)

        if not candidates:
            return []

        scores = []
        for agent in candidates:
            score = self._score_agent(agent, domains, capabilities, skills, prefer_healthy)
            scores.append(score)

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)

        return scores[:limit]

    def _get_candidate_agents(
        self,
        domains: list[str] | None,
        capabilities: list[str] | None,
        skills: list[str] | None,
    ) -> list[AgentDefinition]:
        """Get candidate agents that match any of the requirements."""
        candidates = set()

        # Get by domain
        if domains:
            for domain in domains:
                for agent in self.get_by_domain(domain):
                    candidates.add(agent.role)

        # Get by capability
        if capabilities:
            for cap in capabilities:
                for agent in self.get_by_capability(cap):
                    candidates.add(agent.role)

        # Get by skill
        if skills:
            for skill in skills:
                for agent in self.get_by_skill(skill):
                    candidates.add(agent.role)

        # If no filters, return all agents
        if not domains and not capabilities and not skills:
            return list(self._agents.values())

        return [self._agents[role] for role in candidates if role in self._agents]

    def _score_agent(
        self,
        agent: AgentDefinition,
        domains: list[str] | None,
        capabilities: list[str] | None,
        skills: list[str] | None,
        prefer_healthy: bool,
    ) -> AgentScore:
        """Calculate score for an agent."""
        # Capability match (40%)
        capability_match = 0.0
        if capabilities:
            matched = sum(1 for cap in capabilities if cap in agent.capabilities)
            capability_match = matched / len(capabilities)
        elif agent.capabilities:
            capability_match = 0.5  # Partial credit if has any capabilities

        # Domain priority (30%)
        domain_priority = 0.0
        if domains:
            max_priority = 0
            for domain in domains:
                priority = DOMAIN_PRIORITY.get(domain, {}).get(agent.role, 0)
                max_priority = max(max_priority, priority)
            domain_priority = max_priority / 100.0  # Normalize to 0-1

        # Health score (15%)
        health_score = 1.0
        if prefer_healthy:
            if agent.status == AgentStatus.HEALTHY:
                health_score = 1.0
            elif agent.status == AgentStatus.DEGRADED:
                health_score = 0.7
            elif agent.status == AgentStatus.REGISTERED:
                health_score = 0.8
            else:
                health_score = 0.3

        # Performance score (15%)
        metrics = self._agent_metrics.get(agent.role)
        performance_score = 1.0
        if metrics and metrics.total_invocations > 0:
            # Base on success rate
            performance_score = metrics.success_rate / 100.0

        # Calculate weighted score
        score = (
            capability_match * 0.40
            + domain_priority * 0.30
            + health_score * 0.15
            + performance_score * 0.15
        )

        # Build reasoning
        reasoning_parts = []
        if capability_match > 0:
            reasoning_parts.append(f"capability={capability_match:.0%}")
        if domain_priority > 0:
            reasoning_parts.append(f"domain_priority={domain_priority:.0%}")
        reasoning_parts.append(f"health={health_score:.0%}")
        if metrics and metrics.total_invocations > 0:
            reasoning_parts.append(f"perf={performance_score:.0%}")

        return AgentScore(
            agent=agent,
            score=score,
            capability_match=capability_match,
            domain_priority=domain_priority,
            health_score=health_score,
            performance_score=performance_score,
            reasoning=", ".join(reasoning_parts),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Performance Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def record_invocation(
        self, role: str, duration_ms: int, success: bool
    ) -> None:
        """
        Record an agent invocation for metrics.

        Args:
            role: Agent role
            duration_ms: Duration in milliseconds
            success: Whether invocation was successful
        """
        if role not in self._agent_metrics:
            self._agent_metrics[role] = AgentMetrics()

        self._agent_metrics[role].record_invocation(duration_ms, success)

        self._logger.debug(
            "Recorded invocation",
            role=role,
            duration_ms=duration_ms,
            success=success,
        )

    def get_metrics(self, role: str) -> AgentMetrics | None:
        """
        Get metrics for an agent.

        Args:
            role: Agent role

        Returns:
            AgentMetrics or None if not found
        """
        return self._agent_metrics.get(role)

    def get_all_metrics(self) -> dict[str, AgentMetrics]:
        """
        Get metrics for all agents.

        Returns:
            Dictionary of role -> AgentMetrics
        """
        return dict(self._agent_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # MCP Tool Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_tools_for_domain(self, domain: str) -> list[str]:
        """
        Get MCP tools available for a domain.

        Args:
            domain: Domain name (e.g., "database", "security", "finops")

        Returns:
            List of MCP tool names for the domain
        """
        tool_catalog = self._resolve_tool_catalog()
        if not tool_catalog:
            return []
        return [tool.name for tool in tool_catalog.get_tools_for_domain(domain)]

    def get_all_tools(self) -> dict[str, list[str]]:
        """
        Get all MCP tools organized by domain.

        Returns:
            Dictionary of domain -> list of tool names
        """
        tool_catalog = self._resolve_tool_catalog()
        if not tool_catalog:
            return {}
        return tool_catalog.get_domain_summary()

    def sync_mcp_tools(self, tool_catalog: "ToolCatalog | None" = None) -> int:
        """Refresh agent tool lists from the tool catalog."""
        tool_catalog = tool_catalog or self._resolve_tool_catalog()
        if not tool_catalog:
            return 0

        updated = 0
        for agent in self._agents.values():
            domains = self._get_agent_domains(agent)
            tool_names: set[str] = set()
            for domain in domains:
                for tool in tool_catalog.get_tools_for_domain(domain):
                    tool_names.add(tool.name)
            new_tools = sorted(tool_names)
            if new_tools != agent.mcp_tools:
                agent.mcp_tools = new_tools
                updated += 1

        if updated:
            self._logger.info("Agent tools synchronized", updated=updated)
        return updated

    def resolve_tool_alias(self, tool_name: str) -> str:
        """
        Resolve a tool alias to canonical name.

        Args:
            tool_name: Tool name or alias

        Returns:
            Canonical tool name
        """
        canonical = _normalize_tool_name(tool_name)
        return TOOL_ALIASES.get(canonical, canonical)

    def find_tool_domain(self, tool_name: str) -> str | None:
        """
        Find which domain a tool belongs to.

        Args:
            tool_name: MCP tool name

        Returns:
            Domain name or None if not found
        """
        # Resolve alias first
        canonical = self.resolve_tool_alias(tool_name)
        tool_catalog = self._resolve_tool_catalog()
        if tool_catalog:
            return tool_catalog.get_tool_domain(canonical)
        return None

    def get_agent_for_tool(self, tool_name: str) -> AgentDefinition | None:
        """
        Find the best agent for a specific tool.

        Args:
            tool_name: MCP tool name

        Returns:
            Best matching AgentDefinition or None
        """
        domain = self.find_tool_domain(tool_name)
        if not domain:
            return None

        result = self.find_best_agent(domains=[domain])
        return result.agent if result else None

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get summary of all agent metrics.

        Returns:
            Summary dictionary with aggregate statistics
        """
        total_invocations = 0
        total_successful = 0
        total_failed = 0
        total_duration = 0

        agent_stats = []
        for role, metrics in self._agent_metrics.items():
            if metrics.total_invocations > 0:
                total_invocations += metrics.total_invocations
                total_successful += metrics.successful_invocations
                total_failed += metrics.failed_invocations
                total_duration += metrics.total_duration_ms

                agent_stats.append({
                    "role": role,
                    "invocations": metrics.total_invocations,
                    "success_rate": round(metrics.success_rate, 2),
                    "avg_duration_ms": round(metrics.avg_duration_ms, 2),
                })

        return {
            "total_invocations": total_invocations,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": round(
                (total_successful / total_invocations * 100) if total_invocations > 0 else 100.0,
                2,
            ),
            "total_duration_ms": total_duration,
            "agents": sorted(agent_stats, key=lambda x: x["invocations"], reverse=True),
        }


def initialize_agents(
    agents_path: str = "src/agents",
    tool_catalog: "ToolCatalog | None" = None,
) -> AgentCatalog:
    """
    Initialize agent catalog with auto-discovery.

    This function should be called during application startup.

    Args:
        agents_path: Path to agents directory

    Returns:
        Initialized AgentCatalog
    """
    catalog = AgentCatalog.get_instance(tool_catalog=tool_catalog)
    catalog.auto_discover(agents_path)
    if tool_catalog:
        catalog.sync_mcp_tools(tool_catalog)
    return catalog
