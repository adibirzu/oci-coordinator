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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from src.agents.base import AgentDefinition, AgentStatus, BaseAgent

if TYPE_CHECKING:
    pass

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
    ],
    "observability": [
        "log-analysis",
        "metric-analysis",
        "alerting",
        "pattern-detection",
        "trace-correlation",
        "alarm-management",
    ],
    "infrastructure": [
        "compute-management",
        "network-management",
        "storage-management",
        "resource-scaling",
        "vcn-analysis",
        "instance-troubleshooting",
    ],
}

# MCP Server to domain mapping
MCP_SERVER_DOMAINS = {
    "mcp-oci": ["infrastructure", "observability", "database", "security", "finops"],
    "oci-mcp-security": ["security"],
    "finopsai-mcp": ["finops"],
    "mcp-oci-database-observatory": ["database", "observability"],
    "opsi": ["database"],
}

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
    def get_instance(cls) -> AgentCatalog:
        """
        Get singleton instance of AgentCatalog.

        Returns:
            The shared AgentCatalog instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None

    def __init__(self) -> None:
        """Initialize empty catalog."""
        self._agents: dict[str, AgentDefinition] = {}
        self._agent_classes: dict[str, type[BaseAgent]] = {}
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._domain_index: dict[str, set[str]] = {}  # domain -> set of agent roles
        self._logger = logger.bind(component="AgentCatalog")

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

        Args:
            agent_class: Agent class inheriting from BaseAgent
        """
        try:
            agent_def = agent_class.get_definition()
            agent_def.registered_at = datetime.utcnow()

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
            )
        except Exception as e:
            self._logger.error(
                "Failed to register agent class",
                agent_class=agent_class.__name__,
                error=str(e),
            )

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
    ) -> BaseAgent | None:
        """
        Create an instance of an agent by role.

        Args:
            role: Agent role name
            memory_manager: SharedMemoryManager instance
            tool_catalog: ToolCatalog instance
            config: Agent configuration

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
        return [agent for agent in self._agents.values() if tool_name in agent.mcp_tools]

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


def initialize_agents(agents_path: str = "src/agents") -> AgentCatalog:
    """
    Initialize agent catalog with auto-discovery.

    This function should be called during application startup.

    Args:
        agents_path: Path to agents directory

    Returns:
        Initialized AgentCatalog
    """
    catalog = AgentCatalog.get_instance()
    catalog.auto_discover(agents_path)
    return catalog
