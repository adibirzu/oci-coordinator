"""
Security Threat Agent.

Specialized agent for security analysis, threat detection,
and compliance monitoring in OCI.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from langgraph.graph import END, StateGraph
from opentelemetry import trace

from src.agents.base import (
    AgentDefinition,
    AgentMetadata,
    BaseAgent,
    KafkaTopics,
)
from src.agents.self_healing import SelfHealingMixin
from src.mcp.client import ToolCallResult

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("oci-security-threat-agent")


@dataclass
class SecurityState:
    """State for security analysis workflow."""

    query: str = ""
    compartment_id: str | None = None
    resource_id: str | None = None
    threat_type: str | None = None

    # Analysis results
    cloud_guard_problems: list[dict] = field(default_factory=list)
    security_findings: list[dict] = field(default_factory=list)
    compliance_issues: list[dict] = field(default_factory=list)
    threat_indicators: list[dict] = field(default_factory=list)  # MITRE ATT&CK mapped threats
    remediation_actions: list[dict] = field(default_factory=list)

    # State
    phase: str = "analyze_request"
    risk_score: int = 0
    error: str | None = None
    result: str | None = None


class SecurityThreatAgent(BaseAgent, SelfHealingMixin):
    """
    Security Threat Agent with Self-Healing Capabilities.

    Specializes in OCI security operations:
    - Threat indicator correlation
    - MITRE ATT&CK mapping
    - Cloud Guard problem analysis
    - Security posture assessment
    - Compliance monitoring

    Self-Healing Features:
    - Automatic retry on Cloud Guard API timeouts
    - Parameter correction for security tool calls
    - LLM-powered threat analysis recovery
    """

    def __init__(
        self,
        memory_manager: SharedMemoryManager | None = None,
        tool_catalog: ToolCatalog | None = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        """
        Initialize Security Threat Agent with self-healing.

        Args:
            memory_manager: Shared memory manager
            tool_catalog: Tool catalog for MCP tools
            config: Agent configuration
            llm: LangChain LLM for analysis
        """
        super().__init__(memory_manager, tool_catalog, config)
        self.llm = llm
        self._graph: StateGraph | None = None

        # Initialize self-healing capabilities
        if llm:
            self.init_self_healing(
                llm=llm,
                max_retries=3,
                enable_validation=True,
                enable_correction=True,
            )

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        """Return agent definition for catalog registration."""
        return AgentDefinition(
            agent_id="security-threat-agent",
            role="security-threat-agent",
            capabilities=[
                "threat-detection",
                "compliance-monitoring",
                "security-posture",
                "mitre-mapping",
                "cloud-guard-analysis",
                "sigma-to-ocl",
            ],
            skills=[
                "threat_hunting_workflow",
                "compliance_check",
                "security_assessment",
                "security_assessment_workflow",
                "incident_analysis",
                "security_posture_workflow",
                "security_cloudguard_investigation_workflow",
                "security_cloudguard_remediation_workflow",
                "security_vulnerability_workflow",
                "security_zone_compliance_workflow",
                "security_bastion_audit_workflow",
                "security_bastion_session_cleanup_workflow",
                "security_datasafe_assessment_workflow",
                "security_waf_policy_workflow",
                "security_audit_activity_workflow",
                "security_access_governance_workflow",
                "security_kms_inventory_workflow",
                "security_iam_review_workflow",
                "sigma_to_ocl_conversion",
                "sigma_threat_hunting",
                "mitre_coverage_report",
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.security-threat-agent"],
                produce=["results.security-threat-agent"],
            ),
            health_endpoint="http://localhost:8012/health",
            metadata=AgentMetadata(
                version="1.0.0",
                namespace="oci-coordinator",
                max_iterations=10,
                timeout_seconds=60,
            ),
            description=(
                "Security Expert Agent for threat detection, compliance monitoring, "
                "Cloud Guard problem analysis, MITRE ATT&CK mapping, vulnerability scanning, "
                "WAF policy review, bastion session audit, and Data Safe assessments in OCI."
            ),
            mcp_servers=["oci-gateway", "oci-unified", "oci-mcp-security"],
            mcp_tools=[
                "oci_security_list_users",
                "oci_security_cloudguard_list_problems",
                "oci_security_cloudguard_get_security_score",
                "oci_security_audit_list_events",
                "oci_security_overview",
            ],
        )

    def build_graph(self) -> StateGraph:
        """Build the security analysis workflow graph."""
        graph = StateGraph(SecurityState)

        graph.add_node("analyze_request", self._analyze_request_node)
        graph.add_node("check_cloud_guard", self._check_cloud_guard_node)
        graph.add_node("assess_compliance", self._assess_compliance_node)
        graph.add_node("correlate_threats", self._correlate_threats_node)
        graph.add_node("output", self._output_node)

        graph.set_entry_point("analyze_request")
        graph.add_edge("analyze_request", "check_cloud_guard")
        graph.add_edge("check_cloud_guard", "assess_compliance")
        graph.add_edge("assess_compliance", "correlate_threats")
        graph.add_edge("correlate_threats", "output")
        graph.add_edge("output", END)

        return graph.compile()

    async def _analyze_request_node(self, state: SecurityState) -> dict[str, Any]:
        """Analyze security request."""
        with tracer.start_as_current_span("security.analyze_request") as span:
            span.set_attribute("query", state.query[:200])
            self._logger.info("Analyzing security request", query=state.query[:100])

            # Determine threat type from query
            threat_keywords = {
                "attack": "active_threat",
                "breach": "data_breach",
                "suspicious": "anomaly",
                "compliance": "compliance",
                "audit": "audit",
            }
            threat_type = "general"
            for keyword, ttype in threat_keywords.items():
                if keyword in state.query.lower():
                    threat_type = ttype
                    break

            span.set_attribute("threat_type", threat_type)
            return {"threat_type": threat_type, "phase": "check_cloud_guard"}

    async def _check_cloud_guard_node(self, state: SecurityState) -> dict[str, Any]:
        """Check Cloud Guard for security problems with self-healing."""
        with tracer.start_as_current_span("security.check_cloud_guard") as span:
            self._logger.info("Checking Cloud Guard")

            problems: list[dict[str, Any]] = []
            remediation_actions: list[dict[str, Any]] = []

            if not self.tools:
                return {"cloud_guard_problems": problems, "phase": "assess_compliance"}

            tool_name, tool_args = self._resolve_cloud_guard_tool_args(state)
            if not tool_name:
                self._logger.warning("Cloud Guard tool not available in catalog")
                return {"cloud_guard_problems": problems, "phase": "assess_compliance"}

            try:
                span.set_attribute("tool_name", tool_name)
                # Use self-healing for automatic retry on Cloud Guard API issues
                if self._self_healing_enabled:
                    result = await self.healing_call_tool(
                        tool_name,
                        tool_args,
                        user_intent=state.query,
                        validate=True,
                        correct_on_failure=True,
                    )
                else:
                    result = await self.call_tool(tool_name, tool_args)

                ok, payload, error = self._extract_tool_payload(result)
                if not ok:
                    self._logger.warning("Cloud Guard check failed", error=error or "unknown")
                    span.set_attribute("error", error or "unknown")
                else:
                    problems = self._normalize_cloud_guard_problems(payload)

                remediation_actions = await self._auto_remediate_cloud_guard(
                    problems,
                    user_intent=state.query,
                )
            except Exception as exc:
                self._logger.warning("Cloud Guard check failed", error=str(exc))
                span.set_attribute("error", str(exc))

            span.set_attribute("problems_found", len(problems))
            span.set_attribute("remediation_actions", len(remediation_actions))
            return {
                "cloud_guard_problems": problems,
                "remediation_actions": remediation_actions,
                "phase": "assess_compliance",
            }

    async def _assess_compliance_node(self, state: SecurityState) -> dict[str, Any]:
        """Assess compliance status."""
        with tracer.start_as_current_span("security.assess_compliance") as span:
            self._logger.info("Assessing compliance")

            compliance_issues = []
            # Compliance assessment logic

            span.set_attribute("compliance_issues_count", len(compliance_issues))
            return {"compliance_issues": compliance_issues, "phase": "correlate_threats"}

    async def _correlate_threats_node(self, state: SecurityState) -> dict[str, Any]:
        """Correlate threat indicators and map to MITRE ATT&CK."""
        with tracer.start_as_current_span("security.correlate_threats") as span:
            self._logger.info("Correlating threats with MITRE ATT&CK mapping")
            span.set_attribute("cloud_guard_problems_input", len(state.cloud_guard_problems))

            # MITRE ATT&CK mapping for common Cloud Guard problems
            mitre_mapping = {
                "SUSPICIOUS_LOGIN": {"technique": "T1078", "tactic": "Initial Access", "name": "Valid Accounts"},
                "UNUSUAL_API_ACTIVITY": {"technique": "T1106", "tactic": "Execution", "name": "Native API"},
                "IAM_POLICY_CHANGE": {"technique": "T1098", "tactic": "Persistence", "name": "Account Manipulation"},
                "SECURITY_GROUP_CHANGE": {"technique": "T1562", "tactic": "Defense Evasion", "name": "Impair Defenses"},
                "DATA_EXFILTRATION": {"technique": "T1567", "tactic": "Exfiltration", "name": "Exfiltration Over Web"},
                "RESOURCE_DELETION": {"technique": "T1485", "tactic": "Impact", "name": "Data Destruction"},
                "PRIVILEGED_ACCESS": {"technique": "T1078.004", "tactic": "Privilege Escalation", "name": "Cloud Accounts"},
                "CRYPTO_MINING": {"technique": "T1496", "tactic": "Impact", "name": "Resource Hijacking"},
                "PUBLIC_EXPOSURE": {"technique": "T1190", "tactic": "Initial Access", "name": "Exploit Public-Facing"},
                "BRUTEFORCE": {"technique": "T1110", "tactic": "Credential Access", "name": "Brute Force"},
            }

            # Enrich Cloud Guard problems with MITRE mapping
            threat_indicators = []
            for problem in state.cloud_guard_problems:
                problem_type = problem.get("detector_id", "").upper()
                for key, mapping in mitre_mapping.items():
                    if key in problem_type or key in problem.get("name", "").upper():
                        threat_indicators.append({
                            "problem_id": problem.get("id"),
                            "problem_name": problem.get("name"),
                            "mitre_technique": mapping["technique"],
                            "mitre_tactic": mapping["tactic"],
                            "mitre_name": mapping["name"],
                            "risk_level": problem.get("risk_level", "MEDIUM"),
                            "recommendation": f"Investigate {mapping['tactic']} activity",
                        })
                        break

            # Calculate risk score with MITRE weighting
            risk_score = 0

            # Cloud Guard problems (weighted by severity)
            for problem in state.cloud_guard_problems:
                risk = problem.get("risk_level", "LOW")
                if risk == "CRITICAL":
                    risk_score += 25
                elif risk == "HIGH":
                    risk_score += 15
                elif risk == "MEDIUM":
                    risk_score += 8
                else:
                    risk_score += 3

            # Compliance issues
            risk_score += len(state.compliance_issues) * 5

            # Bonus for MITRE-mapped threats (indicates active attack patterns)
            risk_score += len(threat_indicators) * 5

            risk_score = min(100, risk_score)

            span.set_attribute("threat_indicators_count", len(threat_indicators))
            span.set_attribute("risk_score", risk_score)
            return {
                "threat_indicators": threat_indicators,
                "risk_score": risk_score,
                "phase": "output",
            }

    async def _output_node(self, state: SecurityState) -> dict[str, Any]:
        """Prepare security report with structured formatting."""
        with tracer.start_as_current_span("security.output") as span:
            from src.formatting import (
                ActionButton,
                ListItem,
                MetricValue,
                ResponseFooter,
                Severity,
                StatusIndicator,
            )

            span.set_attribute("risk_score", state.risk_score)
            span.set_attribute("cloud_guard_problems", len(state.cloud_guard_problems))
            span.set_attribute("threat_indicators", len(state.threat_indicators))

            # Determine risk level and severity
            if state.risk_score >= 70:
                risk_level = "Critical"
                severity = "critical"
            elif state.risk_score >= 50:
                risk_level = "High"
                severity = "high"
            elif state.risk_score >= 30:
                risk_level = "Medium"
                severity = "medium"
            else:
                risk_level = "Low"
                severity = "low"

            span.set_attribute("severity", severity)

            # Create structured response
            response = self.create_response(
                title="Security Analysis Results",
                subtitle=f"Threat assessment: {state.threat_type}",
                severity=severity,
                icon="🔒" if state.risk_score < 30 else "⚠️",
            )

            # Add risk metrics
            response.add_metrics(
                "Risk Assessment",
                [
                    MetricValue(
                        label="Risk Score",
                        value=state.risk_score,
                        unit="/100",
                        threshold=30,
                        severity=Severity(severity),
                    ),
                    MetricValue(
                        label="Risk Level",
                        value=risk_level,
                        severity=Severity(severity),
                    ),
                ],
                divider_after=True,
            )

            # Add summary stats
            response.add_section(
                title="Summary",
                fields=[
                    StatusIndicator(
                        label="Cloud Guard Problems",
                        value=str(len(state.cloud_guard_problems)),
                        severity=Severity.HIGH if state.cloud_guard_problems else Severity.SUCCESS,
                    ),
                    StatusIndicator(
                        label="Compliance Issues",
                        value=str(len(state.compliance_issues)),
                        severity=Severity.MEDIUM if state.compliance_issues else Severity.SUCCESS,
                    ),
                    StatusIndicator(
                        label="Threat Type",
                        value=state.threat_type or "general",
                        severity=Severity.INFO,
                    ),
                ],
                divider_after=True,
            )

            # Add MITRE ATT&CK mapped threats first (if any)
            if state.threat_indicators:
                mitre_items = [
                    ListItem(
                        text=f"[{t.get('mitre_technique', '')}] {t.get('mitre_name', 'Unknown')}",
                        details=f"Tactic: {t.get('mitre_tactic', 'N/A')} | {t.get('problem_name', '')}",
                        severity=Severity.CRITICAL if t.get("risk_level") == "CRITICAL" else Severity.HIGH,
                    )
                    for t in state.threat_indicators[:5]
                ]
                response.add_section(
                    title="MITRE ATT&CK Indicators",
                    list_items=mitre_items,
                    divider_after=True,
                )

            # Add Cloud Guard problems
            if state.cloud_guard_problems:
                problem_items = [
                    ListItem(
                        text=p.get("name", "Unknown"),
                        details=f"Risk: {p.get('risk_level', 'N/A')}",
                        severity=Severity.CRITICAL if p.get("risk_level") == "CRITICAL" else Severity.HIGH,
                    )
                    for p in state.cloud_guard_problems[:5]
                ]
                response.add_section(
                    title="Cloud Guard Problems",
                    list_items=problem_items,
                    actions=[
                        ActionButton(
                            label="View in Console",
                            action_id="open_cloud_guard",
                            style="primary",
                        ),
                    ],
                    divider_after=True,
                )

            # Add auto-remediation actions (if any)
            if state.remediation_actions:
                remediation_items = [
                    ListItem(
                        text=f"{a.get('action', 'REMEDIATE')} {a.get('problem_id', 'unknown')}",
                        details=a.get("detail", a.get("error", "Auto-remediation attempted")),
                        severity=Severity.SUCCESS if a.get("success") else Severity.HIGH,
                    )
                    for a in state.remediation_actions[:5]
                ]
                response.add_section(
                    title="Auto-Remediation",
                    list_items=remediation_items,
                    divider_after=True,
                )

            # Add compliance issues
            if state.compliance_issues:
                compliance_items = [
                    ListItem(
                        text=i.get("title", "Unknown issue"),
                        details=i.get("description", ""),
                        severity=Severity.MEDIUM,
                    )
                    for i in state.compliance_issues[:5]
                ]
                response.add_section(
                    title="Compliance Issues",
                    list_items=compliance_items,
                )

            # Add footer
            response.footer = ResponseFooter(
                next_steps=[
                    "Review critical Cloud Guard problems",
                    "Address compliance gaps",
                    "Run `/oci security audit` for full assessment",
                ],
                help_text="Contact security team for incident response",
            )

            return {"result": self.format_response(response)}

    async def invoke(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Execute security analysis workflow."""
        with tracer.start_as_current_span("security.invoke") as span:
            context = context or {}
            span.set_attribute("query", query[:200] if query else "")
            span.set_attribute("compartment_id", context.get("compartment_id", ""))

            graph = self.build_graph()
            initial_state = SecurityState(
                query=query,
                compartment_id=context.get("compartment_id"),
                resource_id=context.get("resource_id"),
            )

            result = await graph.ainvoke(initial_state)
            return result.get("result", "No security issues found.")

    def _resolve_cloud_guard_tool_args(
        self, state: SecurityState
    ) -> tuple[str | None, dict[str, Any]]:
        """Pick the best Cloud Guard list tool and argument shape."""
        tool_name = self._select_tool(
            [
                "oci_security_cloudguard_list_problems",
                "oci_security_list_cloud_guard_problems",
                "oci_security_list_problems",
            ]
        )
        if not tool_name:
            return None, {}

        compartment_id = self._resolve_compartment_id(state)
        if tool_name == "oci_security_cloudguard_list_problems":
            if not compartment_id:
                self._logger.warning("Missing compartment_id for Cloud Guard list")
                return None, {}
            return tool_name, {
                "compartment_id": compartment_id,
                "status": "ACTIVE",
                "limit": 50,
                "response_format": "json",
            }

        if tool_name == "oci_security_list_cloud_guard_problems":
            args = {
                "limit": 50,
                "response_format": "json",
                "lifecycle_state": "ACTIVE",
            }
            if compartment_id:
                args["compartment_id"] = compartment_id
            return tool_name, args

        args = {"limit": 50}
        if compartment_id:
            args["compartment_id"] = compartment_id
        return tool_name, args

    def _resolve_compartment_id(self, state: SecurityState) -> str | None:
        """Resolve compartment id from state or environment."""
        if state.compartment_id:
            return state.compartment_id
        for key in ("OCI_COMPARTMENT_OCID", "OCI_TENANCY_OCID", "OCI_TENANCY_ID"):
            value = os.getenv(key)
            if value:
                return value
        return None

    def _select_tool(self, candidates: list[str]) -> str | None:
        """Select the first available tool from the catalog."""
        if not self.tools:
            return None
        for name in candidates:
            if self.tools.get_tool(name):
                return name
        return None

    def _extract_tool_payload(
        self, result: ToolCallResult | Any
    ) -> tuple[bool, Any, str | None]:
        """Normalize ToolCallResult payload into a usable structure."""
        if isinstance(result, ToolCallResult):
            if not result.success:
                return False, None, result.error or "Tool execution failed"
            payload = result.result
        else:
            payload = result

        if isinstance(payload, dict) and payload.get("error"):
            return False, payload, str(payload.get("error"))

        if isinstance(payload, str):
            cleaned = payload.strip()
            if cleaned:
                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        return True, payload, None

    def _normalize_cloud_guard_problems(self, payload: Any) -> list[dict[str, Any]]:
        """Normalize Cloud Guard problem payloads to a common shape."""
        raw_problems: list[Any] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("problems"), list):
                raw_problems = payload.get("problems", [])
            elif isinstance(payload.get("items"), list):
                raw_problems = payload.get("items", [])
        elif isinstance(payload, list):
            raw_problems = payload

        normalized: list[dict[str, Any]] = []
        for raw in raw_problems:
            if not isinstance(raw, dict):
                continue
            risk_level = raw.get("risk_level") or raw.get("severity") or raw.get("risk")
            name = raw.get("name") or raw.get("problem_name") or raw.get("detector_rule_id")
            detector_id = raw.get("detector_id") or raw.get("problem_name") or raw.get("detector_rule_id")
            status = raw.get("status") or raw.get("lifecycle_state")
            normalized.append(
                {
                    "id": raw.get("id") or raw.get("problem_id"),
                    "name": name or "Cloud Guard Problem",
                    "risk_level": str(risk_level).upper() if risk_level else "UNKNOWN",
                    "status": status,
                    "detector_id": detector_id or "",
                    "recommendation": raw.get("recommendation"),
                    "resource_name": raw.get("resource_name"),
                    "resource_type": raw.get("resource_type"),
                }
            )
        return normalized

    async def _auto_remediate_cloud_guard(
        self, problems: list[dict[str, Any]], user_intent: str | None = None
    ) -> list[dict[str, Any]]:
        """Attempt auto-remediation for Cloud Guard problems when enabled."""
        enabled, reason = self._auto_remediation_enabled()
        if not enabled:
            if reason:
                self._logger.debug("Auto-remediation disabled", reason=reason)
            return []

        tool_name = self._select_tool(["oci_security_cloudguard_remediate_problem"])
        if not tool_name:
            self._logger.info("Remediation tool unavailable in catalog")
            return []

        actions: list[dict[str, Any]] = []
        for problem in problems:
            problem_id = problem.get("id")
            if not problem_id:
                continue
            risk_level = str(problem.get("risk_level", "")).upper()
            status_value = problem.get("status")
            status = str(status_value).upper() if status_value else ""
            if risk_level not in {"CRITICAL", "HIGH"}:
                continue
            if status and status not in {"ACTIVE", "OPEN"}:
                continue

            try:
                if self._self_healing_enabled:
                    result = await self.healing_call_tool(
                        tool_name,
                        {
                            "problem_id": problem_id,
                            "action": "RESOLVE",
                            "comment": "Auto-remediation triggered by coordinator",
                        },
                        user_intent=user_intent,
                        validate=True,
                        correct_on_failure=False,
                    )
                else:
                    result = await self.call_tool(
                        tool_name,
                        {
                            "problem_id": problem_id,
                            "action": "RESOLVE",
                            "comment": "Auto-remediation triggered by coordinator",
                        },
                    )

                ok, payload, error = self._extract_tool_payload(result)
                actions.append(
                    {
                        "problem_id": problem_id,
                        "action": "RESOLVE",
                        "success": ok and not error,
                        "detail": payload if ok else None,
                        "error": error,
                    }
                )
            except Exception as exc:
                actions.append(
                    {
                        "problem_id": problem_id,
                        "action": "RESOLVE",
                        "success": False,
                        "error": str(exc),
                    }
                )

        return actions

    def _auto_remediation_enabled(self) -> tuple[bool, str | None]:
        """Check if auto-remediation is explicitly enabled."""
        config_flag = False
        if self.config:
            config_flag = bool(self.config.get("auto_remediate"))
        env_flag = os.getenv("SECURITY_AUTO_REMEDIATE", "").lower() in {"1", "true", "yes"}

        if not (config_flag or env_flag):
            return False, "flag_not_set"

        allow_mutations = os.getenv("ALLOW_MUTATIONS", "").lower() in {"1", "true", "yes"}
        if not allow_mutations:
            return False, "allow_mutations_disabled"

        return True, None
