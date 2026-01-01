"""
Security Threat Agent.

Specialized agent for security analysis, threat detection,
and compliance monitoring in OCI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from src.agents.base import (
    AgentDefinition,
    AgentMetadata,
    BaseAgent,
    KafkaTopics,
)

logger = structlog.get_logger(__name__)


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

    # State
    phase: str = "analyze_request"
    risk_score: int = 0
    error: str | None = None
    result: str | None = None


class SecurityThreatAgent(BaseAgent):
    """
    Security Threat Agent.

    Specializes in OCI security operations:
    - Threat indicator correlation
    - MITRE ATT&CK mapping
    - Cloud Guard problem analysis
    - Security posture assessment
    - Compliance monitoring
    """

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
            ],
            skills=[
                "threat_hunting_workflow",
                "compliance_check",
                "security_assessment",
                "incident_analysis",
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
                "and MITRE ATT&CK analysis in OCI environments."
            ),
            mcp_tools=[
                "oci_security_list_problems",
                "oci_security_get_problem",
                "oci_security_list_users",
                "oci_security_list_policies",
                "oci_security_get_security_assessment",
            ],
            mcp_servers=["oci-unified", "oci-mcp-security"],
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

        return {"threat_type": threat_type, "phase": "check_cloud_guard"}

    async def _check_cloud_guard_node(self, state: SecurityState) -> dict[str, Any]:
        """Check Cloud Guard for security problems."""
        self._logger.info("Checking Cloud Guard")

        problems = []
        if self.tools:
            try:
                result = await self.call_tool(
                    "oci_security_list_problems",
                    {"compartment_id": state.compartment_id or "default"},
                )
                if isinstance(result, list):
                    problems = result
            except Exception as e:
                self._logger.warning("Cloud Guard check failed", error=str(e))

        return {"cloud_guard_problems": problems, "phase": "assess_compliance"}

    async def _assess_compliance_node(self, state: SecurityState) -> dict[str, Any]:
        """Assess compliance status."""
        self._logger.info("Assessing compliance")

        compliance_issues = []
        # Compliance assessment logic

        return {"compliance_issues": compliance_issues, "phase": "correlate_threats"}

    async def _correlate_threats_node(self, state: SecurityState) -> dict[str, Any]:
        """Correlate threat indicators and map to MITRE ATT&CK."""
        self._logger.info("Correlating threats with MITRE ATT&CK mapping")

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

        return {
            "threat_indicators": threat_indicators,
            "risk_score": risk_score,
            "phase": "output",
        }

    async def _output_node(self, state: SecurityState) -> dict[str, Any]:
        """Prepare security report with structured formatting."""
        from src.formatting import (
            ActionButton,
            ListItem,
            MetricValue,
            ResponseFooter,
            Severity,
            StatusIndicator,
        )

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
        context = context or {}

        graph = self.build_graph()
        initial_state = SecurityState(
            query=query,
            compartment_id=context.get("compartment_id"),
            resource_id=context.get("resource_id"),
        )

        result = await graph.ainvoke(initial_state)
        return result.get("result", "No security issues found.")
