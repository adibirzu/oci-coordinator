"""
Log Analytics DeepSkills.

Advanced skills for log pattern detection, security event analysis,
and MITRE ATT&CK technique identification using OCI Logging Analytics.

Skills:
- LogPatternSkill: Detect patterns and anomalies in log data
- SecurityEventSkill: Analyze security-related events
- MITREAnalysisSkill: Map log events to MITRE ATT&CK techniques
"""

from typing import Any, Dict, List, Optional
import structlog

from src.agents.core.deep_skills import (
    DeepSkill,
    SkillContext,
    SkillResult,
    register_skill,
)

logger = structlog.get_logger()


# =============================================================================
# Log Pattern Detection Skill
# =============================================================================


@register_skill(
    skill_id="logan_pattern_detection",
    name="Log Pattern Detection",
    description="Detect patterns, anomalies, and trends in log data using statistical analysis",
    compatible_agents=["log_analytics", "coordinator"],
    required_mcp_tools=["oci_logan_execute_query", "oci_logan_detect_anomalies"],
    requires_code_execution=True,
    tags=["logs", "patterns", "anomaly", "analytics"],
)
class LogPatternSkill(DeepSkill):
    """
    Analyzes log data to detect patterns and anomalies.

    Uses OCI Logging Analytics queries combined with statistical
    analysis to identify unusual patterns in log data.
    """

    # Query for log volume by source
    LOG_VOLUME_QUERY = """* | stats count as logcount by 'Log Source' | sort -logcount"""

    # Query for error patterns
    ERROR_PATTERN_QUERY = """Severity = 'error' | stats count as errors by 'Log Source', Message | sort -errors | head 20"""

    # Analysis code for pattern detection
    PATTERN_ANALYSIS_CODE = """
import json
from collections import defaultdict
from datetime import datetime

def analyze_log_patterns(volume_data, anomaly_data, error_data):
    \"\"\"
    Analyze log patterns combining volume, anomaly, and error data.

    Returns:
        Dict with pattern analysis results
    \"\"\"
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {},
        "patterns": [],
        "anomalies": [],
        "recommendations": []
    }

    # Parse volume data
    if isinstance(volume_data, str):
        try:
            volume_data = json.loads(volume_data)
        except:
            volume_data = {}

    # Analyze log sources
    sources = volume_data.get("results", []) if isinstance(volume_data, dict) else []
    total_logs = sum(int(s.get("logcount", 0)) for s in sources if isinstance(s, dict))

    results["summary"]["total_log_volume"] = total_logs
    results["summary"]["source_count"] = len(sources)

    # Identify high-volume sources (potential noise)
    high_volume = [
        {"source": s.get("Log Source", "unknown"), "count": int(s.get("logcount", 0))}
        for s in sources
        if isinstance(s, dict) and int(s.get("logcount", 0)) > total_logs * 0.2
    ]
    if high_volume:
        results["patterns"].append({
            "type": "high_volume_sources",
            "description": "Sources generating >20% of total logs",
            "sources": high_volume
        })

    # Process anomaly data
    if isinstance(anomaly_data, str):
        try:
            anomaly_data = json.loads(anomaly_data)
        except:
            anomaly_data = {}

    anomalies = anomaly_data.get("anomalies", []) if isinstance(anomaly_data, dict) else []
    for anomaly in anomalies:
        if isinstance(anomaly, dict):
            results["anomalies"].append({
                "source": anomaly.get("log_source", "unknown"),
                "type": anomaly.get("anomaly_type", "volume"),
                "severity": anomaly.get("severity", "medium"),
                "deviation": anomaly.get("deviation_percent", 0)
            })

    results["summary"]["anomaly_count"] = len(results["anomalies"])

    # Process error data
    if isinstance(error_data, str):
        try:
            error_data = json.loads(error_data)
        except:
            error_data = {}

    errors = error_data.get("results", []) if isinstance(error_data, dict) else []
    error_by_source = defaultdict(int)
    for err in errors:
        if isinstance(err, dict):
            source = err.get("Log Source", "unknown")
            count = int(err.get("errors", 0))
            error_by_source[source] += count

    results["summary"]["total_errors"] = sum(error_by_source.values())
    results["summary"]["error_sources"] = len(error_by_source)

    # Identify problematic sources
    if error_by_source:
        top_error_sources = sorted(
            error_by_source.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        results["patterns"].append({
            "type": "error_concentration",
            "description": "Top sources by error count",
            "sources": [{"source": s, "errors": c} for s, c in top_error_sources]
        })

    # Generate recommendations
    if results["summary"]["anomaly_count"] > 0:
        results["recommendations"].append({
            "priority": "high",
            "action": "Investigate anomalous log sources",
            "detail": f"{results['summary']['anomaly_count']} sources showing unusual patterns"
        })

    if results["summary"]["total_errors"] > 100:
        results["recommendations"].append({
            "priority": "medium",
            "action": "Review error logs",
            "detail": f"{results['summary']['total_errors']} errors detected across {results['summary']['error_sources']} sources"
        })

    if high_volume:
        results["recommendations"].append({
            "priority": "low",
            "action": "Consider log filtering/sampling",
            "detail": f"{len(high_volume)} sources generating majority of log volume"
        })

    return results

# Execute analysis
result = analyze_log_patterns(volume_data, anomaly_data, error_data)
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute log pattern detection analysis."""
        logger.info("log_pattern_skill_executing", session_id=context.session_id)

        try:
            # Get parameters
            time_range = context.parameters.get("time_range", "24h")
            compartment_id = context.parameters.get("compartment_id")

            # Execute log volume query
            volume_result = await self.call_mcp_tool(
                context,
                "oci_logan_execute_query",
                {
                    "query": self.LOG_VOLUME_QUERY,
                    "timeRange": time_range,
                    "format": "json",
                    **({"compartmentId": compartment_id} if compartment_id else {}),
                },
            )

            # Detect anomalies
            anomaly_result = await self.call_mcp_tool(
                context,
                "oci_logan_detect_anomalies",
                {
                    "detection_hours": 24 if time_range == "24h" else 168,
                    **({"compartment_id": compartment_id} if compartment_id else {}),
                },
            )

            # Get error patterns
            error_result = await self.call_mcp_tool(
                context,
                "oci_logan_execute_query",
                {
                    "query": self.ERROR_PATTERN_QUERY,
                    "timeRange": time_range,
                    "format": "json",
                    **({"compartmentId": compartment_id} if compartment_id else {}),
                },
            )

            # Execute analysis code
            analysis = await self.execute_code(
                context,
                self.PATTERN_ANALYSIS_CODE,
                variables={
                    "volume_data": volume_result,
                    "anomaly_data": anomaly_result,
                    "error_data": error_result,
                },
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    error=f"Analysis failed: {analysis.error}",
                    data={"raw_results": {
                        "volume": volume_result,
                        "anomalies": anomaly_result,
                        "errors": error_result,
                    }},
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "time_range": time_range,
                    "compartment_id": compartment_id,
                    "queries_executed": 3,
                },
            )

        except Exception as e:
            logger.error("log_pattern_skill_failed", error=str(e))
            return SkillResult(success=False, error=str(e))

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test log pattern detection connectivity."""
        try:
            # Test Logan health
            health = await self.call_mcp_tool(
                context,
                "oci_logan_health",
                {"detail": False},
            )
            success = health.get("status") == "ok"
            return SkillResult(
                success=success,
                data={"test": "logan_health", "result": health} if success else None,
                error=None if success else "Logan health check failed"
            )
        except Exception as e:
            logger.error("log_pattern_skill_self_test_failed", error=str(e))
            return SkillResult(success=False, error=str(e), error_type="SelfTestError")


# =============================================================================
# Security Event Analysis Skill
# =============================================================================


@register_skill(
    skill_id="logan_security_events",
    name="Security Event Analysis",
    description="Analyze security-related events including authentication, privilege escalation, and network anomalies",
    compatible_agents=["log_analytics", "security", "coordinator"],
    required_mcp_tools=["oci_logan_search_security_events", "oci_logan_run_security_query"],
    requires_code_execution=True,
    tags=["logs", "security", "authentication", "forensics"],
)
class SecurityEventSkill(DeepSkill):
    """
    Analyzes security-related events in log data.

    Combines predefined security queries with custom analysis
    to identify authentication failures, privilege escalation,
    and potential security incidents.
    """

    SECURITY_ANALYSIS_CODE = """
import json
from collections import defaultdict
from datetime import datetime

def analyze_security_events(auth_events, priv_events, network_events):
    \"\"\"
    Analyze security events across authentication, privilege, and network categories.

    Returns:
        Dict with security analysis results
    \"\"\"
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "risk_score": 0,
        "summary": {
            "authentication": {},
            "privilege": {},
            "network": {}
        },
        "incidents": [],
        "recommendations": []
    }

    risk_score = 0

    # Parse authentication events
    if isinstance(auth_events, str):
        try:
            auth_events = json.loads(auth_events)
        except:
            auth_events = {}

    auth_results = auth_events.get("results", []) if isinstance(auth_events, dict) else []
    failed_logins = defaultdict(int)

    for event in auth_results:
        if isinstance(event, dict):
            user = event.get("user", event.get("User Name", "unknown"))
            count = int(event.get("count", event.get("failures", 1)))
            failed_logins[user] += count

    results["summary"]["authentication"]["total_failures"] = sum(failed_logins.values())
    results["summary"]["authentication"]["affected_users"] = len(failed_logins)

    # Flag brute force attempts (>10 failures per user)
    brute_force_suspects = [
        {"user": user, "failures": count}
        for user, count in failed_logins.items()
        if count > 10
    ]
    if brute_force_suspects:
        risk_score += 30
        results["incidents"].append({
            "type": "brute_force_attempt",
            "severity": "high",
            "description": f"{len(brute_force_suspects)} users with >10 failed login attempts",
            "affected": brute_force_suspects
        })

    # Parse privilege events
    if isinstance(priv_events, str):
        try:
            priv_events = json.loads(priv_events)
        except:
            priv_events = {}

    priv_results = priv_events.get("results", []) if isinstance(priv_events, dict) else []
    priv_escalations = []

    for event in priv_results:
        if isinstance(event, dict):
            priv_escalations.append({
                "user": event.get("user", "unknown"),
                "action": event.get("action", event.get("Message", "unknown")),
                "host": event.get("host", event.get("Host Name", "unknown"))
            })

    results["summary"]["privilege"]["escalation_count"] = len(priv_escalations)

    if len(priv_escalations) > 20:
        risk_score += 20
        results["incidents"].append({
            "type": "excessive_privilege_escalation",
            "severity": "medium",
            "description": f"{len(priv_escalations)} privilege escalation events detected",
            "sample": priv_escalations[:5]
        })

    # Parse network events
    if isinstance(network_events, str):
        try:
            network_events = json.loads(network_events)
        except:
            network_events = {}

    network_results = network_events.get("results", []) if isinstance(network_events, dict) else []
    suspicious_ips = set()

    for event in network_results:
        if isinstance(event, dict):
            src_ip = event.get("source_ip", event.get("Source IP", ""))
            if src_ip and not src_ip.startswith(("10.", "192.168.", "172.")):
                suspicious_ips.add(src_ip)

    results["summary"]["network"]["suspicious_external_ips"] = len(suspicious_ips)

    if suspicious_ips:
        risk_score += 10 * min(len(suspicious_ips), 5)  # Cap at 50 points
        results["incidents"].append({
            "type": "external_network_activity",
            "severity": "medium" if len(suspicious_ips) < 10 else "high",
            "description": f"{len(suspicious_ips)} external IPs with suspicious activity",
            "ips": list(suspicious_ips)[:10]
        })

    # Calculate final risk score (0-100)
    results["risk_score"] = min(risk_score, 100)

    # Risk level classification
    if results["risk_score"] >= 70:
        results["risk_level"] = "critical"
    elif results["risk_score"] >= 50:
        results["risk_level"] = "high"
    elif results["risk_score"] >= 30:
        results["risk_level"] = "medium"
    else:
        results["risk_level"] = "low"

    # Generate recommendations
    if brute_force_suspects:
        results["recommendations"].append({
            "priority": "high",
            "action": "Investigate brute force attempts",
            "detail": "Consider blocking suspicious IPs and enabling MFA"
        })

    if len(priv_escalations) > 20:
        results["recommendations"].append({
            "priority": "medium",
            "action": "Review privilege escalation policy",
            "detail": "High volume of sudo/admin operations detected"
        })

    if suspicious_ips:
        results["recommendations"].append({
            "priority": "high",
            "action": "Investigate external IP activity",
            "detail": f"Review traffic from {len(suspicious_ips)} external sources"
        })

    return results

# Execute analysis
result = analyze_security_events(auth_events, priv_events, network_events)
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute security event analysis."""
        logger.info("security_event_skill_executing", session_id=context.session_id)

        try:
            time_range = context.parameters.get("time_range", "24h")
            compartment_id = context.parameters.get("compartment_id")

            # Query failed logins
            auth_result = await self.call_mcp_tool(
                context,
                "oci_logan_run_security_query",
                {
                    "query_type": "failed_logins",
                    "time_range_minutes": 1440 if time_range == "24h" else 10080,
                    **({"compartment_id": compartment_id} if compartment_id else {}),
                },
            )

            # Query privilege escalations
            priv_result = await self.call_mcp_tool(
                context,
                "oci_logan_run_security_query",
                {
                    "query_type": "privileged_operations",
                    "time_range_minutes": 1440 if time_range == "24h" else 10080,
                    **({"compartment_id": compartment_id} if compartment_id else {}),
                },
            )

            # Query network anomalies
            network_result = await self.call_mcp_tool(
                context,
                "oci_logan_run_security_query",
                {
                    "query_type": "network_anomalies",
                    "time_range_minutes": 1440 if time_range == "24h" else 10080,
                    **({"compartment_id": compartment_id} if compartment_id else {}),
                },
            )

            # Execute analysis
            analysis = await self.execute_code(
                context,
                self.SECURITY_ANALYSIS_CODE,
                variables={
                    "auth_events": auth_result,
                    "priv_events": priv_result,
                    "network_events": network_result,
                },
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    error=f"Security analysis failed: {analysis.error}",
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "time_range": time_range,
                    "queries_executed": 3,
                },
            )

        except Exception as e:
            logger.error("security_event_skill_failed", error=str(e))
            return SkillResult(success=False, error=str(e))


# =============================================================================
# MITRE ATT&CK Analysis Skill
# =============================================================================


@register_skill(
    skill_id="logan_mitre_analysis",
    name="MITRE ATT&CK Analysis",
    description="Map log events to MITRE ATT&CK framework techniques for threat detection",
    compatible_agents=["log_analytics", "security", "coordinator"],
    required_mcp_tools=["oci_logan_get_mitre_techniques"],
    requires_code_execution=True,
    tags=["logs", "security", "mitre", "threat", "forensics"],
)
class MITREAnalysisSkill(DeepSkill):
    """
    Maps log events to MITRE ATT&CK framework.

    Identifies tactics and techniques observed in log data
    to support threat hunting and incident response.
    """

    # MITRE technique categories to analyze
    MITRE_CATEGORIES = [
        "initial_access",
        "execution",
        "persistence",
        "privilege_escalation",
        "defense_evasion",
        "credential_access",
        "discovery",
        "lateral_movement",
    ]

    MITRE_ANALYSIS_CODE = """
import json
from collections import defaultdict
from datetime import datetime

# MITRE ATT&CK technique severity mapping
TECHNIQUE_SEVERITY = {
    "T1110": "high",      # Brute Force
    "T1003": "critical",  # OS Credential Dumping
    "T1059": "high",      # Command and Scripting Interpreter
    "T1078": "high",      # Valid Accounts
    "T1098": "high",      # Account Manipulation
    "T1136": "medium",    # Create Account
    "T1543": "high",      # Create or Modify System Process
    "T1548": "critical",  # Abuse Elevation Control Mechanism
    "T1021": "medium",    # Remote Services
    "T1018": "low",       # Remote System Discovery
    "T1082": "low",       # System Information Discovery
    "T1083": "low",       # File and Directory Discovery
}

def analyze_mitre_techniques(technique_data_list):
    \"\"\"
    Analyze MITRE ATT&CK techniques detected in logs.

    Returns:
        Dict with MITRE analysis results
    \"\"\"
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "threat_score": 0,
        "tactics_detected": [],
        "techniques": [],
        "attack_chain": [],
        "recommendations": []
    }

    technique_counts = defaultdict(int)
    tactic_counts = defaultdict(int)
    all_techniques = []

    # Process each category's results
    for category_data in technique_data_list:
        if isinstance(category_data, str):
            try:
                category_data = json.loads(category_data)
            except:
                continue

        if not isinstance(category_data, dict):
            continue

        techniques = category_data.get("techniques", category_data.get("results", []))
        if not isinstance(techniques, list):
            continue

        for tech in techniques:
            if not isinstance(tech, dict):
                continue

            tech_id = tech.get("technique_id", tech.get("id", "unknown"))
            tech_name = tech.get("technique_name", tech.get("name", "unknown"))
            tactic = tech.get("tactic", tech.get("category", "unknown"))
            count = int(tech.get("count", tech.get("occurrences", 1)))

            technique_counts[tech_id] += count
            tactic_counts[tactic] += count
            all_techniques.append({
                "id": tech_id,
                "name": tech_name,
                "tactic": tactic,
                "count": count,
                "severity": TECHNIQUE_SEVERITY.get(tech_id, "medium")
            })

    # Calculate threat score based on technique severity
    threat_score = 0
    for tech in all_techniques:
        severity = tech["severity"]
        if severity == "critical":
            threat_score += 25
        elif severity == "high":
            threat_score += 15
        elif severity == "medium":
            threat_score += 5
        else:
            threat_score += 2

    results["threat_score"] = min(threat_score, 100)

    # Threat level classification
    if results["threat_score"] >= 70:
        results["threat_level"] = "critical"
    elif results["threat_score"] >= 50:
        results["threat_level"] = "high"
    elif results["threat_score"] >= 30:
        results["threat_level"] = "elevated"
    else:
        results["threat_level"] = "low"

    # Top techniques
    sorted_techniques = sorted(all_techniques, key=lambda x: x["count"], reverse=True)
    results["techniques"] = sorted_techniques[:20]

    # Tactics summary
    results["tactics_detected"] = [
        {"tactic": tactic, "events": count}
        for tactic, count in sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    # Identify potential attack chains
    tactic_order = [
        "initial_access", "execution", "persistence",
        "privilege_escalation", "defense_evasion", "credential_access",
        "discovery", "lateral_movement", "collection", "exfiltration"
    ]

    detected_tactics = set(tactic_counts.keys())
    chain_tactics = [t for t in tactic_order if t in detected_tactics or t.replace("_", " ") in detected_tactics]

    if len(chain_tactics) >= 3:
        results["attack_chain"] = {
            "detected": True,
            "phases": chain_tactics,
            "description": f"Potential attack chain detected across {len(chain_tactics)} phases"
        }

    # Generate recommendations
    critical_techniques = [t for t in all_techniques if t["severity"] == "critical"]
    if critical_techniques:
        results["recommendations"].append({
            "priority": "critical",
            "action": "Immediate incident response required",
            "detail": f"{len(critical_techniques)} critical techniques detected: {', '.join(t['id'] for t in critical_techniques[:3])}"
        })

    high_techniques = [t for t in all_techniques if t["severity"] == "high"]
    if high_techniques:
        results["recommendations"].append({
            "priority": "high",
            "action": "Investigate high-severity techniques",
            "detail": f"{len(high_techniques)} high-severity techniques require investigation"
        })

    if results.get("attack_chain", {}).get("detected"):
        results["recommendations"].append({
            "priority": "critical",
            "action": "Review attack chain progression",
            "detail": "Multi-phase attack pattern detected - isolate affected systems"
        })

    return results

# Execute analysis
result = analyze_mitre_techniques(technique_data)
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute MITRE ATT&CK analysis."""
        logger.info("mitre_analysis_skill_executing", session_id=context.session_id)

        try:
            time_range = context.parameters.get("time_range", "30d")
            categories = context.parameters.get("categories", self.MITRE_CATEGORIES)

            # Collect technique data from multiple categories
            technique_results = []

            for category in categories:
                try:
                    result = await self.call_mcp_tool(
                        context,
                        "oci_logan_get_mitre_techniques",
                        {
                            "category": category,
                            "timeRange": time_range,
                            "format": "json",
                        },
                    )
                    technique_results.append(result)
                except Exception as e:
                    logger.warning(
                        "mitre_category_query_failed",
                        category=category,
                        error=str(e),
                    )
                    continue

            if not technique_results:
                return SkillResult(
                    success=False,
                    error="No MITRE technique data retrieved",
                )

            # Execute analysis
            analysis = await self.execute_code(
                context,
                self.MITRE_ANALYSIS_CODE,
                variables={"technique_data": technique_results},
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    error=f"MITRE analysis failed: {analysis.error}",
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "time_range": time_range,
                    "categories_analyzed": len(categories),
                    "categories_with_data": len(technique_results),
                },
            )

        except Exception as e:
            logger.error("mitre_analysis_skill_failed", error=str(e))
            return SkillResult(success=False, error=str(e))

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test MITRE analysis capability."""
        try:
            # Test Logan health
            health = await self.call_mcp_tool(
                context,
                "oci_logan_health",
                {"detail": False},
            )
            success = health.get("status") == "ok"
            return SkillResult(
                success=success,
                data={"test": "logan_mitre_health", "result": health} if success else None,
                error=None if success else "Logan MITRE health check failed"
            )
        except Exception as e:
            logger.error("mitre_skill_self_test_failed", error=str(e))
            return SkillResult(success=False, error=str(e), error_type="SelfTestError")
