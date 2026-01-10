"""
Security DeepSkills.

Advanced skills for cloud security analysis, vulnerability scanning,
and security posture assessment using OCI Cloud Guard and Security services.

Skills:
- CloudGuardSkill: Analyze Cloud Guard problems and recommendations
- VulnerabilityScanSkill: Analyze vulnerability scan results
- SecurityPostureSkill: Comprehensive security posture assessment
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
# Cloud Guard Analysis Skill
# =============================================================================


@register_skill(
    skill_id="security_cloudguard_analysis",
    name="Cloud Guard Analysis",
    description="Analyze Cloud Guard security problems, detectors, and recommendations",
    compatible_agents=["security", "coordinator"],
    required_mcp_tools=[
        "oci_security_cloudguard_list_problems",
        "oci_security_cloudguard_cloudguard_get_security_score",
        "oci_security_cloudguard_cloudguard_list_recommendations",
    ],
    requires_code_execution=True,
    tags=["security", "cloudguard", "compliance", "risk"],
)
class CloudGuardSkill(DeepSkill):
    """
    Analyzes Cloud Guard security data.

    Combines problem detection, security scoring, and recommendations
    to provide actionable security insights.
    """

    CLOUDGUARD_ANALYSIS_CODE = """
import json
from collections import defaultdict
from datetime import datetime

def analyze_cloudguard(problems_data, score_data, recommendations_data):
    \"\"\"
    Analyze Cloud Guard security data.

    Returns:
        Dict with Cloud Guard analysis results
    \"\"\"
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "security_score": {},
        "problems_summary": {},
        "top_problems": [],
        "recommendations": [],
        "risk_assessment": {}
    }

    # Parse security score
    if isinstance(score_data, str):
        try:
            score_data = json.loads(score_data)
        except:
            score_data = {}

    if isinstance(score_data, dict):
        results["security_score"] = {
            "score": score_data.get("score", 0),
            "grade": score_data.get("grade", score_data.get("security_rating", "Unknown")),
            "trend": score_data.get("trend", "stable")
        }

    # Parse problems
    if isinstance(problems_data, str):
        try:
            problems_data = json.loads(problems_data)
        except:
            problems_data = {}

    problems = problems_data.get("problems", problems_data.get("items", []))
    if not isinstance(problems, list):
        problems = []

    # Categorize by severity
    severity_counts = defaultdict(int)
    resource_types = defaultdict(int)
    detector_types = defaultdict(int)

    for problem in problems:
        if not isinstance(problem, dict):
            continue

        severity = problem.get("risk_level", problem.get("severity", "UNKNOWN"))
        severity_counts[severity] += 1

        resource_type = problem.get("resource_type", "Unknown")
        resource_types[resource_type] += 1

        detector = problem.get("detector_id", problem.get("detector_type", "Unknown"))
        detector_types[detector] += 1

    results["problems_summary"] = {
        "total_problems": len(problems),
        "by_severity": dict(severity_counts),
        "by_resource_type": dict(resource_types),
        "by_detector": dict(detector_types)
    }

    # Get top problems
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "MINOR": 4}
    sorted_problems = sorted(
        problems,
        key=lambda x: (severity_order.get(x.get("risk_level", "MINOR"), 5), x.get("time_first_detected", ""))
    )

    results["top_problems"] = [
        {
            "id": p.get("id", "unknown"),
            "name": p.get("problem_name", p.get("name", "Unknown")),
            "severity": p.get("risk_level", p.get("severity", "UNKNOWN")),
            "resource_type": p.get("resource_type", "Unknown"),
            "recommendation": p.get("recommendation", "Review and remediate"),
            "first_detected": p.get("time_first_detected", "Unknown")
        }
        for p in sorted_problems[:10]
    ]

    # Parse recommendations
    if isinstance(recommendations_data, str):
        try:
            recommendations_data = json.loads(recommendations_data)
        except:
            recommendations_data = {}

    recs = recommendations_data.get("recommendations", recommendations_data.get("items", []))
    if not isinstance(recs, list):
        recs = []

    results["recommendations"] = [
        {
            "name": r.get("name", "Unknown"),
            "description": r.get("description", ""),
            "severity": r.get("risk_level", r.get("severity", "MEDIUM")),
            "affected_resources": r.get("resource_count", 0)
        }
        for r in recs[:10]
    ]

    # Risk assessment
    critical_count = severity_counts.get("CRITICAL", 0)
    high_count = severity_counts.get("HIGH", 0)

    if critical_count > 0:
        risk_level = "critical"
        risk_description = f"{critical_count} critical security issues require immediate attention"
    elif high_count > 5:
        risk_level = "high"
        risk_description = f"{high_count} high-severity issues detected"
    elif high_count > 0:
        risk_level = "elevated"
        risk_description = f"{high_count} high-severity issues need review"
    else:
        risk_level = "moderate"
        risk_description = "No critical or high-severity issues detected"

    results["risk_assessment"] = {
        "level": risk_level,
        "description": risk_description,
        "critical_count": critical_count,
        "high_count": high_count,
        "priority_actions": []
    }

    # Generate priority actions
    if critical_count > 0:
        results["risk_assessment"]["priority_actions"].append({
            "action": "Remediate critical security problems immediately",
            "impact": "high",
            "estimated_effort": "varies"
        })

    if severity_counts.get("HIGH", 0) > 10:
        results["risk_assessment"]["priority_actions"].append({
            "action": "Review and prioritize high-severity findings",
            "impact": "high",
            "estimated_effort": "medium"
        })

    return results

# Execute analysis
result = analyze_cloudguard(problems_data, score_data, recommendations_data)
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute Cloud Guard analysis."""
        logger.info("cloudguard_skill_executing", session_id=context.session_id)

        try:
            compartment_id = context.parameters.get("compartment_id")
            if not compartment_id:
                return SkillResult(
                    success=False,
                    error="compartment_id is required",
                )

            severity = context.parameters.get("severity")  # Optional filter

            # Get Cloud Guard problems
            problems_params = {
                "compartment_id": compartment_id,
                "limit": 100,
            }
            if severity:
                problems_params["severity"] = severity

            problems_result = await self.call_mcp_tool(
                context,
                "oci_security_cloudguard_list_problems",
                problems_params,
            )

            # Get security score
            score_result = await self.call_mcp_tool(
                context,
                "oci_security_cloudguard_cloudguard_get_security_score",
                {"compartment_id": compartment_id},
            )

            # Get recommendations
            recommendations_result = await self.call_mcp_tool(
                context,
                "oci_security_cloudguard_cloudguard_list_recommendations",
                {
                    "compartment_id": compartment_id,
                    "limit": 50,
                },
            )

            # Execute analysis
            analysis = await self.execute_code(
                context,
                self.CLOUDGUARD_ANALYSIS_CODE,
                variables={
                    "problems_data": problems_result,
                    "score_data": score_result,
                    "recommendations_data": recommendations_result,
                },
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    error=f"Cloud Guard analysis failed: {analysis.error}",
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "compartment_id": compartment_id,
                    "severity_filter": severity,
                },
            )

        except Exception as e:
            logger.error("cloudguard_skill_failed", error=str(e))
            return SkillResult(success=False, error=str(e))


# =============================================================================
# Vulnerability Scan Skill
# =============================================================================


@register_skill(
    skill_id="security_vulnerability_scan",
    name="Vulnerability Scan Analysis",
    description="Analyze host and container vulnerability scan results",
    compatible_agents=["security", "coordinator"],
    required_mcp_tools=[
        "oci_security_vss_vss_list_host_scans",
        "oci_security_vss_vss_list_container_scans",
    ],
    requires_code_execution=True,
    tags=["security", "vulnerability", "scanning", "containers", "hosts"],
)
class VulnerabilityScanSkill(DeepSkill):
    """
    Analyzes vulnerability scanning results.

    Combines host and container scan data to provide
    comprehensive vulnerability assessment.
    """

    VULNERABILITY_ANALYSIS_CODE = """
import json
from collections import defaultdict
from datetime import datetime

def analyze_vulnerabilities(host_scans, container_scans):
    \"\"\"
    Analyze vulnerability scan results.

    Returns:
        Dict with vulnerability analysis results
    \"\"\"
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "hosts": {},
            "containers": {},
            "total_vulnerabilities": 0
        },
        "severity_distribution": {},
        "top_vulnerabilities": [],
        "affected_resources": [],
        "recommendations": []
    }

    severity_counts = defaultdict(int)
    all_vulns = []

    # Process host scans
    if isinstance(host_scans, str):
        try:
            host_scans = json.loads(host_scans)
        except:
            host_scans = {}

    host_results = host_scans.get("results", host_scans.get("items", []))
    if not isinstance(host_results, list):
        host_results = []

    hosts_scanned = set()
    host_vulns = 0

    for scan in host_results:
        if not isinstance(scan, dict):
            continue

        host_id = scan.get("instance_id", scan.get("target_id", "unknown"))
        hosts_scanned.add(host_id)

        vulns = scan.get("vulnerabilities", scan.get("findings", []))
        if not isinstance(vulns, list):
            continue

        for vuln in vulns:
            if not isinstance(vuln, dict):
                continue

            severity = vuln.get("severity", "MEDIUM").upper()
            severity_counts[severity] += 1
            host_vulns += 1

            all_vulns.append({
                "type": "host",
                "resource_id": host_id,
                "cve_id": vuln.get("cve_id", vuln.get("id", "Unknown")),
                "severity": severity,
                "package": vuln.get("package", vuln.get("component", "Unknown")),
                "description": vuln.get("description", "")[:200]
            })

    results["summary"]["hosts"] = {
        "scanned": len(hosts_scanned),
        "vulnerabilities": host_vulns
    }

    # Process container scans
    if isinstance(container_scans, str):
        try:
            container_scans = json.loads(container_scans)
        except:
            container_scans = {}

    container_results = container_scans.get("results", container_scans.get("items", []))
    if not isinstance(container_results, list):
        container_results = []

    images_scanned = set()
    container_vulns = 0

    for scan in container_results:
        if not isinstance(scan, dict):
            continue

        image_id = scan.get("image_id", scan.get("image", "unknown"))
        images_scanned.add(image_id)

        vulns = scan.get("vulnerabilities", scan.get("findings", []))
        if not isinstance(vulns, list):
            continue

        for vuln in vulns:
            if not isinstance(vuln, dict):
                continue

            severity = vuln.get("severity", "MEDIUM").upper()
            severity_counts[severity] += 1
            container_vulns += 1

            all_vulns.append({
                "type": "container",
                "resource_id": image_id,
                "cve_id": vuln.get("cve_id", vuln.get("id", "Unknown")),
                "severity": severity,
                "package": vuln.get("package", vuln.get("component", "Unknown")),
                "description": vuln.get("description", "")[:200]
            })

    results["summary"]["containers"] = {
        "images_scanned": len(images_scanned),
        "vulnerabilities": container_vulns
    }

    results["summary"]["total_vulnerabilities"] = host_vulns + container_vulns
    results["severity_distribution"] = dict(severity_counts)

    # Sort vulnerabilities by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_vulns = sorted(
        all_vulns,
        key=lambda x: severity_order.get(x["severity"], 4)
    )
    results["top_vulnerabilities"] = sorted_vulns[:20]

    # Identify most affected resources
    resource_vuln_counts = defaultdict(lambda: {"critical": 0, "high": 0, "total": 0})
    for vuln in all_vulns:
        resource_id = vuln["resource_id"]
        resource_vuln_counts[resource_id]["total"] += 1
        if vuln["severity"] == "CRITICAL":
            resource_vuln_counts[resource_id]["critical"] += 1
        elif vuln["severity"] == "HIGH":
            resource_vuln_counts[resource_id]["high"] += 1

    results["affected_resources"] = sorted(
        [
            {"resource_id": rid, **counts}
            for rid, counts in resource_vuln_counts.items()
        ],
        key=lambda x: (x["critical"], x["high"], x["total"]),
        reverse=True
    )[:10]

    # Generate recommendations
    critical_count = severity_counts.get("CRITICAL", 0)
    high_count = severity_counts.get("HIGH", 0)

    if critical_count > 0:
        results["recommendations"].append({
            "priority": "critical",
            "action": "Patch critical vulnerabilities immediately",
            "detail": f"{critical_count} critical CVEs require immediate patching",
            "resources_affected": len([r for r in results["affected_resources"] if r["critical"] > 0])
        })

    if high_count > 10:
        results["recommendations"].append({
            "priority": "high",
            "action": "Schedule high-severity patching",
            "detail": f"{high_count} high-severity vulnerabilities detected"
        })

    if len(images_scanned) > 0 and container_vulns > 50:
        results["recommendations"].append({
            "priority": "medium",
            "action": "Review container base images",
            "detail": "Consider using minimal base images to reduce attack surface"
        })

    return results

# Execute analysis
result = analyze_vulnerabilities(host_scans, container_scans)
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute vulnerability scan analysis."""
        logger.info("vulnerability_scan_skill_executing", session_id=context.session_id)

        try:
            compartment_id = context.parameters.get("compartment_id")
            if not compartment_id:
                return SkillResult(
                    success=False,
                    error="compartment_id is required",
                )

            # Get host scans
            host_result = await self.call_mcp_tool(
                context,
                "oci_security_vss_vss_list_host_scans",
                {
                    "compartment_id": compartment_id,
                    "limit": 100,
                },
            )

            # Get container scans
            container_result = await self.call_mcp_tool(
                context,
                "oci_security_vss_vss_list_container_scans",
                {
                    "compartment_id": compartment_id,
                    "limit": 100,
                },
            )

            # Execute analysis
            analysis = await self.execute_code(
                context,
                self.VULNERABILITY_ANALYSIS_CODE,
                variables={
                    "host_scans": host_result,
                    "container_scans": container_result,
                },
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    error=f"Vulnerability analysis failed: {analysis.error}",
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={"compartment_id": compartment_id},
            )

        except Exception as e:
            logger.error("vulnerability_scan_skill_failed", error=str(e))
            return SkillResult(success=False, error=str(e))


# =============================================================================
# Security Posture Skill
# =============================================================================


@register_skill(
    skill_id="security_posture_assessment",
    name="Security Posture Assessment",
    description="Comprehensive security posture assessment combining multiple security data sources",
    compatible_agents=["security", "coordinator"],
    required_mcp_tools=[
        "oci_security_skill_skill_security_posture_summary",
        "oci_security_audit_audit_list_events",
    ],
    requires_code_execution=True,
    tags=["security", "posture", "audit", "compliance", "assessment"],
)
class SecurityPostureSkill(DeepSkill):
    """
    Comprehensive security posture assessment.

    Combines Cloud Guard, audit events, and security configurations
    to provide a holistic security posture view.
    """

    POSTURE_ANALYSIS_CODE = """
import json
from collections import defaultdict
from datetime import datetime

def analyze_security_posture(posture_summary, audit_events):
    \"\"\"
    Analyze overall security posture.

    Returns:
        Dict with comprehensive security posture assessment
    \"\"\"
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "posture_score": 0,
        "posture_grade": "F",
        "summary": {},
        "risk_areas": [],
        "compliance_gaps": [],
        "recent_activity": [],
        "recommendations": []
    }

    # Process posture summary
    if isinstance(posture_summary, str):
        try:
            posture_summary = json.loads(posture_summary)
        except:
            posture_summary = {}

    if isinstance(posture_summary, dict):
        results["summary"]["cloudguard"] = {
            "security_score": posture_summary.get("security_score", 0),
            "total_problems": posture_summary.get("total_problems", 0),
            "critical_problems": posture_summary.get("critical_problems", 0),
            "high_problems": posture_summary.get("high_problems", 0)
        }

        # Extract risk areas from posture summary
        problems = posture_summary.get("top_problems", [])
        if isinstance(problems, list):
            for problem in problems[:5]:
                if isinstance(problem, dict):
                    results["risk_areas"].append({
                        "area": problem.get("resource_type", "Unknown"),
                        "issue": problem.get("name", "Security issue"),
                        "severity": problem.get("severity", "MEDIUM"),
                        "impact": "high" if problem.get("severity") in ["CRITICAL", "HIGH"] else "medium"
                    })

    # Process audit events
    if isinstance(audit_events, str):
        try:
            audit_events = json.loads(audit_events)
        except:
            audit_events = {}

    events = audit_events.get("events", audit_events.get("items", []))
    if not isinstance(events, list):
        events = []

    # Analyze audit events
    event_types = defaultdict(int)
    suspicious_events = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type", event.get("type", "Unknown"))
        event_types[event_type] += 1

        # Flag suspicious events
        source = event.get("source", "")
        action = event.get("action", "")

        if any(s in action.lower() for s in ["delete", "terminate", "modify policy"]):
            suspicious_events.append({
                "event": event_type,
                "action": action,
                "source": source,
                "time": event.get("event_time", event.get("timestamp", "Unknown"))
            })

    results["summary"]["audit"] = {
        "total_events": len(events),
        "event_types": dict(event_types),
        "suspicious_events": len(suspicious_events)
    }

    results["recent_activity"] = suspicious_events[:10]

    # Calculate posture score
    base_score = 100
    deductions = 0

    # Deduct for Cloud Guard issues
    cloudguard = results["summary"].get("cloudguard", {})
    deductions += cloudguard.get("critical_problems", 0) * 15
    deductions += cloudguard.get("high_problems", 0) * 5
    deductions += min(cloudguard.get("total_problems", 0), 20)

    # Deduct for suspicious audit events
    deductions += len(suspicious_events) * 2

    posture_score = max(0, base_score - deductions)
    results["posture_score"] = posture_score

    # Assign grade
    if posture_score >= 90:
        results["posture_grade"] = "A"
    elif posture_score >= 80:
        results["posture_grade"] = "B"
    elif posture_score >= 70:
        results["posture_grade"] = "C"
    elif posture_score >= 60:
        results["posture_grade"] = "D"
    else:
        results["posture_grade"] = "F"

    # Generate recommendations
    if cloudguard.get("critical_problems", 0) > 0:
        results["recommendations"].append({
            "priority": "critical",
            "category": "Cloud Guard",
            "action": "Address critical security problems",
            "detail": f"{cloudguard['critical_problems']} critical issues require immediate remediation"
        })

    if len(suspicious_events) > 5:
        results["recommendations"].append({
            "priority": "high",
            "category": "Audit",
            "action": "Review suspicious activity",
            "detail": f"{len(suspicious_events)} potentially suspicious events detected"
        })

    if posture_score < 70:
        results["recommendations"].append({
            "priority": "high",
            "category": "Overall",
            "action": "Improve security posture",
            "detail": f"Current grade: {results['posture_grade']} - implement security best practices"
        })

    return results

# Execute analysis
result = analyze_security_posture(posture_summary, audit_events)
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute security posture assessment."""
        logger.info("security_posture_skill_executing", session_id=context.session_id)

        try:
            compartment_id = context.parameters.get("compartment_id")
            if not compartment_id:
                return SkillResult(
                    success=False,
                    error="compartment_id is required",
                )

            hours_back = context.parameters.get("hours_back", 24)

            # Get posture summary
            posture_result = await self.call_mcp_tool(
                context,
                "oci_security_skill_skill_security_posture_summary",
                {
                    "compartment_id": compartment_id,
                    "problem_limit": 20,
                    "recommendation_limit": 10,
                },
            )

            # Get audit events
            audit_result = await self.call_mcp_tool(
                context,
                "oci_security_audit_audit_list_events",
                {
                    "compartment_id": compartment_id,
                    "hours_back": hours_back,
                    "limit": 100,
                },
            )

            # Execute analysis
            analysis = await self.execute_code(
                context,
                self.POSTURE_ANALYSIS_CODE,
                variables={
                    "posture_summary": posture_result,
                    "audit_events": audit_result,
                },
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    error=f"Security posture analysis failed: {analysis.error}",
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "compartment_id": compartment_id,
                    "hours_back": hours_back,
                },
            )

        except Exception as e:
            logger.error("security_posture_skill_failed", error=str(e))
            return SkillResult(success=False, error=str(e))

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test security posture skill connectivity."""
        try:
            health = await self.call_mcp_tool(
                context,
                "oci_security_health",
                {"deep": False},
            )
            success = health.get("status") in ["healthy", "ok"]
            return SkillResult(
                success=success,
                data={"test": "security_health", "result": health} if success else None,
                error=None if success else "Security health check failed"
            )
        except Exception as e:
            logger.error("security_posture_skill_self_test_failed", error=str(e))
            return SkillResult(success=False, error=str(e), error_type="SelfTestError")
