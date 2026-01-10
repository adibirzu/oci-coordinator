"""
OCI Security MCP Tools.

Provides tools for security operations including Cloud Guard problems,
security scoring, and audit events.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from opentelemetry import trace

from src.mcp.server.auth import (
    get_audit_client,
    get_cloud_guard_client,
    get_identity_client,
    get_oci_config,
)

# Get tracer for security tools
_tracer = trace.get_tracer("mcp-oci-security")


async def _list_users_logic(
    compartment_id: str,
    limit: int = 20,
    format: str = "markdown"
) -> str:
    """Internal logic for listing users."""
    client = get_identity_client()

    try:
        response = client.list_users(compartment_id=compartment_id, limit=limit)
        users = response.data

        if format == "json":
            return json.dumps([{"name": u.name, "id": u.id, "state": u.lifecycle_state} for u in users], indent=2)

        lines = ["| Name | State | OCID |", "| --- | --- | --- |"]
        for u in users:
            lines.append(f"| {u.name} | {u.lifecycle_state} | `{u.id}` |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing users: {e}"


async def _list_cloud_guard_problems_logic(
    compartment_id: str | None = None,
    risk_level: str | None = None,
    limit: int = 50,
    profile: str | None = None,
) -> str:
    """List Cloud Guard problems."""
    with _tracer.start_as_current_span("mcp.security.list_cloud_guard_problems") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_cloud_guard_client(profile=profile)

            # Use tenancy as root compartment if not specified
            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)

            # Build request parameters
            # access_level is REQUIRED when compartment_id_in_subtree is True
            kwargs: dict[str, Any] = {
                "compartment_id": compartment,
                "compartment_id_in_subtree": True,
                "access_level": "ACCESSIBLE",
                "limit": limit,
            }
            if risk_level:
                kwargs["risk_level"] = risk_level
                span.set_attribute("risk_level", risk_level)

            # List problems
            response = client.list_problems(**kwargs)
            # response.data may be a ProblemCollection with .items attribute
            problems_data = response.data
            if hasattr(problems_data, "items"):
                problems = problems_data.items
            else:
                problems = problems_data if problems_data else []

            span.set_attribute("problem_count", len(problems))

            # Build typed response
            problem_list = []
            severity_counts: dict[str, int] = {}

            for problem in problems:
                severity = getattr(problem, "risk_level", "UNKNOWN")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

                problem_list.append({
                    "id": problem.id,
                    "name": getattr(problem, "detector_rule_id", "Unknown"),
                    "risk_level": severity,
                    "lifecycle_state": problem.lifecycle_state,
                    "resource_type": getattr(problem, "resource_type", "Unknown"),
                    "resource_name": getattr(problem, "resource_name", "Unknown"),
                    "labels": getattr(problem, "labels", []),
                    "time_first_detected": str(problem.time_first_detected) if hasattr(problem, "time_first_detected") else None,
                })

            return json.dumps({
                "type": "cloud_guard_problems",
                "count": len(problem_list),
                "severity_summary": severity_counts,
                "problems": problem_list,
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error listing Cloud Guard problems: {e}"})


async def _get_security_score_logic(
    compartment_id: str | None = None,
    profile: str | None = None,
) -> str:
    """Get Cloud Guard security score."""
    with _tracer.start_as_current_span("mcp.security.get_security_score") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_cloud_guard_client(profile=profile)

            # Use tenancy as root compartment if not specified
            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)

            # Get security zone scores
            try:
                # List security zones first
                zones_response = client.list_security_zones(compartment_id=compartment, limit=10)
                # response.data may be a SecurityZoneCollection with .items attribute
                zones_data = zones_response.data if hasattr(zones_response, "data") else []
                if hasattr(zones_data, "items"):
                    zones = zones_data.items
                else:
                    zones = zones_data if zones_data else []
            except Exception:
                zones = []

            # Get problem summary for scoring
            # access_level is REQUIRED when compartment_id_in_subtree is True
            try:
                problems_response = client.list_problems(
                    compartment_id=compartment,
                    compartment_id_in_subtree=True,
                    access_level="ACCESSIBLE",
                    limit=500,
                )
                # response.data may be a ProblemCollection with .items attribute
                problems_data = problems_response.data
                if hasattr(problems_data, "items"):
                    problems = problems_data.items
                else:
                    problems = problems_data if problems_data else []
            except Exception:
                problems = []

            # Calculate score based on problem severity
            critical_count = sum(1 for p in problems if getattr(p, "risk_level", "") == "CRITICAL")
            high_count = sum(1 for p in problems if getattr(p, "risk_level", "") == "HIGH")
            medium_count = sum(1 for p in problems if getattr(p, "risk_level", "") == "MEDIUM")
            low_count = sum(1 for p in problems if getattr(p, "risk_level", "") == "LOW")

            # Score calculation: 100 - (critical*15 + high*5 + medium*2 + low*0.5)
            score = max(0, 100 - (critical_count * 15 + high_count * 5 + medium_count * 2 + low_count * 0.5))
            score = round(min(100, score), 1)

            # Grade assignment
            if score >= 90:
                grade = "A"
            elif score >= 80:
                grade = "B"
            elif score >= 70:
                grade = "C"
            elif score >= 60:
                grade = "D"
            else:
                grade = "F"

            return json.dumps({
                "type": "security_score",
                "score": score,
                "grade": grade,
                "problem_counts": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": medium_count,
                    "low": low_count,
                    "total": len(problems),
                },
                "security_zones": len(zones),
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error getting security score: {e}"})


async def _list_audit_events_logic(
    compartment_id: str | None = None,
    hours_back: int = 24,
    limit: int = 100,
    profile: str | None = None,
) -> str:
    """List recent audit events."""
    with _tracer.start_as_current_span("mcp.security.list_audit_events") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_audit_client(profile=profile)

            # Use tenancy as root compartment if not specified
            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)
            span.set_attribute("hours_back", hours_back)

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)

            # List audit events (limit not supported by API, apply manually)
            response = client.list_events(
                compartment_id=compartment,
                start_time=start_time,
                end_time=end_time,
            )
            events = response.data[:limit] if limit else response.data

            span.set_attribute("event_count", len(events))

            # Categorize events
            event_types: dict[str, int] = {}
            event_list = []

            for event in events:
                event_type = getattr(event, "event_type", "Unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1

                # Extract event details
                data = getattr(event, "data", {})
                if hasattr(data, "resource_name"):
                    resource_name = data.resource_name
                else:
                    resource_name = "Unknown"

                event_list.append({
                    "event_id": event.event_id if hasattr(event, "event_id") else str(id(event)),
                    "event_type": event_type,
                    "event_time": str(event.event_time) if hasattr(event, "event_time") else None,
                    "source": getattr(event, "source", "Unknown"),
                    "resource_name": resource_name,
                    "principal_name": getattr(data, "principal_name", "Unknown") if hasattr(data, "principal_name") else "Unknown",
                })

            return json.dumps({
                "type": "audit_events",
                "count": len(event_list),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours_back,
                },
                "event_type_summary": event_types,
                "events": event_list[:50],  # Limit for display
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error listing audit events: {e}"})


async def _security_overview_logic(
    compartment_id: str | None = None,
    profile: str | None = None,
) -> str:
    """Get comprehensive security overview."""
    with _tracer.start_as_current_span("mcp.security.overview") as span:
        try:
            config = get_oci_config(profile=profile)

            # Use tenancy as root compartment if not specified
            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)

            # Get security score
            score_result = await _get_security_score_logic(compartment, profile)
            score_data = json.loads(score_result)

            # Get critical problems
            problems_result = await _list_cloud_guard_problems_logic(
                compartment_id=compartment,
                risk_level="CRITICAL",
                limit=10,
                profile=profile,
            )
            problems_data = json.loads(problems_result)

            # Get recent audit events summary (last 24h)
            audit_result = await _list_audit_events_logic(
                compartment_id=compartment,
                hours_back=24,
                limit=50,
                profile=profile,
            )
            audit_data = json.loads(audit_result)

            # Build overview
            overview = {
                "type": "security_overview",
                "posture": {
                    "score": score_data.get("score", 0),
                    "grade": score_data.get("grade", "N/A"),
                    "total_problems": score_data.get("problem_counts", {}).get("total", 0),
                },
                "critical_issues": {
                    "count": problems_data.get("count", 0),
                    "top_issues": problems_data.get("problems", [])[:5],
                },
                "audit_activity": {
                    "events_24h": audit_data.get("count", 0),
                    "event_types": len(audit_data.get("event_type_summary", {})),
                },
                "recommendations": [],
            }

            # Generate recommendations
            if score_data.get("problem_counts", {}).get("critical", 0) > 0:
                overview["recommendations"].append({
                    "priority": "critical",
                    "action": "Address critical Cloud Guard problems immediately",
                })
            if score_data.get("score", 100) < 70:
                overview["recommendations"].append({
                    "priority": "high",
                    "action": "Review and remediate security findings to improve posture score",
                })

            return json.dumps(overview)

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error getting security overview: {e}"})


def register_security_tools(mcp):
    @mcp.tool()
    async def oci_security_list_users(compartment_id: str, limit: int = 20, format: str = "markdown") -> str:
        """List IAM users in a compartment.

        Args:
            compartment_id: OCID of the compartment
            limit: Maximum users to return (default 20)
            format: Output format ('json' or 'markdown')

        Returns:
            List of IAM users with name, state, and OCID
        """
        return await _list_users_logic(compartment_id, limit, format)

    @mcp.tool()
    async def oci_security_cloudguard_list_problems(
        compartment_id: str | None = None,
        risk_level: str | None = None,
        limit: int = 50,
        profile: str | None = None,
    ) -> str:
        """List Cloud Guard security problems.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root)
            risk_level: Filter by risk level (CRITICAL, HIGH, MEDIUM, LOW)
            limit: Maximum problems to return (default 50)
            profile: OCI profile name

        Returns:
            JSON with Cloud Guard problems including severity summary
        """
        return await _list_cloud_guard_problems_logic(compartment_id, risk_level, limit, profile)

    @mcp.tool()
    async def oci_security_cloudguard_get_security_score(
        compartment_id: str | None = None,
        profile: str | None = None,
    ) -> str:
        """Get Cloud Guard security score.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root)
            profile: OCI profile name

        Returns:
            JSON with security score, grade, and problem counts
        """
        return await _get_security_score_logic(compartment_id, profile)

    @mcp.tool()
    async def oci_security_audit_list_events(
        compartment_id: str | None = None,
        hours_back: int = 24,
        limit: int = 100,
        profile: str | None = None,
    ) -> str:
        """List recent audit events.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root)
            hours_back: How many hours back to search (default 24)
            limit: Maximum events to return (default 100)
            profile: OCI profile name

        Returns:
            JSON with audit events including event type summary
        """
        return await _list_audit_events_logic(compartment_id, hours_back, limit, profile)

    @mcp.tool()
    async def oci_security_overview(
        compartment_id: str | None = None,
        profile: str | None = None,
    ) -> str:
        """Get comprehensive security posture overview.

        Combines Cloud Guard problems, security score, and recent audit activity
        to provide a holistic security view with recommendations.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root)
            profile: OCI profile name

        Returns:
            JSON with security posture score, critical issues, and recommendations
        """
        return await _security_overview_logic(compartment_id, profile)
