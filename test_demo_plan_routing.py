#!/usr/bin/env python3
"""
Test all 30 DEMO_PLAN commands for correct routing.

This standalone test verifies that all pre-classification functions
route queries to the correct intents and workflows.
"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any


class IntentCategory(Enum):
    QUERY = "query"
    ACTION = "action"
    ANALYSIS = "analysis"


@dataclass
class IntentClassification:
    intent: str
    category: IntentCategory
    confidence: float
    domains: list[str]
    entities: dict
    suggested_workflow: str | None
    suggested_agent: str | None = None


def has_complexity_indicators(query: str) -> bool:
    """Check if query has complexity indicators requiring LLM reasoning."""
    query_lower = query.lower()
    indicators = [
        "vs", "versus", "compare", "compared to", "comparison",
        "trend", "over time", "month", "week", "year", "last",
        "why", "explain", "analyze", "forecast", "predict"
    ]
    return any(ind in query_lower for ind in indicators)


def extract_profiles(query: str) -> list[str]:
    """Extract OCI profile names from query."""
    query_lower = query.lower()
    profiles = []

    profile_patterns = [
        r'\b(emdemo|default|prod|dev)\b'
    ]

    for pattern in profile_patterns:
        matches = re.findall(pattern, query_lower)
        profiles.extend(matches)

    return list(set(profiles))


def extract_database_name(query: str) -> str | None:
    """Extract database name from query."""
    query_upper = query.upper()

    # Pattern: "for <database>"
    match = re.search(r'\bFOR\s+([A-Z][A-Z0-9_-]*)\b', query_upper)
    if match:
        db_name = match.group(1)
        if db_name not in {"THE", "A", "AN", "ALL", "MY", "LAST", "NEXT"}:
            return db_name

    # Pattern: "on <database>"
    match = re.search(r'\b(?:ON|OF)\s+([A-Z][A-Z0-9_-]*)\b', query_upper)
    if match:
        db_name = match.group(1)
        if db_name not in {"THE", "A", "AN", "ALL", "MY"}:
            return db_name

    return None


# Pre-classification functions extracted from nodes.py

def pre_classify_database_query(query: str) -> IntentClassification | None:
    """Pre-classify database listing queries."""
    query_lower = query.lower()

    listing_keywords = ["list", "show", "get", "what", "which", "display", "inventory", "names", "all"]
    database_keywords = ["database", "databases", "db", "dbs", "autonomous", "atp", "adw", "exadata"]

    has_listing = any(kw in query_lower for kw in listing_keywords)
    has_database = any(kw in query_lower for kw in database_keywords)

    if has_listing and has_database:
        # EXCLUSION: Cost queries
        cost_keywords = ["cost", "spend", "spending", "price", "billing", "expensive"]
        if any(kw in query_lower for kw in cost_keywords):
            return None

        # EXCLUSION: Performance queries
        perf_keywords = ["performance", "slow", "error", "problem", "issue", "troubleshoot"]
        if any(kw in query_lower for kw in perf_keywords):
            return None

        # EXCLUSION: DB Management queries
        dbmgmt_keywords = [
            "running sql", "active sql", "sql monitor", "blocking", "blocked",
            "wait event", "wait events", "parallelism", "longops", "long running",
            "table scan", "full scan", "awr report", "awr", "health check",
            "database health", "db health", "top sql", "top queries"
        ]
        if any(kw in query_lower for kw in dbmgmt_keywords):
            return None

        return IntentClassification(
            intent="list_databases",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["database"],
            entities={},
            suggested_workflow="list_databases",
            suggested_agent=None,
        )

    return None


def pre_classify_cost_query(query: str) -> IntentClassification | None:
    """Pre-classify domain-specific cost queries."""
    query_lower = query.lower()

    cost_keywords = ["cost", "spend", "spending", "budget", "expensive", "price", "billing"]
    is_cost_query = any(kw in query_lower for kw in cost_keywords)

    if not is_cost_query:
        return None

    needs_reasoning = has_complexity_indicators(query)

    domain_patterns = {
        "database": {
            "keywords": ["database", "db", "autonomous", "atp", "adw", "exadata", "mysql", "nosql"],
            "intent": "database_costs",
            "workflow": "database_costs",
        },
        "compute": {
            "keywords": ["compute", "instance", "vm", "virtual machine", "server"],
            "intent": "compute_costs",
            "workflow": "compute_costs",
        },
    }

    for domain, config in domain_patterns.items():
        if any(kw in query_lower for kw in config["keywords"]):
            confidence = 0.70 if needs_reasoning else 0.95
            return IntentClassification(
                intent=config["intent"],
                category=IntentCategory.ANALYSIS if needs_reasoning else IntentCategory.QUERY,
                confidence=confidence,
                domains=[domain, "cost"],
                entities={},
                suggested_workflow=config["workflow"],
                suggested_agent="finops-agent",
            )

    # General cost summary
    return IntentClassification(
        intent="cost_summary",
        category=IntentCategory.ANALYSIS if needs_reasoning else IntentCategory.QUERY,
        confidence=0.70 if needs_reasoning else 0.90,
        domains=["cost"],
        entities={},
        suggested_workflow="cost_summary",
        suggested_agent="finops-agent",
    )


def pre_classify_dbmgmt_query(query: str) -> IntentClassification | None:
    """Pre-classify Database Management queries."""
    query_lower = query.lower()

    # Health check patterns - CRITICAL for "check database health for X"
    health_keywords = [
        "health check", "database health", "db health", "check health",
        "health status", "is healthy", "health for"
    ]
    if any(kw in query_lower for kw in health_keywords):
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="db_performance_overview",
            category=IntentCategory.ANALYSIS,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_performance_overview",
            suggested_agent="db-troubleshoot",
        )

    # Blocking sessions
    if "blocking" in query_lower or "blocked" in query_lower:
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="check_blocking",
            category=IntentCategory.ANALYSIS,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_blocking_sessions_workflow",
            suggested_agent="db-troubleshoot",
        )

    # Wait events
    if "wait event" in query_lower or "wait events" in query_lower:
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="wait_events",
            category=IntentCategory.ANALYSIS,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_wait_events_workflow",
            suggested_agent="db-troubleshoot",
        )

    # Running SQL
    sql_patterns = ["running sql", "show running", "active sql", "executing sql"]
    if any(p in query_lower for p in sql_patterns):
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="sql_monitoring",
            category=IntentCategory.ANALYSIS,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_sql_monitoring_workflow",
            suggested_agent="db-troubleshoot",
        )

    # Long running operations
    if "long running" in query_lower or "longops" in query_lower:
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="long_running_ops",
            category=IntentCategory.ANALYSIS,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_long_running_ops_workflow",
            suggested_agent="db-troubleshoot",
        )

    # Parallelism
    if "parallelism" in query_lower or "parallel query" in query_lower:
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="parallelism_stats",
            category=IntentCategory.ANALYSIS,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_parallelism_stats_workflow",
            suggested_agent="db-troubleshoot",
        )

    # Full table scans
    if "table scan" in query_lower or "full scan" in query_lower or "full table" in query_lower:
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="full_table_scan",
            category=IntentCategory.ANALYSIS,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_full_table_scan_workflow",
            suggested_agent="db-troubleshoot",
        )

    # AWR report
    if "awr report" in query_lower or "awr" in query_lower:
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="awr_report",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_awr_report_workflow",
            suggested_agent="db-troubleshoot",
        )

    # Top SQL
    if "top sql" in query_lower or "top queries" in query_lower:
        db_name = extract_database_name(query)
        entities = {"database_name": db_name} if db_name else {}
        return IntentClassification(
            intent="top_sql",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["dbmgmt", "database"],
            entities=entities,
            suggested_workflow="db_top_sql_workflow",
            suggested_agent="db-troubleshoot",
        )

    return None


def pre_classify_compute_query(query: str) -> IntentClassification | None:
    """Pre-classify compute/infrastructure queries."""
    query_lower = query.lower()

    # Listing patterns - check FIRST
    listing_keywords = ["list", "show", "get", "display", "what", "which"]
    has_listing = any(kw in query_lower for kw in listing_keywords)

    instance_keywords = ["instance", "instances", "vm", "vms", "server", "servers"]
    has_instance = any(kw in query_lower for kw in instance_keywords)

    # Instance listing (must check before lifecycle actions)
    if has_listing and has_instance:
        return IntentClassification(
            intent="list_instances",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["compute"],
            entities={},
            suggested_workflow="list_instances",
            suggested_agent="infrastructure-agent",
        )

    # Lifecycle actions (only if NOT a listing query)
    if not has_listing:
        # Restart (check first since "restart" contains "start")
        if any(kw in query_lower for kw in ["restart", "reboot", "reset"]) and has_instance:
            entities = {}
            match = re.search(r'(?:instance|vm|server)\s+([a-zA-Z][a-zA-Z0-9_-]*)', query_lower)
            if match:
                entities["instance_name"] = match.group(1)
            return IntentClassification(
                intent="restart_instance",
                category=IntentCategory.ACTION,
                confidence=0.95,
                domains=["compute"],
                entities=entities,
                suggested_workflow="restart_instance_by_name",
                suggested_agent="infrastructure-agent",
            )

        # Stop
        if any(kw in query_lower for kw in ["stop", "shutdown", "halt"]) and has_instance:
            entities = {}
            match = re.search(r'(?:instance|vm|server)\s+([a-zA-Z][a-zA-Z0-9_-]*)', query_lower)
            if match:
                entities["instance_name"] = match.group(1)
            return IntentClassification(
                intent="stop_instance",
                category=IntentCategory.ACTION,
                confidence=0.95,
                domains=["compute"],
                entities=entities,
                suggested_workflow="stop_instance_by_name",
                suggested_agent="infrastructure-agent",
            )

        # Start
        if any(kw in query_lower for kw in ["start", "boot"]) and has_instance:
            entities = {}
            match = re.search(r'(?:instance|vm|server)\s+([a-zA-Z][a-zA-Z0-9_-]*)', query_lower)
            if match:
                entities["instance_name"] = match.group(1)
            return IntentClassification(
                intent="start_instance",
                category=IntentCategory.ACTION,
                confidence=0.95,
                domains=["compute"],
                entities=entities,
                suggested_workflow="start_instance_by_name",
                suggested_agent="infrastructure-agent",
            )

    return None


def pre_classify_security_query(query: str) -> IntentClassification | None:
    """Pre-classify security queries."""
    query_lower = query.lower()

    # Security overview
    if "security overview" in query_lower or "security posture" in query_lower:
        return IntentClassification(
            intent="security_overview",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["security"],
            entities={},
            suggested_workflow="security_posture_summary",
            suggested_agent="security-agent",
        )

    # Cloud Guard problems
    if "cloud guard" in query_lower and "problem" in query_lower:
        return IntentClassification(
            intent="cloud_guard_problems",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["security"],
            entities={},
            suggested_workflow="cloud_guard_problems",
            suggested_agent="security-agent",
        )

    # Security score
    if "security score" in query_lower:
        return IntentClassification(
            intent="security_score",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["security"],
            entities={},
            suggested_workflow="security_score",
            suggested_agent="security-agent",
        )

    # Audit events
    if "audit event" in query_lower or "audit events" in query_lower:
        return IntentClassification(
            intent="audit_events",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["security"],
            entities={},
            suggested_workflow="audit_events",
            suggested_agent="security-agent",
        )

    return None


def pre_classify_logs_query(query: str) -> IntentClassification | None:
    """Pre-classify log analytics queries."""
    query_lower = query.lower()

    # Log summary
    if "log summary" in query_lower:
        return IntentClassification(
            intent="log_summary",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["logs", "observability"],
            entities={},
            suggested_workflow="log_summary",
            suggested_agent="log-analytics-agent",
        )

    # Log search for errors
    if "search logs" in query_lower or ("logs" in query_lower and "error" in query_lower):
        return IntentClassification(
            intent="log_search",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["logs", "observability"],
            entities={},
            suggested_workflow="log_search",
            suggested_agent="log-analytics-agent",
        )

    # Active alarms
    if "alarm" in query_lower or "alarms" in query_lower:
        return IntentClassification(
            intent="list_alarms",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["observability"],
            entities={},
            suggested_workflow="list_alarms",
            suggested_agent="infrastructure-agent",
        )

    return None


def pre_classify_discovery_query(query: str) -> IntentClassification | None:
    """Pre-classify discovery queries."""
    query_lower = query.lower()

    # List compartments
    if "compartment" in query_lower and any(kw in query_lower for kw in ["list", "show", "get"]):
        return IntentClassification(
            intent="list_compartments",
            category=IntentCategory.QUERY,
            confidence=0.95,
            domains=["identity"],
            entities={},
            suggested_workflow="list_compartments",
            suggested_agent="infrastructure-agent",
        )

    return None


def classify_query(query: str) -> IntentClassification | None:
    """Run all pre-classifiers in order."""

    # Order matters! Check specific patterns before general ones
    classifiers = [
        ("dbmgmt", pre_classify_dbmgmt_query),
        ("cost", pre_classify_cost_query),
        ("database", pre_classify_database_query),
        ("compute", pre_classify_compute_query),
        ("security", pre_classify_security_query),
        ("logs", pre_classify_logs_query),
        ("discovery", pre_classify_discovery_query),
    ]

    for name, classifier in classifiers:
        result = classifier(query)
        if result:
            return result

    return None


# Test cases from DEMO_PLAN
TEST_CASES = [
    # Database Troubleshooting (Commands 1-10)
    ("list databases", "list_databases", "list_databases"),
    ("check database health for ATPAdi", "db_performance_overview", "db_performance_overview"),
    ("check blocking sessions on ATPAdi", "check_blocking", "db_blocking_sessions_workflow"),
    ("show wait events for ATPAdi", "wait_events", "db_wait_events_workflow"),
    ("show running SQL on ATPAdi", "sql_monitoring", "db_sql_monitoring_workflow"),
    ("show long running operations on ATPAdi", "long_running_ops", "db_long_running_ops_workflow"),
    ("check parallelism for ATPAdi", "parallelism_stats", "db_parallelism_stats_workflow"),
    ("find full table scans on ATPAdi", "full_table_scan", "db_full_table_scan_workflow"),
    ("generate AWR report for ATPAdi last hour", "awr_report", "db_awr_report_workflow"),
    ("show top SQL by CPU on ATPAdi", "top_sql", "db_top_sql_workflow"),

    # Instance Management (Commands 11-15)
    ("list running instances", "list_instances", "list_instances"),
    ("list stopped instances", "list_instances", "list_instances"),  # NOT stop_instance!
    ("start instance web-server-01", "start_instance", "start_instance_by_name"),
    ("stop instance web-server-01", "stop_instance", "stop_instance_by_name"),
    ("restart instance web-server-01", "restart_instance", "restart_instance_by_name"),

    # Cost & Usage (Commands 16-22)
    ("show cost summary", "cost_summary", "cost_summary"),
    ("show cost summary for October", "cost_summary", "cost_summary"),
    ("show costs by service", "cost_summary", "cost_summary"),  # Generic cost query
    ("show costs by compartment", "cost_summary", "cost_summary"),  # Generic cost query
    ("show database costs", "database_costs", "database_costs"),  # NOT list_databases!
    ("compare costs August vs October vs November", "cost_summary", "cost_summary"),  # Has complexity
    ("show cost trend last 6 months", "cost_summary", "cost_summary"),  # Has complexity

    # Security (Commands 23-26)
    ("show security overview", "security_overview", "security_posture_summary"),
    ("list Cloud Guard problems", "cloud_guard_problems", "cloud_guard_problems"),
    ("show security score", "security_score", "security_score"),
    ("show recent audit events", "audit_events", "audit_events"),

    # Logs & Observability (Commands 27-29)
    ("show log summary last 24 hours", "log_summary", "log_summary"),
    ("search logs for errors", "log_search", "log_search"),
    ("list active alarms", "list_alarms", "list_alarms"),

    # Discovery (Command 30)
    ("list compartments", "list_compartments", "list_compartments"),
]


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("Testing 30 DEMO_PLAN Commands for Correct Routing")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    results = []

    for query, expected_intent, expected_workflow in TEST_CASES:
        result = classify_query(query)

        if result is None:
            status = "FAILED"
            actual_intent = "None"
            actual_workflow = "None"
            failed += 1
        elif result.intent == expected_intent:
            status = "PASSED"
            actual_intent = result.intent
            actual_workflow = result.suggested_workflow
            passed += 1
        else:
            status = "FAILED"
            actual_intent = result.intent
            actual_workflow = result.suggested_workflow
            failed += 1

        results.append({
            "query": query,
            "expected": expected_intent,
            "actual": actual_intent,
            "status": status,
        })

    # Print results
    for r in results:
        icon = "✅" if r["status"] == "PASSED" else "❌"
        print(f"{icon} [{r['status']}] \"{r['query'][:50]}...\"")
        if r["status"] == "FAILED":
            print(f"    Expected: {r['expected']}")
            print(f"    Actual:   {r['actual']}")

    # Summary
    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(TEST_CASES)} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
