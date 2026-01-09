"""
Enterprise Troubleshooting Catalog for Slack.

Provides interactive, categorized quick-action menus with
suggested questions and runbooks for common OCI operations.

The catalog integrates with pre-built workflows for deterministic,
fast responses and guides users to the right questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Category(str, Enum):
    """Troubleshooting categories."""
    DATABASE = "database"
    COST = "cost"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    LOGS = "logs"
    DISCOVERY = "discovery"


@dataclass
class QuickAction:
    """A quick action that triggers a workflow or question."""
    id: str
    label: str
    description: str
    query: str  # The query to execute
    category: Category
    icon: str = ""
    is_workflow: bool = True  # True if maps to deterministic workflow


@dataclass
class Runbook:
    """An enterprise runbook for guided troubleshooting."""
    id: str
    name: str
    description: str
    category: Category
    steps: list[str] = field(default_factory=list)
    quick_actions: list[QuickAction] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)


# Quick actions organized by category
QUICK_ACTIONS: dict[Category, list[QuickAction]] = {
    Category.DATABASE: [
        QuickAction(
            id="db_fleet",
            label="Database Fleet",
            description="View all databases in your tenancy",
            query="list databases",
            category=Category.DATABASE,
            icon=":database:",
        ),
        QuickAction(
            id="db_health",
            label="DB Health Check",
            description="Quick health check across all databases",
            query="database health check",
            category=Category.DATABASE,
            icon=":stethoscope:",
        ),
        QuickAction(
            id="db_performance",
            label="Performance Analysis",
            description="Analyze database performance metrics",
            query="analyze database performance",
            category=Category.DATABASE,
            icon=":chart_with_upwards_trend:",
        ),
        QuickAction(
            id="db_slow_queries",
            label="Slow Queries",
            description="Find and analyze slow running queries",
            query="show slow queries",
            category=Category.DATABASE,
            icon=":turtle:",
        ),
    ],
    Category.COST: [
        QuickAction(
            id="cost_summary",
            label="Cost Summary",
            description="Current month spending breakdown",
            query="show costs",
            category=Category.COST,
            icon=":money_with_wings:",
        ),
        QuickAction(
            id="cost_by_service",
            label="Cost by Service",
            description="Breakdown by OCI service",
            query="cost breakdown by service",
            category=Category.COST,
            icon=":bar_chart:",
        ),
        QuickAction(
            id="db_costs",
            label="Database Costs",
            description="Database-specific spending",
            query="database costs",
            category=Category.COST,
            icon=":database:",
        ),
        QuickAction(
            id="compute_costs",
            label="Compute Costs",
            description="Instance and VM spending",
            query="compute costs",
            category=Category.COST,
            icon=":computer:",
        ),
        QuickAction(
            id="cost_optimization",
            label="Optimize Costs",
            description="Find cost saving opportunities",
            query="resource utilization",
            category=Category.COST,
            icon=":bulb:",
        ),
    ],
    Category.SECURITY: [
        QuickAction(
            id="security_overview",
            label="Security Overview",
            description="Cloud Guard alerts and security posture",
            query="security overview",
            category=Category.SECURITY,
            icon=":shield:",
        ),
        QuickAction(
            id="threats",
            label="Active Threats",
            description="Current threat detections",
            query="show threats",
            category=Category.SECURITY,
            icon=":warning:",
        ),
        QuickAction(
            id="iam_users",
            label="IAM Users",
            description="List identity users",
            query="list users",
            category=Category.SECURITY,
            icon=":busts_in_silhouette:",
        ),
    ],
    Category.INFRASTRUCTURE: [
        QuickAction(
            id="list_compartments",
            label="Compartments",
            description="View compartment hierarchy",
            query="list compartments",
            category=Category.INFRASTRUCTURE,
            icon=":file_folder:",
        ),
        QuickAction(
            id="list_instances",
            label="Compute Instances",
            description="View running instances",
            query="list instances",
            category=Category.INFRASTRUCTURE,
            icon=":desktop_computer:",
        ),
        QuickAction(
            id="list_vcns",
            label="Networks (VCNs)",
            description="View virtual networks",
            query="list vcns",
            category=Category.INFRASTRUCTURE,
            icon=":globe_with_meridians:",
        ),
        QuickAction(
            id="tenancy_info",
            label="Tenancy Info",
            description="Current tenancy details",
            query="tenancy info",
            category=Category.INFRASTRUCTURE,
            icon=":office:",
        ),
        QuickAction(
            id="list_regions",
            label="Regions",
            description="Available OCI regions",
            query="list regions",
            category=Category.INFRASTRUCTURE,
            icon=":world_map:",
        ),
    ],
    Category.LOGS: [
        QuickAction(
            id="recent_errors",
            label="Recent Errors",
            description="Errors from the last hour",
            query="show recent errors",
            category=Category.LOGS,
            icon=":x:",
        ),
        QuickAction(
            id="audit_logs",
            label="Audit Trail",
            description="Recent audit events",
            query="show audit logs",
            category=Category.LOGS,
            icon=":memo:",
        ),
    ],
    Category.DISCOVERY: [
        QuickAction(
            id="resource_summary",
            label="Resource Summary",
            description="Overview of all resources",
            query="discovery summary",
            category=Category.DISCOVERY,
            icon=":mag:",
        ),
        QuickAction(
            id="search_resources",
            label="Search Resources",
            description="Find resources by name",
            query="search resources",
            category=Category.DISCOVERY,
            icon=":mag_right:",
        ),
        QuickAction(
            id="capabilities",
            label="What Can I Do?",
            description="See available capabilities",
            query="what can you do?",
            category=Category.DISCOVERY,
            icon=":question:",
        ),
    ],
}


# Enterprise runbooks for guided troubleshooting
RUNBOOKS: list[Runbook] = [
    Runbook(
        id="db_performance_runbook",
        name="Database Performance Troubleshooting",
        description="Step-by-step guide for diagnosing database performance issues",
        category=Category.DATABASE,
        steps=[
            "Check database fleet health status",
            "Review CPU and memory utilization",
            "Identify slow running queries",
            "Analyze wait events and blocking sessions",
            "Review AWR reports for historical trends",
        ],
        quick_actions=[
            QUICK_ACTIONS[Category.DATABASE][1],  # DB Health Check
            QUICK_ACTIONS[Category.DATABASE][2],  # Performance Analysis
            QUICK_ACTIONS[Category.DATABASE][3],  # Slow Queries
        ],
        follow_up_questions=[
            "What is the current CPU utilization?",
            "Are there any blocking sessions?",
            "Show AWR report for the last 24 hours",
        ],
    ),
    Runbook(
        id="cost_investigation_runbook",
        name="Cost Investigation",
        description="Investigate unexpected cost increases and find optimization opportunities",
        category=Category.COST,
        steps=[
            "Review current month cost summary",
            "Compare with previous month",
            "Identify top spending services",
            "Check for unused or underutilized resources",
            "Review recent resource changes",
        ],
        quick_actions=[
            QUICK_ACTIONS[Category.COST][0],  # Cost Summary
            QUICK_ACTIONS[Category.COST][1],  # Cost by Service
            QUICK_ACTIONS[Category.COST][4],  # Optimize Costs
        ],
        follow_up_questions=[
            "What changed in the last week?",
            "Which compartments are spending the most?",
            "Are there underutilized instances?",
        ],
    ),
    Runbook(
        id="security_incident_runbook",
        name="Security Incident Response",
        description="Investigate and respond to security alerts",
        category=Category.SECURITY,
        steps=[
            "Check Cloud Guard for active problems",
            "Review IAM activity logs",
            "Identify affected resources",
            "Check for suspicious network activity",
            "Document findings and take action",
        ],
        quick_actions=[
            QUICK_ACTIONS[Category.SECURITY][0],  # Security Overview
            QUICK_ACTIONS[Category.SECURITY][1],  # Active Threats
        ],
        follow_up_questions=[
            "Who made changes in the last hour?",
            "Are there failed login attempts?",
            "Show network security changes",
        ],
    ),
]


# Context-aware follow-up suggestions based on query type
FOLLOW_UP_SUGGESTIONS: dict[str, list[str]] = {
    "cost_summary": [
        "Show database costs specifically",
        "Compare with last month",
        "Which compartments are spending the most?",
        "Find optimization opportunities",
    ],
    "list_compartments": [
        "Show resources in a specific compartment",
        "What are the costs per compartment?",
        "List instances in the root compartment",
    ],
    "list_instances": [
        "Show instance metrics",
        "What are the compute costs?",
        "Check instance health",
    ],
    "list_databases": [
        "Check database performance",
        "Show database costs",
        "Analyze slow queries",
    ],
    "security_overview": [
        "Show detailed threat information",
        "Who made recent changes?",
        "Check IAM policies",
    ],
    "database_health": [
        "Analyze CPU utilization",
        "Show slow queries",
        "Check storage usage",
    ],
    "error": [
        "Try a simpler query",
        "Check specific resources",
        "Show available capabilities",
    ],
    # DB Troubleshooting - SQL Monitoring
    "sql_monitoring": [
        "Check blocking sessions",
        "Show wait events",
        "Show top SQL by CPU",
        "Generate AWR report",
    ],
    # DB Troubleshooting - Blocking Sessions
    "blocking_sessions": [
        "Show wait events",
        "Show running SQL",
        "Check long running operations",
        "Generate AWR report",
    ],
    # DB Troubleshooting - Wait Events
    "wait_events": [
        "Check blocking sessions",
        "Show running SQL",
        "Show top SQL by CPU",
        "Generate AWR report",
    ],
    # DB Troubleshooting - Parallelism
    "parallelism_stats": [
        "Show running SQL",
        "Check long running operations",
        "Show top SQL by CPU",
        "Check full table scans",
    ],
    # DB Troubleshooting - Long Running Ops
    "long_running_ops": [
        "Show running SQL",
        "Check parallelism",
        "Show wait events",
        "Check blocking sessions",
    ],
    # DB Troubleshooting - Full Table Scan
    "full_table_scan": [
        "Show top SQL by CPU",
        "Check parallelism",
        "Show running SQL",
        "Generate AWR report",
    ],
    # DB Troubleshooting - AWR Report
    "awr_report": [
        "Show wait events",
        "Show top SQL by CPU",
        "Check blocking sessions",
        "Check long running operations",
    ],
    # DB Troubleshooting - Top SQL
    "top_sql": [
        "Show running SQL",
        "Show wait events",
        "Check full table scans",
        "Generate AWR report",
    ],
    # DB Performance Overview
    "db_performance_overview": [
        "Show wait events",
        "Show top SQL by CPU",
        "Check blocking sessions",
        "Generate AWR report",
    ],
}


def get_category_icon(category: Category) -> str:
    """Get icon for category."""
    icons = {
        Category.DATABASE: ":database:",
        Category.COST: ":moneybag:",
        Category.SECURITY: ":shield:",
        Category.INFRASTRUCTURE: ":cloud:",
        Category.LOGS: ":page_facing_up:",
        Category.DISCOVERY: ":mag:",
    }
    return icons.get(category, ":question:")


def get_category_label(category: Category) -> str:
    """Get human-readable label for category."""
    labels = {
        Category.DATABASE: "Database",
        Category.COST: "Cost & FinOps",
        Category.SECURITY: "Security",
        Category.INFRASTRUCTURE: "Infrastructure",
        Category.LOGS: "Logs & Errors",
        Category.DISCOVERY: "Discovery",
    }
    return labels.get(category, category.value.title())


def build_catalog_blocks(selected_category: Category | None = None) -> list[dict[str, Any]]:
    """
    Build Slack blocks for the troubleshooting catalog.

    If no category selected, shows category overview.
    If category selected, shows quick actions for that category.
    """
    blocks: list[dict[str, Any]] = []

    if selected_category is None:
        # Show category overview
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":books: OCI Troubleshooting Catalog",
                "emoji": True,
            }
        })
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Select a category to see quick actions and common questions:",
            }
        })
        blocks.append({"type": "divider"})

        # Category buttons
        button_elements = []
        for category in Category:
            icon = get_category_icon(category)
            label = get_category_label(category)
            button_elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": f"{icon} {label}",
                    "emoji": True,
                },
                "action_id": f"catalog_category_{category.value}",
                "value": category.value,
            })

        # Split into rows of 3
        for i in range(0, len(button_elements), 3):
            blocks.append({
                "type": "actions",
                "elements": button_elements[i:i+3],
            })

    else:
        # Show quick actions for selected category
        icon = get_category_icon(selected_category)
        label = get_category_label(selected_category)

        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{icon} {label} Quick Actions",
                "emoji": True,
            }
        })
        blocks.append({"type": "divider"})

        # Quick action buttons
        actions = QUICK_ACTIONS.get(selected_category, [])
        for action in actions:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{action.icon} {action.label}*\n{action.description}",
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Run",
                        "emoji": True,
                    },
                    "action_id": f"catalog_action_{action.id}",
                    "value": action.query,
                    "style": "primary",
                }
            })

        # Back button
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "actions",
            "elements": [{
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": ":arrow_left: Back to Categories",
                    "emoji": True,
                },
                "action_id": "catalog_back",
            }]
        })

    return blocks


def build_runbook_blocks(runbook_id: str) -> list[dict[str, Any]]:
    """Build Slack blocks for a specific runbook."""
    runbook = next((r for r in RUNBOOKS if r.id == runbook_id), None)
    if not runbook:
        return [{
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Runbook not found."},
        }]

    blocks: list[dict[str, Any]] = []
    icon = get_category_icon(runbook.category)

    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"{icon} {runbook.name}",
            "emoji": True,
        }
    })

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": runbook.description,
        }
    })

    blocks.append({"type": "divider"})

    # Steps
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*Steps:*",
        }
    })

    step_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(runbook.steps))
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": step_text,
        }
    })

    # Quick action buttons
    if runbook.quick_actions:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Quick Actions:*",
            }
        })

        button_elements = []
        for action in runbook.quick_actions[:5]:
            button_elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": f"{action.icon} {action.label}",
                    "emoji": True,
                },
                "action_id": f"runbook_action_{action.id}",
                "value": action.query,
                "style": "primary",
            })

        blocks.append({
            "type": "actions",
            "elements": button_elements,
        })

    return blocks


def get_follow_up_suggestions(
    query_type: str,
    response_content: str | None = None,
) -> list[str]:
    """
    Get context-aware follow-up question suggestions.

    Args:
        query_type: Type of query that was executed (workflow name or category)
        response_content: Optional response content for smarter suggestions

    Returns:
        List of suggested follow-up questions
    """
    # Map common query patterns to suggestion keys
    query_type_lower = query_type.lower()

    # First, check for exact matches in FOLLOW_UP_SUGGESTIONS
    # This catches specific DB troubleshooting workflows like sql_monitoring, blocking_sessions, etc.
    if query_type_lower in FOLLOW_UP_SUGGESTIONS:
        return FOLLOW_UP_SUGGESTIONS[query_type_lower][:3]

    # DB Troubleshooting workflows - check for specific patterns BEFORE generic "database"
    db_troubleshoot_mappings = {
        "sql_monitor": "sql_monitoring",
        "running_sql": "sql_monitoring",
        "active_sql": "sql_monitoring",
        "blocking": "blocking_sessions",
        "lock_contention": "blocking_sessions",
        "wait_event": "wait_events",
        "parallelism": "parallelism_stats",
        "parallel": "parallelism_stats",
        "long_running": "long_running_ops",
        "longops": "long_running_ops",
        "full_table": "full_table_scan",
        "table_scan": "full_table_scan",
        "awr": "awr_report",
        "top_sql": "top_sql",
        "cpu_sql": "top_sql",
        "expensive_queries": "top_sql",
        "performance_overview": "db_performance_overview",
        "health_check": "db_performance_overview",
    }

    for pattern, suggestion_key in db_troubleshoot_mappings.items():
        if pattern in query_type_lower:
            return FOLLOW_UP_SUGGESTIONS.get(suggestion_key, [])[:3]

    # General category-based suggestions
    if "cost" in query_type_lower:
        return FOLLOW_UP_SUGGESTIONS.get("cost_summary", [])[:3]
    elif "compartment" in query_type_lower:
        return FOLLOW_UP_SUGGESTIONS.get("list_compartments", [])[:3]
    elif "instance" in query_type_lower:
        return FOLLOW_UP_SUGGESTIONS.get("list_instances", [])[:3]
    elif "database" in query_type_lower or "db" in query_type_lower:
        # Generic database queries - fallback to list_databases suggestions
        return FOLLOW_UP_SUGGESTIONS.get("list_databases", [])[:3]
    elif "security" in query_type_lower or "threat" in query_type_lower:
        return FOLLOW_UP_SUGGESTIONS.get("security_overview", [])[:3]
    elif "error" in query_type_lower:
        return FOLLOW_UP_SUGGESTIONS.get("error", [])[:3]
    else:
        # Default suggestions
        return [
            "Show my costs",
            "List compartments",
            "What can you do?",
        ]


def build_follow_up_blocks(suggestions: list[str]) -> list[dict[str, Any]]:
    """Build Slack blocks for follow-up suggestions."""
    if not suggestions:
        return []

    blocks: list[dict[str, Any]] = []
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": ":bulb: *Suggested follow-ups:*",
        }]
    })

    button_elements = []
    for i, suggestion in enumerate(suggestions[:4]):
        button_elements.append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": suggestion[:75],
                "emoji": True,
            },
            "action_id": f"follow_up_{i}",
            "value": suggestion,
        })

    blocks.append({
        "type": "actions",
        "elements": button_elements,
    })

    return blocks


def build_error_recovery_blocks(
    error_type: str,
    original_query: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build Slack blocks for error recovery suggestions.

    Args:
        error_type: Type of error (timeout, auth, not_found, etc.)
        original_query: The query that failed

    Returns:
        Slack blocks with recovery suggestions
    """
    blocks: list[dict[str, Any]] = []

    recovery_suggestions = {
        "timeout": {
            "message": "The operation timed out. This can happen with large datasets.",
            "actions": [
                ("Try simpler query", "list compartments"),
                ("Check system status", "help"),
            ]
        },
        "auth": {
            "message": "Authentication is required or has expired.",
            "actions": [
                ("Show help", "help"),
            ]
        },
        "not_found": {
            "message": "The requested resource was not found.",
            "actions": [
                ("Search resources", "search resources"),
                ("List compartments", "list compartments"),
            ]
        },
        "permission": {
            "message": "You don't have permission for this operation.",
            "actions": [
                ("Show available actions", "what can you do?"),
                ("Check IAM policies", "check iam policies"),
            ]
        },
        "default": {
            "message": "Something went wrong. Here are some alternatives:",
            "actions": [
                ("Show help", "help"),
                ("Open catalog", "catalog"),
                ("List resources", "discovery summary"),
            ]
        },
    }

    recovery = recovery_suggestions.get(error_type, recovery_suggestions["default"])

    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": f":bulb: {recovery['message']}",
        }]
    })

    if recovery["actions"]:
        button_elements = []
        for label, query in recovery["actions"]:
            button_elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": label,
                    "emoji": True,
                },
                "action_id": f"recovery_{label.lower().replace(' ', '_')}",
                "value": query,
            })

        blocks.append({
            "type": "actions",
            "elements": button_elements,
        })

    return blocks


# Follow-up suggestions that require a database name to be specified
DATABASE_NAME_REQUIRED_PATTERNS = [
    "check database performance",
    "analyze slow queries",
    "show slow queries",
    "analyze cpu utilization",
    "check storage usage",
    "database health",
    "troubleshoot",
]


def needs_database_name_prompt(query: str) -> bool:
    """
    Check if a follow-up query needs a database name to be specified.

    Some follow-up suggestions like "Check database performance" need
    a specific database name to work. This function identifies those queries.

    Args:
        query: The follow-up query text

    Returns:
        True if the query needs a database name prompt
    """
    query_lower = query.lower()

    # Check against known patterns that need database names
    for pattern in DATABASE_NAME_REQUIRED_PATTERNS:
        if pattern in query_lower:
            return True

    return False


def build_database_name_modal(
    original_query: str,
    channel_id: str,
    thread_ts: str | None = None,
) -> dict[str, Any]:
    """
    Build a Slack modal to collect a database name for follow-up queries.

    Args:
        original_query: The follow-up query that needs a database name
        channel_id: Channel ID to post the result to
        thread_ts: Optional thread timestamp

    Returns:
        Slack modal view definition
    """
    # Store context in private_metadata for the submission handler
    import json
    private_metadata = json.dumps({
        "query": original_query,
        "channel_id": channel_id,
        "thread_ts": thread_ts,
    })

    return {
        "type": "modal",
        "callback_id": "database_name_modal",
        "title": {
            "type": "plain_text",
            "text": "Database Name",
            "emoji": True,
        },
        "submit": {
            "type": "plain_text",
            "text": "Run Query",
            "emoji": True,
        },
        "close": {
            "type": "plain_text",
            "text": "Cancel",
            "emoji": True,
        },
        "private_metadata": private_metadata,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":database: *Query:* _{original_query}_\n\nPlease specify which database you want to analyze:",
                },
            },
            {
                "type": "input",
                "block_id": "database_name_block",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "database_name_input",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "e.g., FINANCE, HR_DB, prod-db-1",
                    },
                },
                "label": {
                    "type": "plain_text",
                    "text": "Database Name",
                    "emoji": True,
                },
                "hint": {
                    "type": "plain_text",
                    "text": "Enter the database name as shown in OCI Console",
                },
            },
        ],
    }
