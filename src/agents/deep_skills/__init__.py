"""
Agent-Specific Skills (DeepSkills).

This module contains advanced skills for each agent domain:
- db_troubleshoot: Database performance analysis and troubleshooting
- log_analytics: Log pattern detection and security analysis
- security: Cloud security audit and vulnerability analysis
- finops: Cost optimization and budget analysis
- infrastructure: Compute and network management

Each skill:
1. Orchestrates multiple MCP tools
2. Processes results with code execution
3. Can self-test its functionality
4. Reports structured results

Usage:
    from src.agents.deep_skills import DatabaseBlockingSkill, CostAnomalySkill
    from src.agents.core import SkillContext

    # Execute a skill
    skill = DatabaseBlockingSkill()
    result = await skill.execute(context)
"""

# Database troubleshooting skills
from src.agents.deep_skills.db_troubleshoot import (
    DatabaseBlockingSkill,
    WaitEventsSkill,
    TopSQLSkill,
    AWRReportSkill,
    DatabaseHealthCheckSkill,
)

# FinOps skills
from src.agents.deep_skills.finops import (
    CostAnomalySkill,
    CostByServiceSkill,
    BudgetAlertSkill,
)

# Log Analytics skills
from src.agents.deep_skills.log_analytics import (
    LogPatternSkill,
    SecurityEventSkill,
    MITREAnalysisSkill,
)

# Security skills
from src.agents.deep_skills.security import (
    CloudGuardSkill,
    VulnerabilityScanSkill,
    SecurityPostureSkill,
)

# Infrastructure skills
from src.agents.deep_skills.infrastructure import (
    InstanceHealthSkill,
    NetworkAnalysisSkill,
    ComputeMetricsSkill,
)

__all__ = [
    # DB Troubleshoot
    "DatabaseBlockingSkill",
    "WaitEventsSkill",
    "TopSQLSkill",
    "AWRReportSkill",
    "DatabaseHealthCheckSkill",
    # FinOps
    "CostAnomalySkill",
    "CostByServiceSkill",
    "BudgetAlertSkill",
    # Log Analytics
    "LogPatternSkill",
    "SecurityEventSkill",
    "MITREAnalysisSkill",
    # Security
    "CloudGuardSkill",
    "VulnerabilityScanSkill",
    "SecurityPostureSkill",
    # Infrastructure
    "InstanceHealthSkill",
    "NetworkAnalysisSkill",
    "ComputeMetricsSkill",
]
