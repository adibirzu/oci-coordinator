"""
OCI AI Agents module.

Provides the agent infrastructure:
- BaseAgent: Abstract base class for all agents
- AgentCatalog: Auto-discovery and registration
- LangGraphCoordinator: Workflow-first orchestration
- Skills: Reusable workflow definitions and execution
"""

from src.agents.base import (
    AgentDefinition,
    AgentMetadata,
    AgentStatus,
    BaseAgent,
    KafkaTopics,
)
from src.agents.catalog import AgentCatalog, initialize_agents
from src.agents.coordinator import (
    CoordinatorNodes,
    CoordinatorState,
    LangGraphCoordinator,
    create_coordinator,
)
from src.agents.skills import (
    COST_ANALYSIS_WORKFLOW,
    RCA_WORKFLOW,
    SECURITY_ASSESSMENT_WORKFLOW,
    SECURITY_ACCESS_GOVERNANCE_WORKFLOW,
    SECURITY_AUDIT_ACTIVITY_WORKFLOW,
    SECURITY_BASTION_AUDIT_WORKFLOW,
    SECURITY_BASTION_SESSION_CLEANUP_WORKFLOW,
    SECURITY_CLOUDGUARD_INVESTIGATION_WORKFLOW,
    SECURITY_CLOUDGUARD_REMEDIATION_WORKFLOW,
    SECURITY_DATASAFE_ASSESSMENT_WORKFLOW,
    SECURITY_IAM_REVIEW_WORKFLOW,
    SECURITY_KMS_INVENTORY_WORKFLOW,
    SECURITY_POSTURE_WORKFLOW,
    SECURITY_VULNERABILITY_WORKFLOW,
    SECURITY_WAF_POLICY_WORKFLOW,
    SECURITY_ZONE_COMPLIANCE_WORKFLOW,
    SkillDefinition,
    SkillExecutionResult,
    SkillExecutor,
    SkillRegistry,
    SkillStatus,
    SkillStep,
    StepResult,
    register_default_skills,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentDefinition",
    "AgentMetadata",
    "AgentStatus",
    "KafkaTopics",
    # Catalog
    "AgentCatalog",
    "initialize_agents",
    # Coordinator
    "LangGraphCoordinator",
    "create_coordinator",
    "CoordinatorState",
    "CoordinatorNodes",
    # Skills
    "SkillDefinition",
    "SkillStep",
    "SkillExecutor",
    "SkillRegistry",
    "SkillExecutionResult",
    "SkillStatus",
    "StepResult",
    "register_default_skills",
    # Pre-defined skills
    "RCA_WORKFLOW",
    "COST_ANALYSIS_WORKFLOW",
    "SECURITY_ASSESSMENT_WORKFLOW",
    "SECURITY_ACCESS_GOVERNANCE_WORKFLOW",
    "SECURITY_AUDIT_ACTIVITY_WORKFLOW",
    "SECURITY_BASTION_AUDIT_WORKFLOW",
    "SECURITY_BASTION_SESSION_CLEANUP_WORKFLOW",
    "SECURITY_CLOUDGUARD_INVESTIGATION_WORKFLOW",
    "SECURITY_CLOUDGUARD_REMEDIATION_WORKFLOW",
    "SECURITY_DATASAFE_ASSESSMENT_WORKFLOW",
    "SECURITY_IAM_REVIEW_WORKFLOW",
    "SECURITY_KMS_INVENTORY_WORKFLOW",
    "SECURITY_POSTURE_WORKFLOW",
    "SECURITY_VULNERABILITY_WORKFLOW",
    "SECURITY_WAF_POLICY_WORKFLOW",
    "SECURITY_ZONE_COMPLIANCE_WORKFLOW",
]
