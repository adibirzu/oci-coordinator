"""
Code Sandbox Skill.

Allows agents to execute dynamically generated Python/Bash scripts safely
when pre-built MCP tools are missing or insufficient.
"""

from src.agents.skills import SkillDefinition, SkillStep

CODE_EXECUTION_WORKFLOW = SkillDefinition(
    name="code_execution_workflow",
    description="Generates and executes a raw script to achieve a goal",
    steps=[
        SkillStep(
            name="generate_script",
            description="Write the necessary python/bash script using internal knowledge",
            required_tools=["llm_generate_code"], 
            timeout_seconds=30,
        ),
        SkillStep(
            name="execute_script",
            description="Run the script in the secure sandbox",
            required_tools=["execute_sandbox_code"],
            timeout_seconds=120,
        ),
    ],
    required_tools=["llm_generate_code", "execute_sandbox_code"],
    tags=["execution", "dynamic", "code-sandbox"],
    estimated_duration_seconds=150,
)
