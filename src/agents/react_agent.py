"""
ReAct Agent with MCP Tool Integration.

Implements a Reason-Action-Observation loop that can use MCP tools
to query real OCI data instead of just generating text.

Usage:
    from src.agents.react_agent import OCIReActAgent

    agent = OCIReActAgent(llm=get_llm(), tool_catalog=ToolCatalog.get_instance())
    response = await agent.run("show me instances in adrian_birzu compartment")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.cache.oci_resource_cache import OCIResourceCache
from src.mcp.catalog import ToolCatalog
from src.observability.tracing import get_tracer

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from src.memory.manager import SharedMemoryManager


@dataclass
class AgentStep:
    """Single step in agent execution."""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_final: bool = False


@dataclass
class AgentResult:
    """Result of agent execution."""

    success: bool
    response: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# System prompt for ReAct agent - Action-oriented
REACT_SYSTEM_PROMPT = """You are an autonomous OCI Operations Agent. You MUST use tools to get real data - NEVER ask the user for OCIDs or information you can discover yourself.

## CRITICAL: BE PROACTIVE
- When a user asks about a compartment by NAME, use the compartment context below to get its OCID
- ALWAYS call tools first to get real data - do not ask the user for information
- If you need to list resources, call the appropriate tool immediately
- NEVER respond with "I don't have access" or "please provide OCID" - USE THE TOOLS

## Compartment Context (Name → OCID Mapping)

{compartment_context}

## Available Tools

{tools_context}

## Tool Usage Format

To call a tool, respond with EXACTLY this JSON format:
```json
{{
  "thought": "Brief reason for this action",
  "action": "tool_name",
  "action_input": {{"param": "value"}}
}}
```

After receiving tool results, either call another tool or provide final answer:
```json
{{
  "thought": "I now have the data to answer",
  "final_answer": "Here are the results: ..."
}}
```

## Example Workflows

**User asks: "Show me instances in adrian_birzu compartment"**
1. Look up adrian_birzu in compartment context → get OCID
2. Call: `{{"action": "oci_compute_list_instances", "action_input": {{"compartment_id": "ocid1.compartment.oc1..xxxx"}}}}`
3. Format and return results

**User asks: "What databases do I have?"**
1. Call: `{{"action": "oci_database_list_autonomous", "action_input": {{}}}}`
2. If need more details, call `oci_database_get_autonomous` for each
3. Summarize findings

**User asks: "Show compartment resources"**
1. For the compartment, call multiple tools in sequence:
   - `oci_compute_list_instances`
   - `oci_database_list_autonomous`
   - `oci_network_list_vcns`
2. Aggregate and present summary

## Rules
1. ALWAYS use tools - never make up data or ask user for OCIDs
2. Use compartment context to resolve names to OCIDs
3. If a tool fails, try alternatives or explain the specific error
4. Format responses in Slack markdown (*bold*, `code`, bullet points)
5. Be specific: show counts, states, and resource names
"""


class OCIReActAgent:
    """
    ReAct Agent for OCI operations with MCP tool integration.

    Implements Reason-Action-Observation loop:
    1. Reason: Think about what to do
    2. Action: Call an MCP tool
    3. Observation: Process tool result
    4. Repeat until final answer
    """

    def __init__(
        self,
        llm: Any,
        tool_catalog: ToolCatalog | None = None,
        resource_cache: OCIResourceCache | None = None,
        memory_manager: SharedMemoryManager | None = None,
        agent_id: str = "oci-react-agent",
        max_iterations: int = 5,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: LangChain LLM instance
            tool_catalog: MCP tool catalog for API calls
            resource_cache: OCI resource cache for context
            memory_manager: Shared memory manager for runtime feedback
            agent_id: Agent identifier
            max_iterations: Maximum tool call iterations
        """
        self.llm = llm
        self.tool_catalog = tool_catalog or ToolCatalog.get_instance()
        self.resource_cache = resource_cache
        self.memory = memory_manager
        self.agent_id = agent_id
        self.max_iterations = max_iterations
        self._tracer = get_tracer(agent_id)
        self._logger = logger.bind(agent=agent_id)
        if self.memory:
            try:
                self.tool_catalog.set_memory_manager(self.memory)
            except Exception as exc:
                self._logger.debug(
                    "Failed to attach memory manager to tool catalog",
                    error=str(exc),
                )

    async def _get_compartment_context(self) -> str:
        """Get compartment context from TenancyManager for the system prompt."""
        try:
            from src.oci.tenancy_manager import TenancyManager

            manager = TenancyManager.get_instance()
            await manager.initialize()

            compartments = await manager.list_compartments()

            if not compartments:
                return "No compartments discovered yet. Use oci_list_compartments tool."

            context_lines = []
            for comp in compartments[:30]:  # Limit to 30 for prompt size
                context_lines.append(f"- **{comp.name}** ({comp.tenancy_profile}): `{comp.id}`")

            return "\n".join(context_lines)

        except Exception as e:
            self._logger.error("Failed to get compartment context", error=str(e))
            # Fallback to resource cache if available
            return await self._get_cache_context_fallback()

    async def _get_cache_context_fallback(self) -> str:
        """Fallback to resource cache for compartment context."""
        if not self.resource_cache:
            return "No compartment context available. Discovery may be needed."

        try:
            await self.resource_cache.initialize()
            compartments = await self.resource_cache.get_compartments()

            if not compartments:
                return "No compartments cached. Run discovery first."

            context_lines = []
            for comp in compartments[:20]:  # Limit to 20
                name = comp.get("name", "Unknown")
                ocid = comp.get("id", "")
                context_lines.append(f"- **{name}**: `{ocid}`")

            return "\n".join(context_lines)

        except Exception as e:
            self._logger.error("Failed to get cache context", error=str(e))
            return f"Cache error: {e!s}"

    def _get_tools_context(self) -> str:
        """Get available tools context for the system prompt."""
        if not self.tool_catalog:
            return "No tools available."

        # Get all available tools (up to 100) for better visibility
        tools = self.tool_catalog.list_tools()

        if not tools:
            return "No tools available."

        context_lines = []
        for tool in tools[:100]:  # Limit to 100 for prompt size
            name = tool.name
            desc = (tool.description or "")[:100]
            context_lines.append(f"- **{name}**: {desc}")

        return "\n".join(context_lines)

    def _parse_agent_response(self, response: str) -> dict[str, Any]:
        """Parse agent response to extract action or final answer.

        Handles:
        - Single JSON object in ```json``` block
        - Single JSON object in plain text
        - Multiple JSON objects (takes the LAST one with final_answer, or FIRST one with action)
        """
        # Try to extract JSON from ```json``` block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Handle multiple JSON objects in response
        # OCA/gpt5 sometimes returns iterations concatenated
        json_objects = []
        depth = 0
        start_idx = None

        for i, char in enumerate(response):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    json_str = response[start_idx:i+1]
                    try:
                        parsed = json.loads(json_str)
                        json_objects.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None

        if json_objects:
            # Check if there's BOTH an action AND final_answer in the objects
            # This happens when LLM hallucates the answer without waiting for tool
            has_action = any("action" in obj for obj in json_objects)
            has_final = any("final_answer" in obj for obj in json_objects)

            if has_action and has_final:
                # Execute action first - LLM is hallucinating the final answer
                # We need actual tool results, not premature conclusions
                for obj in json_objects:
                    if "action" in obj:
                        return obj

            # If only final_answer, use it (last one if multiple)
            if has_final:
                for obj in reversed(json_objects):
                    if "final_answer" in obj:
                        return obj

            # Otherwise return first object with action
            for obj in json_objects:
                if "action" in obj:
                    return obj

            # Return first valid object
            return json_objects[0]

        # Fallback: treat entire response as final answer
        return {"thought": "Unable to parse structured response", "final_answer": response}

    async def _execute_tool(
        self, action: str, action_input: dict[str, Any]
    ) -> str:
        """Execute an MCP tool and return observation."""
        if not self.tool_catalog:
            return "Error: Tool catalog not available"

        with self._tracer.start_as_current_span(f"tool.{action}") as span:
            span.set_attribute("tool.name", action)
            span.set_attribute("tool.input", json.dumps(action_input)[:500])

            try:
                result = await self.tool_catalog.execute(action, action_input)

                if result.success:
                    span.set_attribute("tool.success", True)
                    # Format result for observation
                    if isinstance(result.result, (dict, list)):
                        return json.dumps(result.result, indent=2, default=str)[:3000]
                    return str(result.result)[:3000]
                else:
                    span.set_attribute("tool.success", False)
                    span.set_attribute("tool.error", result.error or "Unknown error")
                    return f"Tool Error: {result.error}"

            except Exception as e:
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", str(e))
                self._logger.error("Tool execution failed", tool=action, error=str(e))
                return f"Error executing {action}: {e!s}"

    async def run(self, query: str, user_id: str | None = None) -> AgentResult:
        """
        Run the ReAct agent on a query.

        Args:
            query: User query
            user_id: Optional user ID for context

        Returns:
            AgentResult with response and execution steps
        """
        with self._tracer.start_as_current_span("react_agent.run") as span:
            span.set_attribute("query", query[:200])
            if user_id:
                span.set_attribute("user_id", user_id)

            steps: list[AgentStep] = []
            tool_calls: list[dict[str, Any]] = []

            try:
                # Build system prompt with context
                compartment_context = await self._get_compartment_context()
                tools_context = self._get_tools_context()

                system_prompt = REACT_SYSTEM_PROMPT.format(
                    compartment_context=compartment_context,
                    tools_context=tools_context,
                )
                if self.memory:
                    feedback_text = await self.memory.get_feedback_text()
                    if feedback_text:
                        system_prompt = (
                            f"{system_prompt}\n\nRuntime feedback directives:\n{feedback_text}"
                        )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query),
                ]

                # ReAct loop
                for iteration in range(self.max_iterations):
                    self._logger.debug("ReAct iteration", iteration=iteration)

                    # Get LLM response
                    response = await self.llm.ainvoke(messages)
                    response_text = response.content

                    # Parse response
                    parsed = self._parse_agent_response(response_text)
                    thought = parsed.get("thought", "")

                    # Check for final answer
                    if "final_answer" in parsed:
                        step = AgentStep(
                            thought=thought,
                            is_final=True,
                        )
                        steps.append(step)

                        span.set_attribute("iterations", iteration + 1)
                        span.set_attribute("success", True)

                        final_answer = parsed["final_answer"]

                        return AgentResult(
                            success=True,
                            response=final_answer,
                            steps=steps,
                            tool_calls=tool_calls,
                        )

                    # Execute tool action
                    action = parsed.get("action")
                    action_input = parsed.get("action_input", {})

                    if not action:
                        # No action specified, treat as final answer
                        return AgentResult(
                            success=True,
                            response=response_text,
                            steps=steps,
                            tool_calls=tool_calls,
                        )

                    # Record step
                    step = AgentStep(
                        thought=thought,
                        action=action,
                        action_input=action_input,
                    )

                    # Execute tool
                    observation = await self._execute_tool(action, action_input)
                    step.observation = observation
                    steps.append(step)

                    tool_calls.append({
                        "tool": action,
                        "input": action_input,
                        "output": observation[:500],
                    })

                    # Add to messages for next iteration
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content=f"Observation: {observation}"))

                # Max iterations reached
                self._logger.warning("Max iterations reached", max=self.max_iterations)
                span.set_attribute("max_iterations_reached", True)

                # Return last response as answer
                return AgentResult(
                    success=True,
                    response="I've gathered the following information:\n\n" + "\n".join(
                        f"- {step.observation[:200]}..." if step.observation else ""
                        for step in steps
                    ),
                    steps=steps,
                    tool_calls=tool_calls,
                )

            except Exception as e:
                self._logger.error("ReAct agent failed", error=str(e))
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                return AgentResult(
                    success=False,
                    response=f"Agent error: {e!s}",
                    steps=steps,
                    tool_calls=tool_calls,
                    error=str(e),
                )


class SpecializedReActAgent(OCIReActAgent):
    """
    Specialized ReAct agent for specific domains.

    Extends base agent with domain-specific prompts and tool preferences.
    """

    def __init__(
        self,
        domain: str,
        llm: Any,
        tool_catalog: ToolCatalog | None = None,
        resource_cache: OCIResourceCache | None = None,
        prompts_dir: str = "prompts",
        **kwargs,
    ):
        """
        Initialize specialized agent.

        Args:
            domain: Agent domain (database, infrastructure, finops, security)
            llm: LangChain LLM
            tool_catalog: MCP tool catalog
            resource_cache: OCI resource cache
            prompts_dir: Directory containing prompt files
        """
        agent_id = f"oci-{domain}-agent"
        super().__init__(
            llm=llm,
            tool_catalog=tool_catalog,
            resource_cache=resource_cache,
            agent_id=agent_id,
            **kwargs,
        )
        self.domain = domain
        self._prompts_dir = Path(prompts_dir)
        self.domain_prompt = self._load_domain_prompt()

    def _load_domain_prompt(self) -> str:
        """Load domain-specific prompt from file."""
        # Try specific domain file first
        prompt_file = self._prompts_dir / f"{self.domain}.md"

        # Fallback mappings for common domains if exact match not found
        if not prompt_file.exists():
            # Try mapping common aliases
            aliases = {
                "infrastructure": "05-INFRASTRUCTURE-AGENT.md",
                "database": "01-DB-TROUBLESHOOT-AGENT.md",
                "finops": "04-FINOPS-AGENT.md",
                "security": "03-SECURITY-THREAT-AGENT.md",
                "observability": "02-LOG-ANALYTICS-AGENT.md",
            }
            if self.domain in aliases:
                prompt_file = self._prompts_dir / aliases[self.domain]
            else:
                 # Check for generic agent prompt
                 prompt_file = self._prompts_dir / "01-GENERIC-AGENT.md"

        if prompt_file.exists():
            try:
                self._logger.info(f"Loading prompt from {prompt_file}")
                return prompt_file.read_text().strip()
            except Exception as e:
                self._logger.error(f"Failed to load prompt from {prompt_file}", error=str(e))

        # severe fallback if files missing
        return f"You are an OCI {self.domain.title()} Agent. Use available tools to assist the user."

    async def _get_compartment_context(self) -> str:
        """Get context with domain-specific additions and dynamic tool discovery."""
        base_context = await super()._get_compartment_context()

        # Get dynamically discovered tools for this domain
        domain_tools_context = await self._get_domain_tools_context()

        return f"{self.domain_prompt}\n\n## Domain Tools\n{domain_tools_context}\n\n## Compartments\n{base_context}"

    async def _get_domain_tools_context(self) -> str:
        """Get dynamically discovered tools for this domain.

        Uses the ToolCatalog.get_tools_for_domain() to discover
        available tools at runtime, ensuring agents always have
        access to the latest tools without hardcoding.
        """
        if not self.tool_catalog:
            return "No tools available for this domain."

        try:
            # Ensure catalog is fresh
            await self.tool_catalog.ensure_fresh()

            # Get tools for this domain
            domain_tools = self.tool_catalog.get_tools_for_domain(self.domain)

            if not domain_tools:
                # Fall back to search
                search_results = self.tool_catalog.search_tools(
                    domain=self.domain,
                    max_tier=3,
                    limit=15,
                )
                if search_results:
                    lines = []
                    for tool in search_results:
                        lines.append(f"- **{tool['name']}**: {tool['description'][:80]}")
                    return "\n".join(lines)
                return f"No tools found for domain: {self.domain}"

            # Format tool list
            lines = []
            for tool in domain_tools[:15]:  # Limit to 15 tools
                desc = (tool.description or "")[:80]
                lines.append(f"- **{tool.name}**: {desc}")

            return "\n".join(lines)

        except Exception as e:
            self._logger.error("Failed to get domain tools", error=str(e))
            return f"Error discovering tools: {e}"
