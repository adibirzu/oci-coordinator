"""
LangGraph nodes for the Coordinator.

Implements the graph nodes:
- input_node: Process initial query
- classifier_node: Classify intent
- router_node: Determine routing (workflow vs agent vs direct)
- workflow_node: Execute deterministic workflow
- agent_node: Delegate to specialized agent
- action_node: Execute tool calls
- output_node: Prepare final response
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.coordinator.state import (
    AgentContext,
    CoordinatorState,
    IntentCategory,
    IntentClassification,
    RoutingType,
    ToolCall,
    ToolResult,
    determine_routing,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.agents.catalog import AgentCatalog
    from src.memory.manager import SharedMemoryManager
    from src.mcp.catalog import ToolCatalog

logger = structlog.get_logger(__name__)


class CoordinatorNodes:
    """
    Collection of LangGraph nodes for the Coordinator.

    Implements workflow-first routing:
    - 70%+ requests → deterministic workflows
    - Remaining → agentic LLM reasoning
    """

    def __init__(
        self,
        llm: "BaseChatModel",
        tool_catalog: "ToolCatalog",
        agent_catalog: "AgentCatalog",
        memory: "SharedMemoryManager",
        workflow_registry: dict[str, Any] | None = None,
    ):
        """
        Initialize coordinator nodes.

        Args:
            llm: LangChain chat model for reasoning
            tool_catalog: Catalog of MCP tools
            agent_catalog: Catalog of specialized agents
            memory: Shared memory manager
            workflow_registry: Map of workflow names to workflow functions
        """
        self.llm = llm
        self.tool_catalog = tool_catalog
        self.agent_catalog = agent_catalog
        self.memory = memory
        self.workflow_registry = workflow_registry or {}
        self._logger = logger.bind(component="CoordinatorNodes")

    # ─────────────────────────────────────────────────────────────────────────
    # Input Node
    # ─────────────────────────────────────────────────────────────────────────

    async def input_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Process initial input.

        Extracts query from messages and prepares for classification.

        Args:
            state: Current coordinator state

        Returns:
            State updates
        """
        self._logger.debug("Processing input", message_count=len(state.messages))

        # Extract query from last human message
        query = ""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return {"error": "No query found in messages"}

        return {
            "query": query,
            "iteration": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Classifier Node
    # ─────────────────────────────────────────────────────────────────────────

    async def classifier_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Classify user intent.

        Uses LLM to understand:
        - What the user wants
        - Which domain(s) are involved
        - Confidence level
        - Workflow/agent suggestions

        Args:
            state: Current coordinator state

        Returns:
            State updates with intent classification
        """
        self._logger.info("Classifying intent", query=state.query[:100])

        # Build classification prompt
        classification_prompt = self._build_classification_prompt(state.query)

        try:
            response = await self.llm.ainvoke([HumanMessage(content=classification_prompt)])

            # Parse classification from response
            intent = self._parse_classification(response.content, state.query)

            self._logger.info(
                "Intent classified",
                intent=intent.intent,
                category=intent.category.value,
                confidence=intent.confidence,
                domains=intent.domains,
            )

            return {"intent": intent}

        except Exception as e:
            self._logger.error("Classification failed", error=str(e))
            # Return low-confidence fallback
            return {
                "intent": IntentClassification(
                    intent="unknown",
                    category=IntentCategory.UNKNOWN,
                    confidence=0.3,
                    domains=[],
                )
            }

    def _build_classification_prompt(self, query: str) -> str:
        """Build the intent classification prompt."""
        # Get available workflows and agents for context
        available_workflows = list(self.workflow_registry.keys())
        available_agents = [
            agent.role for agent in self.agent_catalog.list_all()
        ]

        return f"""Classify the following user query for an OCI (Oracle Cloud Infrastructure) management system.

Query: "{query}"

Available Workflows: {available_workflows}
Available Agents: {available_agents}

Respond with a JSON object:
{{
    "intent": "<specific intent like list_instances, analyze_performance, troubleshoot_db>",
    "category": "<query|action|analysis|troubleshoot|unknown>",
    "confidence": <0.0 to 1.0>,
    "domains": ["<domain1>", "<domain2>"],
    "entities": {{"entity_name": "value"}},
    "suggested_workflow": "<workflow name or null>",
    "suggested_agent": "<agent role or null>"
}}

Categories:
- query: Information retrieval (list, get, describe)
- action: Perform operation (start, stop, create, delete)
- analysis: Complex analysis (cost analysis, performance review)
- troubleshoot: Diagnose issues (why is X slow, fix Y)
- unknown: Cannot determine

Domains: compute, network, database, security, cost, observability, storage

Return only the JSON object, no other text."""

    def _parse_classification(
        self, response: str, query: str
    ) -> IntentClassification:
        """Parse LLM response into IntentClassification."""
        import json

        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(
                    line for line in lines if not line.startswith("```")
                )

            data = json.loads(response)

            return IntentClassification(
                intent=data.get("intent", "unknown"),
                category=IntentCategory(data.get("category", "unknown")),
                confidence=float(data.get("confidence", 0.5)),
                domains=data.get("domains", []),
                entities=data.get("entities", {}),
                suggested_workflow=data.get("suggested_workflow"),
                suggested_agent=data.get("suggested_agent"),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._logger.warning(
                "Failed to parse classification",
                error=str(e),
                response=response[:200],
            )
            # Fallback classification
            return IntentClassification(
                intent="unknown",
                category=IntentCategory.UNKNOWN,
                confidence=0.4,
                domains=[],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Router Node
    # ─────────────────────────────────────────────────────────────────────────

    async def router_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Determine routing based on intent classification.

        Implements workflow-first design:
        - High confidence + workflow match → WORKFLOW
        - Medium confidence + agent match → AGENT
        - Low confidence → ESCALATE
        - Otherwise → DIRECT

        Args:
            state: Current coordinator state

        Returns:
            State updates with routing decision
        """
        if not state.intent:
            self._logger.warning("No intent for routing")
            return {
                "error": "No intent classification available",
            }

        routing = determine_routing(state.intent)

        self._logger.info(
            "Routing decision",
            routing_type=routing.routing_type.value,
            target=routing.target,
            confidence=routing.confidence,
            reasoning=routing.reasoning,
        )

        # Prepare agent context if routing to agent
        agent_context = None
        if routing.routing_type == RoutingType.AGENT:
            agent_context = AgentContext(
                query=state.query,
                intent=state.intent,
                previous_results=state.tool_results,
            )

        return {
            "routing": routing,
            "agent_context": agent_context,
            "current_agent": routing.target if routing.routing_type == RoutingType.AGENT else None,
            "workflow_name": routing.target if routing.routing_type == RoutingType.WORKFLOW else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow Node
    # ─────────────────────────────────────────────────────────────────────────

    async def workflow_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Execute deterministic workflow.

        Workflows are pre-defined sequences of operations that don't
        require LLM reasoning - they execute deterministically.

        Args:
            state: Current coordinator state

        Returns:
            State updates with workflow result
        """
        workflow_name = state.workflow_name

        if not workflow_name:
            return {"error": "No workflow specified"}

        workflow = self.workflow_registry.get(workflow_name)
        if not workflow:
            self._logger.warning("Workflow not found", workflow=workflow_name)
            # Fallback to agentic
            return {
                "routing": state.routing._replace(routing_type=RoutingType.AGENT)
                if state.routing
                else None,
                "error": f"Workflow '{workflow_name}' not found",
            }

        self._logger.info("Executing workflow", workflow=workflow_name)

        try:
            start_time = time.time()
            result = await workflow(
                query=state.query,
                entities=state.intent.entities if state.intent else {},
                tool_catalog=self.tool_catalog,
                memory=self.memory,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            self._logger.info(
                "Workflow completed",
                workflow=workflow_name,
                duration_ms=duration_ms,
            )

            return {
                "final_response": result,
                "workflow_state": {"completed": True, "duration_ms": duration_ms},
            }

        except Exception as e:
            self._logger.error(
                "Workflow failed",
                workflow=workflow_name,
                error=str(e),
            )
            # Try fallback if available
            if state.routing and state.routing.fallback:
                return {
                    "routing": state.routing.fallback,
                    "workflow_name": None,
                    "current_agent": state.routing.fallback.target,
                }
            return {"error": f"Workflow failed: {e}"}

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Node
    # ─────────────────────────────────────────────────────────────────────────

    async def agent_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Invoke LLM for reasoning or delegate to specialized agent.

        Args:
            state: Current coordinator state

        Returns:
            State updates with agent response
        """
        self._logger.debug(
            "Agent node",
            iteration=state.iteration,
            current_agent=state.current_agent,
        )

        # If delegating to specialized agent
        if state.current_agent:
            return await self._invoke_specialized_agent(state)

        # Otherwise, use coordinator LLM
        return await self._invoke_coordinator_llm(state)

    async def _invoke_specialized_agent(
        self, state: CoordinatorState
    ) -> dict[str, Any]:
        """Delegate to a specialized agent."""
        agent_role = state.current_agent

        if not agent_role:
            return {"error": "No agent specified"}

        # Get agent from catalog with output format config
        agent_config = {
            "output_format": state.output_format,
            "channel_type": state.channel_type,
        }

        agent_instance = self.agent_catalog.instantiate(
            role=agent_role,
            memory_manager=self.memory,
            tool_catalog=self.tool_catalog,
            config=agent_config,
        )

        if not agent_instance:
            self._logger.warning("Agent not found", role=agent_role)
            return {"error": f"Agent '{agent_role}' not available"}

        self._logger.info(
            "Delegating to agent",
            agent_role=agent_role,
            output_format=state.output_format,
        )

        try:
            start_time = time.time()
            context = state.agent_context.to_dict() if state.agent_context else {}
            context["output_format"] = state.output_format
            context["channel_type"] = state.channel_type

            result = await agent_instance.invoke(state.query, context)
            duration_ms = int((time.time() - start_time) * 1000)

            # Record metrics
            self.agent_catalog.record_invocation(
                role=agent_role,
                duration_ms=duration_ms,
                success=True,
            )

            self._logger.info(
                "Agent completed",
                agent_role=agent_role,
                duration_ms=duration_ms,
            )

            return {
                "agent_response": result,
                "final_response": result,
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Record failed invocation
            self.agent_catalog.record_invocation(
                role=agent_role,
                duration_ms=duration_ms,
                success=False,
            )

            self._logger.error(
                "Agent failed",
                agent_role=agent_role,
                error=str(e),
            )
            return {"error": f"Agent failed: {e}"}

    async def _invoke_coordinator_llm(
        self, state: CoordinatorState
    ) -> dict[str, Any]:
        """Invoke the coordinator's LLM for reasoning."""
        # Build messages for LLM
        messages = list(state.messages)

        # Add tool results as messages
        for result in state.tool_results:
            messages.append(
                ToolMessage(
                    content=str(result.result),
                    tool_call_id=result.tool_call_id,
                )
            )

        try:
            response = await self.llm.ainvoke(messages)

            self._logger.debug(
                "LLM response",
                has_tool_calls=bool(
                    hasattr(response, "tool_calls") and response.tool_calls
                ),
            )

            # Extract tool calls if any
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", f"call_{i}"),
                        name=tc.get("name", ""),
                        arguments=tc.get("args", {}),
                    )
                    for i, tc in enumerate(response.tool_calls)
                ]

            return {
                "messages": [response],
                "tool_calls": tool_calls,
                "tool_results": [],  # Clear previous results
                "iteration": state.iteration + 1,
            }

        except Exception as e:
            self._logger.error("LLM invocation failed", error=str(e))
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"Error: {e}")],
                "tool_calls": [],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Action Node
    # ─────────────────────────────────────────────────────────────────────────

    async def action_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Execute pending tool calls.

        Args:
            state: Current coordinator state

        Returns:
            State updates with tool results
        """
        if not state.tool_calls:
            return {"tool_results": []}

        self._logger.info(
            "Executing tools",
            tool_count=len(state.tool_calls),
        )

        results = []
        for tool_call in state.tool_calls:
            start_time = time.time()

            try:
                result = await self._execute_tool(tool_call)
                duration_ms = int((time.time() - start_time) * 1000)

                results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=result,
                        success=True,
                        duration_ms=duration_ms,
                    )
                )

                self._logger.debug(
                    "Tool executed",
                    tool=tool_call.name,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                self._logger.error(
                    "Tool execution failed",
                    tool=tool_call.name,
                    error=str(e),
                )

                results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=None,
                        success=False,
                        duration_ms=duration_ms,
                        error=str(e),
                    )
                )

        return {
            "tool_results": results,
            "tool_calls": [],  # Clear after processing
        }

    async def _execute_tool(self, tool_call: ToolCall) -> Any:
        """Execute a single tool via the tool catalog."""
        if not self.tool_catalog:
            raise RuntimeError("Tool catalog not initialized")

        tool_def = self.tool_catalog.get_tool(tool_call.name)
        if not tool_def:
            raise ValueError(f"Tool not found: {tool_call.name}")

        # Execute through catalog
        return await self.tool_catalog.execute(tool_call.name, tool_call.arguments)

    # ─────────────────────────────────────────────────────────────────────────
    # Output Node
    # ─────────────────────────────────────────────────────────────────────────

    async def output_node(self, state: CoordinatorState) -> dict[str, Any]:
        """
        Prepare final output.

        Extracts the final response from state and prepares for return.

        Args:
            state: Current coordinator state

        Returns:
            State updates with final response
        """
        self._logger.debug(
            "Output node",
            iterations=state.iteration,
            has_error=bool(state.error),
        )

        # If we already have a final response, return it
        if state.final_response:
            return {}

        # Extract from last AI message
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                return {"final_response": msg.content}

        # Fallback
        if state.error:
            return {"final_response": f"Error: {state.error}"}

        return {"final_response": "No response generated."}


# ─────────────────────────────────────────────────────────────────────────────
# Routing Functions for Conditional Edges
# ─────────────────────────────────────────────────────────────────────────────


def should_continue_after_router(state: CoordinatorState) -> str:
    """
    Determine next node after routing decision.

    Returns:
        Node name: "workflow", "agent", "direct", or "escalate"
    """
    if state.error:
        return "output"

    if not state.routing:
        return "agent"  # Default to agentic

    routing_type = state.routing.routing_type

    if routing_type == RoutingType.WORKFLOW:
        return "workflow"
    elif routing_type == RoutingType.AGENT:
        return "agent"
    elif routing_type == RoutingType.ESCALATE:
        return "output"  # Escalate goes directly to output
    else:
        return "agent"  # DIRECT uses agent node


def should_continue_after_agent(state: CoordinatorState) -> str:
    """
    Determine if we should continue to action or end.

    Returns:
        "action" if tool calls pending, "output" otherwise
    """
    if state.error:
        return "output"

    if state.final_response:
        return "output"

    if state.iteration >= state.max_iterations:
        return "output"

    if state.has_pending_tools():
        return "action"

    return "output"


def should_loop_from_action(state: CoordinatorState) -> str:
    """
    Determine if we should loop back to agent after action.

    Returns:
        "agent" to continue, "output" to end
    """
    if state.error:
        return "output"

    if state.iteration >= state.max_iterations:
        return "output"

    return "agent"
