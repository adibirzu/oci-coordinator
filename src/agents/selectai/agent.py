"""
SelectAI Agent for Oracle Autonomous Database.

Specialized agent for natural language database interaction using
Oracle Autonomous Database's SelectAI capabilities (DBMS_CLOUD_AI).

Features:
- Natural language to SQL (NL2SQL) translation
- Chat with database context
- Text summarization and translation
- AI agent orchestration (DBMS_CLOUD_AI_AGENT)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog
from langgraph.graph import END, StateGraph

from src.agents.base import (
    AgentDefinition,
    AgentMetadata,
    BaseAgent,
    KafkaTopics,
)
from src.agents.self_healing import SelfHealingMixin
from src.mcp.client import ToolCallResult

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


# SelectAI action types
SelectAIAction = Literal["showsql", "runsql", "explainsql", "narrate", "summarize", "translate", "chat"]


@dataclass
class SelectAIState:
    """State for SelectAI workflow."""

    query: str = ""
    intent: str = "nl2sql"  # nl2sql, chat, summarize, agent_run, profile_manage

    # Database configuration
    database_id: str | None = None
    profile_name: str | None = None
    schema_name: str = "ADMIN"

    # SelectAI parameters
    action: SelectAIAction = "runsql"
    tables: list[dict] = field(default_factory=list)

    # Agent execution (for DBMS_CLOUD_AI_AGENT)
    agent_name: str | None = None
    team_name: str | None = None
    session_id: str | None = None

    # Results
    generated_sql: str | None = None
    query_results: list[dict] = field(default_factory=list)
    llm_response: str | None = None
    agent_response: dict | None = None

    # Reasoning chain for transparency
    reasoning_chain: list[str] = field(default_factory=list)

    # State management
    phase: str = "analyze_intent"
    error: str | None = None
    result: str | None = None

    # OCI profile for auth
    oci_profile: str | None = None


class SelectAIAgent(BaseAgent, SelfHealingMixin):
    """
    SelectAI Agent for natural language database interaction.

    Uses Oracle Autonomous Database's SelectAI capabilities for:
    - Natural language to SQL translation
    - Chat with database context
    - Text summarization
    - AI agent orchestration

    This agent integrates with DBMS_CLOUD_AI and DBMS_CLOUD_AI_AGENT
    packages in Oracle Autonomous Database.

    Example Usage:
        agent = SelectAIAgent(tool_catalog=catalog, llm=llm)
        result = await agent.invoke(
            "Show me top 10 customers by revenue",
            context={"database_id": "ocid1..."}
        )
    """

    def __init__(
        self,
        memory_manager: SharedMemoryManager | None = None,
        tool_catalog: ToolCatalog | None = None,
        config: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        """
        Initialize SelectAI Agent.

        Args:
            memory_manager: Shared memory manager
            tool_catalog: Tool catalog for MCP tools
            config: Agent configuration including:
                - default_profile: Default SelectAI profile name
                - default_database: Default ADB OCID
                - connection_type: ords, wallet, or db-observatory
            llm: LangChain LLM for intent analysis
        """
        super().__init__(memory_manager, tool_catalog, config)
        self.llm = llm
        self._graph: StateGraph | None = None

        # Configuration
        self._default_profile = self.config.get("default_profile", "OCI_GENAI")
        self._default_database = self.config.get("default_database")
        self._connection_type = self.config.get("connection_type", "db-observatory")

        # Initialize self-healing
        if llm:
            self.init_self_healing(
                llm=llm,
                max_retries=2,
                enable_validation=True,
                enable_correction=True,
            )

    @classmethod
    def get_definition(cls) -> AgentDefinition:
        """Return agent definition for catalog registration."""
        return AgentDefinition(
            agent_id="selectai-agent",
            role="selectai-agent",
            capabilities=[
                "nl2sql",
                "data-chat",
                "text-summarization",
                "ai-agent-orchestration",
                "database-qa",
            ],
            skills=[
                "nl2sql_workflow",
                "data_exploration",
                "report_generation",
                "custom_agent_execution",
            ],
            kafka_topics=KafkaTopics(
                consume=["commands.selectai-agent"],
                produce=["results.selectai-agent"],
            ),
            health_endpoint="http://localhost:8016/health",
            metadata=AgentMetadata(
                version="1.0.0",
                namespace="oci-coordinator",
                max_iterations=10,
                timeout_seconds=120,
            ),
            description=(
                "SelectAI Agent for natural language database queries (NL2SQL), "
                "chat with database context, AI agent orchestration, and report "
                "generation using Oracle Autonomous Database SelectAI."
            ),
            mcp_servers=["selectai", "database-observatory"],
            mcp_tools=[
                "oci_selectai_generate",
                "oci_selectai_list_profiles",
                "oci_selectai_get_profile_tables",
                "oci_selectai_run_agent",
                "oci_selectai_ping",
                "oci_selectai_test_connection",
            ],
        )

    def build_graph(self) -> StateGraph:
        """Build the SelectAI workflow graph."""
        graph = StateGraph(SelectAIState)

        # Add nodes
        graph.add_node("analyze_intent", self._analyze_intent_node)
        graph.add_node("resolve_profile", self._resolve_profile_node)
        graph.add_node("execute_nl2sql", self._execute_nl2sql_node)
        graph.add_node("execute_chat", self._execute_chat_node)
        graph.add_node("execute_agent", self._execute_agent_node)
        graph.add_node("output", self._output_node)

        # Set entry point
        graph.set_entry_point("analyze_intent")

        # Add routing
        graph.add_conditional_edges(
            "analyze_intent",
            self._route_by_intent,
            {
                "nl2sql": "resolve_profile",
                "chat": "resolve_profile",
                "summarize": "resolve_profile",
                "agent_run": "execute_agent",
                "error": "output",
            },
        )

        # After profile resolution, route to execution
        graph.add_conditional_edges(
            "resolve_profile",
            self._route_after_profile,
            {
                "nl2sql": "execute_nl2sql",
                "chat": "execute_chat",
                "summarize": "execute_chat",
                "error": "output",
            },
        )

        # All execution paths lead to output
        graph.add_edge("execute_nl2sql", "output")
        graph.add_edge("execute_chat", "output")
        graph.add_edge("execute_agent", "output")
        graph.add_edge("output", END)

        return graph.compile()

    def _route_by_intent(self, state: SelectAIState) -> str:
        """Route based on detected intent."""
        if state.error:
            return "error"
        return state.intent

    def _route_after_profile(self, state: SelectAIState) -> str:
        """Route after profile resolution."""
        if state.error:
            return "error"
        return state.intent

    async def _analyze_intent_node(self, state: SelectAIState) -> dict[str, Any]:
        """Analyze query to determine intent and action type."""
        self._logger.info("Analyzing SelectAI query intent", query=state.query[:100])

        query_lower = state.query.lower()
        reasoning = []

        # Detect intent based on keywords
        intent = "nl2sql"  # Default
        action: SelectAIAction = "runsql"

        # Check for agent/team execution
        if any(kw in query_lower for kw in ["run agent", "execute agent", "run team", "run the"]):
            intent = "agent_run"
            reasoning.append("Detected agent execution request")

            # Try to extract team/agent name
            team_match = re.search(r"run (?:the )?(\w+) (?:agent|team)", query_lower)
            if team_match:
                state.team_name = team_match.group(1).upper()
                reasoning.append(f"Extracted team name: {state.team_name}")

        # Check for chat/conversation
        elif any(kw in query_lower for kw in ["chat", "tell me about", "explain", "what is", "why", "how does"]):
            intent = "chat"
            action = "chat"
            reasoning.append("Detected chat/conversation intent")

        # Check for summarization
        elif any(kw in query_lower for kw in ["summarize", "summary", "overview", "brief"]):
            intent = "summarize"
            action = "narrate"
            reasoning.append("Detected summarization request")

        # Check for SQL generation without execution
        elif any(kw in query_lower for kw in ["show sql", "generate sql", "sql for", "write sql"]):
            intent = "nl2sql"
            action = "showsql"
            reasoning.append("Detected SQL generation request (show only)")

        # Check for explanation
        elif any(kw in query_lower for kw in ["explain sql", "explain the query"]):
            intent = "nl2sql"
            action = "explainsql"
            reasoning.append("Detected SQL explanation request")

        # Default to NL2SQL with execution
        else:
            intent = "nl2sql"
            action = "runsql"
            reasoning.append("Defaulting to NL2SQL with query execution")

        self._logger.info(
            "Intent analysis complete",
            intent=intent,
            action=action,
            reasoning_count=len(reasoning),
        )

        return {
            "intent": intent,
            "action": action,
            "reasoning_chain": reasoning,
            "phase": "resolve_profile",
        }

    async def _resolve_profile_node(self, state: SelectAIState) -> dict[str, Any]:
        """Resolve database and profile configuration."""
        self._logger.info("Resolving SelectAI profile")

        reasoning = list(state.reasoning_chain)

        # Use provided or default database
        database_id = state.database_id or self._default_database
        if not database_id:
            reasoning.append("No database specified, will attempt discovery")
        else:
            reasoning.append(f"Using database: {database_id[:30]}...")

        # Use provided or default profile
        profile_name = state.profile_name or self._default_profile
        reasoning.append(f"Using profile: {profile_name}")

        # If we have tools, try to list available profiles
        if self.tools and database_id:
            try:
                profiles_result = await self.call_tool(
                    "oci_selectai_list_profiles",
                    {"database_id": database_id},
                )
                if isinstance(profiles_result, list) and profiles_result:
                    available = [p.get("name") for p in profiles_result]
                    reasoning.append(f"Available profiles: {', '.join(available[:5])}")

                    # Validate profile exists
                    if profile_name not in available:
                        reasoning.append(f"Profile '{profile_name}' not found, using first available")
                        profile_name = available[0]
            except Exception as e:
                reasoning.append(f"Could not list profiles: {str(e)[:50]}")

        return {
            "database_id": database_id,
            "profile_name": profile_name,
            "reasoning_chain": reasoning,
            "phase": "execute",
        }

    async def _execute_nl2sql_node(self, state: SelectAIState) -> dict[str, Any]:
        """Execute NL2SQL via DBMS_CLOUD_AI.GENERATE."""
        self._logger.info(
            "Executing NL2SQL",
            action=state.action,
            profile=state.profile_name,
        )

        reasoning = list(state.reasoning_chain)
        generated_sql = None
        query_results = []
        error = None

        if not self.tools:
            error = "Tool catalog not initialized"
            reasoning.append(f"Error: {error}")
            return {
                "error": error,
                "reasoning_chain": reasoning,
            }

        try:
            # Call SelectAI generate tool
            result = await self.call_tool(
                "oci_selectai_generate",
                {
                    "prompt": state.query,
                    "profile_name": state.profile_name,
                    "action": state.action,
                    "database_id": state.database_id,
                },
            )

            # Parse result
            if isinstance(result, ToolCallResult):
                if not result.success:
                    error = result.error or "SelectAI execution failed"
                else:
                    result = result.result

            if isinstance(result, dict):
                generated_sql = result.get("sql")
                query_results = result.get("results", [])
                reasoning.append(f"SelectAI generated SQL: {generated_sql[:100] if generated_sql else 'N/A'}...")
                reasoning.append(f"Results: {len(query_results)} rows returned")
            elif isinstance(result, str):
                # May be raw SQL or error
                if result.upper().startswith("SELECT") or result.upper().startswith("WITH"):
                    generated_sql = result
                    reasoning.append(f"Generated SQL: {result[:100]}...")
                else:
                    error = result
                    reasoning.append(f"Error from SelectAI: {result[:100]}")

        except ValueError as e:
            # Tool not found - try fallback through db-observatory
            reasoning.append(f"SelectAI tool not available: {str(e)[:50]}")
            reasoning.append("Attempting fallback via database-observatory...")

            try:
                # Build DBMS_CLOUD_AI.GENERATE call
                sql = self._build_generate_sql(
                    prompt=state.query,
                    profile_name=state.profile_name,
                    action=state.action,
                )

                db_result = await self.call_tool(
                    "oci_database_execute_sql",
                    {
                        "database_id": state.database_id,
                        "sql": sql,
                        "schema": state.schema_name,
                    },
                )

                if isinstance(db_result, dict):
                    generated_sql = db_result.get("result")
                    reasoning.append("Executed via database-observatory fallback")
                elif isinstance(db_result, str):
                    generated_sql = db_result
                    reasoning.append("Received SQL via database-observatory")

            except Exception as fallback_error:
                error = f"Both SelectAI and fallback failed: {str(fallback_error)[:100]}"
                reasoning.append(error)

        except Exception as e:
            error = f"SelectAI execution failed: {str(e)}"
            reasoning.append(error)
            self._logger.error("SelectAI execution error", error=str(e))

        return {
            "generated_sql": generated_sql,
            "query_results": query_results,
            "error": error,
            "reasoning_chain": reasoning,
            "phase": "output",
        }

    async def _execute_chat_node(self, state: SelectAIState) -> dict[str, Any]:
        """Execute chat/narrate via DBMS_CLOUD_AI.GENERATE."""
        self._logger.info(
            "Executing chat/narrate",
            action=state.action,
            profile=state.profile_name,
        )

        reasoning = list(state.reasoning_chain)
        llm_response = None
        error = None

        if not self.tools:
            error = "Tool catalog not initialized"
            return {"error": error, "reasoning_chain": reasoning}

        try:
            result = await self.call_tool(
                "oci_selectai_generate",
                {
                    "prompt": state.query,
                    "profile_name": state.profile_name,
                    "action": state.action,  # chat or narrate
                    "database_id": state.database_id,
                },
            )

            if isinstance(result, ToolCallResult):
                if not result.success:
                    error = result.error
                else:
                    result = result.result

            if isinstance(result, dict):
                llm_response = result.get("response") or result.get("text")
            elif isinstance(result, str):
                llm_response = result

            if llm_response:
                reasoning.append(f"Received response ({len(llm_response)} chars)")

        except Exception as e:
            error = f"Chat execution failed: {str(e)}"
            reasoning.append(error)

        return {
            "llm_response": llm_response,
            "error": error,
            "reasoning_chain": reasoning,
            "phase": "output",
        }

    async def _execute_agent_node(self, state: SelectAIState) -> dict[str, Any]:
        """Execute SelectAI agent via DBMS_CLOUD_AI_AGENT.RUN_TEAM."""
        self._logger.info(
            "Executing SelectAI agent",
            team=state.team_name,
            session=state.session_id,
        )

        reasoning = list(state.reasoning_chain)
        agent_response = None
        error = None

        if not state.team_name:
            error = "No team/agent name specified"
            reasoning.append(error)
            return {"error": error, "reasoning_chain": reasoning}

        if not self.tools:
            error = "Tool catalog not initialized"
            return {"error": error, "reasoning_chain": reasoning}

        try:
            result = await self.call_tool(
                "oci_selectai_run_agent",
                {
                    "database_id": state.database_id,
                    "team_name": state.team_name,
                    "user_prompt": state.query,
                    "session_id": state.session_id,
                },
            )

            if isinstance(result, ToolCallResult):
                if not result.success:
                    error = result.error
                else:
                    agent_response = result.result

            elif isinstance(result, dict):
                agent_response = result
                reasoning.append(f"Agent executed successfully")
            elif isinstance(result, str):
                agent_response = {"response": result}

        except Exception as e:
            error = f"Agent execution failed: {str(e)}"
            reasoning.append(error)

        return {
            "agent_response": agent_response,
            "error": error,
            "reasoning_chain": reasoning,
            "phase": "output",
        }

    async def _output_node(self, state: SelectAIState) -> dict[str, Any]:
        """Format and output results."""
        from src.formatting import (
            CodeBlock,
            ListItem,
            MetricValue,
            ResponseFooter,
            Severity,
            TableData,
            TableRow,
        )

        self._logger.info("Preparing SelectAI output", intent=state.intent)

        # Determine severity and title based on outcome
        if state.error:
            severity = "high"
            title = "SelectAI Error"
        else:
            severity = "success"
            title = self._get_title_for_intent(state.intent)

        response = self.create_response(
            title=title,
            subtitle=f"Profile: {state.profile_name or 'N/A'}",
            severity=severity,
        )

        # Add error section if present
        if state.error:
            response.add_section(
                title="Error",
                list_items=[ListItem(text=state.error, severity=Severity.HIGH)],
            )
            return {"result": self.format_response(response)}

        # Format based on intent
        if state.intent == "nl2sql":
            # Show generated SQL
            if state.generated_sql:
                response.add_code_block(
                    CodeBlock(
                        title="Generated SQL",
                        language="sql",
                        code=state.generated_sql,
                    )
                )

            # Show query results as table
            if state.query_results:
                if state.query_results and len(state.query_results) > 0:
                    first_row = state.query_results[0]
                    if isinstance(first_row, dict):
                        headers = list(first_row.keys())
                        rows = [
                            TableRow(cells=[str(row.get(h, "")) for h in headers])
                            for row in state.query_results[:20]  # Limit rows
                        ]

                        table = TableData(
                            title="Query Results",
                            headers=headers,
                            rows=rows,
                        )
                        response.add_table("Results", table)

                response.add_metrics(
                    "Summary",
                    [MetricValue(label="Rows", value=str(len(state.query_results)))],
                )

        elif state.intent in ("chat", "summarize"):
            if state.llm_response:
                response.add_section(
                    title="Response",
                    text=state.llm_response,
                )

        elif state.intent == "agent_run":
            if state.agent_response:
                if isinstance(state.agent_response, dict):
                    response.add_section(
                        title="Agent Response",
                        text=state.agent_response.get("response", json.dumps(state.agent_response, indent=2)),
                    )
                else:
                    response.add_section(
                        title="Agent Response",
                        text=str(state.agent_response),
                    )

        # Add reasoning chain for transparency (collapsed)
        if state.reasoning_chain and len(state.reasoning_chain) <= 5:
            reasoning_items = [
                ListItem(text=step, severity=Severity.INFO)
                for step in state.reasoning_chain
            ]
            response.add_section(
                title="Reasoning",
                list_items=reasoning_items,
            )

        # Footer
        response.footer = ResponseFooter(
            help_text="Use 'show sql' for SQL only, 'explain' for detailed explanation",
        )

        return {"result": self.format_response(response)}

    def _get_title_for_intent(self, intent: str) -> str:
        """Get display title for intent."""
        titles = {
            "nl2sql": "SelectAI Query Results",
            "chat": "SelectAI Chat",
            "summarize": "Data Summary",
            "agent_run": "Agent Execution Results",
        }
        return titles.get(intent, "SelectAI Response")

    def _build_generate_sql(
        self,
        prompt: str,
        profile_name: str,
        action: SelectAIAction,
    ) -> str:
        """Build DBMS_CLOUD_AI.GENERATE SQL statement for fallback execution."""
        # Escape single quotes in prompt
        escaped_prompt = prompt.replace("'", "''")

        return f"""
SELECT DBMS_CLOUD_AI.GENERATE(
    prompt => '{escaped_prompt}',
    profile_name => '{profile_name}',
    action => '{action}'
) AS response FROM DUAL
"""

    async def invoke(self, query: str, context: dict[str, Any] | None = None) -> str:
        """
        Execute SelectAI workflow.

        Args:
            query: Natural language query or chat message
            context: Additional context including:
                - database_id: Target ADB OCID
                - profile_name: SelectAI profile name
                - action: Force specific action (showsql, runsql, etc.)
                - team_name: For agent execution
                - session_id: Conversation session ID

        Returns:
            Formatted response string
        """
        context = context or {}
        metadata = context.get("metadata", {})

        graph = self.build_graph()

        initial_state = SelectAIState(
            query=query,
            database_id=context.get("database_id", self._default_database),
            profile_name=context.get("profile_name", self._default_profile),
            action=context.get("action", "runsql"),
            team_name=context.get("team_name"),
            session_id=context.get("session_id"),
            oci_profile=metadata.get("oci_profile"),
        )

        result = await graph.ainvoke(initial_state)
        return result.get("result", "No response from SelectAI.")
