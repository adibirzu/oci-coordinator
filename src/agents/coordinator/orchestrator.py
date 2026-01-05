"""
Parallel Orchestrator for Multi-Agent Execution.

Implements the orchestrator-worker pattern from Anthropic's best practices:
- Lead agent (coordinator) decomposes tasks
- 3-5 subagents execute in parallel
- Results are aggregated and synthesized

Features:
- Dynamic task decomposition based on complexity
- Parallel execution with bounded concurrency
- Result synthesis from multiple agents
- Graceful degradation on failures

Usage:
    from src.agents.coordinator.orchestrator import ParallelOrchestrator

    orchestrator = ParallelOrchestrator(
        agent_catalog=catalog,
        tool_catalog=tools,
        llm=llm,
    )

    result = await orchestrator.execute(
        query="Analyze database performance and cost for the production compartment",
        context={"compartment_id": "ocid1..."},
    )
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.protocol import (
    AgentMessage,
    AgentResult,
    MessageBus,
    MessagePriority,
    MessageStatus,
    TaskSpecification,
    get_message_bus,
)
from src.mcp.dynamic_manager import (
    DynamicToolManager,
    UpdateEvent,
    ToolChangeEvent,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.agents.catalog import AgentCatalog
    from src.mcp.catalog import ToolCatalog
    from src.memory.manager import SharedMemoryManager

logger = structlog.get_logger(__name__)


class ComplexityLevel(str, Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Single agent, <5 tool calls
    MODERATE = "moderate"  # Single agent, 5-10 tool calls
    COMPLEX = "complex"  # Multiple agents, parallel execution
    VERY_COMPLEX = "very_complex"  # Requires decomposition + synthesis


@dataclass
class ComplexityAssessment:
    """Result of task complexity assessment."""

    level: ComplexityLevel
    score: float  # 0.0 to 1.0
    domains_involved: list[str]
    estimated_agents: int
    estimated_tool_calls: int
    decomposable: bool
    suggested_parallelism: int
    reasoning: str


@dataclass
class SubtaskDefinition:
    """Definition of a subtask for parallel execution."""

    task_id: str
    intent: str
    description: str
    target_agent: str
    domains: list[str]
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0  # Higher = more important


@dataclass
class OrchestratorResult:
    """Result from orchestrator execution."""

    success: bool
    response: str
    agent_results: list[AgentResult]
    execution_time_ms: int
    total_tool_calls: int
    agents_used: list[str]
    error: str | None = None


class ParallelOrchestrator:
    """
    Orchestrates parallel execution of multiple agents.

    Implements Anthropic's best practices:
    - Dynamic complexity assessment
    - Task decomposition for complex queries
    - Bounded parallel execution (3-5 agents)
    - Result synthesis with context awareness
    """

    # Complexity thresholds
    SIMPLE_THRESHOLD = 0.3
    MODERATE_THRESHOLD = 0.5
    COMPLEX_THRESHOLD = 0.7

    # Execution limits
    DEFAULT_MAX_AGENTS = 5
    DEFAULT_TIMEOUT = 60

    # Loop prevention
    MAX_ITERATIONS = 10  # Maximum orchestration iterations
    MAX_RECURSION_DEPTH = 3  # Maximum nesting of agent calls

    def __init__(
        self,
        agent_catalog: AgentCatalog,
        tool_catalog: ToolCatalog,
        llm: BaseChatModel,
        memory: SharedMemoryManager | None = None,
        message_bus: MessageBus | None = None,
        max_concurrent_agents: int = DEFAULT_MAX_AGENTS,
        default_timeout: int = DEFAULT_TIMEOUT,
        dynamic_manager: DynamicToolManager | None = None,
    ):
        """
        Initialize the parallel orchestrator.

        Args:
            agent_catalog: Catalog of available agents
            tool_catalog: Catalog of available tools
            llm: LLM for reasoning and synthesis
            memory: Shared memory manager
            message_bus: Message bus for A2A communication
            max_concurrent_agents: Maximum agents running in parallel
            default_timeout: Default timeout per agent (seconds)
            dynamic_manager: Dynamic tool manager for hot-reload support
        """
        self.agent_catalog = agent_catalog
        self.tool_catalog = tool_catalog
        self.llm = llm
        self.memory = memory
        self.message_bus = message_bus or get_message_bus()
        self.max_concurrent = max_concurrent_agents
        self.default_timeout = default_timeout
        self._logger = logger.bind(component="ParallelOrchestrator")

        # Dynamic tool manager for auto-updates
        self._dynamic_manager = dynamic_manager or DynamicToolManager.get_instance(
            tool_catalog=tool_catalog,
            agent_catalog=agent_catalog,
        )

        # Loop prevention state
        self._iteration_count = 0
        self._active_tasks: set[str] = set()  # Track active task IDs
        self._seen_queries: dict[str, int] = {}  # Track repeated queries

        # Register agent handlers with message bus
        self._register_agent_handlers()

        # Register for dynamic tool updates
        self._register_dynamic_callbacks()

    def _register_agent_handlers(self) -> None:
        """Register all agents as message handlers."""
        for agent_def in self.agent_catalog.list_all():
            self.message_bus.register_handler(
                agent_def.agent_id,
                self._create_agent_handler(agent_def),
            )
            self._logger.debug(
                "Registered agent handler",
                agent_id=agent_def.agent_id,
            )

    def _register_dynamic_callbacks(self) -> None:
        """Register callbacks for dynamic tool/skill updates."""
        # Listen for tool additions
        self._dynamic_manager.on_update(
            UpdateEvent.TOOLS_ADDED,
            self._handle_tools_updated,
        )

        # Listen for runbook registrations
        self._dynamic_manager.on_update(
            UpdateEvent.RUNBOOKS_REGISTERED,
            self._handle_runbooks_registered,
        )

        # Listen for sync completions
        self._dynamic_manager.on_update(
            UpdateEvent.SYNC_COMPLETE,
            self._handle_sync_complete,
        )

        self._logger.info("Dynamic callbacks registered")

    async def _handle_tools_updated(self, event: ToolChangeEvent) -> None:
        """Handle tool update events."""
        self._logger.info(
            "Tools updated event received",
            tools_count=len(event.tools_affected),
            details=event.details,
        )

        # Refresh tool catalog to pick up new tools
        await self.tool_catalog.ensure_fresh(force=True)

        # Re-register any agent handlers that might use new tools
        self._logger.debug("Tool catalog refreshed after update")

    async def _handle_runbooks_registered(self, event: ToolChangeEvent) -> None:
        """Handle runbook registration events."""
        self._logger.info(
            "Runbooks registered event received",
            runbooks=event.tools_affected,
            details=event.details,
        )

        # Runbooks are now available as skills
        # Agents can discover them via the skill registry

    async def _handle_sync_complete(self, event: ToolChangeEvent) -> None:
        """Handle sync completion events."""
        self._logger.info(
            "Sync complete event received",
            timestamp=event.timestamp.isoformat(),
            details=event.details,
        )

    async def start_dynamic_manager(self) -> None:
        """
        Start the dynamic tool manager for auto-sync.

        Call this during application startup to enable hot-reload
        of tools, skills, and runbooks.
        """
        await self._dynamic_manager.start()
        self._logger.info("Dynamic tool manager started")

    async def stop_dynamic_manager(self) -> None:
        """Stop the dynamic tool manager."""
        await self._dynamic_manager.stop()
        self._logger.info("Dynamic tool manager stopped")

    async def register_runbook(
        self,
        runbook_id: str,
        name: str,
        description: str,
        category: str,
        executor: callable | None = None,
    ) -> None:
        """
        Register a runbook as an executable skill.

        Args:
            runbook_id: Unique runbook identifier
            name: Human-readable name
            description: Runbook description
            category: Category (monitoring, troubleshooting, etc.)
            executor: Optional custom executor function
        """
        await self._dynamic_manager.register_runbook_skill(
            runbook_id=runbook_id,
            name=name,
            description=description,
            category=category,
            executor=executor,
        )
        self._logger.info(
            "Runbook registered via orchestrator",
            runbook_id=runbook_id,
            name=name,
        )

    def get_dynamic_status(self) -> dict[str, Any]:
        """Get status of dynamic tool manager."""
        return self._dynamic_manager.get_status()

    def _create_agent_handler(self, agent_def) -> callable:
        """Create a message handler for an agent."""

        async def handler(message: AgentMessage) -> AgentResult:
            """Handle message by invoking the agent."""
            start_time = time.time()

            try:
                # Instantiate the agent
                agent_instance = self.agent_catalog.instantiate(
                    role=agent_def.role,
                    memory_manager=self.memory,
                    tool_catalog=self.tool_catalog,
                    llm=self.llm,
                )

                if not agent_instance:
                    return message.create_response(
                        success=False,
                        result=None,
                        error=f"Could not instantiate agent: {agent_def.agent_id}",
                    )

                # Build context from message
                context = {
                    **message.context,
                    **message.payload,
                    "task_boundaries": message.boundaries,
                    "output_format": message.output_format,
                }

                # Invoke agent
                result = await agent_instance.invoke(
                    query=message.intent,
                    context=context,
                )

                execution_time = int((time.time() - start_time) * 1000)

                return AgentResult(
                    message_id=message.message_id,
                    task_id=message.task_id,
                    sender=agent_def.agent_id,
                    recipient=message.sender,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time,
                )

            except Exception as e:
                execution_time = int((time.time() - start_time) * 1000)
                self._logger.error(
                    "Agent handler failed",
                    agent_id=agent_def.agent_id,
                    error=str(e),
                )
                return AgentResult(
                    message_id=message.message_id,
                    task_id=message.task_id,
                    sender=agent_def.agent_id,
                    recipient=message.sender,
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time,
                    status=MessageStatus.FAILED,
                )

        return handler

    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        force_parallel: bool = False,
    ) -> OrchestratorResult:
        """
        Execute a query with automatic complexity handling.

        Args:
            query: User query
            context: Additional context
            force_parallel: Force parallel execution even for simple queries

        Returns:
            OrchestratorResult with aggregated response
        """
        start_time = time.time()
        context = context or {}
        task_id = context.get("task_id", str(uuid.uuid4())[:8])

        # Loop prevention check
        query_hash = hash(query[:100])  # Use first 100 chars for dedup
        self._seen_queries[query_hash] = self._seen_queries.get(query_hash, 0) + 1

        if self._seen_queries[query_hash] > 3:
            self._logger.warning(
                "Loop detected: repeated query",
                query_hash=query_hash,
                count=self._seen_queries[query_hash],
            )
            return OrchestratorResult(
                success=False,
                response="Request loop detected. This query has been processed multiple times. Please rephrase or try a different question.",
                agent_results=[],
                execution_time_ms=0,
                total_tool_calls=0,
                agents_used=[],
                error="loop_detected",
            )

        # Check for maximum iterations
        self._iteration_count += 1
        if self._iteration_count > self.MAX_ITERATIONS:
            self._logger.warning(
                "Maximum iterations exceeded",
                count=self._iteration_count,
                max=self.MAX_ITERATIONS,
            )
            return OrchestratorResult(
                success=False,
                response=f"Maximum orchestration iterations ({self.MAX_ITERATIONS}) exceeded. Please simplify your request.",
                agent_results=[],
                execution_time_ms=0,
                total_tool_calls=0,
                agents_used=[],
                error="max_iterations_exceeded",
            )

        # Check for task already in progress (reentrant call)
        if task_id in self._active_tasks:
            self._logger.warning(
                "Reentrant task detected",
                task_id=task_id,
            )
            return OrchestratorResult(
                success=False,
                response="This task is already being processed. Please wait for completion.",
                agent_results=[],
                execution_time_ms=0,
                total_tool_calls=0,
                agents_used=[],
                error="reentrant_task",
            )

        # Mark task as active
        self._active_tasks.add(task_id)

        try:
            self._logger.info(
                "Starting orchestration",
                query_length=len(query),
                force_parallel=force_parallel,
                iteration=self._iteration_count,
                task_id=task_id,
            )

            # Step 1: Assess complexity
            complexity = await self._assess_complexity(query, context)

            self._logger.info(
                "Complexity assessed",
                level=complexity.level.value,
                score=complexity.score,
                domains=complexity.domains_involved,
                suggested_agents=complexity.estimated_agents,
            )

            # Step 2: Route based on complexity
            if (
                complexity.level in (ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE)
                and not force_parallel
            ):
                # Single agent execution
                result = await self._execute_single_agent(query, context, complexity)
            else:
                # Parallel multi-agent execution
                result = await self._execute_parallel(query, context, complexity)

            # Calculate total execution time
            execution_time = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time

            self._logger.info(
                "Orchestration complete",
                success=result.success,
                execution_time_ms=execution_time,
                agents_used=result.agents_used,
                total_tool_calls=result.total_tool_calls,
            )

            return result

        finally:
            # Clean up active task tracking
            self._active_tasks.discard(task_id)

    async def _assess_complexity(
        self,
        query: str,
        context: dict[str, Any],
    ) -> ComplexityAssessment:
        """
        Assess the complexity of a query.

        Uses heuristics and optionally LLM for assessment.
        """
        # Quick heuristic assessment
        domains = self._detect_domains(query)
        keywords = self._count_action_keywords(query)

        # Calculate base score
        score = 0.0

        # Multiple domains = higher complexity
        score += min(len(domains) * 0.15, 0.45)

        # Action keywords indicate complexity
        score += min(keywords * 0.1, 0.3)

        # Query length as a proxy
        if len(query) > 200:
            score += 0.15
        elif len(query) > 100:
            score += 0.1

        # Context entities add complexity
        if context:
            score += min(len(context) * 0.05, 0.15)

        # Determine level
        if score < self.SIMPLE_THRESHOLD:
            level = ComplexityLevel.SIMPLE
        elif score < self.MODERATE_THRESHOLD:
            level = ComplexityLevel.MODERATE
        elif score < self.COMPLEX_THRESHOLD:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.VERY_COMPLEX

        # Calculate estimates
        estimated_agents = max(1, min(len(domains), self.max_concurrent))
        estimated_tool_calls = int(score * 10) + 3

        return ComplexityAssessment(
            level=level,
            score=min(score, 1.0),
            domains_involved=domains,
            estimated_agents=estimated_agents,
            estimated_tool_calls=estimated_tool_calls,
            decomposable=level in (ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX),
            suggested_parallelism=estimated_agents,
            reasoning=f"Detected {len(domains)} domains with {keywords} action keywords",
        )

    def _detect_domains(self, query: str) -> list[str]:
        """Detect OCI domains mentioned in query."""
        query_lower = query.lower()
        domain_keywords = {
            "database": ["database", "db", "awr", "sql", "autonomous", "atp", "adw"],
            "compute": ["instance", "vm", "compute", "server", "shape"],
            "network": ["vcn", "subnet", "network", "firewall", "security list", "nsg"],
            "security": ["security", "iam", "policy", "cloud guard", "threat", "audit"],
            "cost": ["cost", "budget", "spending", "billing", "finops", "pricing"],
            "storage": ["storage", "bucket", "object", "block volume", "file"],
            "observability": ["log", "metric", "alarm", "monitoring", "apm", "trace"],
        }

        detected = []
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(domain)

        return detected or ["general"]

    def _count_action_keywords(self, query: str) -> int:
        """Count action-related keywords in query."""
        query_lower = query.lower()
        action_words = [
            "analyze", "check", "compare", "troubleshoot", "investigate",
            "find", "list", "show", "get", "describe", "audit", "optimize",
            "diagnose", "fix", "monitor", "review", "assess",
        ]
        return sum(1 for word in action_words if word in query_lower)

    async def _execute_single_agent(
        self,
        query: str,
        context: dict[str, Any],
        complexity: ComplexityAssessment,
    ) -> OrchestratorResult:
        """Execute with a single agent."""
        # Find best agent for the primary domain
        primary_domain = complexity.domains_involved[0]
        agent_def = self._find_agent_for_domain(primary_domain)

        if not agent_def:
            return OrchestratorResult(
                success=False,
                response=f"No agent available for domain: {primary_domain}",
                agent_results=[],
                execution_time_ms=0,
                total_tool_calls=0,
                agents_used=[],
                error=f"No agent for {primary_domain}",
            )

        # Create and send message
        message = AgentMessage(
            sender="coordinator",
            recipient=agent_def.agent_id,
            intent=query,
            payload=context,
            boundaries=[f"Focus on {primary_domain} domain"],
            context=context,
            timeout_seconds=self.default_timeout,
        )

        result = await self.message_bus.send(message)

        if result and result.success:
            return OrchestratorResult(
                success=True,
                response=str(result.result),
                agent_results=[result],
                execution_time_ms=result.execution_time_ms or 0,
                total_tool_calls=len(result.tool_calls),
                agents_used=[agent_def.agent_id],
            )
        else:
            error = result.error if result else "No result returned"
            return OrchestratorResult(
                success=False,
                response=f"Agent execution failed: {error}",
                agent_results=[result] if result else [],
                execution_time_ms=result.execution_time_ms if result else 0,
                total_tool_calls=0,
                agents_used=[agent_def.agent_id],
                error=error,
            )

    async def _execute_parallel(
        self,
        query: str,
        context: dict[str, Any],
        complexity: ComplexityAssessment,
    ) -> OrchestratorResult:
        """Execute with multiple agents in parallel."""
        # Step 1: Decompose into subtasks
        subtasks = await self._decompose_task(query, context, complexity)

        self._logger.info(
            "Task decomposed",
            subtask_count=len(subtasks),
            domains=complexity.domains_involved,
        )

        # Step 2: Create messages for each subtask
        messages = []
        for subtask in subtasks:
            message = AgentMessage(
                sender="coordinator",
                recipient=subtask.target_agent,
                task_id=subtask.task_id,
                intent=subtask.description,
                payload=context,
                boundaries=[f"Focus on: {subtask.intent}"],
                context={
                    **context,
                    "parent_query": query,
                    "subtask_id": subtask.task_id,
                },
                priority=MessagePriority.HIGH if subtask.priority > 0 else MessagePriority.NORMAL,
                timeout_seconds=self.default_timeout,
            )
            messages.append(message)

        # Step 3: Execute in parallel
        results = await self.message_bus.send_parallel(
            messages,
            max_concurrent=self.max_concurrent,
        )

        # Step 4: Synthesize results
        synthesized = await self._synthesize_results(query, results, context)

        # Calculate totals
        total_tool_calls = sum(len(r.tool_calls) for r in results)
        agents_used = list(set(r.sender for r in results))

        return OrchestratorResult(
            success=all(r.success for r in results),
            response=synthesized,
            agent_results=results,
            execution_time_ms=0,  # Will be set by caller
            total_tool_calls=total_tool_calls,
            agents_used=agents_used,
            error=None if all(r.success for r in results) else "Some agents failed",
        )

    async def _decompose_task(
        self,
        query: str,
        context: dict[str, Any],
        complexity: ComplexityAssessment,
    ) -> list[SubtaskDefinition]:
        """
        Decompose a complex task into subtasks.

        Uses domain detection to assign to appropriate agents.
        """
        subtasks = []

        for i, domain in enumerate(complexity.domains_involved):
            agent_def = self._find_agent_for_domain(domain)
            if not agent_def:
                continue

            # Create domain-specific subtask
            subtask = SubtaskDefinition(
                task_id=f"subtask-{uuid.uuid4().hex[:8]}",
                intent=f"{domain}_analysis",
                description=self._create_domain_prompt(query, domain),
                target_agent=agent_def.agent_id,
                domains=[domain],
                priority=len(complexity.domains_involved) - i,  # First domain = highest priority
            )
            subtasks.append(subtask)

        return subtasks

    def _create_domain_prompt(self, query: str, domain: str) -> str:
        """Create a domain-focused prompt from the original query."""
        domain_focus = {
            "database": "Focus on database performance, queries, and health metrics.",
            "compute": "Focus on compute instances, shapes, and resource utilization.",
            "network": "Focus on network configuration, VCNs, and connectivity.",
            "security": "Focus on security findings, IAM policies, and threats.",
            "cost": "Focus on cost analysis, spending trends, and optimization.",
            "storage": "Focus on storage usage, buckets, and data management.",
            "observability": "Focus on logs, metrics, and monitoring data.",
        }

        focus = domain_focus.get(domain, f"Focus on {domain}-related aspects.")
        return f"{query}\n\n{focus}"

    def _find_agent_for_domain(self, domain: str):
        """Find the best agent for a domain."""
        domain_to_capability = {
            "database": "database-analysis",
            "compute": "compute-management",
            "network": "network-analysis",
            "security": "threat-detection",
            "cost": "cost-analysis",
            "storage": "storage-management",
            "observability": "log-search",
            "general": "compute-management",  # Default
        }

        capability = domain_to_capability.get(domain, "compute-management")
        agents = self.agent_catalog.get_by_capability(capability)

        return agents[0] if agents else None

    async def _synthesize_results(
        self,
        query: str,
        results: list[AgentResult],
        context: dict[str, Any],
    ) -> str:
        """
        Synthesize results from multiple agents into a coherent response.

        Uses LLM to combine and summarize agent outputs.
        """
        if not results:
            return "No results from agents."

        # If only one result, return it directly
        if len(results) == 1 and results[0].success:
            return str(results[0].result)

        # Build synthesis prompt
        agent_outputs = []
        for i, result in enumerate(results):
            status = "SUCCESS" if result.success else "FAILED"
            content = result.result if result.success else result.error
            agent_outputs.append(
                f"### Agent {i+1} ({result.sender}) [{status}]\n{content}"
            )

        synthesis_prompt = f"""Synthesize the following agent outputs into a coherent response.

Original Query: {query}

Agent Outputs:
{chr(10).join(agent_outputs)}

Instructions:
1. Combine insights from all successful agents
2. Highlight key findings across domains
3. Note any failures or missing information
4. Provide actionable recommendations
5. Use clear formatting with headers and bullet points

Synthesized Response:"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a synthesis agent that combines outputs from multiple specialized agents into a coherent, actionable response."),
                HumanMessage(content=synthesis_prompt),
            ])
            return response.content

        except Exception as e:
            self._logger.error("Synthesis failed", error=str(e))
            # Fallback: concatenate results
            return "\n\n---\n\n".join(
                f"**{r.sender}**: {r.result if r.success else f'Error: {r.error}'}"
                for r in results
            )

    async def execute_with_plan(
        self,
        query: str,
        plan: TaskSpecification,
        context: dict[str, Any] | None = None,
    ) -> OrchestratorResult:
        """
        Execute a pre-defined task plan.

        Useful when the coordinator has already planned the decomposition.

        Args:
            query: Original query
            plan: Pre-defined task specification
            context: Additional context

        Returns:
            OrchestratorResult
        """
        context = context or {}

        # Convert plan to messages
        messages = plan.to_messages("coordinator", context)

        # Route messages to appropriate agents
        for message in messages:
            if message.recipient == "auto":
                # Auto-route based on intent
                domains = self._detect_domains(message.intent)
                agent_def = self._find_agent_for_domain(domains[0])
                if agent_def:
                    message.recipient = agent_def.agent_id

        # Execute
        if plan.parallel and len(messages) > 1:
            results = await self.message_bus.send_parallel(messages)
        else:
            results = []
            for msg in messages:
                result = await self.message_bus.send(msg)
                if result:
                    results.append(result)

        # Synthesize
        synthesized = await self._synthesize_results(query, results, context)

        return OrchestratorResult(
            success=all(r.success for r in results),
            response=synthesized,
            agent_results=results,
            execution_time_ms=0,
            total_tool_calls=sum(len(r.tool_calls) for r in results),
            agents_used=list(set(r.sender for r in results)),
        )
