# OCI Coordinator Enhancement Plan

## Executive Summary

This document outlines comprehensive enhancements for the OCI AI Agent Coordinator, focusing on:
1. **LangGraph Optimization** - Improved workflow routing and state management
2. **LLM Usage Optimization** - Strategic model tiering and token efficiency
3. **Advanced Skills Framework** - Deep agents with code execution capabilities
4. **DB Observability Enhancement** - SQLcl MCP server direct integration
5. **Naming Convention Standardization** - Consistent patterns across all components

---

## 1. Current State Analysis

### 1.1 Architecture Overview
```
Current: 6 Agents → 35+ Workflows → 395+ MCP Tools → 4 MCP Servers
```

| Component | Current State | Gap |
|-----------|---------------|-----|
| Agents | 6 specialized agents | Missing code execution in agents |
| Skills | 20+ workflow definitions | No dynamic tool chaining |
| MCP Tools | 395+ tools | No tiered classification |
| LLM Usage | Single model per request | No model orchestration |
| DB Agent | API-based queries only | No SQLcl direct execution |

### 1.2 Identified Issues
1. **Routing Errors**: Intent classification failures at 15%+ rate
2. **Token Waste**: Full tool catalogs loaded for simple queries
3. **Naming Inconsistency**: Mixed conventions (kebab-case, snake_case, camelCase)
4. **Static Skills**: No runtime skill composition
5. **Missing Tool Tiers**: No cache vs API vs execution classification

---

## 2. LangGraph Enhancements

### 2.1 Enhanced State Graph
```python
# New coordinator state with enhanced routing
class EnhancedCoordinatorState(TypedDict):
    # Core state
    messages: Annotated[list, add_messages]
    user_input: str
    channel_context: ChannelContext

    # Enhanced routing
    intent: IntentClassification
    confidence_score: float
    routing_decision: RoutingDecision
    fallback_chain: list[str]

    # Execution tracking
    current_workflow: Optional[str]
    current_agent: Optional[str]
    tool_calls: list[ToolCall]
    execution_trace: list[ExecutionStep]

    # Self-healing
    error_count: int
    recovery_attempts: list[RecoveryAttempt]
    success_indicators: list[str]
```

### 2.2 Intelligent Routing Layer
```python
class IntelligentRouter:
    """Enhanced routing with confidence-based fallback chains."""

    def __init__(self):
        self.workflow_registry = WorkflowRegistry()
        self.skill_registry = SkillRegistry()
        self.intent_cache = TTLCache(maxsize=1000, ttl=3600)

    async def route(self, state: EnhancedCoordinatorState) -> RoutingDecision:
        """Multi-stage routing with confidence scoring."""
        # Stage 1: Exact workflow match (fastest)
        if exact_match := self._exact_workflow_match(state.user_input):
            return RoutingDecision(
                type="workflow",
                target=exact_match,
                confidence=0.99
            )

        # Stage 2: Skill pattern match
        if skill_match := await self._skill_pattern_match(state):
            return RoutingDecision(
                type="skill",
                target=skill_match.skill_id,
                confidence=skill_match.confidence,
                fallback="agentic" if skill_match.confidence < 0.85 else None
            )

        # Stage 3: LLM classification (most flexible)
        classification = await self._llm_classify(state)
        return RoutingDecision(
            type=classification.route_type,
            target=classification.target,
            confidence=classification.confidence,
            suggested_tools=classification.tools,
            model_tier=self._select_model_tier(classification)
        )
```

### 2.3 Workflow Composition
```python
class DynamicWorkflowComposer:
    """Compose workflows from skill building blocks."""

    def compose(self, skill_ids: list[str], context: dict) -> CompiledGraph:
        """Create a LangGraph from skill composition."""
        builder = StateGraph(WorkflowState)

        # Add skill nodes
        for i, skill_id in enumerate(skill_ids):
            skill = self.skill_registry.get(skill_id)
            node_name = f"skill_{i}_{skill_id}"
            builder.add_node(node_name, skill.create_node())

            # Chain to previous
            if i == 0:
                builder.add_edge(START, node_name)
            else:
                prev_node = f"skill_{i-1}_{skill_ids[i-1]}"
                builder.add_conditional_edges(
                    prev_node,
                    self._should_continue,
                    {True: node_name, False: END}
                )

        # Final edge
        builder.add_edge(f"skill_{len(skill_ids)-1}_{skill_ids[-1]}", END)

        return builder.compile()
```

---

## 3. LLM Usage Optimization

### 3.1 Three-Tier Model Strategy
```python
class ModelTierStrategy:
    """Strategic model selection based on task complexity."""

    TIERS = {
        "tier1_critical": {
            "model": "claude-opus-4-20250514",  # or "o1-preview"
            "use_cases": [
                "security_assessment",
                "root_cause_analysis",
                "complex_debugging",
                "architecture_decisions"
            ],
            "max_tokens": 16000,
            "temperature": 0.2
        },
        "tier2_complex": {
            "model": "claude-sonnet-4-20250514",
            "use_cases": [
                "intent_classification",
                "code_analysis",
                "report_generation",
                "multi_step_reasoning"
            ],
            "max_tokens": 8000,
            "temperature": 0.3
        },
        "tier3_operational": {
            "model": "claude-3-5-haiku-latest",
            "use_cases": [
                "simple_queries",
                "data_formatting",
                "status_checks",
                "validation"
            ],
            "max_tokens": 4000,
            "temperature": 0.1
        }
    }

    def select_tier(self, context: RequestContext) -> str:
        """Select optimal model tier based on request analysis."""
        complexity_score = self._calculate_complexity(context)

        if complexity_score >= 0.8 or context.requires_reasoning:
            return "tier1_critical"
        elif complexity_score >= 0.4 or context.multi_step:
            return "tier2_complex"
        else:
            return "tier3_operational"
```

### 3.2 Token Efficiency Patterns
```python
class TokenOptimizer:
    """Optimize token usage across the system."""

    def __init__(self):
        self.tool_embeddings = self._load_tool_embeddings()
        self.prompt_templates = PromptTemplateRegistry()

    def optimize_tool_context(
        self,
        query: str,
        max_tools: int = 10
    ) -> list[ToolDefinition]:
        """Load only relevant tools based on query embedding."""
        query_embedding = self._embed(query)

        # Semantic search for relevant tools
        similarities = cosine_similarity(
            query_embedding,
            self.tool_embeddings
        )

        top_indices = np.argsort(similarities)[-max_tools:]
        return [self.all_tools[i] for i in top_indices]

    def compress_history(
        self,
        messages: list[BaseMessage],
        max_tokens: int = 4000
    ) -> list[BaseMessage]:
        """Compress conversation history while preserving key context."""
        # Keep system message and last N messages
        system_msg = messages[0] if messages[0].type == "system" else None
        recent = messages[-5:]  # Last 5 messages

        # Summarize middle section if needed
        middle = messages[1:-5] if len(messages) > 6 else []
        if middle and self._count_tokens(middle) > max_tokens:
            summary = self._summarize_messages(middle)
            middle = [AIMessage(content=f"[Previous context summary: {summary}]")]

        return [m for m in [system_msg] + middle + recent if m]
```

### 3.3 Caching Strategy
```python
class LLMCacheStrategy:
    """Multi-layer caching for LLM responses."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.local_cache = TTLCache(maxsize=500, ttl=300)  # 5 min local

    async def get_or_generate(
        self,
        prompt_hash: str,
        generator: Callable,
        ttl: int = 3600
    ) -> str:
        """Check caches before generating."""
        # L1: Local cache
        if result := self.local_cache.get(prompt_hash):
            return result

        # L2: Redis cache
        if result := await self.redis.get(f"llm:{prompt_hash}"):
            self.local_cache[prompt_hash] = result
            return result

        # L3: Generate
        result = await generator()
        await self.redis.setex(f"llm:{prompt_hash}", ttl, result)
        self.local_cache[prompt_hash] = result
        return result
```

---

## 4. Advanced Skills Framework

### 4.1 Deep Agent Skills Architecture
```python
class DeepAgentSkill:
    """
    Skills with code execution capabilities.
    Based on multi-agent skills patterns from LangChain.
    """

    skill_id: str
    name: str
    description: str
    required_tools: list[str]
    code_templates: dict[str, str]  # Reusable code snippets

    # Progressive disclosure - only load when activated
    _knowledge_loaded: bool = False
    _knowledge_base: Optional[KnowledgeBase] = None

    async def activate(self) -> None:
        """Load skill knowledge on demand."""
        if not self._knowledge_loaded:
            self._knowledge_base = await self._load_knowledge()
            self._knowledge_loaded = True

    async def execute(
        self,
        context: SkillContext,
        code_executor: CodeExecutor
    ) -> SkillResult:
        """Execute skill with code execution support."""
        await self.activate()

        # Build execution plan
        plan = await self._build_plan(context)

        results = []
        for step in plan.steps:
            if step.type == "mcp_tool":
                result = await self._call_mcp_tool(step.tool_name, step.params)
            elif step.type == "code_execution":
                result = await code_executor.execute(
                    code=step.code,
                    context=context.to_execution_context()
                )
            elif step.type == "llm_analysis":
                result = await self._llm_analyze(step.prompt, results)

            results.append(StepResult(step=step, result=result))

            # Check for early termination
            if step.critical and not result.success:
                break

        return SkillResult(steps=results, success=all(r.success for r in results))
```

### 4.2 Agent Skill Definitions

#### Database Troubleshoot Agent Skills
```python
DB_TROUBLESHOOT_SKILLS = [
    DeepAgentSkill(
        skill_id="db_performance_rca",
        name="Database Performance Root Cause Analysis",
        description="Deep analysis of database performance issues with actionable recommendations",
        required_tools=[
            "oci_dbmgmt_get_awr_report",
            "oci_dbmgmt_get_top_sql",
            "oci_opsi_analyze_cpu",
            "oci_database_execute_sql"
        ],
        code_templates={
            "wait_event_analysis": '''
                # Analyze wait events from AWR data
                wait_events = awr_data.get("wait_events", [])
                critical_waits = [w for w in wait_events if w["pct_time"] > 10]
                return {
                    "critical_waits": critical_waits,
                    "recommendation": generate_wait_recommendation(critical_waits)
                }
            ''',
            "sql_tuning_advice": '''
                # Generate SQL tuning recommendations
                top_sql = sql_stats[:10]
                return [
                    {
                        "sql_id": sql["sql_id"],
                        "issue": detect_sql_issue(sql),
                        "recommendation": generate_sql_fix(sql)
                    }
                    for sql in top_sql if sql["elapsed_time_sec"] > threshold
                ]
            '''
        },
        steps=[
            SkillStep(
                name="gather_metrics",
                type="mcp_tool",
                tool_name="oci_opsi_get_performance_summary",
                description="Collect current performance metrics"
            ),
            SkillStep(
                name="get_awr",
                type="mcp_tool",
                tool_name="oci_dbmgmt_get_awr_report_auto",
                description="Generate AWR report for analysis"
            ),
            SkillStep(
                name="analyze_waits",
                type="code_execution",
                code_template="wait_event_analysis",
                description="Analyze wait event patterns"
            ),
            SkillStep(
                name="get_top_sql",
                type="mcp_tool",
                tool_name="oci_dbmgmt_get_top_sql",
                description="Identify resource-intensive SQL"
            ),
            SkillStep(
                name="sql_analysis",
                type="code_execution",
                code_template="sql_tuning_advice",
                description="Generate SQL tuning recommendations"
            ),
            SkillStep(
                name="synthesize",
                type="llm_analysis",
                prompt="Synthesize findings into actionable RCA report",
                model_tier="tier1_critical"
            )
        ]
    ),

    DeepAgentSkill(
        skill_id="db_blocking_resolution",
        name="Database Blocking Session Resolution",
        description="Identify and resolve database blocking chains",
        required_tools=[
            "oci_database_execute_sql",
            "oci_dbmgmt_list_attention_logs"
        ],
        code_templates={
            "blocking_tree": '''
                # Build blocking tree from V$SESSION data
                def build_tree(sessions, root_sid=None):
                    tree = {}
                    for s in sessions:
                        if s["blocking_session"] == root_sid:
                            tree[s["sid"]] = {
                                "info": s,
                                "blockers": build_tree(sessions, s["sid"])
                            }
                    return tree
                return build_tree(session_data)
            '''
        }
    ),

    DeepAgentSkill(
        skill_id="db_capacity_planning",
        name="Database Capacity Planning",
        description="Analyze trends and forecast capacity needs",
        required_tools=[
            "oci_opsi_get_resource_utilization",
            "oci_cost_database_drilldown"
        ]
    )
]
```

#### Log Analytics Agent Skills
```python
LOG_ANALYTICS_SKILLS = [
    DeepAgentSkill(
        skill_id="logan_security_investigation",
        name="Security Event Investigation",
        description="Deep investigation of security events with MITRE ATT&CK mapping",
        required_tools=[
            "oci_logan_execute_query",
            "oci_logan_get_mitre_techniques",
            "oci_logan_analyze_ip_activity"
        ],
        code_templates={
            "timeline_builder": '''
                # Build event timeline from query results
                events = sorted(results, key=lambda x: x["timestamp"])
                return {
                    "timeline": events,
                    "clusters": cluster_by_time(events, window="5m"),
                    "patterns": detect_patterns(events)
                }
            ''',
            "threat_scoring": '''
                # Score threat level based on MITRE mapping
                mitre_hits = [e for e in events if e.get("mitre_technique")]
                return {
                    "score": calculate_threat_score(mitre_hits),
                    "techniques": list(set(e["mitre_technique"] for e in mitre_hits)),
                    "kill_chain_stage": determine_kill_chain_stage(mitre_hits)
                }
            '''
        }
    ),

    DeepAgentSkill(
        skill_id="logan_anomaly_detection",
        name="Log Anomaly Detection",
        description="Detect anomalies in log patterns using statistical analysis",
        required_tools=[
            "oci_logan_execute_advanced_analytics",
            "oci_logan_execute_statistical_analysis"
        ]
    )
]
```

#### Security Agent Skills
```python
SECURITY_SKILLS = [
    DeepAgentSkill(
        skill_id="security_posture_assessment",
        name="Comprehensive Security Posture Assessment",
        description="Full security assessment with remediation recommendations",
        required_tools=[
            "oci_security_cloudguard_list_problems",
            "oci_security_vss_list_host_scans",
            "oci_security_datasafe_list_assessments",
            "oci_security_audit_list_events"
        ],
        code_templates={
            "risk_aggregation": '''
                # Aggregate risk scores across sources
                cg_risk = sum(p["risk_score"] for p in cloudguard_problems)
                vss_risk = sum(v["cvss_score"] for v in vulnerabilities)

                return {
                    "overall_score": normalize(cg_risk + vss_risk),
                    "breakdown": {
                        "cloud_guard": cg_risk,
                        "vulnerabilities": vss_risk
                    },
                    "grade": score_to_grade(cg_risk + vss_risk)
                }
            '''
        }
    )
]
```

#### FinOps Agent Skills
```python
FINOPS_SKILLS = [
    DeepAgentSkill(
        skill_id="finops_cost_optimization",
        name="Cost Optimization Analysis",
        description="Identify cost savings opportunities across OCI services",
        required_tools=[
            "oci_cost_service_drilldown",
            "oci_cost_detect_anomalies",
            "finops_rightsizing",
            "finops_commitment_alerts"
        ],
        code_templates={
            "savings_calculator": '''
                # Calculate potential savings
                compute_savings = calculate_rightsizing_savings(instances)
                commitment_savings = calculate_commitment_opportunities(usage)
                waste_reduction = identify_waste(resources)

                return {
                    "total_monthly_savings": sum([
                        compute_savings["monthly"],
                        commitment_savings["monthly"],
                        waste_reduction["monthly"]
                    ]),
                    "recommendations": prioritize_recommendations([
                        *compute_savings["items"],
                        *commitment_savings["items"],
                        *waste_reduction["items"]
                    ])
                }
            '''
        }
    )
]
```

#### Infrastructure Agent Skills
```python
INFRASTRUCTURE_SKILLS = [
    DeepAgentSkill(
        skill_id="infra_health_assessment",
        name="Infrastructure Health Assessment",
        description="Comprehensive health check across compute, network, and storage",
        required_tools=[
            "oci_compute_list_instances",
            "oci_network_list_vcns",
            "oci_observability_list_alarms"
        ]
    ),

    DeepAgentSkill(
        skill_id="infra_troubleshoot_instance",
        name="Instance Troubleshooting",
        description="Deep troubleshooting for compute instances",
        required_tools=[
            "oci_compute_get_instance",
            "oci_observability_get_instance_metrics",
            "oci_observability_execute_log_query"
        ]
    )
]
```

### 4.3 Code Execution Engine
```python
class AgentCodeExecutor:
    """
    Secure code execution for agent skills.
    Runs in sandboxed environment with limited capabilities.
    """

    ALLOWED_IMPORTS = {
        "json", "datetime", "re", "collections",
        "statistics", "math", "itertools", "functools"
    }

    def __init__(self):
        self.sandbox = RestrictedPython()
        self.result_cache = TTLCache(maxsize=100, ttl=60)

    async def execute(
        self,
        code: str,
        context: ExecutionContext,
        timeout: float = 30.0
    ) -> ExecutionResult:
        """Execute code in sandboxed environment."""
        # Validate code safety
        if not self._validate_code(code):
            return ExecutionResult(
                success=False,
                error="Code validation failed: unsafe patterns detected"
            )

        # Prepare execution context
        safe_globals = self._build_safe_globals(context)

        try:
            # Execute with timeout
            async with asyncio.timeout(timeout):
                result = await asyncio.to_thread(
                    self.sandbox.exec,
                    code,
                    safe_globals
                )

            return ExecutionResult(success=True, result=result)

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error=f"Execution timeout after {timeout}s"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )

    def _validate_code(self, code: str) -> bool:
        """Check code for unsafe patterns."""
        unsafe_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"import\s+sys",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\("
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, code):
                return False
        return True
```

---

## 5. DB Observability Enhancement

### 5.1 SQLcl Direct Integration
```python
class SQLclDirectExecutor:
    """
    Direct SQLcl MCP server integration for DB agent.
    Enables real-time SQL execution and analysis.
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp = mcp_client
        self.query_templates = SQLQueryTemplates()
        self.result_formatter = SQLResultFormatter()

    async def execute_diagnostic_query(
        self,
        query_type: DiagnosticQueryType,
        database_id: str,
        parameters: dict
    ) -> DiagnosticResult:
        """Execute pre-defined diagnostic queries via SQLcl."""

        # Get query template
        template = self.query_templates.get(query_type)
        sql = template.render(**parameters)

        # Execute via MCP
        result = await self.mcp.call_tool(
            "oci_database_execute_sql",
            {
                "sql": sql,
                "connection": database_id
            }
        )

        # Format and enrich result
        formatted = self.result_formatter.format(result, query_type)
        enriched = await self._enrich_with_recommendations(formatted)

        return DiagnosticResult(
            query_type=query_type,
            raw_result=result,
            formatted=formatted,
            recommendations=enriched.recommendations
        )


class SQLQueryTemplates:
    """Pre-built SQL templates for common diagnostics."""

    TEMPLATES = {
        DiagnosticQueryType.BLOCKING_SESSIONS: '''
            SELECT
                s.sid,
                s.serial#,
                s.username,
                s.status,
                s.blocking_session,
                s.event,
                s.wait_class,
                s.seconds_in_wait,
                s.sql_id,
                (SELECT sql_text FROM v$sql WHERE sql_id = s.sql_id AND ROWNUM = 1) as sql_text
            FROM v$session s
            WHERE s.blocking_session IS NOT NULL
               OR s.sid IN (SELECT blocking_session FROM v$session WHERE blocking_session IS NOT NULL)
            ORDER BY s.blocking_session NULLS FIRST
        ''',

        DiagnosticQueryType.WAIT_EVENTS: '''
            SELECT
                wait_class,
                event,
                total_waits,
                total_timeouts,
                time_waited,
                average_wait,
                time_waited_micro
            FROM v$system_event
            WHERE wait_class != 'Idle'
            ORDER BY time_waited DESC
            FETCH FIRST {{ top_n | default(20) }} ROWS ONLY
        ''',

        DiagnosticQueryType.SQL_MONITOR: '''
            SELECT
                sql_id,
                sql_exec_id,
                status,
                username,
                module,
                elapsed_time/1000000 as elapsed_secs,
                cpu_time/1000000 as cpu_secs,
                buffer_gets,
                disk_reads,
                sql_text
            FROM v$sql_monitor
            WHERE status = 'EXECUTING'
               OR (status = 'DONE' AND elapsed_time/1000000 > {{ min_elapsed | default(60) }})
            ORDER BY elapsed_time DESC
            FETCH FIRST {{ top_n | default(10) }} ROWS ONLY
        ''',

        DiagnosticQueryType.LONG_RUNNING_OPS: '''
            SELECT
                sid,
                serial#,
                opname,
                target,
                sofar,
                totalwork,
                ROUND(sofar/NULLIF(totalwork,0)*100, 2) as pct_complete,
                time_remaining,
                elapsed_seconds,
                message
            FROM v$session_longops
            WHERE sofar < totalwork
               OR time_remaining > 0
            ORDER BY start_time DESC
        ''',

        DiagnosticQueryType.PARALLEL_QUERIES: '''
            SELECT
                qcsid,
                qcserial#,
                qcinst_id,
                server_group,
                server_set,
                degree,
                req_degree,
                actual_degree,
                SUM(elapsed_time)/1000000 as total_elapsed_secs
            FROM v$px_session ps, v$session s
            WHERE ps.sid = s.sid
            GROUP BY qcsid, qcserial#, qcinst_id, server_group, server_set,
                     degree, req_degree, actual_degree
        ''',

        DiagnosticQueryType.FULL_TABLE_SCANS: '''
            SELECT
                object_owner,
                object_name,
                full_scans,
                index_scans,
                ROUND(full_scans/(NULLIF(full_scans + index_scans, 0))*100, 2) as fts_pct
            FROM (
                SELECT
                    owner as object_owner,
                    object_name,
                    NVL(SUM(CASE WHEN operation = 'TABLE ACCESS' AND options = 'FULL'
                            THEN executions END), 0) as full_scans,
                    NVL(SUM(CASE WHEN operation LIKE 'INDEX%'
                            THEN executions END), 0) as index_scans
                FROM v$sql_plan_statistics_all sp
                JOIN dba_objects o ON sp.object# = o.object_id
                WHERE o.object_type = 'TABLE'
                GROUP BY owner, object_name
            )
            WHERE full_scans > {{ min_scans | default(100) }}
            ORDER BY full_scans DESC
            FETCH FIRST {{ top_n | default(20) }} ROWS ONLY
        '''
    }
```

### 5.2 Enhanced DB Troubleshoot Agent
```python
class EnhancedDbTroubleshootAgent(DbTroubleshootAgent):
    """
    Enhanced DB Troubleshoot Agent with SQLcl direct execution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sqlcl_executor = SQLclDirectExecutor(self.mcp_client)
        self.code_executor = AgentCodeExecutor()

    async def execute_deep_skill(
        self,
        skill: DeepAgentSkill,
        context: SkillContext
    ) -> SkillResult:
        """Execute a deep skill with code execution support."""

        await skill.activate()
        results = []

        for step in skill.steps:
            step_result = await self._execute_step(step, context, results)
            results.append(step_result)

            # Update context with step results
            context.step_results[step.name] = step_result

            if step.critical and not step_result.success:
                self.logger.warning(
                    f"Critical step {step.name} failed, aborting skill",
                    skill_id=skill.skill_id
                )
                break

        return SkillResult(
            skill_id=skill.skill_id,
            steps=results,
            success=all(r.success for r in results if r.step.critical)
        )

    async def _execute_step(
        self,
        step: SkillStep,
        context: SkillContext,
        previous_results: list[StepResult]
    ) -> StepResult:
        """Execute individual skill step."""

        if step.type == "mcp_tool":
            # Direct MCP tool call
            params = self._resolve_params(step.params, context, previous_results)
            result = await self._call_mcp_tool(step.tool_name, params)
            return StepResult(step=step, result=result, success=result.get("success", True))

        elif step.type == "sqlcl_query":
            # SQLcl direct execution
            result = await self.sqlcl_executor.execute_diagnostic_query(
                query_type=step.query_type,
                database_id=context.database_id,
                parameters=step.params
            )
            return StepResult(step=step, result=result.formatted, success=result.success)

        elif step.type == "code_execution":
            # Sandboxed code execution
            code = skill.code_templates.get(step.code_template)
            exec_context = ExecutionContext(
                variables={
                    **context.to_dict(),
                    "previous_results": [r.result for r in previous_results]
                }
            )
            result = await self.code_executor.execute(code, exec_context)
            return StepResult(step=step, result=result.result, success=result.success)

        elif step.type == "llm_analysis":
            # LLM analysis with appropriate tier
            tier = step.model_tier or "tier2_complex"
            result = await self._llm_analyze(
                prompt=step.prompt,
                context=context,
                previous_results=previous_results,
                model_tier=tier
            )
            return StepResult(step=step, result=result, success=True)
```

### 5.3 DB Workflow Enhancement
```python
# Enhanced DB workflows with SQLcl integration
ENHANCED_DB_WORKFLOWS = {
    "db_blocking_sessions_workflow": {
        "steps": [
            {
                "type": "sqlcl_query",
                "query_type": DiagnosticQueryType.BLOCKING_SESSIONS,
                "description": "Get blocking session hierarchy"
            },
            {
                "type": "code_execution",
                "code_template": "blocking_tree",
                "description": "Build blocking tree visualization"
            },
            {
                "type": "llm_analysis",
                "prompt": "Analyze blocking chain and recommend resolution",
                "model_tier": "tier2_complex"
            }
        ]
    },

    "db_wait_events_workflow": {
        "steps": [
            {
                "type": "mcp_tool",
                "tool_name": "oci_dbmgmt_summarize_awr_wait_events",
                "params": {"top_n": 20}
            },
            {
                "type": "sqlcl_query",
                "query_type": DiagnosticQueryType.WAIT_EVENTS,
                "description": "Get real-time wait events"
            },
            {
                "type": "code_execution",
                "code_template": "wait_event_analysis",
                "description": "Correlate AWR and real-time data"
            }
        ]
    },

    "db_sql_monitoring_workflow": {
        "steps": [
            {
                "type": "sqlcl_query",
                "query_type": DiagnosticQueryType.SQL_MONITOR,
                "params": {"min_elapsed": 30, "top_n": 15}
            },
            {
                "type": "mcp_tool",
                "tool_name": "oci_dbmgmt_get_top_sql",
                "params": {"sort_by": "elapsed_time", "limit": 10}
            },
            {
                "type": "code_execution",
                "code_template": "sql_tuning_advice",
                "description": "Generate tuning recommendations"
            }
        ]
    }
}
```

---

## 6. Naming Convention Standardization

### 6.1 Unified Naming Standards
```yaml
# Naming Convention Reference

agents:
  class_name: PascalCase  # DbTroubleshootAgent
  agent_id: kebab-case    # db-troubleshoot
  capability: kebab-case  # performance-analysis

skills:
  skill_id: snake_case    # db_performance_rca
  step_name: snake_case   # gather_metrics

workflows:
  workflow_id: snake_case # db_health_check_workflow
  intent: snake_case      # check_db_health

mcp_tools:
  tool_name: snake_case   # oci_dbmgmt_get_awr_report
  pattern: oci_{domain}_{action}

files:
  module: snake_case      # troubleshoot.py
  class: PascalCase       # DbTroubleshootAgent

environment:
  variables: SCREAMING_SNAKE_CASE  # REDIS_URL
```

### 6.2 Validation & Migration
```python
class NamingValidator:
    """Validate and enforce naming conventions."""

    PATTERNS = {
        "agent_class": r"^[A-Z][a-zA-Z]+Agent$",
        "agent_id": r"^[a-z]+(-[a-z]+)*$",
        "skill_id": r"^[a-z]+(_[a-z]+)*$",
        "mcp_tool": r"^oci_[a-z]+_[a-z_]+$",
        "workflow_id": r"^[a-z]+(_[a-z]+)*_workflow$"
    }

    def validate(self, item_type: str, name: str) -> ValidationResult:
        """Validate name against convention."""
        pattern = self.PATTERNS.get(item_type)
        if not pattern:
            return ValidationResult(valid=True, warning="Unknown item type")

        if re.match(pattern, name):
            return ValidationResult(valid=True)

        suggested = self._suggest_fix(item_type, name)
        return ValidationResult(
            valid=False,
            error=f"Name '{name}' doesn't match pattern '{pattern}'",
            suggested=suggested
        )

    def audit_codebase(self, root_path: Path) -> AuditReport:
        """Audit entire codebase for naming violations."""
        violations = []

        # Check agent files
        for agent_file in root_path.glob("src/agents/**/*.py"):
            violations.extend(self._audit_file(agent_file))

        # Check MCP tools
        for tool_file in root_path.glob("src/mcp/server/tools/*.py"):
            violations.extend(self._audit_mcp_tools(tool_file))

        return AuditReport(violations=violations)
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Implement EnhancedCoordinatorState
- [ ] Create IntelligentRouter with confidence scoring
- [ ] Set up ModelTierStrategy
- [ ] Create TokenOptimizer
- [ ] Run naming convention audit

### Phase 2: Skills Framework (Week 3-4)
- [ ] Implement DeepAgentSkill base class
- [ ] Create AgentCodeExecutor with sandbox
- [ ] Define skills for all 6 agents
- [ ] Implement skill activation/progressive disclosure

### Phase 3: DB Enhancement (Week 5-6)
- [ ] Implement SQLclDirectExecutor
- [ ] Create SQL query templates
- [ ] Enhance DbTroubleshootAgent with SQLcl
- [ ] Add DB-specific code execution templates

### Phase 4: LLM Optimization (Week 7-8)
- [ ] Implement LLMCacheStrategy
- [ ] Add semantic tool search
- [ ] Create conversation compression
- [ ] Tune model tier thresholds

### Phase 5: Testing & Refinement (Week 9-10)
- [ ] End-to-end testing of all skills
- [ ] Performance benchmarking
- [ ] Error recovery testing
- [ ] Documentation updates

---

## 8. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Routing Accuracy | 85% | 95%+ |
| Token Efficiency | Baseline | -40% |
| Response Latency (P95) | ~15s | <8s |
| Skill Success Rate | N/A | 90%+ |
| DB Query Success | 80% | 95%+ |

---

## Appendix A: Skill Registry

```python
# Complete skill registry for all agents
SKILL_REGISTRY = {
    "db-troubleshoot": DB_TROUBLESHOOT_SKILLS,
    "log-analytics": LOG_ANALYTICS_SKILLS,
    "security": SECURITY_SKILLS,
    "finops": FINOPS_SKILLS,
    "infrastructure": INFRASTRUCTURE_SKILLS,
    "error-analysis": ERROR_ANALYSIS_SKILLS
}
```

## Appendix B: MCP Tool Tiers

| Tier | Latency | Tools | Use Case |
|------|---------|-------|----------|
| 1 - Cache | <100ms | `*_get_cached_*`, `*_search_*` | Quick lookups |
| 2 - API | 1-5s | `*_list_*`, `*_get_*` | Data retrieval |
| 3 - Compute | 5-30s | `*_analyze_*`, `*_execute_*` | Heavy analysis |
| 4 - Report | 30s+ | `*_get_awr_report*` | Full reports |

## Appendix C: Error Recovery Strategies

```python
ERROR_RECOVERY = {
    "mcp_timeout": {
        "strategy": "retry_with_backoff",
        "max_attempts": 3,
        "backoff_multiplier": 2
    },
    "llm_rate_limit": {
        "strategy": "fallback_model",
        "fallback_order": ["tier2", "tier3"]
    },
    "sql_syntax_error": {
        "strategy": "self_heal",
        "analyzer": "sql_syntax_analyzer"
    },
    "permission_denied": {
        "strategy": "escalate_to_user",
        "message_template": "Need elevated permissions for {action}"
    }
}
```
