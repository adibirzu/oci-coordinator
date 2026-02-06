# Architecture Details

## LangGraph Coordinator (`src/agents/coordinator/`)

**Graph Nodes**: input → classifier → router → workflow|parallel|agent → action → output

**Routing Thresholds:**
| Threshold | Value | Action |
|-----------|-------|--------|
| Workflow | ≥ 0.80 | Direct workflow execution |
| Parallel | ≥ 0.60 + 2+ domains | Multi-agent parallel |
| Agent | ≥ 0.60 | Single agent with tools |
| Clarify | 0.30 - 0.60 | Ask clarifying question |
| Escalate | < 0.30 | Human handoff |

## Pre-built Workflows (`src/agents/coordinator/workflows.py`)

35+ deterministic workflows, 100+ intent aliases:

| Category | Workflow | Intent Aliases |
|----------|----------|----------------|
| Identity | `list_compartments` | show_compartments, get_compartments |
| Cost | `cost_summary` | get_costs, how_much_spent, monthly_cost |
| DB Mgmt | `top_sql` | db_top_sql, high_cpu_sql |
| DB Mgmt | `awr_report` | generate_awr, ash_report |
| OPSI | `addm_findings` | opsi_addm, database_diagnostics |
| OPSI | `capacity_forecast` | opsi_forecast, growth_forecast |
| Infra | `provision_instance` | create_vm, launch_instance |
| Infra | `list_shapes` | available_shapes, list_compute_shapes |

## Parallel Orchestrator (`src/agents/coordinator/orchestrator.py`)

- Multi-agent parallel execution for cross-domain queries
- Automatic task decomposition by domain
- Bounded concurrency (3-5 agents max)

## Shared Memory (`src/memory/`)

| Layer | Backend | Purpose | TTL |
|-------|---------|---------|-----|
| Hot Cache | Redis | Session state, tool results | 1 hour |
| Checkpoints | LangGraph MemorySaver | Graph state | Session |

## Resilience Infrastructure (`src/resilience/`)

**Bulkhead Partitions:** DATABASE: 3, INFRASTRUCTURE: 5, COST: 2, SECURITY: 3, LLM: 5, DEFAULT: 10

| Component | Purpose |
|-----------|---------|
| `DeadLetterQueue` | Persist failed operations |
| `Bulkhead` | Resource isolation |
| `HealthMonitor` | Component health tracking |

## Self-Healing Framework (`src/agents/self_healing/`)

| Component | Purpose |
|-----------|---------|
| `ErrorAnalyzer` | Categorize errors, suggest recovery |
| `ParameterCorrector` | Fix incorrect tool parameters |
| `RetryStrategy` | Smart retry with exponential backoff |
| `SelfHealingMixin` | Mixin for agents to inherit self-healing |

## Observability (`src/observability/`)

- OpenTelemetry tracing to OCI APM via OTLP
- OTLP log export to OCI APM for span-level log correlation (APM "Logs" tab)
- Per-agent OCI Logging with trace_id correlation (Log Analytics persistence)
- 3 parallel log pipelines: OTLP→APM, OCILogging→LogAnalytics, Console→stdout
- GenAI semantic conventions for LLM call instrumentation (OracleCodeAssistInstrumentor)
- Token usage, latency, and error tracking per LLM call
