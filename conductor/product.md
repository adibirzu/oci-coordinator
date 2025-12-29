# OCI AI Agent Coordinator System - Product Definition

## Executive Summary

The OCI AI Agent Coordinator is an intelligent orchestration system that manages specialized AI agents for Oracle Cloud Infrastructure operations. It acts as a central hub that receives requests from multiple channels (Slack, Teams, Web, API) and routes them to appropriate specialized agents for execution.

---

## 1. Product Vision

### 1.1 Problem Statement
- Operations teams face fragmented tooling across OCI services
- Troubleshooting requires context-switching between multiple consoles
- No unified interface for cross-domain correlation (logs, metrics, costs, security)
- Manual processes for routine operational tasks

### 1.2 Solution
An AI-powered coordinator that:
- Understands natural language requests
- Routes to specialized agents based on intent classification
- Orchestrates multi-step workflows across agents
- Maintains context and conversation state
- Provides unified access through familiar channels

### 1.3 Value Proposition
- **Reduced MTTR**: Faster incident resolution through automated correlation
- **Unified Experience**: Single interface for all OCI operations
- **Proactive Insights**: AI-driven anomaly detection and recommendations
- **Extensibility**: Modular architecture for adding new capabilities

---

## 2. Product Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT CHANNELS                                     │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│  Slack   │  Teams   │   Web    │   API    │  Email   │   OCI Events        │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────────┬──────────┘
     │          │          │          │          │                │
     └──────────┴──────────┴────┬─────┴──────────┴────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │                       │
                    │  LANGGRAPH CONDUCTOR  │
                    │  (Tool Catalog-aware) │
                    │                       │
                    └───────────┬───────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┬───────────┐
        │           │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐
│ Database  │ │ OCI GenAI │ │ OCI GenAI │ │ OCI GenAI │ │ OCI GenAI │ │ Custom  │
│Observatory│ │ Log Agent │ │ Sec Agent │ │ Fin Agent │ │ Infra Agt │ │ LangChn │
│  Agent    │ │ (Managed) │ │ (Managed) │ │ (Managed) │ │ (Managed) │ │  Tool   │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────┬────┘
      │             │             │             │             │            │
      └─────────────┴─────────────┴──────┬──────┴─────────────┴────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │      UNIFIED MCP SERVER LAYER           │
                    │  (Progressive Disclosure & Skills)      │
                    │  • Compute   • Network    • DB          │
                    │  • Cost      • Security   • Observability│
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │      ORACLE CLOUD INFRASTRUCTURE        │
                    │  ─────────────────────────────────────  │
                    │  Databases | Compute | Network | IAM    │
                    └─────────────────────────────────────────┘
```

### 2.2 Component Description

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Input Channels | Receive and normalize user requests | Slack Bot, Teams Bot, Web API |
| LangGraph Conductor | Orchestration, state persistence, and tool routing | LangGraph + ToolCatalog |
| Database Observatory | Specialized multi-agent system for DB observability | OCI Database Observatory Agent |
| OCI Managed Agents | Domain-specific task execution using OCI GenAI | OCI GenAI Agents Service |
| Unified MCP Server | OCI API abstraction with progressive disclosure | FastMCP + Skills Architecture |
| Context Store | Checkpoints and thread history | Redis |

---

## 3. Agent Definitions

### 3.1 Coordinator (LangGraph)

**Purpose**: Central state machine that manages the conversation lifecycle using a `ToolCatalog`.



**Capabilities**:

- **State Management**: Persists conversation state (`CoordinatorState`) using `MemorySaver` / Redis.

- **Dynamic Tool Routing**: Uses `ToolCatalog` to discover and bind tools/skills.

- **Skill Execution**: Can invoke high-level skills (e.g., "Troubleshoot Instance") via MCP.

- **Graph Flow**: `input -> agent -> (action -> agent)* -> output` loop.



### 3.2 Specialized OCI Agents (Managed)



#### Database Observatory Agent

- **Type**: Dedicated Sub-System (Reference: `oci-database-observatory`)

- **Capabilities**:

    - **Multi-DB Support**: Oracle, MySQL, PostgreSQL.

    - **Observability**: OCI APM integration, OpenTelemetry tracing.

    - **Advanced Diagnostics**: SQL tuning, AWR analysis, wait event interpretation.

- **Integration**: Accessed via MCP or direct API call.

#### Log Analytics Agent
- **Type**: OCI GenAI Agent
- **Tools**: `oci-mcp-observability` tools.
- **Knowledge**: Log patterns and error codes.

#### Security & Threat Agent
- **Type**: OCI GenAI Agent
- **Tools**: `oci-mcp-security` tools.
- **Knowledge**: MITRE ATT&CK, OCI Security Guidelines.

#### FinOps Agent
- **Type**: OCI GenAI Agent
- **Tools**: `oci-mcp-cost` tools.
- **Knowledge**: OCI Pricing, Cost optimization strategies.

#### Infrastructure Agent
- **Type**: OCI GenAI Agent
- **Tools**: `oci-mcp-compute`, `oci-mcp-network` tools.
- **Knowledge**: OCI Architecture Framework.

---

## 4. Functional Requirements

### 4.1 Core Features (MVP)

| ID | Feature | Priority | Agent |
|----|---------|----------|-------|
| F01 | Natural language query processing | P0 | Coordinator |
| F02 | Intent classification with confidence | P0 | Coordinator |
| F03 | Multi-channel input support | P0 | Coordinator |
| F04 | Database bottleneck analysis | P0 | DB Agent |
| F05 | Log search and error analysis | P0 | Log Agent |
| F06 | Cost summary and trends | P0 | FinOps Agent |
| F07 | Response formatting per channel | P0 | Coordinator |

### 4.2 Extended Features (Phase 2)

| ID | Feature | Priority | Agent |
|----|---------|----------|-------|
| F08 | Multi-agent workflow orchestration | P1 | Coordinator |
| F09 | Security threat hunting | P1 | Security Agent |
| F10 | Cross-domain correlation | P1 | Multiple |
| F11 | Proactive alerting | P1 | All |
| F12 | Runbook automation | P1 | All |
| F13 | Knowledge base integration | P1 | Coordinator |

### 4.3 Advanced Features (Phase 3)

| ID | Feature | Priority | Agent |
|----|---------|----------|-------|
| F14 | Self-healing actions | P2 | All |
| F15 | Predictive analytics | P2 | All |
| F16 | Custom agent creation | P2 | Coordinator |
| F17 | Multi-tenancy support | P2 | All |

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Response latency: < 5 seconds for simple queries
- Throughput: 100 concurrent requests
- Agent handoff: < 500ms

### 5.2 Security
- OCI IAM integration
- Least privilege access per agent
- Audit logging for all operations
- Secrets management via OCI Vault

### 5.3 Reliability
- 99.9% availability target
- Graceful degradation on agent failure
- Request retry with exponential backoff
- Circuit breaker pattern for MCP calls

### 5.4 Scalability
- Horizontal scaling of agents
- Queue-based request processing
- Stateless agent design

---

## 6. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Intent Classification Accuracy | > 95% | Manual review sampling |
| Mean Time to Resolution | -40% | Ticket closure time |
| User Satisfaction | > 4.5/5 | Feedback surveys |
| Agent Response Time | < 5s | P95 latency |
| Automation Rate | > 70% | Resolved without escalation |

---

## 7. Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Coordinator agent with basic routing
- DB Troubleshoot agent
- Log Analytics agent
- Slack integration

### Phase 2: Expansion (Weeks 5-8)
- FinOps agent
- Security agent
- Teams integration
- Multi-agent workflows

### Phase 3: Enhancement (Weeks 9-12)
- Web UI
- Knowledge base integration
- Proactive alerting
- Custom agent framework

### Phase 4: Optimization (Weeks 13-16)
- Performance tuning
- ML-based intent improvement
- Self-healing capabilities
- Production hardening

---

## 8. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LLM hallucination | High | Medium | Tool-first approach, validation layer |
| API rate limiting | Medium | Medium | Caching, request batching |
| Security breach | High | Low | Least privilege, audit logs |
| Agent timeout | Medium | Medium | Circuit breakers, fallbacks |
| Scope creep | Medium | High | Strict phase boundaries |

---

## 9. Dependencies

### 9.1 External Dependencies
- OCI API access and permissions
- LLM provider (Claude/Gemini) API
- MCP server infrastructure
- Message platform APIs (Slack/Teams)

### 9.2 Internal Dependencies
- OCI tenancy configuration
- Network connectivity to OCI
- Authentication setup
- Runbook documentation

---

## 10. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | | | |
| Technical Lead | | | |
| Security Review | | | |
| Architecture Review | | | |
