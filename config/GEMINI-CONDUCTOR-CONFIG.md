# Gemini Conductor Configuration

## Overview
Configuration for using Gemini Conductor to orchestrate the multi-agent OCI AI system.

---

## Gemini Conductor Product Definition

```yaml
# gemini-conductor-product.yaml

product:
  name: "OCI AI Agent Coordinator"
  version: "1.0.0"
  description: |
    An intelligent orchestration system that coordinates specialized AI agents 
    for Oracle Cloud Infrastructure operations. Receives requests from multiple 
    channels and routes them to appropriate agents for execution.

objectives:
  primary:
    - "Provide unified natural language interface for OCI operations"
    - "Route requests to specialized agents based on intent"
    - "Coordinate multi-agent workflows"
    - "Aggregate and format responses for users"
    
  success_metrics:
    - metric: "Intent Classification Accuracy"
      target: ">95%"
    - metric: "Response Time (P95)"
      target: "<5 seconds"
    - metric: "User Satisfaction"
      target: ">4.5/5"

stakeholders:
  users:
    - role: "Cloud Operations Engineers"
      needs: ["Fast troubleshooting", "Infrastructure management"]
    - role: "FinOps Analysts"
      needs: ["Cost visibility", "Optimization recommendations"]
    - role: "Security Engineers"
      needs: ["Threat detection", "Compliance monitoring"]
    - role: "DBAs"
      needs: ["Database troubleshooting", "Performance analysis"]

constraints:
  technical:
    - "Must integrate with existing OCI tenancies"
    - "Must support multiple input channels (Slack, Teams, Web)"
    - "Must use MCP servers for OCI API access"
  security:
    - "All operations must be audited"
    - "Must respect OCI IAM policies"
    - "No credential storage in agents"
  operational:
    - "Must handle agent failures gracefully"
    - "Must support horizontal scaling"
```

---

## Track Generation Configuration

```yaml
# tracks/track-definition.yaml

tracks:
  - id: "track-01-foundation"
    name: "Foundation & Core Infrastructure"
    description: "Set up project structure, MCP servers, and basic routing"
    duration_weeks: 2
    dependencies: []
    
    milestones:
      - id: "M1.1"
        name: "Project Setup"
        tasks:
          - "Initialize Python project with Poetry"
          - "Configure linting (ruff, black)"
          - "Set up testing framework (pytest)"
          - "Configure CI/CD pipeline"
          - "Create Docker development environment"
          
      - id: "M1.2"
        name: "MCP Server Integration"
        tasks:
          - "Deploy oci-mcp-cost server"
          - "Deploy oci-mcp-db server"
          - "Deploy oci-mcp-compute server"
          - "Deploy oci-mcp-network server"
          - "Deploy oci-mcp-security server"
          - "Deploy oci-mcp-observability server"
          - "Create MCP server health monitoring"
          
      - id: "M1.3"
        name: "Basic Coordinator"
        tasks:
          - "Implement coordinator prompt loading"
          - "Create intent classification logic"
          - "Implement basic routing to agents"
          - "Create response formatting utilities"

    deliverables:
      - "Working development environment"
      - "All MCP servers running and healthy"
      - "Basic coordinator routing test"

  - id: "track-02-agents"
    name: "Specialized Agent Implementation"
    description: "Implement all specialized agents with their prompts"
    duration_weeks: 4
    dependencies: ["track-01-foundation"]
    
    milestones:
      - id: "M2.1"
        name: "DB Troubleshoot Agent"
        tasks:
          - "Implement agent prompt"
          - "Create MCP tool integration"
          - "Implement diagnostic queries"
          - "Add response formatting"
          - "Create unit tests"
          
      - id: "M2.2"
        name: "Log Analytics Agent"
        tasks:
          - "Implement agent prompt"
          - "Create query builder"
          - "Implement pattern detection"
          - "Add MITRE mapping"
          - "Create unit tests"
          
      - id: "M2.3"
        name: "Security & Threat Agent"
        tasks:
          - "Implement agent prompt"
          - "Create threat hunting flows"
          - "Integrate Cloud Guard"
          - "Add threat intelligence"
          - "Create unit tests"
          
      - id: "M2.4"
        name: "FinOps Agent"
        tasks:
          - "Implement agent prompt"
          - "Create cost analysis logic"
          - "Implement optimization recommendations"
          - "Add forecasting"
          - "Create unit tests"
          
      - id: "M2.5"
        name: "Infrastructure Agent"
        tasks:
          - "Implement agent prompt"
          - "Create resource management"
          - "Implement safety controls"
          - "Add capacity planning"
          - "Create unit tests"

    deliverables:
      - "All 5 specialized agents working"
      - "Agent-to-agent communication"
      - "Comprehensive test coverage"

  - id: "track-03-channels"
    name: "Input Channel Integration"
    description: "Integrate Slack, Teams, and Web interfaces"
    duration_weeks: 2
    dependencies: ["track-02-agents"]
    
    milestones:
      - id: "M3.1"
        name: "Slack Integration"
        tasks:
          - "Create Slack bot application"
          - "Implement slash commands"
          - "Create message formatting"
          - "Add interactive components"
          - "Handle rate limiting"
          
      - id: "M3.2"
        name: "Teams Integration"
        tasks:
          - "Create Teams bot application"
          - "Implement adaptive cards"
          - "Create response formatting"
          - "Handle authentication"
          
      - id: "M3.3"
        name: "Web API"
        tasks:
          - "Create FastAPI application"
          - "Implement WebSocket support"
          - "Add authentication (JWT)"
          - "Create OpenAPI documentation"

    deliverables:
      - "Working Slack bot"
      - "Working Teams bot"
      - "Web API with documentation"

  - id: "track-04-orchestration"
    name: "Advanced Orchestration"
    description: "Multi-agent workflows and context management"
    duration_weeks: 2
    dependencies: ["track-03-channels"]
    
    milestones:
      - id: "M4.1"
        name: "Context Management"
        tasks:
          - "Implement conversation state"
          - "Create context persistence (Redis)"
          - "Handle session management"
          - "Implement context passing between agents"
          
      - id: "M4.2"
        name: "Multi-Agent Workflows"
        tasks:
          - "Implement sequential workflows"
          - "Implement parallel workflows"
          - "Create workflow definitions"
          - "Add workflow monitoring"
          
      - id: "M4.3"
        name: "Error Handling & Fallbacks"
        tasks:
          - "Implement circuit breakers"
          - "Create fallback responses"
          - "Add retry logic"
          - "Create error reporting"

    deliverables:
      - "Working multi-agent workflows"
      - "Persistent context"
      - "Resilient error handling"

  - id: "track-05-production"
    name: "Production Hardening"
    description: "Security, monitoring, and deployment"
    duration_weeks: 2
    dependencies: ["track-04-orchestration"]
    
    milestones:
      - id: "M5.1"
        name: "Security Hardening"
        tasks:
          - "Implement OCI Vault integration"
          - "Add audit logging"
          - "Create IAM policy templates"
          - "Security testing"
          
      - id: "M5.2"
        name: "Monitoring & Observability"
        tasks:
          - "Add metrics (Prometheus)"
          - "Create dashboards (Grafana)"
          - "Implement alerting"
          - "Add tracing"
          
      - id: "M5.3"
        name: "Deployment"
        tasks:
          - "Create Kubernetes manifests"
          - "Configure OKE deployment"
          - "Set up CI/CD"
          - "Create runbooks"

    deliverables:
      - "Production-ready deployment"
      - "Monitoring and alerting"
      - "Operations documentation"

  - id: "track-06-enhancement"
    name: "Enhancement & Optimization"
    description: "Performance tuning and feature additions"
    duration_weeks: 2
    dependencies: ["track-05-production"]
    
    milestones:
      - id: "M6.1"
        name: "Performance Optimization"
        tasks:
          - "Profile and optimize agents"
          - "Implement caching"
          - "Optimize MCP calls"
          - "Load testing"
          
      - id: "M6.2"
        name: "Knowledge Base"
        tasks:
          - "Create runbook database"
          - "Implement RAG pipeline"
          - "Add knowledge retrieval"
          
      - id: "M6.3"
        name: "Custom Agent Framework"
        tasks:
          - "Create agent template"
          - "Document extension points"
          - "Create agent development guide"

    deliverables:
      - "Optimized performance"
      - "Knowledge base integration"
      - "Agent development documentation"
```

---

## Claude Code Implementation Tasks

```yaml
# claude-code-tasks.yaml

implementation:
  repository:
    name: "oci-ai-agent-coordinator"
    structure:
      - "src/"
      - "src/coordinator/"
      - "src/agents/"
      - "src/agents/db_troubleshoot/"
      - "src/agents/log_analytics/"
      - "src/agents/security/"
      - "src/agents/finops/"
      - "src/agents/infrastructure/"
      - "src/channels/"
      - "src/channels/slack/"
      - "src/channels/teams/"
      - "src/channels/web/"
      - "src/mcp/"
      - "src/common/"
      - "tests/"
      - "config/"
      - "prompts/"
      - "deploy/"
      - "docs/"

  phase1_tasks:
    name: "Foundation Setup"
    tasks:
      - task: "PROJECT_INIT"
        description: "Initialize project with Poetry"
        files:
          - "pyproject.toml"
          - "README.md"
          - ".gitignore"
          - ".env.example"
        commands:
          - "poetry init --name oci-ai-coordinator --python ^3.11"
          - "poetry add anthropic fastapi uvicorn redis pydantic"
          
      - task: "MCP_CLIENT"
        description: "Create MCP client wrapper"
        files:
          - "src/mcp/client.py"
          - "src/mcp/config.py"
          - "src/mcp/health.py"
        dependencies:
          - "httpx"
          - "tenacity"

      - task: "COORDINATOR_CORE"
        description: "Implement coordinator core"
        files:
          - "src/coordinator/main.py"
          - "src/coordinator/intent.py"
          - "src/coordinator/router.py"
          - "src/coordinator/context.py"
          - "src/coordinator/formatter.py"

  phase2_tasks:
    name: "Agent Implementation"
    tasks:
      - task: "AGENT_BASE"
        description: "Create base agent class"
        files:
          - "src/agents/base.py"
          - "src/agents/registry.py"
          - "src/agents/response.py"
          
      - task: "DB_AGENT"
        description: "Implement DB Troubleshoot Agent with comprehensive RCA workflow"
        files:
          - "src/agents/database/__init__.py"
          - "src/agents/database/troubleshoot.py"
          - "src/skills/troubleshoot_database.py"
        workflows:
          - "db_blocking_sessions_workflow"      # v$session, v$lock queries
          - "db_wait_events_workflow"            # OPSI wait event analysis
          - "db_sql_monitoring_workflow"         # v$sql_monitor real-time
          - "db_long_running_ops_workflow"       # gv$session_longops
          - "db_parallelism_stats_workflow"      # req_degree vs actual
          - "db_full_table_scan_workflow"        # v$sql_plan + dba_tables
          - "db_awr_report_workflow"             # AWR snapshot analysis
        mcp_tools:
          - "oci_database_execute_sql"           # SQLcl queries (database-observatory)
          - "oci_opsi_summarize_resource_stats"  # CPU/Memory metrics (oci-unified)
          - "oci_dbmgmt_get_wait_events"         # Wait events (oci-unified)
          - "oci_dbmgmt_get_awr_report"          # AWR reports (oci-unified)
          - "oci_opsi_search_databases"          # Database discovery (database-observatory)
          
      - task: "LOG_AGENT"
        description: "Implement Log Analytics Agent for OCI Logging Analytics"
        files:
          - "src/agents/log_analytics/__init__.py"
          - "src/agents/log_analytics/agent.py"
        mcp_tools:
          - "oci_logan_execute_query"            # Logan query execution
          - "oci_logan_search_security_events"   # Security event search
          - "oci_logan_get_mitre_techniques"     # MITRE ATT&CK mapping
          - "oci_logan_analyze_ip_activity"      # IP analysis
          - "oci_logan_list_log_sources"         # Log source discovery

      - task: "SECURITY_AGENT"
        description: "Implement Security Agent for Cloud Guard, VSS, WAF"
        files:
          - "src/agents/security/__init__.py"
          - "src/agents/security/agent.py"
        mcp_tools:
          - "oci_security_cloudguard_list_problems"  # Cloud Guard problems
          - "oci_security_vss_list_host_scans"       # VSS host scans
          - "oci_security_waf_list_firewalls"        # WAF policies
          - "oci_security_bastion_list"              # Bastion sessions
          - "oci_security_datasafe_list_assessments" # Data Safe

      - task: "FINOPS_AGENT"
        description: "Implement FinOps Agent for cost analysis and optimization"
        files:
          - "src/agents/finops/__init__.py"
          - "src/agents/finops/agent.py"
        mcp_tools:
          - "oci_cost_by_compartment"            # Cost breakdown (finopsai)
          - "oci_cost_service_drilldown"         # Service costs (finopsai)
          - "oci_cost_monthly_trend"             # Trends & forecast (finopsai)
          - "finops_detect_anomalies"            # Cost anomaly detection (finopsai)
          - "finops_rightsizing"                 # Optimization recommendations (finopsai)

      - task: "INFRA_AGENT"
        description: "Implement Infrastructure Agent for compute, network, storage"
        files:
          - "src/agents/infrastructure/__init__.py"
          - "src/agents/infrastructure/agent.py"
        mcp_tools:
          - "oci_compute_list_instances"         # Compute instances (oci-infrastructure)
          - "oci_compute_launch_instance"        # Instance provisioning (oci-infrastructure)
          - "oci_blockstorage_list_volumes"      # Block storage (oci-infrastructure)
          - "oci_vcn_list_vcns"                  # VCN listing (oci-infrastructure)
          - "oci_vcn_list_subnets"               # Subnet listing (oci-infrastructure)

      - task: "ERROR_ANALYSIS_AGENT"
        description: "Implement Error Analysis Agent for self-healing and diagnostics"
        files:
          - "src/agents/error_analysis/__init__.py"
          - "src/agents/error_analysis/agent.py"
          - "src/agents/self_healing/corrector.py"
          - "src/agents/self_healing/validator.py"
        capabilities:
          - "Tool failure analysis"
          - "MCP timeout detection"
          - "Alternative tool suggestion"
          - "Self-healing retry logic"

      - task: "SELECTAI_AGENT"
        description: "Implement SelectAI Agent for natural language to SQL"
        files:
          - "src/agents/selectai/__init__.py"
          - "src/agents/selectai/agent.py"
        capabilities:
          - "Natural language to SQL via OCI GenAI"
          - "ATP/ADW database integration"
          - "Profile-based query routing"

  phase3_tasks:
    name: "Channel Integration"
    tasks:
      - task: "SLACK_BOT"
        description: "Create Slack bot with Socket Mode"
        files:
          - "src/channels/__init__.py"
          - "src/channels/slack.py"
          - "src/formatting/slack.py"
        dependencies:
          - "slack-sdk"
          - "slack-bolt"
        features:
          - "Socket Mode for real-time events"
          - "Thread-aware responses"
          - "mrkdwn formatting"
          - "Table rendering with limits"
          
      - task: "TEAMS_BOT"
        description: "Create Teams bot"
        files:
          - "src/channels/teams/__init__.py"
          - "src/channels/teams/bot.py"
          - "src/channels/teams/cards.py"
        dependencies:
          - "botbuilder-core"
          
      - task: "WEB_API"
        description: "Create Web API"
        files:
          - "src/channels/web/__init__.py"
          - "src/channels/web/api.py"
          - "src/channels/web/websocket.py"
          - "src/channels/web/auth.py"

  phase4_tasks:
    name: "Orchestration & Production"
    tasks:
      - task: "WORKFLOWS"
        description: "Multi-agent workflows"
        files:
          - "src/coordinator/workflows.py"
          - "src/coordinator/parallel.py"
          - "src/coordinator/sequential.py"
          
      - task: "DEPLOYMENT"
        description: "Kubernetes deployment"
        files:
          - "deploy/kubernetes/namespace.yaml"
          - "deploy/kubernetes/configmap.yaml"
          - "deploy/kubernetes/secrets.yaml"
          - "deploy/kubernetes/deployment.yaml"
          - "deploy/kubernetes/service.yaml"
          - "deploy/kubernetes/ingress.yaml"
```

---

## Gemini Conductor Prompts

### Main Conductor Prompt

```markdown
You are an AI project conductor orchestrating the development of the OCI AI Agent Coordinator system. Your role is to:

1. **Track Progress**: Monitor which tracks and milestones are complete
2. **Generate Tasks**: Break down milestones into actionable development tasks
3. **Manage Dependencies**: Ensure tasks are executed in the correct order
4. **Quality Control**: Verify deliverables meet requirements
5. **Adapt Plans**: Adjust based on feedback and blockers

## Current Project Status
Track the implementation status:
- [ ] Track 1: Foundation (Weeks 1-2)
- [ ] Track 2: Agents (Weeks 3-6)
- [ ] Track 3: Channels (Weeks 7-8)
- [ ] Track 4: Orchestration (Weeks 9-10)
- [ ] Track 5: Production (Weeks 11-12)
- [ ] Track 6: Enhancement (Weeks 13-14)

## Task Generation Rules
When generating implementation tasks:
1. Include clear acceptance criteria
2. Reference relevant prompt files
3. List required MCP tools
4. Specify test requirements
5. Note any blockers or dependencies

## Quality Gates
Before marking a milestone complete:
- All unit tests passing
- Integration tests passing
- Documentation updated
- Code reviewed
- No critical bugs
```

### Task Generator Prompt

```markdown
Generate implementation tasks for the OCI AI Agent Coordinator project.

## Context
You are creating development tasks for Claude Code to implement. Each task should:
- Be self-contained when possible
- Have clear inputs and outputs
- Reference the appropriate prompt files
- Include testing requirements

## Task Template
```yaml
task:
  id: "TASK-XXX"
  title: "Descriptive task title"
  description: |
    Detailed description of what needs to be implemented
  
  inputs:
    - "List of input files/prompts"
    
  outputs:
    - "List of files to create"
    
  implementation_notes:
    - "Key implementation details"
    
  testing:
    - "Test requirements"
    
  acceptance_criteria:
    - "Criterion 1"
    - "Criterion 2"
    
  estimated_time: "X hours"
  priority: "high|medium|low"
```
```

---

## Integration with Claude Code

### Claude Code Session Template

```markdown
# Claude Code Implementation Session

## Session Context
- **Project**: OCI AI Agent Coordinator
- **Track**: [Current Track]
- **Milestone**: [Current Milestone]
- **Task**: [Task ID and Description]

## Reference Documents
Load these prompts before implementation:
1. `/prompts/00-COORDINATOR-AGENT.md` - Main coordinator
2. `/prompts/01-DB-TROUBLESHOOT-AGENT.md` - DB agent
3. `/prompts/02-LOG-ANALYTICS-AGENT.md` - Log agent
4. `/prompts/03-SECURITY-THREAT-AGENT.md` - Security agent
5. `/prompts/04-FINOPS-AGENT.md` - FinOps agent
6. `/prompts/05-INFRASTRUCTURE-AGENT.md` - Infra agent

## MCP Server References
Available MCP servers and their tools are documented in the agent prompts.

## Implementation Instructions
[Specific implementation instructions for this task]

## Testing Requirements
[Test cases to implement]

## Deliverables
[Expected output files]
```

### Handoff Document Template

```markdown
# Implementation Handoff: [Feature/Task Name]

## Summary
Brief description of what was implemented.

## Files Created/Modified
- `path/to/file1.py` - Description
- `path/to/file2.py` - Description

## Architecture Decisions
Key decisions made during implementation:
1. Decision 1 - Rationale
2. Decision 2 - Rationale

## Configuration Required
```yaml
# Required configuration
key: value
```

## Testing Status
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Manual testing completed

## Known Issues/TODO
- Issue 1
- TODO 1

## Next Steps
1. Next task to implement
2. Dependencies to resolve
```
