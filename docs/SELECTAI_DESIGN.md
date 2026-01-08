# SelectAI Agent Design Document

## Overview

This document describes the design for integrating Oracle Autonomous Database's SelectAI capabilities into the oci-coordinator. SelectAI provides natural language to SQL translation, chat with database context, and AI-powered agents.

## Background

Oracle Autonomous Database includes two key packages for AI integration:
- **DBMS_CLOUD_AI**: Core AI capabilities (NL2SQL, chat, summarize, translate)
- **DBMS_CLOUD_AI_AGENT**: Agentic framework for creating AI agents with tools

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OCI Coordinator                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  FinOps Agent   │    │   DB Troubleshoot│    │ SelectAI Agent  │ ◄───NEW │
│  │                 │    │   Agent          │    │                 │         │
│  └────────┬────────┘    └────────┬─────────┘    └────────┬────────┘         │
│           │                      │                       │                   │
│           └──────────────────────┼───────────────────────┘                   │
│                                  │                                           │
│                      ┌───────────┴───────────┐                               │
│                      │    Tool Catalog       │                               │
│                      │    (MCP Client)       │                               │
│                      └───────────┬───────────┘                               │
│                                  │                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼──────┐    ┌───────────▼──────────┐   ┌───────▼────────┐
    │ FinOps MCP  │    │ Database Observatory │   │ SelectAI MCP   │◄───NEW
    │ Server      │    │ MCP Server           │   │ Server         │
    └─────────────┘    └──────────────────────┘   └───────┬────────┘
                                                          │
                                                  ┌───────▼────────┐
                                                  │ Autonomous DB  │
                                                  │ DBMS_CLOUD_AI  │
                                                  └────────────────┘
```

### Design Principles

1. **Non-Breaking**: New agent added alongside existing agents
2. **Modular**: SelectAI functionality encapsulated in its own module
3. **Consistent**: Follows existing agent patterns (BaseAgent, StateGraph, etc.)
4. **Extensible**: Supports both stateless calls and agentic workflows

## SelectAI Capabilities

### 1. DBMS_CLOUD_AI.GENERATE (Stateless)

Primary function for quick NL2SQL and chat operations:

```sql
SELECT DBMS_CLOUD_AI.GENERATE(
    prompt => 'Show top 10 customers by revenue',
    profile_name => 'MY_PROFILE',
    action => 'runsql'
) AS response FROM DUAL;
```

**Actions**:
| Action | Description |
|--------|-------------|
| `showsql` | Return generated SQL without executing |
| `runsql` | Execute generated SQL and return results |
| `explainsql` | Generate SQL with explanation |
| `narrate` | Natural language summary of data |
| `summarize` | Summarize text/documents |
| `translate` | Translate between languages |
| `chat` | General conversation with context |

### 2. AI Profiles

Profiles configure LLM providers and database context:

```sql
DBMS_CLOUD_AI.CREATE_PROFILE(
    profile_name => 'MY_PROFILE',
    attributes => '{
        "provider": "oci",
        "credential_name": "OCI_CRED",
        "object_list": [
            {"owner": "SALES", "name": "CUSTOMERS"},
            {"owner": "SALES", "name": "ORDERS"}
        ]
    }'
);
```

**Supported Providers**:
- OCI Generative AI
- OpenAI
- Azure OpenAI
- Cohere
- Google
- Anthropic

### 3. DBMS_CLOUD_AI_AGENT (Agentic)

For complex multi-step workflows with tools:

```sql
-- Create Agent
DBMS_CLOUD_AI_AGENT.CREATE_AGENT(
    agent_name => 'SALES_ANALYST',
    attributes => '{
        "profile_name": "MY_PROFILE",
        "role": "You are a sales data analyst"
    }'
);

-- Create Tool
DBMS_CLOUD_AI_AGENT.CREATE_TOOL(
    tool_name => 'GET_REVENUE',
    attributes => '{
        "description": "Get revenue by region",
        "type": "SQL",
        "statements": ["SELECT region, SUM(amount) FROM orders GROUP BY region"]
    }'
);

-- Create Task
DBMS_CLOUD_AI_AGENT.CREATE_TASK(
    task_name => 'ANALYZE_SALES',
    agent_name => 'SALES_ANALYST',
    tool_names => '["GET_REVENUE"]',
    instruction_template => 'Analyze ${region} sales data'
);

-- Run Agent
DBMS_CLOUD_AI_AGENT.RUN_TEAM(
    team_name => 'SALES_TEAM',
    user_prompt => 'What were last quarter revenues?'
);
```

## Implementation

### 1. Directory Structure

```
src/agents/selectai/
├── __init__.py
├── agent.py           # SelectAIAgent class
├── profiles.py        # AI profile management
└── tools.py           # Helper functions for DBMS_CLOUD_AI calls

src/mcp/server/tools/
├── selectai.py        # NEW: MCP tools for SelectAI
```

### 2. SelectAIAgent Class

```python
class SelectAIAgent(BaseAgent, SelfHealingMixin):
    """
    SelectAI Agent for natural language database interaction.

    Capabilities:
    - Natural language to SQL (NL2SQL)
    - Chat with database context
    - Text summarization
    - AI agent orchestration
    """

    @classmethod
    def get_definition(cls) -> AgentDefinition:
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
                "SelectAI Agent for natural language database queries, "
                "chat with database context, and AI agent orchestration "
                "using Oracle Autonomous Database SelectAI."
            ),
            mcp_servers=["selectai", "database-observatory"],
        )
```

### 3. MCP Tools

#### Tool: `oci_selectai_generate`

Execute DBMS_CLOUD_AI.GENERATE for NL2SQL and chat:

```python
@tool
async def oci_selectai_generate(
    prompt: str,
    profile_name: str,
    action: Literal["showsql", "runsql", "explainsql", "narrate", "chat"] = "runsql",
    database_id: str | None = None,
) -> dict:
    """
    Execute SelectAI GENERATE function.

    Args:
        prompt: Natural language query or chat message
        profile_name: Name of the AI profile to use
        action: Action type (showsql, runsql, explainsql, narrate, chat)
        database_id: Target Autonomous Database OCID

    Returns:
        Generated SQL and/or query results
    """
```

#### Tool: `oci_selectai_list_profiles`

List available AI profiles:

```python
@tool
async def oci_selectai_list_profiles(
    database_id: str,
) -> list[dict]:
    """List AI profiles configured in the database."""
```

#### Tool: `oci_selectai_create_profile`

Create new AI profile:

```python
@tool
async def oci_selectai_create_profile(
    database_id: str,
    profile_name: str,
    provider: str,
    credential_name: str,
    tables: list[dict],
    model: str | None = None,
) -> dict:
    """Create a new SelectAI profile with specified tables."""
```

#### Tool: `oci_selectai_run_agent`

Execute a SelectAI agent:

```python
@tool
async def oci_selectai_run_agent(
    database_id: str,
    team_name: str,
    user_prompt: str,
    session_id: str | None = None,
) -> dict:
    """Execute a SelectAI agent team with the given prompt."""
```

### 4. Agent Catalog Updates

Add to `DOMAIN_CAPABILITIES`:

```python
"selectai": [
    "nl2sql",
    "data-chat",
    "text-summarization",
    "ai-agent-orchestration",
    "database-qa",
],
```

Add to `DOMAIN_PRIORITY`:

```python
"selectai": {
    "selectai-agent": 100,
    "db-troubleshoot-agent": 20,
},
```

### 5. Workflow Graph

```
┌─────────────────┐
│  analyze_query  │
│ (detect intent) │
└────────┬────────┘
         │
    ┌────▼────┐
    │ router  │
    └────┬────┘
         │
    ┌────┴────────────────────┬─────────────────────┐
    │                         │                     │
┌───▼───┐              ┌──────▼──────┐      ┌───────▼────────┐
│ nl2sql│              │   chat      │      │   agent_run    │
│ node  │              │   node      │      │   node         │
└───┬───┘              └──────┬──────┘      └───────┬────────┘
    │                         │                     │
    └─────────────────────────┼─────────────────────┘
                              │
                      ┌───────▼───────┐
                      │    output     │
                      │    node       │
                      └───────────────┘
```

## Connection Strategy

### Option 1: Database Link through Database Observatory

Leverage existing Database Observatory connection infrastructure:

```python
# Use existing SQL execution capability
result = await self.call_tool(
    "oci_database_execute_sql",
    {
        "database_id": database_id,
        "sql": f"SELECT DBMS_CLOUD_AI.GENERATE(...) FROM DUAL",
    }
)
```

**Pros**: No new connection management needed
**Cons**: Depends on Database Observatory MCP

### Option 2: Direct OCI REST API (Preferred)

Use Autonomous Database REST endpoints:

```python
# Use ADB Data API for SQL execution
async def execute_selectai(
    database_id: str,
    sql: str,
    schema: str = "ADMIN",
) -> dict:
    # POST to https://{adb-id}.adb.{region}.oraclecloudapps.com/ords/{schema}/_/sql
```

**Pros**: Independent, uses standard ADB REST API
**Cons**: Requires ORDS configuration

### Option 3: Python cx_Oracle/oracledb

Direct Python driver connection:

```python
import oracledb

async def connect_and_execute(
    wallet_location: str,
    service_name: str,
    user: str,
    password: str,
    sql: str,
) -> dict:
    connection = oracledb.connect(
        user=user,
        password=password,
        dsn=service_name,
        config_dir=wallet_location,
        wallet_location=wallet_location,
        wallet_password=password,
    )
    # Execute DBMS_CLOUD_AI calls
```

**Pros**: Full control, works with all DB features
**Cons**: Requires wallet management

## Security Considerations

1. **Credential Management**: AI profile credentials stored in OCI Vault
2. **Access Control**: Only users with EXECUTE on DBMS_CLOUD_AI can use SelectAI
3. **Rate Limiting**: LLM providers have rate limits; implement backoff
4. **Data Privacy**: SelectAI queries may send table metadata to LLM providers

## Configuration

### Environment Variables

```bash
# SelectAI Configuration
SELECTAI_DEFAULT_PROFILE=MY_PROFILE
SELECTAI_DEFAULT_DATABASE=ocid1.autonomousdatabase...
SELECTAI_CONNECTION_TYPE=ords  # ords, wallet, db-observatory
SELECTAI_ORDS_BASE_URL=https://xxx.adb.region.oraclecloudapps.com

# For wallet-based connections
SELECTAI_WALLET_LOCATION=/path/to/wallet
SELECTAI_WALLET_PASSWORD_SECRET=ocid1.vaultsecret...
```

### Profile YAML Configuration

```yaml
# config/selectai_profiles.yaml
profiles:
  default:
    database_id: ${SELECTAI_DEFAULT_DATABASE}
    profile_name: OCI_GENAI_PROFILE

  sales_analysis:
    database_id: ocid1.autonomousdatabase.oc1...
    profile_name: SALES_PROFILE
    tables:
      - owner: SALES
        name: CUSTOMERS
      - owner: SALES
        name: ORDERS
```

## Usage Examples

### Example 1: Simple NL2SQL

```
User: "Show me customers with orders over $10000"

SelectAI Agent:
1. Detects NL2SQL intent
2. Calls oci_selectai_generate(prompt=..., action="runsql")
3. Returns formatted results
```

### Example 2: Data Chat

```
User: "What's the trend in our sales data?"

SelectAI Agent:
1. Detects chat/narrate intent
2. Calls oci_selectai_generate(prompt=..., action="narrate")
3. Returns LLM-generated narrative with data context
```

### Example 3: Agent Execution

```
User: "Run the quarterly sales analysis agent"

SelectAI Agent:
1. Detects agent execution intent
2. Calls oci_selectai_run_agent(team_name="QUARTERLY_ANALYSIS", ...)
3. Returns agent execution results
```

## Testing Strategy

1. **Unit Tests**: Mock DBMS_CLOUD_AI responses
2. **Integration Tests**: Test against sandbox ADB instance
3. **E2E Tests**: Full workflow from Slack to ADB and back

## Migration Path

### Phase 1: Core NL2SQL
- Implement `oci_selectai_generate` tool
- Basic SelectAIAgent with showsql/runsql

### Phase 2: Profile Management
- List/create/manage AI profiles
- Support multiple databases

### Phase 3: Agent Orchestration
- CREATE_AGENT, CREATE_TOOL, CREATE_TASK
- RUN_TEAM for multi-step workflows

### Phase 4: Advanced Features
- RAG with Vector Search
- Conversation history/sessions
- Custom tool integration

## Open Questions

1. **Connection Strategy**: Which approach (ORDS, wallet, db-observatory)?
2. **Profile Management**: User-managed or auto-configured?
3. **Multi-tenancy**: How to handle multiple databases/profiles?
4. **Error Handling**: How to surface SQL errors from SelectAI?

## References

- [Oracle SelectAI Agents Documentation](https://docs.oracle.com/en-us/iaas/autonomous-database-serverless/doc/about-select-ai-agents.html)
- [DBMS_CLOUD_AI Package](https://docs.oracle.com/en-us/iaas/autonomous-database-serverless/doc/dbms-cloud-ai-package.html)
- [DBMS_CLOUD_AI_AGENT Package](https://docs.oracle.com/en-us/iaas/autonomous-database-serverless/doc/dbms-cloud-ai-agent.html)
