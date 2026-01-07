# Reference: Context Engineering Principles

Based on Manus's context engineering patterns, adapted for OCI operations.

## The 6 Core Principles

### 1. Filesystem as External Memory

> "Markdown is my 'working memory' on disk."

**Problem:** MCP tools return large JSON outputs that saturate context windows.

**Solution for OCI:**
- Store tool outputs in `.plan/notes.md`
- Keep only summaries and paths in context
- Agent can "look up" full data when needed

**Example:**
```python
# Instead of keeping 500-line JSON in context:
result = await run_tool("oci_cost_by_compartment", {...})
# Save to file, keep summary
save_to_notes(result, section="cost_analysis")
summary = extract_summary(result)  # Top 5 compartments, total cost
```

### 2. Attention Manipulation Through Repetition

**Problem:** After 10+ MCP tool calls, models forget the original user request.

**Solution:** Re-read `task_plan.md` before major decisions:
```
Start of context: [Original goal - far away, forgotten]
...many tool calls...
End of context: [Recently read task_plan.md - gets ATTENTION!]
```

### 3. Keep Failure Traces

**Problem:** OCI API errors get hidden by retry logic. Same errors repeat.

**Solution:** Log ALL errors in the plan file:
```markdown
## Errors Encountered
- [2025-01-07] oci.exceptions.ServiceError: 401 - Invalid auth
  → Switched from DEFAULT to EMDEMO profile
- [2025-01-07] Timeout querying OPSI metrics
  → Reduced time range from 30d to 7d
```

### 4. Avoid Few-Shot Overfitting

**Problem:** Repetitive tool-call patterns cause drift.

**Solution for OCI agents:**
- Vary the order of diagnostic steps when appropriate
- Don't blindly follow the same troubleshooting path
- Recalibrate on each new context

### 5. Stable Prefixes for Cache Optimization

**Problem:** LLM input is expensive (100:1 input-to-output ratio).

**Solution:**
- Put static system prompt FIRST
- Append-only context for tool outputs
- Consistent JSON serialization

### 6. Append-Only Context

**Problem:** Modifying previous messages invalidates KV-cache.

**Solution:** NEVER edit previous messages. Always append.

## OCI-Specific Patterns

### Pattern: Cascading Discovery

For finding the right resource:
```
1. List compartments → save to notes
2. Filter by name pattern
3. List resources in target compartment → save to notes
4. Get specific resource details
```

### Pattern: Time-Windowed Analysis

For cost/performance analysis:
```
1. Start with 7-day window (fast, recent)
2. If pattern unclear, extend to 30 days
3. For anomaly investigation, narrow to specific day
4. Save each result to notes with timestamp
```

### Pattern: Profile-Aware Operations

For multi-tenancy:
```markdown
## OCI Context
- Profile: EMDEMO (production metrics)
- Fallback: DEFAULT (development)

## Tool Outputs
| Tool | Profile | Summary |
|------|---------|---------|
| list_databases | EMDEMO | 15 databases found |
```

## Token Savings Estimates

| Pattern | Before | After | Savings |
|---------|--------|-------|---------|
| Raw tool output in context | 5000 tokens | 200 tokens | 96% |
| Re-reading plan vs forgetting | N/A | 300 tokens | Goal retention |
| Error logging vs retry loops | 3x calls | 1x calls | 66% fewer calls |

## When File-Based Planning Helps Most

1. **Complex troubleshooting** - 10+ tool calls, multiple data sources
2. **Cost analysis** - Large JSON outputs, comparisons over time
3. **Security audits** - Multiple domains to check, many findings
4. **Multi-profile operations** - Context switching between tenancies
