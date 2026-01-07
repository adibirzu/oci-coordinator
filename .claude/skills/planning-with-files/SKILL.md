---
name: planning-with-files
description: Use persistent markdown files for planning, progress tracking, and knowledge storage. Reduces LLM token usage by storing context in files instead of conversation history. Use for complex OCI operations, multi-step troubleshooting, research tasks, or when tasks span many tool calls.
---

# Planning with Files for OCI Operations

Work like Manus: Use persistent markdown files as your "working memory on disk."

## Quick Start

Before ANY complex OCI task:

1. **Create `.plan/task_plan.md`** in the working directory
2. **Define phases** with checkboxes
3. **Update after each phase** - mark [x] and change status
4. **Read before deciding** - refresh goals in attention window

## The 3-File Pattern

For every non-trivial OCI operation, create THREE files:

| File | Purpose | When to Update |
|------|---------|----------------|
| `.plan/task_plan.md` | Track phases and progress | After each phase |
| `.plan/notes.md` | Store findings, tool outputs | During investigation |
| `.plan/[deliverable].md` | Final output/report | At completion |

## Core Workflow

```
Loop 1: Create task_plan.md with goal and phases
Loop 2: Execute tools → save outputs to notes.md → update task_plan.md
Loop 3: Read notes.md → analyze → update task_plan.md
Loop 4: Generate final report/deliverable
```

### The Loop in Detail

**Before each major action:**
```bash
Read .plan/task_plan.md  # Refresh goals in attention window
```

**After each phase:**
```bash
Edit .plan/task_plan.md  # Mark [x], update status
```

**When storing tool outputs:**
```bash
Append .plan/notes.md    # Don't stuff context, store in file
```

## task_plan.md Template for OCI

```markdown
# Task Plan: [Brief Description]

## Goal
[One sentence describing the end state]

## OCI Context
- **Tenancy**: [name/OCID if relevant]
- **Compartment**: [name if relevant]
- **Profile**: [OCI profile being used]

## Phases
- [ ] Phase 1: Gather context and plan
- [ ] Phase 2: Execute discovery/investigation
- [ ] Phase 3: Analyze findings
- [ ] Phase 4: Generate report/take action

## Key Questions
1. [Question to answer]
2. [Question to answer]

## Tool Outputs
| Tool | Summary | Details Location |
|------|---------|------------------|
| [tool_name] | [one-line summary] | notes.md#section |

## Decisions Made
- [Decision]: [Rationale]

## Errors Encountered
- [Error]: [Resolution]

## Status
**Currently in Phase X** - [What I'm doing now]
```

## notes.md Template for OCI

```markdown
# Notes: [Operation Name]

## Tool Outputs

### [Tool Name] - [Timestamp]
```json
[Condensed/summarized output - NOT full raw output]
```

Key findings:
- [Finding 1]
- [Finding 2]

## Synthesized Findings

### [Category]
- [Finding]
- [Finding]

## Metrics/Data Points
| Metric | Value | Notes |
|--------|-------|-------|
| [name] | [value] | [context] |
```

## Critical Rules

### 1. Store Tool Outputs, Not Raw Data
Large MCP tool outputs go to notes.md with summaries. Keep only paths in context.

**Wrong:**
```
Tool returned 500 lines of JSON → stuff into context
```

**Right:**
```
Tool returned 500 lines of JSON →
  Save to notes.md
  Extract key metrics
  Continue with "see notes.md#section for details"
```

### 2. Read Before Decide
Before any major decision (which workflow to use, which action to take), read the plan file.

### 3. Update After Act
After completing any phase, immediately update the plan file.

### 4. Compress Reversibly
When summarizing outputs, ensure enough detail remains to make decisions.

### 5. Log All Errors
Every OCI API error, timeout, or failure goes in "Errors Encountered" section.

## OCI-Specific Patterns

### For Cost Analysis Tasks
```markdown
## Phases
- [ ] Phase 1: Identify tenancy and time range
- [ ] Phase 2: Run oci_cost_by_compartment
- [ ] Phase 3: Run oci_cost_service_drilldown
- [ ] Phase 4: Analyze anomalies
- [ ] Phase 5: Generate cost report
```

### For Database Troubleshooting
```markdown
## Phases
- [ ] Phase 1: Identify database and symptoms
- [ ] Phase 2: Check database status/metrics
- [ ] Phase 3: Query logs for errors
- [ ] Phase 4: Analyze AWR if available
- [ ] Phase 5: Generate troubleshooting report
```

### For Security Audits
```markdown
## Phases
- [ ] Phase 1: Define audit scope
- [ ] Phase 2: Run Cloud Guard problem check
- [ ] Phase 3: Check IAM policies
- [ ] Phase 4: Review security zones
- [ ] Phase 5: Generate security report
```

## When to Use This Pattern

**Use for:**
- Multi-step OCI operations (3+ tool calls)
- Troubleshooting tasks
- Cost analysis across time periods
- Security audits
- Any task requiring organization

**Skip for:**
- Simple single-query tasks
- Quick lookups (list instances, get status)
- Tasks with <3 tool calls

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| Stuff raw tool output in context | Save to notes.md, keep summary |
| Forget original goal after 10+ calls | Re-read task_plan.md |
| Hide errors and retry silently | Log errors in plan file |
| Start executing immediately | Create plan file FIRST |
| Keep intermediate state in memory | Persist to .plan/ directory |
