# Examples: Planning with Files for OCI Operations

## Example 1: Cost Spike Investigation

**User Request:** "Why did our OCI costs spike last week?"

### Loop 1: Create Plan
```bash
mkdir -p .plan
Write .plan/task_plan.md
```

```markdown
# Task Plan: Cost Spike Investigation

## Goal
Identify root cause of cost increase in the past week.

## OCI Context
- **Tenancy**: Default
- **Time Range**: Last 7 days vs previous 7 days

## Phases
- [x] Phase 1: Create this plan
- [ ] Phase 2: Get cost summary for both periods (CURRENT)
- [ ] Phase 3: Drill down by service
- [ ] Phase 4: Identify anomalies
- [ ] Phase 5: Generate cost spike report

## Key Questions
1. Which services showed the biggest increase?
2. Which compartments are affected?
3. Was this a one-time spike or gradual increase?

## Status
**Currently in Phase 2** - Querying cost data
```

### Loop 2: Gather Cost Data
```bash
Read .plan/task_plan.md           # Refresh goals
Run oci_cost_by_compartment       # This week
Append .plan/notes.md             # Save output
Run oci_cost_by_compartment       # Last week
Append .plan/notes.md             # Save output
Edit .plan/task_plan.md           # Mark Phase 2 complete
```

**notes.md after Loop 2:**
```markdown
# Notes: Cost Spike Investigation

## Cost Data - This Week (Jan 1-7)
Total: $12,450
Top compartments:
1. production: $8,200 (+45%)
2. staging: $2,100 (flat)
3. dev: $1,500 (-10%)

## Cost Data - Last Week (Dec 25-31)
Total: $9,100
Top compartments:
1. production: $5,650
2. staging: $2,100
3. dev: $1,350
```

### Loop 3: Service Drilldown
```bash
Read .plan/task_plan.md           # Refresh goals
Run oci_cost_service_drilldown    # Focus on production
Append .plan/notes.md             # Save output
Edit .plan/task_plan.md           # Update status
```

### Loop 4: Generate Report
```bash
Read .plan/task_plan.md           # Verify complete
Read .plan/notes.md               # Get all findings
Write .plan/cost_spike_report.md  # Final deliverable
```

---

## Example 2: Database Troubleshooting

**User Request:** "Database ORCL01 is slow, help me investigate"

### task_plan.md
```markdown
# Task Plan: Database Performance Investigation

## Goal
Identify root cause of performance degradation on ORCL01.

## OCI Context
- **Profile**: EMDEMO
- **Database**: ORCL01
- **Reported Issue**: Slow queries

## Phases
- [x] Phase 1: Get database details and status
- [x] Phase 2: Check recent metrics (CPU, I/O, memory)
- [x] Phase 3: Query logs for errors
- [ ] Phase 4: Get AWR report (CURRENT)
- [ ] Phase 5: Synthesize findings and recommendations

## Tool Outputs
| Tool | Summary | Location |
|------|---------|----------|
| get_database | ORCL01 is AVAILABLE, ATP | notes.md#database-info |
| analyze_cpu | CPU avg 78%, peak 95% | notes.md#metrics |
| execute_log_query | 23 ORA-errors in 24h | notes.md#log-errors |

## Errors Encountered
- First AWR attempt timeout → Reduced time range to 1 hour

## Status
**Currently in Phase 4** - Generating AWR report with 1-hour window
```

### notes.md
```markdown
# Notes: ORCL01 Investigation

## Database Info
- Name: ORCL01
- Type: Autonomous Transaction Processing
- OCPUs: 4
- Storage: 1TB (62% used)
- Status: AVAILABLE

## Metrics (Last 24h)
| Metric | Avg | Peak | Trend |
|--------|-----|------|-------|
| CPU | 78% | 95% | Increasing |
| Memory | 65% | 72% | Stable |
| I/O Ops | 1200/s | 3500/s | Spiky |

## Log Errors (Last 24h)
Top errors:
1. ORA-01555 (Snapshot too old) - 15 occurrences
2. ORA-04031 (Shared pool) - 5 occurrences
3. ORA-00060 (Deadlock) - 3 occurrences

Pattern: Errors cluster between 2-4 PM UTC
```

---

## Example 3: Security Audit

**User Request:** "Run a security check on our production compartment"

### The 3-File Pattern for Security

**task_plan.md:**
```markdown
# Task Plan: Production Security Audit

## Goal
Comprehensive security assessment of production compartment.

## Phases
- [x] Phase 1: Identify compartment and scope
- [x] Phase 2: Check Cloud Guard problems
- [ ] Phase 3: Review IAM policies (CURRENT)
- [ ] Phase 4: Check security zones
- [ ] Phase 5: Generate security report

## Findings Summary
- Cloud Guard: 3 CRITICAL, 7 HIGH problems
- IAM: [Pending]
- Security Zones: [Pending]

## Status
**Currently in Phase 3** - Enumerating IAM policies
```

**notes.md:**
```markdown
# Notes: Production Security Audit

## Cloud Guard Problems
### CRITICAL (3)
1. Public bucket with sensitive data - bucket-logs-prod
2. Instance without security list - vm-legacy-01
3. API key older than 90 days - user/admin

### HIGH (7)
1. VCN with 0.0.0.0/0 ingress - vcn-prod
2. Database without encryption - db-legacy
[... truncated, see full list in Cloud Guard console]

## IAM Policy Analysis
[To be filled in Phase 3]
```

**security_report.md:** (final deliverable)
```markdown
# Security Audit Report: Production Compartment

## Executive Summary
- **Risk Level**: HIGH
- **Critical Issues**: 3
- **High Issues**: 7
- **Immediate Actions Required**: 3

## Critical Findings
[Detailed findings with remediation steps]
```

---

## Example 4: Error Recovery Pattern

When OCI API calls fail, DON'T hide it:

### Before (Wrong)
```
Action: oci_cost_by_compartment (30 days)
Error: Timeout
Action: oci_cost_by_compartment (30 days)  # Silent retry
Action: oci_cost_by_compartment (30 days)  # Another retry
```

### After (Correct)
```
Action: oci_cost_by_compartment (30 days)
Error: Timeout

# Update task_plan.md:
## Errors Encountered
- Cost query timeout for 30-day range → Will try 7-day range

Action: oci_cost_by_compartment (7 days)
Success! → Save to notes.md
```

---

## The Read-Before-Decide Pattern

**Always read your plan before major decisions:**

```
[10+ tool calls have happened...]
[Context is getting long...]
[Original goal might be forgotten...]

→ Read .plan/task_plan.md      # This brings goals back into attention!
→ Now make the decision         # Goals are fresh in context
```

This is how you handle complex OCI operations without losing track of the original objective.
