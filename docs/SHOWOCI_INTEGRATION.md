# ShowOCI Integration Guide

This document describes how to use the ShowOCI-style resource discovery and caching capabilities in the OCI AI Agent Coordinator.

## Overview

The ShowOCI integration provides comprehensive OCI resource discovery similar to Oracle's [ShowOCI tool](https://github.com/oracle/oci-python-sdk/tree/master/examples/showoci). It enables:

- **Full tenancy discovery**: Scan all compartments, regions, and resource types
- **Redis caching**: Store discovered resources for fast agent access
- **Periodic refresh**: Automatically update cache on a configurable schedule
- **MCP tools**: Query cached resources through the unified MCP server

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OCI Tenancy                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │Compute  │  │ Network │  │Database │  │ Storage │  ...        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘             │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                          │
              ┌───────────▼───────────┐
              │   ShowOCI Runner      │
              │   (OCI SDK Direct)    │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Cache Loader        │
              │   (Redis Writer)      │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Redis Cache         │
              │   ┌─────────────────┐ │
              │   │ oci:compartments│ │
              │   │ oci:instances:* │ │
              │   │ oci:databases:* │ │
              │   │ oci:vcns:*      │ │
              │   └─────────────────┘ │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   MCP Discovery Tools │
              │   - oci_discovery_run │
              │   - oci_discovery_get │
              │   - oci_discovery_... │
              └───────────────────────┘
```

## Configuration

### Environment Variables

Add these to your `.env.local` file:

```bash
# Enable ShowOCI cache on startup
SHOWOCI_CACHE_ENABLED=true

# OCI profiles to discover (comma-separated)
OCI_PROFILES=DEFAULT,prod-tenancy

# Cache refresh interval in hours (0 to disable)
SHOWOCI_REFRESH_HOURS=4

# Redis connection
REDIS_URL=redis://localhost:6379
```

### OCI Profiles

Configure your OCI profiles in `~/.oci/config`:

```ini
[DEFAULT]
user=ocid1.user.oc1..aaaa...
fingerprint=xx:xx:xx:...
tenancy=ocid1.tenancy.oc1..aaaa...
region=eu-frankfurt-1
key_file=~/.oci/oci_api_key.pem

[prod-tenancy]
user=ocid1.user.oc1..bbbb...
fingerprint=yy:yy:yy:...
tenancy=ocid1.tenancy.oc1..bbbb...
region=us-ashburn-1
key_file=~/.oci/prod_key.pem
```

## MCP Discovery Tools

The following tools are available through the unified MCP server:

### oci_discovery_run

Run a full discovery scan:

```json
{
  "tool": "oci_discovery_run",
  "arguments": {
    "profile": "DEFAULT",
    "resource_types": "compute,network,database",
    "compartment_id": null,
    "load_to_cache": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "profile": "DEFAULT",
  "duration_seconds": 45.2,
  "compartments_scanned": 90,
  "resource_counts": {
    "compartments": 90,
    "instances": 25,
    "vcns": 12,
    "subnets": 48,
    "autonomous_databases": 5,
    "block_volumes": 15
  }
}
```

### oci_discovery_get_cached

Get cached resources:

```json
{
  "tool": "oci_discovery_get_cached",
  "arguments": {
    "resource_type": "instances",
    "compartment_name": "Adrian_Birzu",
    "format": "markdown"
  }
}
```

**Resource Types:**
- `instances` - Compute instances
- `databases` - Autonomous databases
- `vcns` - Virtual Cloud Networks
- `subnets` - Subnets
- `compartments` - Compartments

### oci_discovery_refresh

Force a cache refresh:

```json
{
  "tool": "oci_discovery_refresh",
  "arguments": {
    "profiles": "DEFAULT,prod-tenancy"
  }
}
```

### oci_discovery_summary

Get a resource summary:

```json
{
  "tool": "oci_discovery_summary",
  "arguments": {
    "compartment_name": "Production"
  }
}
```

### oci_discovery_search

Search cached resources by name:

```json
{
  "tool": "oci_discovery_search",
  "arguments": {
    "query": "web-server",
    "resource_type": "instance"
  }
}
```

### oci_discovery_cache_status

Get cache health and statistics:

```json
{
  "tool": "oci_discovery_cache_status"
}
```

## Python API Usage

### Basic Discovery

```python
from src.showoci import ShowOCIRunner, ShowOCIConfig

# Configure runner
config = ShowOCIConfig(
    profile="DEFAULT",
    resource_types=["compute", "network"],
)

# Run discovery
runner = ShowOCIRunner(config=config)
result = await runner.run_discovery()

if result.success:
    # Access discovered resources
    for instance in result.get_instances():
        print(f"{instance['display_name']}: {instance['lifecycle_state']}")

    for vcn in result.get_vcns():
        print(f"{vcn['display_name']}: {vcn['cidr_block']}")
```

### Cache Loading

```python
from src.showoci import ShowOCICacheLoader

# Initialize loader
loader = ShowOCICacheLoader(
    redis_url="redis://localhost:6379",
    profiles=["DEFAULT", "prod-tenancy"],
)

# Run full load
result = await loader.run_full_load()
print(f"Loaded {result['total_resources']['instances']} instances")

# Start periodic refresh
await loader.start_scheduler(interval_hours=4)
```

### Querying Cache

```python
from src.cache.oci_resource_cache import OCIResourceCache

# Get cache instance
cache = OCIResourceCache.get_instance()
await cache.initialize()

# Get resources
compartment = await cache.get_compartment_by_name("Adrian_Birzu")
instances = await cache.get_instances(compartment["id"])

# Search resources
results = await cache.search_resources("web-server", resource_type="instance")
```

## Cache Keys

The following Redis key patterns are used:

| Key Pattern | Description | TTL |
|-------------|-------------|-----|
| `oci:compartments` | All compartments | 4 hours |
| `oci:compartment:name:{name}` | Compartment by name | 4 hours |
| `oci:compartment:id:{ocid}` | Compartment by OCID | 4 hours |
| `oci:instances:{compartment_id}` | Instances by compartment | 30 min |
| `oci:databases:{compartment_id}` | Databases by compartment | 30 min |
| `oci:vcns:{compartment_id}` | VCNs by compartment | 30 min |
| `oci:discovery:last_run` | Last discovery timestamp | - |
| `oci:discovery:stats` | Discovery statistics | - |

## Agent Integration

Agents can use the discovery tools to answer resource queries:

### Infrastructure Agent Example

```markdown
User: "Show me all instances in Adrian_Birzu compartment"

Agent Flow:
1. Call oci_discovery_get_cached(resource_type="instances", compartment_name="Adrian_Birzu")
2. If cache empty, call oci_discovery_run(compartment_id="...", load_to_cache=true)
3. Format and return results
```

### Best Practices

1. **Use cache first**: Always try `oci_discovery_get_cached` before running a full discovery
2. **Scope discovery**: Use `compartment_id` to limit discovery scope when possible
3. **Check cache status**: Use `oci_discovery_cache_status` to verify cache freshness
4. **Resource types**: Only discover needed resource types for performance

## Troubleshooting

### Cache Not Populating

1. Check Redis is running: `redis-cli ping`
2. Verify OCI credentials: `oci iam user list --limit 1`
3. Check environment variables are set

### Discovery Timeouts

Large tenancies may take longer to discover:

```bash
# Increase discovery timeout (seconds)
export SHOWOCI_TIMEOUT=900  # 15 minutes
```

### Permission Errors

Ensure the OCI user has these policies:

```
Allow group Administrators to read all-resources in tenancy
```

Or more granular:

```
Allow group Agents to read instances in tenancy
Allow group Agents to read autonomous-databases in tenancy
Allow group Agents to read virtual-network-family in tenancy
Allow group Agents to read compartments in tenancy
```

## References

- [ShowOCI Original Tool](https://github.com/oracle/oci-python-sdk/tree/master/examples/showoci)
- [OCI Python SDK](https://oracle-cloud-infrastructure-python-sdk.readthedocs.io/)
- [OCI IAM Policies](https://docs.oracle.com/en-us/iaas/Content/Identity/Concepts/policyreference.htm)
