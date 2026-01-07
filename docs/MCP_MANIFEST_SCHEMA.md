# MCP Manifest Schema

This document defines the canonical manifest contract used by the OCI Coordinator to discover tools, tiers, and policy metadata from MCP servers.

## Goals

- Provide a single source of truth for tool metadata, tiers, and safety rules.
- Enable dynamic catalog refresh without hardcoded lists.
- Support validation, caching, and routing decisions.

## Resource

Servers must expose a resource at `server://manifest` that returns a JSON document matching this schema.

## Top-Level Fields

- `schema_version` (string, required): Schema version, e.g. "1.0".
- `server_id` (string, required): Stable server identifier.
- `server_name` (string, required): Human readable name.
- `server_version` (string, required): Server version.
- `generated_at` (string, required): ISO 8601 timestamp in UTC.
- `manifest_hash` (string, optional): Content hash for change detection.
- `domains` (array, required): Summary of domains and counts.
- `tools` (array, required): Tool metadata.
- `skills` (array, optional): Skill metadata.
- `policies` (object, optional): Server policy hints, e.g. mutation guard env vars.

## Compatibility Fields (Optional)

Servers may also include legacy or extra fields without breaking validation:

- `name` (string): Legacy server name.
- `version` (string): Legacy server version.
- `description` (string): Human-readable description.
- `capabilities` (object): Grouped tool/skill summaries.
- `environment_variables` (array): Required environment variables.
- `usage_guide` (string): Quick start usage text.

## Domain Entry

Each entry in `domains` must include:

- `name` (string, required): Domain name, e.g. "compute".
- `tool_count` (int, required)
- `skill_count` (int, required)

## Tool Entry

Each entry in `tools` must include:

- `name` (string, required): Tool name.
- `description` (string, required)
- `domain` (string, required): Primary domain.
- `tier` (int, required): 1, 2, 3, or 4.
- `risk` (string, required): none, low, medium, high.
- `read_only` (bool, required)
- `idempotent` (bool, required)
- `mutates` (bool, required)
- `requires_confirmation` (bool, required)
- `latency_ms` (int, optional): Typical latency estimate.
- `cache_ttl_seconds` (int, optional)
- `input_schema` (object, optional): JSON schema for args.
- `aliases` (array, optional): Alternate names.
- `timeouts` (object, optional): e.g. `{ "default_seconds": 60 }`.

## Skill Entry

Each entry in `skills` must include:

- `name` (string, required)
- `description` (string, required)
- `domains` (array, required)

## Policy Hints

`policies` may include:

- `allow_mutations_env` (string): Environment variable controlling write ops.

## Conformance Rules

- Tool names must follow `oci_{domain}_{action}` unless legacy alias.
- Tier 4 tools must set `requires_confirmation=true` and `mutates=true`.
- `read_only=true` implies `mutates=false`.

## Coordinator Config Location

The coordinator keeps the canonical mappings in:

- `config/catalog/domain_prefixes.json`
- `config/catalog/server_domains.json`
- `config/catalog/tool_aliases.json`
- `config/catalog/tool_tiers.yaml`

## Example

```json
{
  "schema_version": "1.0",
  "server_id": "mcp-oci",
  "server_name": "OCI MCP Server",
  "server_version": "2.5.0",
  "generated_at": "2026-01-02T12:00:00Z",
  "domains": [{"name": "compute", "tool_count": 5, "skill_count": 1}],
  "tools": [
    {
      "name": "oci_compute_list_instances",
      "description": "List compute instances",
      "domain": "compute",
      "tier": 2,
      "risk": "low",
      "read_only": true,
      "idempotent": true,
      "mutates": false,
      "requires_confirmation": false,
      "cache_ttl_seconds": 600
    }
  ],
  "skills": [],
  "policies": {"allow_mutations_env": "ALLOW_MUTATIONS"}
}
```
