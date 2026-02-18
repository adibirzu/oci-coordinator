# OKE Deployment and MCP Interconnect

This document defines how coordinator agents and MCP servers interconnect on OCI OKE.

## Target topology

```text
[Users/API/Slack]
      |
      v
[oci-coordinator deployment]
      |
      | Streamable HTTP (/mcp)
      v
[oci-mcp-gateway service]
      |
      +--> [mcp-oci in-process backend]         (enabled by default)
      +--> [database-observatory MCP service]   (optional)
      +--> [finopsai MCP service]               (optional)
      +--> [oci-mcp-security MCP service]       (optional)
```

## Why this is the recommended OKE pattern

- The coordinator keeps a single MCP client integration point (`oci-gateway`).
- Backend server lifecycle and auth are centralized in gateway.
- Horizontal scaling is cleaner: coordinator replicas stay stateless and gateway runs with `--stateless`.
- Direct local stdio MCP dependencies are removed from coordinator pods.

## MCP servers used in OKE

- Coordinator profile: `config/mcp_servers.oke.yaml`
  - Enabled server: `oci-gateway` only
  - Domains served: identity, compute, network, cost, security, observability, database, dbmgmt, finops, logan, opsi
- Gateway backend defaults (`deploy/oke/gateway-configmap.yaml`)
  - `oci` (in-process `mcp-oci`) enabled
  - `db-observatory`, `finops`, `security` remote backends disabled by default, can be enabled as services are deployed

Set deployment mode:

```bash
OCI_DEPLOYMENT_MODE=oke
```

For OCI-DEMO federated environments:

```bash
OCI_DEPLOYMENT_MODE=oci-demo
```

Runtime server ordering profiles are defined in `config/catalog/runtime_profiles.yaml`.

## Identity model

- Preferred for gateway in OKE: workload/resource principal auth for OCI SDK access.
- Keep coordinator as an MCP client; do not grant broad OCI IAM permissions directly to coordinator pods.
- Restrict IAM policies to least privilege by service/function.

## Inter-service connectivity

- Coordinator -> Gateway: internal `ClusterIP` service (`oci-mcp-gateway:9000`).
- Gateway -> OCI APIs: outbound HTTPS (443).
- Optional gateway -> remote MCP services: internal service DNS names.
- Network policies:
  - Gateway ingress accepts traffic from coordinator pods only.
  - Coordinator egress allows gateway + DNS + HTTPS.

## Deployment assets

- Kubernetes manifests: `deploy/oke/`
- Coordinator image build: `Dockerfile`
- OKE MCP config profile: `config/mcp_servers.oke.yaml`

## Rollout notes

1. Deploy gateway first and verify MCP health.
2. Deploy coordinator with `MCP_CONFIG_FILE=/app/config/mcp_servers.oke.yaml`.
3. Confirm `/mcp/servers` in coordinator API reports `oci-gateway` healthy.
4. Scale coordinator and gateway via HPA after baseline traffic validation.
