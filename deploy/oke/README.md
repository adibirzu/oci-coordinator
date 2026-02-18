# OCI OKE Deployment Bundle

This bundle deploys:
- `oci-coordinator` (API + all agents in one deployment)
- `oci-mcp-gateway` (single MCP entrypoint for all agent tool calls)

## Why this topology

- Agents run inside the coordinator process; no per-agent pod sprawl.
- Coordinator talks only to `oci-gateway` using Streamable HTTP.
- Gateway handles backend aggregation and OCI auth mode per backend.
- OKE profile uses `config/mcp_servers.oke.yaml` to avoid local stdio dependencies.

## Files

- `coordinator-*.yaml`: coordinator runtime, scaling, and service
- `gateway-*.yaml`: MCP gateway runtime, scaling, and service
- `networkpolicy.yaml`: least-privilege east/west traffic controls
- `coordinator-secret.example.yaml`: coordinator secret template
- `gateway-configmap.yaml`: gateway backend config template
- `kustomization.yaml`: apply all resources together

## Prerequisites

1. Build and push coordinator image from this repo:

```bash
docker build -t <registry>/oci-coordinator:<tag> .
docker push <registry>/oci-coordinator:<tag>
```

2. Provide a gateway image (from `adibirzu/mcp-oci`) with `oci-mcp-gateway` CLI.

3. Update image references in:
- `coordinator-deployment.yaml`
- `gateway-deployment.yaml`

4. Create secrets from templates:

```bash
cp deploy/oke/coordinator-secret.example.yaml deploy/oke/coordinator-secret.yaml
# Edit values, then:
kubectl apply -f deploy/oke/coordinator-secret.yaml
```

## Deploy

```bash
kubectl apply -k deploy/oke
```

## Verify

```bash
kubectl -n oci-ai get pods
kubectl -n oci-ai get svc
kubectl -n oci-ai logs deploy/oci-coordinator --tail=200
kubectl -n oci-ai logs deploy/oci-mcp-gateway --tail=200
```

## OKE identity best practice

- Use OKE workload identity or resource principals for gateway backend OCI calls.
- Attach minimum IAM policies to the dynamic group / workload identity principal.
- Keep coordinator as an unprivileged MCP client; let gateway hold OCI access.

## Production hardening checklist

- Enable gateway JWT/Bearer auth (instead of open/internal mode).
- Keep gateway as `ClusterIP`; expose only coordinator/API via ingress.
- Set per-namespace `ResourceQuota` and `LimitRange`.
- Use OCI Vault + external secrets controller for secret rotation.
- Add pod anti-affinity across availability domains.
