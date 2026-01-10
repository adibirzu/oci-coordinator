# LLM Provider Configuration Guide

## Overview

The OCI AI Coordinator supports multiple LLM providers for agent reasoning, intent classification, and natural language processing. This document covers configuration for all supported providers.

## Supported Providers

| Provider | Model | Use Case | Environment Variable |
|----------|-------|----------|---------------------|
| **Oracle Code Assist** | oca/gpt-4.1 | Primary (OCI native) | `LLM_PROVIDER=oracle_code_assist` |
| **Anthropic Claude** | claude-3-5-sonnet | Alternative | `LLM_PROVIDER=anthropic` |
| **OpenAI** | gpt-4o | Alternative | `LLM_PROVIDER=openai` |
| **Google Gemini** | gemini-2.0-flash | Alternative | `LLM_PROVIDER=google` |

---

## Oracle Code Assist (Primary)

Oracle Code Assist is the recommended provider for OCI-native deployments.

### Configuration

```bash
# .env or environment
LLM_PROVIDER=oracle_code_assist
OCA_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
OCA_REGION=us-chicago-1
OCA_TENANCY_OCID=ocid1.tenancy.oc1..xxx
OCA_COMPARTMENT_OCID=ocid1.compartment.oc1..xxx
```

### Usage

```python
from src.llm.factory import create_llm

llm = create_llm(provider="oracle_code_assist")
response = await llm.ainvoke("Analyze database performance")
```

### Observability

OCA calls are automatically instrumented with GenAI semantic conventions:

```python
from src.observability import OracleCodeAssistInstrumentor

with OracleCodeAssistInstrumentor.chat_span(model="oca/gpt5") as ctx:
    ctx.set_prompt("User query", role="user")
    # ... LLM call ...
    ctx.set_completion(response)
    ctx.set_tokens(input=100, output=200)
```

---

## Anthropic Claude

### Configuration

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
```

### Models

- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307`

---

## OpenAI

### Configuration

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
```

### Models

- `gpt-4o` (recommended)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

---

## Google Gemini

### Configuration

```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=xxx
```

### Models

- `gemini-2.0-flash-exp` (recommended for 2026)
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### Usage

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
)
```

---

## LLM Factory

The `src/llm/factory.py` provides a unified interface for creating LLM instances:

```python
from src.llm.factory import create_llm, get_llm_config

# Create LLM based on environment
llm = create_llm()

# Create specific provider
llm = create_llm(provider="anthropic")

# Get configuration
config = get_llm_config()
print(config.provider, config.model)
```

---

## Health Monitoring

All LLM providers have health checks:

```python
from src.llm.health import check_llm_health

# Check current provider
status = await check_llm_health()
print(status.is_healthy, status.latency_ms)

# Check specific provider
status = await check_llm_health(provider="oracle_code_assist")
```

---

## OpenTelemetry Integration

LLM calls are instrumented following OpenTelemetry GenAI semantic conventions:

### Attributes Captured

| Attribute | Description |
|-----------|-------------|
| `gen_ai.system` | Provider name (e.g., `oracle_code_assist`) |
| `gen_ai.request.model` | Model ID |
| `gen_ai.usage.input_tokens` | Input token count |
| `gen_ai.usage.output_tokens` | Output token count |
| `gen_ai.response.finish_reason` | Completion reason |

### Example Span

```json
{
  "name": "llm.chat",
  "attributes": {
    "gen_ai.system": "oracle_code_assist",
    "gen_ai.request.model": "oca/gpt5",
    "gen_ai.usage.input_tokens": 150,
    "gen_ai.usage.output_tokens": 250,
    "gen_ai.request.temperature": 0.7
  }
}
```

---

## Best Practices

1. **Use Oracle Code Assist** for OCI deployments - lowest latency, native integration
2. **Enable OpenTelemetry** for production observability
3. **Set reasonable timeouts** (30s for complex queries)
4. **Use streaming** for long responses to improve UX
5. **Cache responses** where appropriate (Redis via `SharedMemoryManager`)

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `OCA_ENDPOINT not set` | Missing config | Set `OCA_ENDPOINT` environment variable |
| `Rate limit exceeded` | Too many requests | Implement backoff, use caching |
| `Model not found` | Invalid model ID | Check provider documentation for valid models |
| `Timeout` | Slow response | Increase timeout, use streaming |

### Debug Logging

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m src.main
```
