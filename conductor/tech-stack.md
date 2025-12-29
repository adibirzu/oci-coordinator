# Technology Stack

## 1. Programming Languages & Runtimes
- **Python (>=3.11)**: Primary language for all agent and API implementation.

## 2. Core Frameworks & Libraries
- **FastAPI / Uvicorn**: High-performance web framework for the Web API and agent coordination.
- **LangGraph**: State graph orchestration for multi-agent workflows (Workflow-first architecture).
- **LangChain**: Framework for building agents and tooling.
- **OCA LangChain Client**: Python client for Oracle Cloud Agent (Primary LLM).
- **OpenTelemetry**: Standard observability framework (SDK, OTLP Exporter).
- **Pydantic**: Data validation and settings management.
- **Redis**: In-memory store for LangGraph state checkpoints and shared context.
- **Tenacity**: For retry logic and resilience.

## 3. OCI Integration & Infrastructure
- **LLM Providers**:
    - **OCA (Primary)**: Oracle Cloud Agent.
    - **OCI GenAI**: Llama 3, Cohere.
    - **External (Supported)**: Anthropic Claude, OpenAI GPT-4, Google Gemini.
- **OCI APM**: Application Performance Monitoring (OpenTelemetry endpoint).
- **OCI SDK for Python**: Underlying library for OCI interactions.
- **MCP (Unified Server)**: Unified FastMCP server exposing "Skills" and "Tools".
- **OCI Vault**: Secure management of secrets and API keys.
- **Oracle Container Engine for Kubernetes (OKE)**: Production deployment environment.

## 4. Input Channel SDKs
- **Slack SDK / Slack Bolt**: For Slack bot integration.
- **Bot Builder Core**: For Microsoft Teams bot integration.

## 5. Development & Operations
- **Poetry**: Dependency management and packaging.
- **Ruff**: Fast Python linter.
- **Black**: Python code formatter.
- **Pytest**: Testing framework.
- **Docker**: Containerization for development and deployment.
- **Prometheus / Grafana**: Monitoring and observability.
