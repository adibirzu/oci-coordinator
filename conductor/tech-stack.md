# Technology Stack

## 1. Programming Languages & Runtimes
- **Python (>=3.11)**: Primary language for all agent and API implementation.

## 2. Core Frameworks & Libraries
- **FastAPI / Uvicorn**: High-performance web framework for the Web API and agent coordination.
- **LangGraph**: State graph orchestration for multi-agent workflows, cycles, and persistence.
- **LangChain**: Framework for building agents and tooling.
- **Pydantic**: Data validation and settings management.
- **Redis**: In-memory store for LangGraph state checkpoints and shared context.
- **Tenacity**: For retry logic and resilience.

## 3. OCI Integration & Infrastructure
- **OCI Generative AI Agents**: Managed service for RAG and specialized agent capabilities.
- **OCI Generative AI Inference**: For direct LLM invocation where needed.
- **OCI SDK for Python**: Underlying library for OCI interactions.
- **MCP (Model Context Protocol)**: Standard for integrating custom tools with agents.
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
