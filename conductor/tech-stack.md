# Technology Stack

## 1. Programming Languages & Runtimes
- **Python (>=3.11)**: Primary language for all agent and API implementation.

## 2. Core Frameworks & Libraries
- **FastAPI / Uvicorn**: High-performance web framework for the Web API and agent coordination.
- **Anthropic SDK**: Client library for interacting with Claude models.
- **Pydantic**: Data validation and settings management.
- **Redis**: In-memory context store for session management and conversation state.
- **Tenacity**: For retry logic and resilience.
- **HTTPX**: Asynchronous HTTP client.

## 3. OCI Integration & Infrastructure
- **MCP (Model Context Protocol)**: Architecture for tool execution and OCI API abstraction.
- **OCI SDK for Python**: Underlying library for OCI interactions.
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
