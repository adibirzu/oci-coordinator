# OCI Agent Stack Architecture (GCP Equivalent)

This document maps the project's architecture directly to the **Google Cloud Platform (GCP) Agentic Stack Architecture**, demonstrating how to replicate that well-known blueprint using Oracle Cloud Infrastructure (OCI) services, LangGraph integrations, and Model Context Protocol (MCP) servers.

## Architecture Mapping Table

| GCP Agent Stack Layer  | GCP Service / Framework   | OCI Equivalent Service / Framework  | Implementation in this Project                             |
| ---------------------- | ------------------------- | ----------------------------------- | ---------------------------------------------------------- |
| **Foundation Models**  | Vertex AI (Gemini)        | OCI Generative AI (Cohere/Llama)    | `ChatOCIGenAI` via LangChain (`src/llm/factory.py`)        |
| **Agent Orchestrator** | LangChain & LangGraph     | LangChain & LangGraph on OCI        | `LangGraphCoordinator` (`src/agents/coordinator/graph.py`) |
| **Memory / Vector DB** | Firestore / Vector Search | Autonomous Database (JSON + Vector) | Persistent Memory via ADB (`src/memory/adb.py`)            |
| **Hot Cache**          | Cloud Memorystore         | OCI Cache (Redis)                   | SharedMemoryManager via Redis (`src/memory/redis.py`)      |
| **Tools & Enablement** | Vertex AI Extensions      | Model Context Protocol (MCP)        | MCP Servers mapped to OCI SDKs (`src/mcp/`)                |
| **Deployment Runtime** | GKE / Cloud Run           | Oracle Kubernetes Engine (OKE)      | Helm Charts (`deploy/helm/oci-agent-stack`)                |
| **Compute / Infra**    | Compute Engine            | OCI Compute Instances               | Underlying Infrastructure via IaC                          |

## Implementation Details

### 1. Foundation Models (OCI Generative AI)

Instead of invoking Gemini models via Vertex AI, the `LLMFactory` initializes **OCI Generative AI** using `langchain-community` integrations:

- Models: Cohere Command R+ / Llama 3
- Integration: `ChatOCIGenAI(model_id="cohere.command-r-plus")`
- Seamlessly fits into LangGraph as a `BaseChatModel`.

### 2. Orchestration (LangGraph Workflow-First)

The project replicates the GCP _Agent Engine_ workflow-first routing behavior. Instead of utilizing an LLM for simple queries, the Coordinator graph routes intents deterministically where possible (>70% match) and falls back to pure Agentic (ReAct) behavior otherwise.
This is implemented using LangGraph's state machine, effectively replacing Google's Agent Development Kit (ADK) with pure open-source LangGraph tuned for OCI.

### 3. Agent Capabilities (MCP Servers)

GCP enables AI Agents with custom API Extensions natively attached to Vertex. We replicate this "Enablement Layer" by utilizing the industry-standard **Model Context Protocol (MCP)**.

- Each core domain (Compute, Network, Security, Data) exposes an independent MCP server.
- The agents query the `ToolCatalog` dynamically.

### 4. Container-Native Runtime (OKE + Helm)

GCP traditionally targets GKE or Cloud Run for hosting LangChain applications. This project targets **Oracle Kubernetes Engine (OKE)** using Helm.
The Helm chart (`deploy/helm/oci-agent-stack/`) encapsulates both the coordinator and the MCP gateway, ensuring autoscaling, liveness probes, and fault tolerance at the orchestration level.
