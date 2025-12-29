# Product Guidelines

## 1. User Experience (UX) Principles

### 1.1 Voice and Tone
- **Professional & Technical**: Communication should be concise, data-driven, and focused on efficiency, suitable for technical operations teams.
- **Authoritative yet Collaborative**: Agents should provide clear recommendations while allowing user control.
- **Context-Aware**: Responses must maintain conversation history and cross-domain context.

### 1.2 Interaction Model
- **Natural Language Interface**: Users interact via natural language queries through familiar channels (Slack, Teams, Web).
- **Unified Entry Point**: A single Coordinator Agent manages routing, eliminating the need to switch between tools.
- **Proactive**: The system should not just respond to queries but offer insights (e.g., cost anomalies, security threats).

## 2. Stakeholders & Needs

- **Cloud Operations Engineers**: Need fast troubleshooting, infrastructure management, and reduced MTTR.
- **FinOps Analysts**: Require cost visibility, optimization recommendations, and budget tracking.
- **Security Engineers**: Focus on threat detection, compliance monitoring, and posture assessment.
- **DBAs**: Need deep database performance analysis and troubleshooting capabilities.

## 3. Operational Constraints & Standards

### 3.1 Technical Constraints
- **OCI Integration**: Must deeply integrate with existing OCI tenancies and APIs.
- **MCP Architecture**: All OCI interactions must be mediated through MCP (Model Context Protocol) servers.
- **Multi-Channel**: Must support Slack, Microsoft Teams, and a Web API simultaneously.

### 3.2 Security & Compliance
- **Least Privilege**: Agents must operate with the minimum required OCI IAM permissions.
- **Auditability**: All agent actions and decisions must be logged and auditable.
- **No Credential Storage**: Agents must not store long-lived credentials; use OCI Vault and IAM authentication.

### 3.3 Reliability & Scale
- **Graceful Degradation**: System must handle agent failures without crashing the entire workflow.
- **Horizontal Scalability**: Architecture must support scaling agents to handle increased load.
- **Performance**: High intent classification accuracy (>95%) and fast response times (<5s P95).
