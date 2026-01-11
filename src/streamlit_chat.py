"""Streamlit Chat Application with OCI APM Instrumentation.

A fully instrumented chat interface that sends all telemetry data
to OCI APM for observability and monitoring.

Run with:
    streamlit run src/streamlit_chat.py

Environment Variables:
    COORDINATOR_API_URL: Backend API URL (default: http://127.0.0.1:3001)
    OCI_APM_ENDPOINT: OCI APM endpoint for tracing
    OCI_APM_PRIVATE_DATA_KEY: OCI APM data key
"""

import os
import time
import uuid
from datetime import datetime, timezone

import httpx
import streamlit as st
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from src.observability.tracing import (
    init_tracing,
    shutdown_tracing,
    truncate,
)

# Configuration
COORDINATOR_API_URL = os.getenv("COORDINATOR_API_URL", "http://127.0.0.1:3001")

# GenAI Semantic Convention attribute keys
GENAI_REQUEST_MODEL = "gen_ai.request.model"
GENAI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GENAI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GENAI_RESPONSE_MODEL = "gen_ai.response.model"
GENAI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GENAI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GENAI_OPERATION_NAME = "gen_ai.operation.name"
GENAI_SYSTEM = "gen_ai.system"


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"streamlit-user-{uuid.uuid4().hex[:8]}"
    if "tracer" not in st.session_state:
        st.session_state.tracer = init_tracing(component="streamlit-chat")
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0


def get_tracer() -> trace.Tracer:
    """Get the initialized tracer."""
    return st.session_state.tracer


def send_chat_message(message: str) -> dict:
    """Send message to coordinator API with full tracing.

    Args:
        message: User message to send

    Returns:
        Response dictionary from the API
    """
    tracer = get_tracer()

    # Create root span for the chat request
    with tracer.start_as_current_span(
        "chat.request",
        kind=SpanKind.CLIENT,
    ) as span:
        # Set request attributes
        span.set_attribute("user.id", st.session_state.user_id)
        span.set_attribute("session.thread_id", st.session_state.thread_id)
        span.set_attribute("channel", "streamlit")
        span.set_attribute("message.length", len(message))
        span.set_attribute(GENAI_OPERATION_NAME, "chat")
        span.set_attribute(GENAI_SYSTEM, "oci-coordinator")

        # Track request timing
        start_time = time.time()

        try:
            # Prepare request payload
            payload = {
                "message": message,
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "channel": "streamlit",
                "metadata": {
                    "source": "streamlit-chat",
                    "request_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }

            # Create child span for HTTP call
            with tracer.start_as_current_span(
                "http.post",
                kind=SpanKind.CLIENT,
            ) as http_span:
                http_span.set_attribute("http.method", "POST")
                http_span.set_attribute("http.url", f"{COORDINATOR_API_URL}/chat")
                http_span.set_attribute("http.request.body.size", len(str(payload)))

                # Make API call
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        f"{COORDINATOR_API_URL}/chat",
                        json=payload,
                    )

                http_span.set_attribute("http.status_code", response.status_code)
                http_span.set_attribute("http.response.body.size", len(response.content))

                if response.status_code != 200:
                    http_span.set_status(Status(StatusCode.ERROR))
                    http_span.set_attribute("error.type", "http_error")
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )

            # Parse response
            result = response.json()

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Set response attributes
            span.set_attribute("response.length", len(result.get("response", "")))
            span.set_attribute("response.duration_ms", duration_ms)
            span.set_attribute("response.agent", result.get("agent", "unknown"))
            span.set_attribute("response.content_type", result.get("content_type", "text"))

            # Set tool usage
            tools_used = result.get("tools_used", [])
            if tools_used:
                span.set_attribute("tools.count", len(tools_used))
                span.set_attribute("tools.names", ",".join(tools_used[:10]))  # Limit to 10

            # Set token usage if available (estimated)
            input_tokens = len(message.split()) * 1.3  # Rough estimate
            output_tokens = len(result.get("response", "").split()) * 1.3
            span.set_attribute(GENAI_USAGE_INPUT_TOKENS, int(input_tokens))
            span.set_attribute(GENAI_USAGE_OUTPUT_TOKENS, int(output_tokens))
            span.set_attribute("gen_ai.usage.total_tokens", int(input_tokens + output_tokens))

            # Update session metrics
            st.session_state.total_tokens += int(input_tokens + output_tokens)
            st.session_state.request_count += 1

            # Set success status
            span.set_status(Status(StatusCode.OK))

            return result

        except httpx.ConnectError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", "connection_error")
            span.set_attribute("error.message", truncate(str(e)))
            return {
                "response": f"Connection error: Could not reach coordinator at {COORDINATOR_API_URL}",
                "thread_id": st.session_state.thread_id,
                "agent": None,
                "tools_used": [],
                "duration_ms": int((time.time() - start_time) * 1000),
                "content_type": "error",
            }

        except httpx.HTTPStatusError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", "http_error")
            span.set_attribute("error.message", truncate(str(e)))
            return {
                "response": f"API error: {e}",
                "thread_id": st.session_state.thread_id,
                "agent": None,
                "tools_used": [],
                "duration_ms": int((time.time() - start_time) * 1000),
                "content_type": "error",
            }

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", truncate(str(e)))
            return {
                "response": f"Error: {e}",
                "thread_id": st.session_state.thread_id,
                "agent": None,
                "tools_used": [],
                "duration_ms": int((time.time() - start_time) * 1000),
                "content_type": "error",
            }


def render_message(role: str, content: str, metadata: dict | None = None):
    """Render a chat message with optional metadata.

    Args:
        role: Message role (user/assistant)
        content: Message content
        metadata: Optional metadata (agent, tools, duration)
    """
    with st.chat_message(role):
        st.markdown(content)

        if metadata and role == "assistant":
            cols = st.columns(3)
            if metadata.get("agent"):
                cols[0].caption(f"Agent: {metadata['agent']}")
            if metadata.get("duration_ms"):
                cols[1].caption(f"Duration: {metadata['duration_ms']}ms")
            if metadata.get("tools_used"):
                cols[2].caption(f"Tools: {len(metadata['tools_used'])}")


def render_sidebar():
    """Render the sidebar with session info and controls."""
    with st.sidebar:
        st.header("Session Info")

        # Session metrics
        st.metric("Thread ID", st.session_state.thread_id[:8] + "...")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Total Tokens (est.)", st.session_state.total_tokens)
        st.metric("Requests", st.session_state.request_count)

        st.divider()

        # Configuration
        st.subheader("Configuration")
        st.caption(f"API: {COORDINATOR_API_URL}")

        # Tracing status
        tracer = get_tracer()
        if tracer:
            st.success("OCI APM: Connected")
        else:
            st.warning("OCI APM: Not configured")

        st.divider()

        # Actions
        if st.button("Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.total_tokens = 0
            st.session_state.request_count = 0
            st.rerun()

        if st.button("New Thread", type="primary"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="OCI Coordinator Chat",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session
    init_session_state()

    # Header
    st.title("OCI Coordinator Chat")
    st.caption("Chat interface with full OCI APM tracing")

    # Render sidebar
    render_sidebar()

    # Display chat history
    for msg in st.session_state.messages:
        render_message(
            msg["role"],
            msg["content"],
            msg.get("metadata"),
        )

    # Chat input
    if prompt := st.chat_input("Ask about your OCI infrastructure..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        # Display user message
        render_message("user", prompt)

        # Send to API and get response
        with st.spinner("Processing..."):
            result = send_chat_message(prompt)

        # Add assistant response to history
        assistant_msg = {
            "role": "assistant",
            "content": result["response"],
            "metadata": {
                "agent": result.get("agent"),
                "tools_used": result.get("tools_used", []),
                "duration_ms": result.get("duration_ms"),
                "content_type": result.get("content_type"),
            },
        }
        st.session_state.messages.append(assistant_msg)

        # Display assistant response
        render_message(
            "assistant",
            result["response"],
            assistant_msg["metadata"],
        )

        # Handle structured data (tables, etc.)
        if result.get("content_type") == "table" and result.get("structured_data"):
            with st.expander("View Table Data"):
                st.json(result["structured_data"])


# Cleanup on session end (best effort)
def cleanup():
    """Cleanup tracing on shutdown."""
    try:
        shutdown_tracing()
    except Exception:
        pass


if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)
    main()
