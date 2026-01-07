"""
Streamlit chat integration helpers.

Provides lightweight helpers for rendering StructuredResponse objects
in Streamlit chat UIs without hard dependencies on Streamlit at import time.
"""

from __future__ import annotations

from typing import Any

from src.formatting import FormatterRegistry, OutputFormat, StructuredResponse


def format_streamlit_message(response: StructuredResponse) -> dict[str, Any]:
    """Format a StructuredResponse into a Streamlit-friendly payload."""
    markdown = FormatterRegistry.format(response, OutputFormat.MARKDOWN)
    return {
        "markdown": markdown,
        "raw_data": response.raw_data,
        "attachments": response.attachments,
    }


def render_streamlit_response(response: StructuredResponse, role: str = "assistant") -> None:
    """Render a StructuredResponse inside a Streamlit chat container."""
    try:
        import streamlit as st
    except Exception as exc:
        raise RuntimeError("streamlit is not installed; run `pip install streamlit`.") from exc

    payload = format_streamlit_message(response)
    if hasattr(st, "chat_message"):
        with st.chat_message(role):
            st.markdown(payload["markdown"])
    else:
        st.markdown(payload["markdown"])
