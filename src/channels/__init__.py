"""Input channels for the OCI AI Agent Coordinator.

Provides integrations with various messaging platforms (Slack, Teams, Web, API)
for receiving user requests and sending agent responses.
"""

from src.channels.slack import SlackHandler, create_slack_app
from src.channels.streamlit import format_streamlit_message, render_streamlit_response

__all__ = [
    "SlackHandler",
    "create_slack_app",
    "format_streamlit_message",
    "render_streamlit_response",
]
