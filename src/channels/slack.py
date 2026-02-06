"""Slack integration for OCI AI Agent Coordinator.

Handles incoming Slack messages and routes them to the coordinator
for processing by specialized agents.

Phase 4 Enhancement:
- Uses shared AsyncRuntime instead of asyncio.run() per handler
- Prevents MCP connection issues from event loop recreation
- Maintains single event loop for all async operations

Usage:
    from src.channels.slack import create_slack_app

    # Create and start the Slack app
    app = create_slack_app()
    app.start(port=3000)  # Or use Socket Mode
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable
from typing import Any

import structlog
from opentelemetry import trace

from src.channels.async_runtime import run_async
from src.channels.conversation import get_conversation_manager
from src.channels.slack_catalog import (
    Category,
    build_catalog_blocks,
    build_database_name_modal,
    build_error_recovery_blocks,
    build_follow_up_blocks,
    get_follow_up_suggestions,
    needs_database_name_prompt,
)
from src.formatting.parser import ResponseParser
from src.formatting.slack import SlackFormatter
from src.observability import get_trace_id, init_observability
from src.observability.tracing import get_tracer
from src.oci.profile_manager import ProfileManager

logger = structlog.get_logger(__name__)

# Pending authentication context storage
# Stores user/channel/query info when auth is requested so we can notify them when auth succeeds
_pending_auth_contexts: dict[str, dict] = {}

# Module-level reference to SlackHandler instance for auth callback resumption
_slack_handler_instance: SlackHandler | None = None


def set_slack_handler_instance(handler: SlackHandler) -> None:
    """Store global reference to SlackHandler for auth callback resumption.

    Called during SlackHandler initialization to enable `notify_auth_success()`
    to resume pending requests after authentication completes.
    """
    global _slack_handler_instance
    _slack_handler_instance = handler
    logger.info("SlackHandler instance registered for auth callback", handler_id=id(handler))


def _has_llm_channel_overrides(channel_type: str) -> bool:
    """Return True if channel-specific LLM overrides are set in the environment."""
    suffix = channel_type.upper()
    keys = [
        f"LLM_PROVIDER_{suffix}",
        f"LLM_MODEL_{suffix}",
        f"LLM_TEMPERATURE_{suffix}",
        f"LLM_MAX_TOKENS_{suffix}",
        f"LLM_BASE_URL_{suffix}",
        f"OCA_MODEL_{suffix}",
        f"ANTHROPIC_MODEL_{suffix}",
        f"OPENAI_MODEL_{suffix}",
        f"OPENAI_BASE_URL_{suffix}",
        f"LLM_API_KEY_{suffix}",
        f"OPENAI_API_KEY_{suffix}",
        f"ANTHROPIC_API_KEY_{suffix}",
    ]
    return any(os.getenv(key) for key in keys)


def store_pending_auth_context(user_id: str, channel_id: str, thread_ts: str | None, query: str) -> None:
    """Store pending auth context for a user.

    Called when showing auth URL to user. Context is retrieved when auth succeeds
    to send a follow-up notification.

    Args:
        user_id: Slack user ID
        channel_id: Channel where auth was requested
        thread_ts: Thread timestamp (if in thread)
        query: Original query that triggered auth requirement
    """
    _pending_auth_contexts[user_id] = {
        "channel_id": channel_id,
        "thread_ts": thread_ts,
        "query": query,
        "timestamp": __import__("time").time(),
    }
    logger.debug("Stored pending auth context", user_id=user_id, channel=channel_id)


def get_pending_auth_context(user_id: str) -> dict | None:
    """Get and remove pending auth context for a user.

    Args:
        user_id: Slack user ID

    Returns:
        Context dict or None if no pending context
    """
    return _pending_auth_contexts.pop(user_id, None)


def get_all_pending_auth_contexts() -> dict[str, dict]:
    """Get all pending auth contexts (for broadcast notification).

    Returns:
        Dict of user_id -> context
    """
    return _pending_auth_contexts.copy()


def clear_pending_auth_context(user_id: str) -> None:
    """Clear a pending auth context after processing.

    Args:
        user_id: Slack user ID
    """
    _pending_auth_contexts.pop(user_id, None)


async def _resume_pending_request_async(
    user_id: str,
    channel_id: str,
    thread_ts: str | None,
    query: str,
    bot_token: str,
) -> None:
    """Resume a pending request after authentication succeeds.

    Invokes the coordinator to process the original query and sends
    the response back to the channel/thread.

    Args:
        user_id: Slack user ID
        channel_id: Channel where request originated
        thread_ts: Thread timestamp
        query: Original query text
        bot_token: Slack bot token
    """
    from slack_sdk.web.async_client import AsyncWebClient

    from src.formatting.parser import ResponseParser
    from src.formatting.slack import SlackFormatter

    client = AsyncWebClient(token=bot_token)

    try:
        # Send "processing" indicator
        thinking_response = await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=":hourglass_flowing_sand: Processing your request...",
            blocks=[
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": ":hourglass_flowing_sand: *Processing your request...* | Resuming after authentication",
                        }
                    ],
                }
            ],
        )
        thinking_ts = thinking_response.get("ts")

        # Check if we have a SlackHandler instance with coordinator
        if _slack_handler_instance is None:
            logger.warning("No SlackHandler instance available for resuming request")
            await client.chat_update(
                channel=channel_id,
                ts=thinking_ts,
                text="Unable to process request - please try again.",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":warning: Unable to process request automatically. Please send your request again.",
                        }
                    }
                ],
            )
            return

        # Invoke the coordinator through SlackHandler
        response = await _slack_handler_instance._invoke_coordinator(
            text=query,
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
        )

        # Delete the thinking message
        try:
            await client.chat_delete(channel=channel_id, ts=thinking_ts)
        except Exception:
            pass  # Ignore if deletion fails

        if response and response.get("type") == "agent_response":
            msg_text = response.get("message", "")

            # Format response for Slack
            parser = ResponseParser()
            formatter = SlackFormatter()
            parse_result = parser.parse(msg_text)
            formatted = formatter.format_response(parse_result.response)
            blocks = formatted.get("blocks", [])
            fallback_text = msg_text[:300] if msg_text else "Response processed"

            # Add thinking summary if available
            all_blocks = []
            thinking_summary = response.get("thinking_summary")
            if thinking_summary:
                all_blocks.append({
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f":brain: {thinking_summary}",
                    }]
                })

            if blocks:
                all_blocks.extend(blocks)

            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=fallback_text,
                blocks=all_blocks if all_blocks else None,
            )

            logger.info(
                "Successfully resumed pending request",
                user_id=user_id,
                channel=channel_id,
                agent_id=response.get("agent_id"),
            )
        else:
            # No valid response
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="I couldn't process that request. Please try again.",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":x: I couldn't process your request. Please try sending it again.",
                        }
                    }
                ],
            )

    except Exception as e:
        logger.error(
            "Failed to resume pending request",
            user_id=user_id,
            channel=channel_id,
            error=str(e),
        )
        # Try to notify user of failure
        try:
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f"Error processing request: {str(e)[:100]}",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f":x: Error processing your request. Please try again.\n`{str(e)[:100]}`",
                        }
                    }
                ],
            )
        except Exception:
            pass


def notify_auth_success(bot_token: str | None = None) -> None:
    """Send Slack notification to all users with pending auth contexts and resume their requests.

    Called by the OCA callback server when authentication succeeds.
    Notifies users they're authenticated and automatically continues processing
    their original request.

    Args:
        bot_token: Slack bot token (uses env var if not provided)
    """
    import time

    from src.channels.async_runtime import run_async

    token = bot_token or os.getenv("SLACK_BOT_TOKEN")
    if not token:
        logger.warning("Cannot notify auth success: no bot token")
        return

    contexts = get_all_pending_auth_contexts()
    if not contexts:
        logger.debug("No pending auth contexts to notify")
        return

    try:
        from slack_sdk import WebClient
        client = WebClient(token=token)

        for user_id, context in contexts.items():
            channel_id = context.get("channel_id")
            thread_ts = context.get("thread_ts")
            query = context.get("query", "")
            timestamp = context.get("timestamp", 0)

            # Skip if context is too old (> 30 minutes)
            if time.time() - timestamp > 1800:
                logger.debug("Skipping stale auth context", user_id=user_id)
                clear_pending_auth_context(user_id)
                continue

            try:
                # Build auth success message
                blocks = [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":white_check_mark: *Authentication successful!*\n\nYou're now logged in. Processing your request...",
                        }
                    },
                ]

                # Add original query context if available
                if query:
                    blocks.append({
                        "type": "context",
                        "elements": [{
                            "type": "mrkdwn",
                            "text": f":speech_balloon: _Your request:_ \"{query[:100]}{'...' if len(query) > 100 else ''}\""
                        }]
                    })

                # Send notification to channel/thread
                client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="Authentication successful! Processing your request...",
                    blocks=blocks,
                )
                logger.info("Sent auth success notification", user_id=user_id, channel=channel_id)

                # Clear context BEFORE resuming to prevent double-processing
                clear_pending_auth_context(user_id)

                # Actually resume the pending request
                logger.info(
                    "Attempting to resume pending request",
                    user_id=user_id,
                    has_query=bool(query),
                    query_preview=query[:50] if query else None,
                    has_handler=_slack_handler_instance is not None,
                )
                if query and _slack_handler_instance is not None:
                    try:
                        run_async(
                            _resume_pending_request_async(
                                user_id=user_id,
                                channel_id=channel_id,
                                thread_ts=thread_ts,
                                query=query,
                                bot_token=token,
                            ),
                            timeout=300,  # 5 minute timeout
                        )
                        logger.info("Resumed pending request after auth", user_id=user_id)
                    except Exception as resume_error:
                        logger.error(
                            "Failed to resume pending request",
                            user_id=user_id,
                            error=str(resume_error),
                        )
                        # Notify user of failure to resume
                        try:
                            client.chat_postMessage(
                                channel=channel_id,
                                thread_ts=thread_ts,
                                text="Failed to continue with your request automatically. Please send it again.",
                                blocks=[{
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": ":warning: I couldn't automatically continue with your request. Please send it again.",
                                    }
                                }],
                            )
                        except Exception:
                            pass
                elif not query:
                    logger.debug("No query to resume for user", user_id=user_id)
                elif _slack_handler_instance is None:
                    logger.warning("No SlackHandler instance - cannot resume request", user_id=user_id)

            except Exception as e:
                logger.warning("Failed to notify user of auth success", user_id=user_id, error=str(e))
                clear_pending_auth_context(user_id)

    except ImportError:
        logger.error("slack_sdk not available for auth notification")
    except Exception as e:
        logger.error("Failed to send auth success notifications", error=str(e))


def build_profile_selection_blocks(profiles: list, current_profile: str | None = None) -> list[dict]:
    """Build Slack blocks for OCI profile selection.

    Args:
        profiles: List of ProfileInfo objects
        current_profile: Currently active profile name

    Returns:
        Slack Block Kit blocks for profile selection
    """
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":gear: Select OCI Profile",
                "emoji": True,
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Multiple OCI profiles are available. Please select which tenancy you want to work with:",
            }
        },
    ]

    # Build profile buttons (max 5 per actions block)
    profile_buttons = []
    for profile in profiles:
        label = profile.display_name or profile.name
        style = "primary" if profile.name == current_profile else None

        button = {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": f":cloud: {label}" if profile.name != current_profile else f":white_check_mark: {label}",
                "emoji": True,
            },
            "action_id": f"select_profile_{profile.name}",
            "value": profile.name,
        }
        if style:
            button["style"] = style
        profile_buttons.append(button)

    if profile_buttons:
        blocks.append({
            "type": "actions",
            "elements": profile_buttons[:5],
        })

    # Add profile details
    blocks.append({"type": "divider"})
    for profile in profiles:
        emoji = ":white_check_mark:" if profile.name == current_profile else ":cloud:"
        region_display = profile.region.replace("-", " ").title() if profile.region else "Unknown"
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"{emoji} *{profile.display_name or profile.name}*: `{region_display}` | `{profile.tenancy_ocid[:35]}...`",
            }]
        })

    return blocks


def build_profile_indicator_block(profile_name: str, profile_region: str) -> dict:
    """Build a compact profile indicator context block.

    Args:
        profile_name: Active profile name
        profile_region: Profile region

    Returns:
        Slack context block showing current profile
    """
    region_display = profile_region.replace("-", " ").title() if profile_region else "Unknown"
    return {
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": f":cloud: *Profile:* {profile_name} ({region_display}) | Type `profile` to change",
        }]
    }


def get_oca_auth_url() -> str:
    """Generate OCA OAuth authorization URL."""
    import base64
    import hashlib
    import secrets
    from pathlib import Path
    from urllib.parse import urlencode

    IDCS_CLIENT_ID = "a8331954c0cf48ba99b5dd223a14c6ea"
    IDCS_OAUTH_URL = "https://idcs-9dc693e80d9b469480d7afe00e743931.identity.oraclecloud.com"
    CACHE_DIR = Path(os.getenv("OCA_CACHE_DIR", Path.home() / ".oca"))

    callback_host = os.getenv("OCA_CALLBACK_HOST", "127.0.0.1")
    callback_port = os.getenv("OCA_CALLBACK_PORT", "48801")
    redirect_uri = f"http://{callback_host}:{callback_port}/auth/oca"

    # Generate PKCE verifier and challenge
    verifier = secrets.token_urlsafe(40)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")

    # Generate state and nonce for CSRF protection and OpenID Connect
    # IDCS requires these for proper OAuth/OIDC flows
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)

    # Save verifier and state for later use
    CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    verifier_path = CACHE_DIR / "verifier.txt"
    verifier_path.write_text(verifier)
    verifier_path.chmod(0o600)

    # Save state for CSRF verification on callback
    state_path = CACHE_DIR / "state.txt"
    state_path.write_text(state)
    state_path.chmod(0o600)

    params = {
        "response_type": "code",
        "client_id": IDCS_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "openid offline_access",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "nonce": nonce,
    }

    return f"{IDCS_OAUTH_URL}/oauth2/v1/authorize?{urlencode(params)}"

# Slack formatter for response conversion
_slack_formatter = SlackFormatter()
_response_parser = ResponseParser()


def build_welcome_blocks() -> list[dict]:
    """Build welcome message blocks using Block Kit."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Hey there :wave: I'm *OCI Coordinator*. I help you manage and troubleshoot Oracle Cloud Infrastructure resources.\nHere are some ways I can help:"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*1️⃣ Database Troubleshooting*\nAnalyze AWR reports, identify slow queries, and troubleshoot database performance issues."
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*2️⃣ Infrastructure Management*\nManage compute instances, networks, and other OCI resources."
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*3️⃣ Cost Analysis*\nAnalyze spending, track budgets, and get cost optimization recommendations."
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*4️⃣ Security & Compliance*\nDetect threats, audit security configurations, and ensure compliance."
            }
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":bulb: *Tip:* Mention me in a channel or send a DM to get started!\n:question: Type `help` to see available commands"
                }
            ]
        }
    ]


def build_help_blocks() -> list[dict]:
    """Build help message blocks using Block Kit with catalog integration."""
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":book: OCI Coordinator Help",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "I help you manage and troubleshoot Oracle Cloud Infrastructure. Ask questions naturally or use the catalog for guided troubleshooting."
            }
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:rocket: Quick Start*"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": ":books: Open Catalog", "emoji": True},
                    "action_id": "show_catalog",
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": ":moneybag: Show Costs", "emoji": True},
                    "action_id": "quick_cost",
                    "value": "show costs",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": ":file_folder: Compartments", "emoji": True},
                    "action_id": "quick_compartments",
                    "value": "list compartments",
                },
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:database: Database*\n• `database health check` - Quick health status\n• `show slow queries` - Find problematic queries\n• `analyze database performance` - Deep metrics"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:cloud: Infrastructure*\n• `list instances` - Compute inventory\n• `list vcns` - Network topology\n• `tenancy info` - Tenancy details"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:moneybag: Cost & FinOps*\n• `show costs` - Current spending\n• `cost by service` - Breakdown\n• `find optimization opportunities`"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:shield: Security*\n• `security overview` - Cloud Guard status\n• `show threats` - Active detections\n• `list users` - IAM users"
            }
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":bulb: Type `catalog` for interactive troubleshooting | :zap: Powered by OCA"
                }
            ]
        }
    ]


def build_auth_required_blocks(auth_url: str) -> list[dict]:
    """Build authentication required message blocks."""
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":lock: Authentication Required",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "I need to authenticate with Oracle Code Assist (OCA) to process your request.\n\nClick the button below to log in with Oracle SSO:"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": ":key: Login with Oracle SSO",
                        "emoji": True
                    },
                    "url": auth_url,
                    "action_id": "oca_login_button",
                    "style": "primary"
                }
            ]
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":white_check_mark: OAuth callback server is running automatically"
                }
            ]
        }
    ]


def build_response_blocks(
    title: str,
    content: str,
    agent_name: str | None = None,
    status: str = "success",
) -> list[dict]:
    """Build a formatted response using Block Kit."""
    emoji = ":white_check_mark:" if status == "success" else ":x:" if status == "error" else ":hourglass:"

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {title}",
                "emoji": True
            }
        }
    ]

    # Add agent context if provided
    if agent_name:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":robot_face: _Agent: {agent_name}_"
                }
            ]
        })

    # Add main content - split into sections if long
    content_chunks = _split_text_safely(content, 2800)

    # We can send up to 50 blocks in a message
    for chunk in content_chunks[:40]:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": chunk
            }
        })

    if len(content_chunks) > 40:
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": ":warning: _Response truncated due to extreme length._"
            }]
        })

    return blocks

def _split_text_safely(text: str, limit: int) -> list[str]:
    """
    Split text into chunks that respect newlines and word boundaries.
    
    Args:
        text: absolute text content
        limit: max chars per chunk
        
    Returns:
        List of text chunks
    """
    chunks = []
    current_chunk = []
    current_length = 0

    # Pre-split by existing newlines to preserve paragraph structure
    # filtering empty lines might change format, so be careful.
    # splitlines(keepends=True) keeps the \n which is useful.
    lines = text.splitlines(keepends=True)

    for line in lines:
        line_len = len(line)

        # If adding this line exceeds limit
        if current_length + line_len > limit:
            # If the current chunk has content, save it
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_length = 0

            # If the line itself is bigger than limit, we must force split it
            if line_len > limit:
                # Split by space
                words = line.split(' ')
                temp_line = []
                temp_len = 0
                for word in words:
                    # restore space that split removed (except last one?)
                    # simpler: just strictly slice if word is huge, or build up.
                    # let's try strict slice for massive lines if normal word split fails
                    w_len = len(word) + 1 # +1 for space

                    if temp_len + w_len > limit:
                        if temp_line:
                            chunks.append(" ".join(temp_line))
                            temp_line = []
                            temp_len = 0

                        # If single word is huge
                        if len(word) > limit:
                            # Force slice
                            for i in range(0, len(word), limit):
                                chunks.append(word[i:i+limit])
                        else:
                            temp_line.append(word)
                            temp_len += w_len
                    else:
                        temp_line.append(word)
                        temp_len += w_len

                if temp_line:
                    # Add remainder of the split line to start of next chunk
                    current_chunk = [" ".join(temp_line)]
                    current_length = len(current_chunk[0])
            else:
                # Line fits in a new chunk
                current_chunk.append(line)
                current_length += line_len
        else:
            # Line fits in current chunk
            current_chunk.append(line)
            current_length += line_len

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

    return blocks


def build_error_blocks(error: str, suggestion: str | None = None) -> list[dict]:
    """Build error message blocks."""
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":x: Error",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"```{error[:2900]}```"
            }
        }
    ]

    if suggestion:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":bulb: *Suggestion:* {suggestion}"
                }
            ]
        })

    return blocks


class SlackHandler:
    """Handler for Slack events and message processing.

    Integrates with Slack Bolt for event handling and the coordinator
    for agent routing.

    Supports both sync and async modes:
    - Sync mode: Uses regular `App` with `SocketModeHandler` (blocking)
    - Async mode: Uses `AsyncApp` with `AsyncSocketModeHandler` (non-blocking)
    """

    def __init__(
        self,
        bot_token: str | None = None,
        app_token: str | None = None,
        signing_secret: str | None = None,
    ):
        """Initialize Slack handler.

        Args:
            bot_token: Slack bot token (xoxb-...)
            app_token: Slack app token for Socket Mode (xapp-...)
            signing_secret: Slack signing secret for request verification
        """
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.app_token = app_token or os.getenv("SLACK_APP_TOKEN")
        self.signing_secret = signing_secret or os.getenv("SLACK_SIGNING_SECRET")

        self._app = None  # Sync app (lazy init)
        self._async_app = None  # Async app (lazy init)
        self._coordinator = None
        self._conversation_manager = get_conversation_manager()
        self._bot_user_id: str | None = None
        self._recent_event_ids: dict[str, float] = {}
        self._event_dedupe_ttl = float(os.getenv("SLACK_EVENT_DEDUP_TTL_SECONDS", "120"))

        # Ensure observability is initialized (safe to call multiple times -
        # init_tracing() has a guard that returns existing tracer if already initialized)
        # The coordinator typically initializes first in main.py, so this is usually a no-op.
        init_observability(agent_name="slack-handler")

        # Get tracer for slack handler - uses coordinator's TracerProvider service name
        # but tracer name helps identify spans in code. Use our get_tracer helper.
        self._tracer = get_tracer("slack-handler")

        # Register this instance globally for auth callback resumption
        set_slack_handler_instance(self)

    def _is_duplicate_event(self, event: dict) -> bool:
        """Return True if the event was recently processed."""
        event_key = event.get("client_msg_id") or event.get("event_ts") or event.get("ts")
        if not event_key:
            return False
        now = time.time()
        cutoff = now - self._event_dedupe_ttl
        for key, ts in list(self._recent_event_ids.items()):
            if ts < cutoff:
                self._recent_event_ids.pop(key, None)
        if event_key in self._recent_event_ids:
            return True
        self._recent_event_ids[event_key] = now
        return False

    def _is_bot_mention(self, text: str) -> bool:
        """Check if the message explicitly mentions this bot."""
        if not text or not self._bot_user_id:
            return False
        return f"<@{self._bot_user_id}>" in text

    @property
    def app(self):
        """Get the sync Slack Bolt app instance."""
        if self._app is None:
            self._app = self._create_app()
        return self._app

    @property
    def async_app(self):
        """Get the async Slack Bolt app instance."""
        if self._async_app is None:
            self._async_app = self._create_async_app()
        return self._async_app

    async def _safe_client_call(self, method, max_retries: int = 3, **kwargs):
        """Safely call a Slack client method with retry logic.

        When running in sync mode through run_async(), the client is still
        the sync WebClient which returns SlackResponse directly (not a coroutine).
        This helper handles both cases properly with exponential backoff retry.

        Args:
            method: The client method to call (e.g., client.chat_postMessage)
            max_retries: Maximum retry attempts (default: 3)
            **kwargs: Arguments to pass to the method

        Returns:
            The result (SlackResponse) from the client method
        """
        import asyncio
        import inspect

        from slack_sdk.errors import SlackApiError

        last_error = None
        for attempt in range(max_retries):
            try:
                result = method(**kwargs)
                # If the result is a coroutine (async client), await it
                if inspect.iscoroutine(result):
                    result = await result
                if hasattr(result, "get") and result.get("ok") is False:
                    error = result.get("error", "unknown_error")
                    raise RuntimeError(f"Slack API error: {error}")
                # Otherwise (sync client), return directly
                return result
            except SlackApiError as e:
                last_error = e
                if e.response.get("error") == "ratelimited":
                    # Get retry-after header, default to exponential backoff
                    retry_after = int(e.response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(
                        "Slack rate limited, waiting",
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                else:
                    raise
            except TimeoutError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Slack API timeout, retrying",
                        wait_time=wait_time,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

        if last_error:
            raise last_error

    def _build_thinking_blocks(self, steps: list) -> list[dict]:
        """Build Slack blocks for thinking progress display.

        Args:
            steps: List of ThinkingStep objects

        Returns:
            Slack Block Kit blocks for the thinking progress
        """
        from src.agents.coordinator.transparency import PHASE_EMOJIS

        if not steps:
            return [{
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": ":hourglass_flowing_sand: *Processing your request...*",
                }]
            }]

        # Build step text with phase emojis
        step_texts = []
        for step in steps[-5:]:  # Show last 5 steps
            emoji = PHASE_EMOJIS.get(step.phase, ":gear:")
            step_texts.append(f"{emoji} {step.message}")

        blocks = [{
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": ":brain: *Thinking Process*",
            }]
        }]

        if step_texts:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(step_texts),
                }
            })

        return blocks

    async def _safe_say(self, say, max_retries: int = 3, **kwargs):
        """Safely call the Slack say function with retry logic.

        When running in sync mode through run_async(), the say function
        returns SlackResponse directly. In async mode, it returns a coroutine.
        This helper handles both cases with exponential backoff retry.

        Args:
            say: The say function from Slack Bolt
            max_retries: Maximum retry attempts (default: 3)
            **kwargs: Arguments to pass to say (text, blocks, thread_ts, etc.)

        Returns:
            The result from say
        """
        import asyncio
        import inspect

        from slack_sdk.errors import SlackApiError

        last_error = None
        for attempt in range(max_retries):
            try:
                result = say(**kwargs)
                # If the result is a coroutine (async mode), await it
                if inspect.iscoroutine(result):
                    result = await result
                if hasattr(result, "get") and result.get("ok") is False:
                    error = result.get("error", "unknown_error")
                    raise RuntimeError(f"Slack say failed: {error}")
                return result
            except SlackApiError as e:
                last_error = e
                if e.response.get("error") == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(
                        "Slack say rate limited, waiting",
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                else:
                    raise
            except TimeoutError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Slack say timeout, retrying",
                        wait_time=wait_time,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                # Catch-all for unexpected errors
                logger.error("Unexpected error in _safe_say", error=str(e), error_type=type(e).__name__)
                raise

        if last_error:
            raise last_error

    async def _send_plain_text_fallback(
        self,
        client: Any,
        channel: str,
        thread_ts: str | None,
        text: str,
    ) -> None:
        """Send a minimal plain-text response if Block Kit send fails."""
        try:
            await self._safe_client_call(
                client.chat_postMessage,
                channel=channel,
                thread_ts=thread_ts,
                text=text[:2900],
            )
        except Exception as e:
            logger.error("Slack fallback send failed", error=str(e))

    async def _safe_say_with_fallback(
        self,
        say: Callable,
        client: Any,
        channel: str,
        thread_ts: str | None,
        text: str,
        blocks: list[dict] | None = None,
    ) -> None:
        """Send response with Block Kit, fallback to plain text on failure."""
        try:
            await self._safe_say(
                say,
                text=text,
                blocks=blocks,
                thread_ts=thread_ts,
            )
        except Exception as e:
            logger.error("Slack say failed, falling back", error=str(e))
            await self._send_plain_text_fallback(
                client=client,
                channel=channel,
                thread_ts=thread_ts,
                text=text,
            )

    def _create_app(self):
        """Create and configure the sync Slack Bolt app."""
        try:
            from slack_bolt import App

            app = App(
                token=self.bot_token,
                signing_secret=self.signing_secret,
            )

            # Register event handlers (sync versions)
            self._register_handlers(app)

            logger.info("Slack sync app created successfully")
            return app

        except ImportError:
            logger.error(
                "slack_bolt not installed",
                help="Run: poetry add slack-bolt",
            )
            raise

    def _create_async_app(self):
        """Create and configure the async Slack Bolt app.

        Uses AsyncApp which is required for AsyncSocketModeHandler.
        """
        try:
            from slack_bolt.async_app import AsyncApp

            app = AsyncApp(
                token=self.bot_token,
                signing_secret=self.signing_secret,
            )

            # Register async event handlers
            self._register_async_handlers(app)

            logger.info("Slack async app created successfully")
            return app

        except ImportError:
            logger.error(
                "slack_bolt async not available",
                help="Ensure slack-bolt is up to date: poetry update slack-bolt",
            )
            raise

    def _register_handlers(self, app) -> None:
        """Register Slack event handlers.

        Uses shared AsyncRuntime instead of asyncio.run() to maintain
        a single event loop across all handlers. This prevents MCP
        connection issues that occur when creating new event loops.
        """
        # Ensure AsyncRuntime is started before registering handlers
        import time

        from src.channels.async_runtime import AsyncRuntime

        runtime = AsyncRuntime.get_instance()
        # Wait for loop to be fully started
        for _ in range(10):
            if runtime.is_running:
                break
            time.sleep(0.1)

        if not runtime.is_running:
            logger.error("Failed to start AsyncRuntime - Slack handlers may not work")
        else:
            logger.info("AsyncRuntime started successfully", loop_id=id(runtime.loop))

        @app.event("app_mention")
        def handle_mention(event: dict, say: Callable, client: Any) -> None:
            """Handle @mentions of the bot."""
            logger.info(
                ">>> APP_MENTION event received",
                user=event.get("user"),
                text_preview=event.get("text", "")[:50],
            )
            # Use shared async runtime instead of asyncio.run()
            run_async(self._process_message(event, say, client))

        @app.event("message")
        def handle_message(event: dict, say: Callable, client: Any) -> None:
            """Handle direct messages and thread replies."""
            text = event.get("text", "")
            channel_type = event.get("channel_type")
            thread_ts = event.get("thread_ts")
            channel = event.get("channel")
            is_dm = channel_type == "im" or (channel.startswith("D") if channel else False)
            logger.info(
                ">>> MESSAGE event received",
                user=event.get("user"),
                channel_type=channel_type,
                subtype=event.get("subtype"),
                bot_id=event.get("bot_id"),
                text_preview=text[:50] if text else None,
            )
            # Skip bot's own messages and message_changed events
            if event.get("bot_id") or event.get("subtype"):
                return

            # Handle DMs and thread replies
            # Note: Do NOT check _is_bot_mention here - app_mention handler already handles @mentions
            # Adding _is_bot_mention causes duplicate processing (both app_mention AND message fire)
            if is_dm or thread_ts:
                run_async(self._process_message(event, say, client))
            else:
                # Skip non-threaded channel messages - app_mention handles @mentions
                pass

        @app.action(re.compile(r"^agent_action_.*"))
        def handle_action(ack: Callable, body: dict, client: Any) -> None:
            """Handle button clicks and other interactions."""
            ack()
            run_async(self._process_action(body, client))

        @app.action("oca_login_button")
        def handle_oca_login(ack: Callable, body: dict, client: Any) -> None:
            """Handle OCA login button click (URL button - just acknowledge)."""
            ack()
            # The button has a URL, so clicking it opens the OAuth page
            # No additional action needed here
            logger.info("OCA login button clicked", user=body.get("user", {}).get("id"))

        @app.action("show_catalog")
        def handle_show_catalog(ack: Callable, body: dict, client: Any) -> None:
            """Handle catalog button click - show troubleshooting catalog."""
            ack()
            run_async(self._handle_catalog_action(body, client, None))

        @app.action(re.compile(r"^catalog_category_.*"))
        def handle_catalog_category(ack: Callable, body: dict, client: Any) -> None:
            """Handle catalog category selection."""
            ack()
            action = body.get("actions", [{}])[0]
            category = action.get("value")
            run_async(self._handle_catalog_action(body, client, category))

        @app.action("catalog_back")
        def handle_catalog_back(ack: Callable, body: dict, client: Any) -> None:
            """Handle catalog back button."""
            ack()
            run_async(self._handle_catalog_action(body, client, None))

        @app.action(re.compile(r"^catalog_action_.*"))
        def handle_catalog_quick_action(ack: Callable, body: dict, client: Any) -> None:
            """Handle quick action button from catalog."""
            ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            run_async(self._handle_quick_action(body, client, query))

        @app.action(re.compile(r"^runbook_action_.*"))
        def handle_runbook_action(ack: Callable, body: dict, client: Any) -> None:
            """Handle runbook action button."""
            ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            run_async(self._handle_quick_action(body, client, query))

        @app.action(re.compile(r"^follow_up_.*"))
        def handle_follow_up(ack: Callable, body: dict, client: Any) -> None:
            """Handle follow-up suggestion button."""
            ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            run_async(self._handle_quick_action(body, client, query))

        @app.action(re.compile(r"^recovery_.*"))
        def handle_recovery(ack: Callable, body: dict, client: Any) -> None:
            """Handle error recovery button."""
            ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            run_async(self._handle_quick_action(body, client, query))

        @app.action(re.compile(r"^quick_.*"))
        def handle_quick_start(ack: Callable, body: dict, client: Any) -> None:
            """Handle quick start buttons from help menu."""
            ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            run_async(self._handle_quick_action(body, client, query))

        @app.action(re.compile(r"^select_profile_.*"))
        def handle_profile_selection(ack: Callable, body: dict, client: Any) -> None:
            """Handle OCI profile selection button."""
            ack()
            action = body.get("actions", [{}])[0]
            profile = action.get("value", "")
            run_async(self._handle_profile_selection(body, client, profile))

        @app.action("show_profile_selector")
        def handle_show_profiles(ack: Callable, body: dict, client: Any) -> None:
            """Handle show profile selector button."""
            ack()
            run_async(self._handle_show_profile_selector(body, client))

        @app.command("/oci")
        def handle_command(ack: Callable, body: dict, respond: Callable) -> None:
            """Handle /oci slash command."""
            ack()
            run_async(self._process_command(body, respond))

        logger.info("Slack handlers registered with shared AsyncRuntime")

    def _register_async_handlers(self, app) -> None:
        """Register async Slack event handlers for AsyncApp.

        These handlers run natively in the async event loop,
        no need for AsyncRuntime bridge.
        """
        @app.event("app_mention")
        async def handle_mention(event: dict, say, client) -> None:
            """Handle @mentions of the bot."""
            logger.info(">>> APP_MENTION event received", user=event.get("user"), text_preview=event.get("text", "")[:50])
            await self._process_message(event, say, client)

        @app.event("message")
        async def handle_message(event: dict, say, client) -> None:
            """Handle direct messages and thread replies."""
            text = event.get("text", "")
            channel_type = event.get("channel_type")
            thread_ts = event.get("thread_ts")
            subtype = event.get("subtype")
            bot_id = event.get("bot_id")
            user = event.get("user")
            channel = event.get("channel")
            is_dm = channel_type == "im" or (channel.startswith("D") if channel else False)

            logger.info(">>> MESSAGE event received", user=user, channel_type=channel_type, subtype=subtype, bot_id=bot_id, text_preview=text[:50] if text else None)

            # Skip bot's own messages and message_changed events
            if bot_id or subtype:
                logger.debug(">>> MESSAGE skipped (bot or subtype)", bot_id=bot_id, subtype=subtype)
                return

            # Handle DMs and thread replies
            # Note: Do NOT check _is_bot_mention here - app_mention handler already handles @mentions
            # Adding _is_bot_mention causes duplicate processing (both app_mention AND message fire)
            if is_dm or thread_ts:
                await self._process_message(event, say, client)
            else:
                # Skip non-threaded channel messages - app_mention handles @mentions
                pass

        @app.action(re.compile(r"^agent_action_.*"))
        async def handle_action(ack, body: dict, client) -> None:
            """Handle button clicks and other interactions."""
            await ack()
            await self._process_action(body, client)

        @app.action("oca_login_button")
        async def handle_oca_login(ack, body: dict, client) -> None:
            """Handle OCA login button click (URL button - just acknowledge)."""
            await ack()
            # The button has a URL, so clicking it opens the OAuth page
            # No additional action needed here
            logger.info("OCA login button clicked", user=body.get("user", {}).get("id"))

        @app.action("show_catalog")
        async def handle_show_catalog(ack, body: dict, client) -> None:
            """Handle catalog button click - show troubleshooting catalog."""
            await ack()
            await self._handle_catalog_action(body, client, None)

        @app.action(re.compile(r"^catalog_category_.*"))
        async def handle_catalog_category(ack, body: dict, client) -> None:
            """Handle catalog category selection."""
            await ack()
            action = body.get("actions", [{}])[0]
            category = action.get("value")
            await self._handle_catalog_action(body, client, category)

        @app.action("catalog_back")
        async def handle_catalog_back(ack, body: dict, client) -> None:
            """Handle catalog back button."""
            await ack()
            await self._handle_catalog_action(body, client, None)

        @app.action(re.compile(r"^catalog_action_.*"))
        async def handle_catalog_quick_action(ack, body: dict, client) -> None:
            """Handle quick action button from catalog."""
            await ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            await self._handle_quick_action(body, client, query)

        @app.action(re.compile(r"^runbook_action_.*"))
        async def handle_runbook_action(ack, body: dict, client) -> None:
            """Handle runbook action button."""
            await ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            await self._handle_quick_action(body, client, query)

        @app.action(re.compile(r"^follow_up_.*"))
        async def handle_follow_up(ack, body: dict, client) -> None:
            """Handle follow-up suggestion button.

            If the follow-up query requires a database name (e.g., "Check database performance"),
            opens a modal to collect the database name. Otherwise, executes the query directly.
            """
            await ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")

            # Check if this follow-up needs a database name
            if needs_database_name_prompt(query):
                # Get context for the modal
                channel_id = body.get("channel", {}).get("id", "")
                message = body.get("message", {})
                thread_ts = message.get("thread_ts") or message.get("ts")
                trigger_id = body.get("trigger_id")

                if trigger_id:
                    # Open modal to collect database name
                    modal = build_database_name_modal(query, channel_id, thread_ts)
                    try:
                        await client.views_open(trigger_id=trigger_id, view=modal)
                        logger.info(
                            "Opened database name modal",
                            query=query,
                            channel=channel_id,
                        )
                        return
                    except Exception as e:
                        logger.error("Failed to open database name modal", error=str(e))
                        # Fall through to execute without database name

            await self._handle_quick_action(body, client, query)

        @app.action(re.compile(r"^recovery_.*"))
        async def handle_recovery(ack, body: dict, client) -> None:
            """Handle error recovery button."""
            await ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            await self._handle_quick_action(body, client, query)

        @app.action(re.compile(r"^quick_.*"))
        async def handle_quick_start(ack, body: dict, client) -> None:
            """Handle quick start buttons from help menu."""
            await ack()
            action = body.get("actions", [{}])[0]
            query = action.get("value", "")
            await self._handle_quick_action(body, client, query)

        @app.action(re.compile(r"^select_profile_.*"))
        async def handle_profile_selection(ack, body: dict, client) -> None:
            """Handle OCI profile selection button."""
            await ack()
            action = body.get("actions", [{}])[0]
            profile = action.get("value", "")
            await self._handle_profile_selection(body, client, profile)

        @app.action("show_profile_selector")
        async def handle_show_profiles(ack, body: dict, client) -> None:
            """Handle show profile selector button."""
            await ack()
            await self._handle_show_profile_selector(body, client)

        @app.command("/oci")
        async def handle_command(ack, body: dict, respond) -> None:
            """Handle /oci slash command."""
            await ack()
            await self._process_command(body, respond)

        @app.view("database_name_modal")
        async def handle_database_name_submission(ack, body: dict, client, view) -> None:
            """Handle database name modal submission.

            Extracts the database name from the modal, combines it with the original
            query, and executes the enhanced query.
            """
            import json  # Import at handler level for exception handler access

            logger.info(
                "Database name modal submission received",
                view_id=view.get("id"),
                callback_id=view.get("callback_id"),
            )
            await ack()

            try:
                # Extract database name from the input
                values = view.get("state", {}).get("values", {})
                logger.debug("Modal values", values=values)
                db_name_block = values.get("database_name_block", {})
                db_name_input = db_name_block.get("database_name_input", {})
                database_name = db_name_input.get("value", "").strip()

                if not database_name:
                    logger.warning("Database name modal submitted without a name")
                    return

                # Get original context from private_metadata
                private_metadata = view.get("private_metadata", "{}")
                context = json.loads(private_metadata)
                original_query = context.get("query", "")
                channel_id = context.get("channel_id", "")
                thread_ts = context.get("thread_ts")

                if not channel_id:
                    logger.error("No channel_id in modal private_metadata")
                    return

                # Combine the original query with the database name
                enhanced_query = f"{original_query} for {database_name}"

                logger.info(
                    "Processing database query from modal",
                    original_query=original_query,
                    database_name=database_name,
                    enhanced_query=enhanced_query,
                    channel=channel_id,
                )

                # Build a synthetic body for _handle_quick_action
                # This matches the structure expected from button click events
                user_info = body.get("user", {})
                synthetic_body = {
                    "channel": {"id": channel_id},
                    "message": {"thread_ts": thread_ts, "ts": thread_ts},
                    "user": user_info,
                }

                logger.info(
                    "Executing database query via _handle_quick_action",
                    channel=channel_id,
                    thread_ts=thread_ts,
                    user=user_info.get("id"),
                    query=enhanced_query,
                )

                # Process the enhanced query using the standard quick action handler
                await self._handle_quick_action(
                    body=synthetic_body,
                    client=client,
                    query=enhanced_query,
                )

                logger.info("Database query execution completed", query=enhanced_query)

            except Exception as e:
                logger.error("Error processing database name modal", error=str(e))
                # Try to notify the user of the error
                try:
                    channel_id = json.loads(view.get("private_metadata", "{}")).get("channel_id")
                    if channel_id:
                        await client.chat_postMessage(
                            channel=channel_id,
                            text=f":warning: Error processing request: {str(e)}",
                        )
                except Exception:
                    pass

        logger.info("Slack async handlers registered")

    async def _process_message(
        self,
        event: dict,
        say: Callable,
        client: Any,
    ) -> None:
        """Process an incoming message.

        Args:
            event: Slack event data
            say: Function to send a response
            client: Slack client
        """
        with self._tracer.start_as_current_span("slack_message") as span:
            user = event.get("user", "unknown")
            channel = event.get("channel")
            text = event.get("text", "")
            channel_type = event.get("channel_type")
            thread_ts = event.get("thread_ts")
            reply_in_thread = os.getenv("SLACK_REPLY_IN_THREAD", "true").lower() in ("1", "true", "yes")
            is_dm = channel_type == "im" or (channel.startswith("D") if channel else False)
            if is_dm:
                thread_ts = None
            elif thread_ts is None and reply_in_thread:
                thread_ts = event.get("ts")

            # Deduplicate events to prevent stacking messages
            if self._is_duplicate_event(event):
                logger.info("Ignoring duplicate event",
                           event_id=event.get("event_id"),
                           msg_id=event.get("client_msg_id"))
                return

            # Remove bot mention from text
            text = self._clean_message(text)
            text_lower = text.lower().strip()
            span.set_attribute("slack.user", user)
            span.set_attribute("slack.channel", channel)
            span.set_attribute("message.length", len(text))

            logger.info(
                "Processing Slack message",
                user=user,
                channel=channel,
                text_preview=text[:50],
                trace_id=get_trace_id(),
            )

            try:
                # Handle special commands first
                if text_lower in ("help", "?", "commands"):
                    await self._safe_say(
                        say,
                        text="OCI Coordinator Help",
                        blocks=build_help_blocks(),
                        thread_ts=thread_ts,
                    )
                    return

                if text_lower in ("hello", "hi", "hey", "start", "welcome"):
                    await self._safe_say(
                        say,
                        text="Welcome to OCI Coordinator!",
                        blocks=build_welcome_blocks(),
                        thread_ts=thread_ts,
                    )
                    return

                # Handle catalog command
                if text_lower in ("catalog", "menu", "runbook", "runbooks", "troubleshoot"):
                    await self._safe_say(
                        say,
                        text="OCI Troubleshooting Catalog",
                        blocks=build_catalog_blocks(),
                        thread_ts=thread_ts,
                    )
                    return

                # Handle profile command - show profile selector
                if text_lower in ("profile", "profiles", "switch profile", "change profile", "tenancy"):
                    profile_manager = ProfileManager.get_instance()
                    await profile_manager.initialize()
                    profiles = profile_manager.list_profiles()
                    current_profile = await profile_manager.get_active_profile(user)

                    if len(profiles) <= 1:
                        # Only one profile, no need for selection
                        await self._safe_say(
                            say,
                            text=f"Using OCI profile: {current_profile}",
                            blocks=[{
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f":cloud: Only one OCI profile is configured: *{current_profile}*",
                                }
                            }],
                            thread_ts=thread_ts,
                        )
                    else:
                        await self._safe_say(
                            say,
                            text="Select OCI Profile",
                            blocks=build_profile_selection_blocks(profiles, current_profile),
                            thread_ts=thread_ts,
                        )
                    return

                # Require explicit profile selection when multiple profiles exist
                profile_manager = ProfileManager.get_instance()
                await profile_manager.initialize()
                profile_context = await profile_manager.get_profile_context(user)
                if profile_context.get("needs_selection"):
                    profiles = profile_manager.list_profiles()
                    current_profile = profile_context.get("profile")
                    await self._safe_say(
                        say,
                        text="Select OCI Profile",
                        blocks=build_profile_selection_blocks(profiles, current_profile),
                        thread_ts=thread_ts,
                    )
                    return

                # Initialize conversation context for this thread
                await self._conversation_manager.get_context(thread_ts, channel, user)

                # Check OCA authentication status
                from src.llm.oca import is_oca_authenticated

                auth_status = is_oca_authenticated()
                if not auth_status:
                    auth_url = get_oca_auth_url()
                    # Store pending auth context so we can notify user when auth succeeds
                    store_pending_auth_context(user, channel, thread_ts, text)
                    await self._safe_say(
                        say,
                        text="Authentication required. Please log in with Oracle SSO.",
                        blocks=build_auth_required_blocks(auth_url),
                        thread_ts=thread_ts,
                    )
                    return

                # 3-second ack pattern: Send immediate "thinking" message
                # This ensures Slack doesn't timeout while we process
                thinking_ts = None
                thinking_steps = []  # Collect thinking steps for live updates

                try:
                    # Use _safe_client_call to handle both sync and async clients
                    thinking_response = await self._safe_client_call(
                        client.chat_postMessage,
                        channel=channel,
                        thread_ts=thread_ts,
                        text=":hourglass_flowing_sand: Processing your request...",
                        blocks=[
                            {
                                "type": "context",
                                "elements": [
                                    {
                                        "type": "mrkdwn",
                                        "text": ":hourglass_flowing_sand: *Processing your request...* | Analyzing intent and routing to agents",
                                    }
                                ],
                            }
                        ],
                    )
                    thinking_ts = thinking_response.get("ts")
                except Exception as e:
                    logger.warning("Failed to send thinking message", error=str(e))

                # Create thinking update callback for real-time progress
                async def on_thinking_update(step):
                    """Update the thinking message with new step."""
                    nonlocal thinking_steps
                    thinking_steps.append(step)

                    if thinking_ts:
                        try:
                            blocks = self._build_thinking_blocks(thinking_steps)
                            await self._safe_client_call(
                                client.chat_update,
                                channel=channel,
                                ts=thinking_ts,
                                text=f":brain: {step.message}",
                                blocks=blocks,
                            )
                        except Exception as e:
                            # Don't fail processing if update fails
                            logger.debug("Failed to update thinking message", error=str(e))

                # Process with coordinator (may take time)
                response = None
                error_msg = None
                try:
                    # Default 300s to accommodate high-latency LLM providers (US-EMEA OCA ~15-20s/call)
                    coordinator_timeout_s = float(os.getenv("SLACK_COORDINATOR_TIMEOUT_SECONDS", "300"))
                    response = await asyncio.wait_for(
                        self._invoke_coordinator(
                            text=text,
                            user_id=user,
                            channel_id=channel,
                            thread_ts=thread_ts,
                            on_thinking_update=on_thinking_update,
                        ),
                        timeout=coordinator_timeout_s,
                    )
                except TimeoutError:
                    error_msg = (
                        "Coordinator timed out after "
                        f"{os.getenv('SLACK_COORDINATOR_TIMEOUT_SECONDS', '300')}s"
                    )
                    logger.error("Coordinator timed out", timeout_s=coordinator_timeout_s)
                except Exception as e:
                    error_msg = str(e)
                    import traceback
                    traceback.print_exc()
                    logger.error("Coordinator failed", error=error_msg)

                # Delete the thinking message now that we have a response
                if thinking_ts:
                    try:
                        await self._safe_client_call(client.chat_delete, channel=channel, ts=thinking_ts)
                    except Exception:
                        pass  # Ignore if delete fails (e.g., already deleted)

                # Handle coordinator error
                if error_msg:
                    await self._safe_say_with_fallback(
                        say=say,
                        client=client,
                        channel=channel,
                        thread_ts=thread_ts,
                        text=f"Error processing request: {error_msg[:100]}",
                        blocks=build_error_blocks(f"Request failed: {error_msg}"),
                    )
                    return

                # Format and send response
                if response:
                    # Check if it's an auth error
                    if response.get("type") == "auth_required":
                        auth_url = get_oca_auth_url()
                        store_pending_auth_context(user, channel, thread_ts, text)
                        await self._safe_say(
                            say,
                            text="Authentication required",
                            blocks=build_auth_required_blocks(auth_url),
                            thread_ts=thread_ts,
                        )
                    elif response.get("type") == "error":
                        error_msg = response.get("message", "Unknown error")
                        if "authentication" in error_msg.lower():
                            auth_url = get_oca_auth_url()
                            store_pending_auth_context(user, channel, thread_ts, text)
                            await self._safe_say(
                                say,
                                text="Authentication required",
                                blocks=build_auth_required_blocks(auth_url),
                                thread_ts=thread_ts,
                            )
                        else:
                            await self._safe_say(
                                say,
                                text=f"Error: {error_msg[:100]}",
                                blocks=build_error_blocks(error_msg),
                                thread_ts=thread_ts,
                            )
                    else:
                        # Format successful response
                        formatted = self._format_response(response)
                        msg_text = response.get("message", "")

                        # Use formatted summary for notifications (clean, no JSON)
                        fallback_text = formatted.get("summary", "Response")

                        # Handle empty or whitespace-only responses
                        if not msg_text or not msg_text.strip():
                            msg_text = (
                                "I processed your request but the response was empty. "
                                "Please try rephrasing your question."
                            )
                            fallback_text = msg_text
                        # Get follow-up suggestions based on query type
                        routing_type = response.get("agent_id", text)
                        suggestions = get_follow_up_suggestions(routing_type)
                        follow_up_blocks = build_follow_up_blocks(suggestions)

                        # Build all blocks with thinking summary (if available)
                        all_blocks = []

                        # Add thinking summary as context (collapsed)
                        thinking_summary = response.get("thinking_summary")
                        if thinking_summary:
                            all_blocks.append({
                                "type": "context",
                                "elements": [{
                                    "type": "mrkdwn",
                                    "text": f":brain: {thinking_summary}",
                                }]
                            })

                        # Add main response blocks
                        response_blocks = formatted.get("blocks", []) or []
                        if response_blocks:
                            all_blocks.extend(response_blocks)

                        # Add follow-up suggestions
                        if follow_up_blocks:
                            all_blocks.extend(follow_up_blocks)

                        await self._safe_say_with_fallback(
                            say=say,
                            client=client,
                            channel=channel,
                            thread_ts=thread_ts,
                            text=fallback_text,
                            blocks=all_blocks if all_blocks else None,
                        )
                        # Update conversation memory
                        try:
                            await self._conversation_manager.add_message(thread_ts, "user", text)
                            await self._conversation_manager.add_message(
                                thread_ts, "assistant", msg_text[:500],
                                metadata={"agent_id": response.get("agent_id")}
                            )
                            await self._conversation_manager.update_query_type(thread_ts, routing_type)
                        except Exception as mem_err:
                            logger.warning("Failed to update conversation memory", error=str(mem_err))

                        # Send file attachments if present
                        attachments = response.get("attachments", [])
                        if attachments:
                            for attachment in attachments:
                                if isinstance(attachment, dict):
                                    self._send_file_attachment(
                                        client=client,
                                        channel=channel,
                                        file_content=attachment.get("content", ""),
                                        filename=attachment.get("filename", "attachment"),
                                        thread_ts=thread_ts,
                                        title=attachment.get("title"),
                                        comment=attachment.get("comment"),
                                    )
                                else:
                                    # FileAttachment dataclass
                                    self._send_file_attachment(
                                        client=client,
                                        channel=channel,
                                        file_content=attachment.content,
                                        filename=attachment.filename,
                                        thread_ts=thread_ts,
                                        title=attachment.title,
                                        comment=attachment.comment,
                                    )

                        # Add success reaction
                        await self._safe_client_call(
                            client.reactions_add,
                            channel=channel,
                            timestamp=event.get("ts"),
                            name="white_check_mark",
                        )
                else:
                    # No response - show error with recovery options
                    error_blocks = build_error_blocks(
                        "I couldn't process that request.",
                        "Try rephrasing your question or type `help` for available commands."
                    )
                    recovery_blocks = build_error_recovery_blocks("default", text)
                    error_blocks.extend(recovery_blocks)

                    await self._safe_say_with_fallback(
                        say=say,
                        client=client,
                        channel=channel,
                        thread_ts=thread_ts,
                        text="I couldn't process that request.",
                        blocks=error_blocks,
                    )

                span.set_attribute("response.success", True)

            except ValueError as e:
                error_msg = str(e)
                logger.error(
                    "Error processing message",
                    error=error_msg,
                    trace_id=get_trace_id(),
                )
                span.set_attribute("error", True)
                span.set_attribute("error.message", error_msg)

                # Check if it's an auth error
                if "authentication" in error_msg.lower() or "requires authentication" in error_msg.lower():
                    auth_url = get_oca_auth_url()
                    store_pending_auth_context(user, channel, thread_ts, text)
                    await self._safe_say(
                        say,
                        text="Authentication required",
                        blocks=build_auth_required_blocks(auth_url),
                        thread_ts=thread_ts,
                    )
                else:
                    await self._safe_say_with_fallback(
                        say=say,
                        client=client,
                        channel=channel,
                        thread_ts=thread_ts,
                        text=f"Error: {error_msg[:100]}",
                        blocks=build_error_blocks(error_msg[:200]),
                    )

                # Add error reaction
                try:
                    await self._safe_client_call(
                        client.reactions_remove,
                        channel=channel,
                        timestamp=event.get("ts"),
                        name="hourglass_flowing_sand",
                    )
                except Exception:
                    pass
                try:
                    await self._safe_client_call(
                        client.reactions_add,
                        channel=channel,
                        timestamp=event.get("ts"),
                        name="x",
                    )
                except Exception:
                    pass

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(
                    "Error processing message",
                    error=str(e),
                    trace_id=get_trace_id(),
                )
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                # Send error response
                await self._safe_say_with_fallback(
                    say=say,
                    client=client,
                    channel=channel,
                    thread_ts=thread_ts,
                    text=f"Error: {str(e)[:100]}",
                    blocks=build_error_blocks(str(e)[:200]),
                )

                # Add error reaction
                try:
                    await self._safe_client_call(
                        client.reactions_remove,
                        channel=channel,
                        timestamp=event.get("ts"),
                        name="hourglass_flowing_sand",
                    )
                except Exception:
                    pass
                try:
                    await self._safe_client_call(
                        client.reactions_add,
                        channel=channel,
                        timestamp=event.get("ts"),
                        name="x",
                    )
                except Exception:
                    pass

    async def _process_command(
        self,
        body: dict,
        respond: Callable,
    ) -> None:
        """Process a slash command.

        Args:
            body: Command body
            respond: Function to send response
        """
        with self._tracer.start_as_current_span("slack_command") as span:
            user = body.get("user_id", "unknown")
            command = body.get("command", "/oci")
            text = body.get("text", "").strip()

            span.set_attribute("slack.user", user)
            span.set_attribute("slack.command", command)

            logger.info(
                "Processing slash command",
                user=user,
                command=command,
                text=text[:50],
            )

            try:
                # Handle help command
                if not text or text.lower() == "help":
                    respond(blocks=build_help_blocks())
                    return

                # Check OCA authentication status
                from src.llm.oca import is_oca_authenticated

                channel_id = body.get("channel_id")
                if not is_oca_authenticated():
                    auth_url = get_oca_auth_url()
                    store_pending_auth_context(user, channel_id, None, text)
                    respond(blocks=build_auth_required_blocks(auth_url))
                    return

                # Process with coordinator
                response = await self._invoke_coordinator(
                    text=text,
                    user_id=user,
                    channel_id=channel_id,
                )

                if response:
                    # Check for auth errors
                    if response.get("type") == "auth_required":
                        auth_url = get_oca_auth_url()
                        store_pending_auth_context(user, channel_id, None, text)
                        respond(blocks=build_auth_required_blocks(auth_url))
                    elif response.get("type") == "error":
                        error_msg = response.get("message", "Unknown error")
                        if "authentication" in error_msg.lower():
                            auth_url = get_oca_auth_url()
                            store_pending_auth_context(user, channel_id, None, text)
                            respond(blocks=build_auth_required_blocks(auth_url))
                        else:
                            respond(blocks=build_error_blocks(error_msg))
                    else:
                        formatted = self._format_response(response)
                        respond(blocks=formatted.get("blocks", []))
                else:
                    respond(blocks=build_error_blocks(
                        "I couldn't process that command.",
                        "Try `/oci help` for available commands."
                    ))

            except ValueError as e:
                error_msg = str(e)
                logger.error("Error processing command", error=error_msg)
                if "authentication" in error_msg.lower():
                    auth_url = get_oca_auth_url()
                    store_pending_auth_context(user, channel_id, None, text)
                    respond(blocks=build_auth_required_blocks(auth_url))
                else:
                    respond(blocks=build_error_blocks(error_msg[:200]))

            except Exception as e:
                logger.error("Error processing command", error=str(e))
                respond(blocks=build_error_blocks(str(e)[:200]))

    async def _process_action(
        self,
        body: dict,
        client: Any,
    ) -> None:
        """Process a button click or interactive action.

        Args:
            body: Action body
            client: Slack client
        """
        with self._tracer.start_as_current_span("slack_action") as span:
            action = body.get("actions", [{}])[0]
            action_id = action.get("action_id", "")
            action_value = action.get("value", "")

            span.set_attribute("slack.action_id", action_id)
            span.set_attribute("slack.action_value", action_value)

            logger.info(
                "Processing action",
                action_id=action_id,
                value=action_value,
            )

            # Handle different action types
            if action_id.startswith("agent_action_drill_"):
                # Drill-down action
                await self._handle_drill_down(action_value, body, client)
            elif action_id.startswith("agent_action_refresh_"):
                # Refresh action
                await self._handle_refresh(action_value, body, client)
            else:
                logger.warning("Unknown action", action_id=action_id)

    async def _handle_catalog_action(
        self,
        body: dict,
        client: Any,
        category: str | None,
    ) -> None:
        """Handle catalog navigation actions.

        Args:
            body: Slack action body
            client: Slack client
            category: Category to show, or None for main catalog
        """
        channel = body.get("channel", {}).get("id")
        thread_ts = body.get("message", {}).get("thread_ts") or body.get("message", {}).get("ts")

        # Convert category string to enum if provided
        cat_enum = None
        if category:
            try:
                cat_enum = Category(category)
            except ValueError:
                logger.warning("Invalid category", category=category)

        # Build catalog blocks
        blocks = build_catalog_blocks(cat_enum)

        # Send or update message
        try:
            await self._safe_client_call(
                client.chat_postMessage,
                channel=channel,
                thread_ts=thread_ts,
                text="OCI Troubleshooting Catalog",
                blocks=blocks,
            )
        except Exception as e:
            logger.error("Failed to send catalog", error=str(e))

    async def _handle_quick_action(
        self,
        body: dict,
        client: Any,
        query: str,
    ) -> None:
        """Handle quick action button clicks.

        Processes the query as if the user typed it.

        Args:
            body: Slack action body
            client: Slack client
            query: Query to execute
        """
        channel = body.get("channel", {}).get("id")
        thread_ts = body.get("message", {}).get("thread_ts") or body.get("message", {}).get("ts")
        user_id = body.get("user", {}).get("id", "unknown")

        logger.info("Processing quick action", query=query, user=user_id)

        # Send thinking message
        thinking_ts = None
        try:
            thinking_response = await self._safe_client_call(
                client.chat_postMessage,
                channel=channel,
                thread_ts=thread_ts,
                text=f":hourglass_flowing_sand: Running: {query}",
                blocks=[{
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f":hourglass_flowing_sand: *Running:* `{query}`",
                    }],
                }],
            )
            thinking_ts = thinking_response.get("ts")
        except Exception as e:
            logger.warning("Failed to send thinking message", error=str(e))

        # Process the query
        try:
            response = await self._invoke_coordinator(
                text=query,
                user_id=user_id,
                channel_id=channel,
                thread_ts=thread_ts,
            )

            # Delete thinking message
            if thinking_ts:
                try:
                    await self._safe_client_call(client.chat_delete, channel=channel, ts=thinking_ts)
                except Exception:
                    pass

            # Format and send response
            if response:
                formatted = self._format_response(response)

                # Use formatted summary for notifications (clean, no JSON)
                fallback_text = formatted.get("summary", "Response")

                # Get raw message for conversation memory
                msg_text = response.get("message", fallback_text)

                # Get follow-up suggestions based on query type
                routing_type = response.get("routing_type", query)
                suggestions = get_follow_up_suggestions(routing_type)
                follow_up_blocks = build_follow_up_blocks(suggestions)

                # Combine response blocks with follow-up suggestions
                all_blocks = formatted.get("blocks", [])
                if follow_up_blocks:
                    all_blocks.extend(follow_up_blocks)

                await self._safe_client_call(
                    client.chat_postMessage,
                    channel=channel,
                    thread_ts=thread_ts,
                    text=fallback_text,
                    blocks=all_blocks if all_blocks else None,
                )

                # Update conversation memory
                await self._conversation_manager.add_message(thread_ts, "user", query)
                await self._conversation_manager.add_message(
                    thread_ts, "assistant", msg_text[:500],
                    metadata={"query_type": routing_type}
                )
                await self._conversation_manager.update_query_type(thread_ts, routing_type)
            else:
                # Error response with recovery suggestions
                error_blocks = build_error_blocks(
                    "I couldn't process that request.",
                    "Try a different query or use the catalog for suggestions."
                )
                recovery_blocks = build_error_recovery_blocks("default", query)
                error_blocks.extend(recovery_blocks)

                await self._safe_client_call(
                    client.chat_postMessage,
                    channel=channel,
                    thread_ts=thread_ts,
                    text="Error processing request",
                    blocks=error_blocks,
                )

        except Exception as e:
            logger.error("Quick action failed", query=query, error=str(e))

            # Delete thinking message
            if thinking_ts:
                try:
                    await self._safe_client_call(client.chat_delete, channel=channel, ts=thinking_ts)
                except Exception:
                    pass

            # Determine error type for recovery suggestions
            error_str = str(e).lower()
            if "timeout" in error_str:
                error_type = "timeout"
            elif "auth" in error_str or "permission" in error_str:
                error_type = "permission"
            else:
                error_type = "default"

            error_blocks = build_error_blocks(str(e)[:200])
            recovery_blocks = build_error_recovery_blocks(error_type, query)
            error_blocks.extend(recovery_blocks)

            await self._safe_client_call(
                client.chat_postMessage,
                channel=channel,
                thread_ts=thread_ts,
                text=f"Error: {str(e)[:100]}",
                blocks=error_blocks,
            )

    async def _handle_profile_selection(
        self,
        body: dict,
        client: Any,
        profile: str,
    ) -> None:
        """Handle OCI profile selection.

        Updates the user's active profile and confirms the change.

        Args:
            body: Slack action body
            client: Slack client
            profile: Selected profile name
        """
        channel = body.get("channel", {}).get("id")
        thread_ts = body.get("message", {}).get("thread_ts") or body.get("message", {}).get("ts")
        user_id = body.get("user", {}).get("id", "unknown")

        logger.info("Profile selection", profile=profile, user=user_id)

        try:
            # Update user's active profile
            profile_manager = ProfileManager.get_instance()
            await profile_manager.initialize()

            success = await profile_manager.set_active_profile(user_id, profile)

            if success:
                profile_info = profile_manager.get_profile(profile)
                region_display = profile_info.region.replace("-", " ").title() if profile_info and profile_info.region else "Unknown"

                # Send confirmation
                await self._safe_client_call(
                    client.chat_postMessage,
                    channel=channel,
                    thread_ts=thread_ts,
                    text=f"Profile switched to {profile}",
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f":white_check_mark: *Profile switched to {profile}*\n\nRegion: `{region_display}`\n\nAll subsequent queries will use this profile.",
                            }
                        },
                        {
                            "type": "context",
                            "elements": [{
                                "type": "mrkdwn",
                                "text": ":bulb: Type `profile` to change profiles later.",
                            }]
                        }
                    ],
                )
            else:
                await self._safe_client_call(
                    client.chat_postMessage,
                    channel=channel,
                    thread_ts=thread_ts,
                    text="Failed to switch profile",
                    blocks=build_error_blocks(
                        f"Could not switch to profile '{profile}'",
                        "Please try again or contact support."
                    ),
                )

        except Exception as e:
            logger.error("Profile selection failed", profile=profile, error=str(e))
            await self._safe_client_call(
                client.chat_postMessage,
                channel=channel,
                thread_ts=thread_ts,
                text=f"Error: {str(e)[:100]}",
                blocks=build_error_blocks(str(e)[:200]),
            )

    async def _handle_show_profile_selector(
        self,
        body: dict,
        client: Any,
    ) -> None:
        """Show the profile selector UI.

        Args:
            body: Slack action body
            client: Slack client
        """
        channel = body.get("channel", {}).get("id")
        thread_ts = body.get("message", {}).get("thread_ts") or body.get("message", {}).get("ts")
        user_id = body.get("user", {}).get("id", "unknown")

        try:
            profile_manager = ProfileManager.get_instance()
            await profile_manager.initialize()

            profiles = profile_manager.list_profiles()
            current_profile = await profile_manager.get_active_profile(user_id)

            await self._safe_client_call(
                client.chat_postMessage,
                channel=channel,
                thread_ts=thread_ts,
                text="Select OCI Profile",
                blocks=build_profile_selection_blocks(profiles, current_profile),
            )

        except Exception as e:
            logger.error("Show profile selector failed", error=str(e))
            await self._safe_client_call(
                client.chat_postMessage,
                channel=channel,
                thread_ts=thread_ts,
                text=f"Error: {str(e)[:100]}",
                blocks=build_error_blocks(str(e)[:200]),
            )

    async def _invoke_coordinator(
        self,
        text: str,
        user_id: str,
        channel_id: str | None = None,
        thread_ts: str | None = None,
        on_thinking_update: Any | None = None,
    ) -> dict | None:
        """Invoke the coordinator to process a request.

        Tries LangGraph coordinator first for intent classification and
        workflow-first routing. Falls back to keyword routing if coordinator
        is not available or fails.

        Args:
            text: User message text
            user_id: Slack user ID
            channel_id: Slack channel ID
            thread_ts: Thread timestamp for threading
            on_thinking_update: Optional callback for real-time thinking updates

        Returns:
            Agent response or None
        """
        from src.agents.catalog import AgentCatalog

        with self._tracer.start_as_current_span("invoke_coordinator") as span:
            span.set_attribute("input.text", text[:100])
            span.set_attribute("user.id", user_id)

            # Get user's active OCI profile
            profile_context = None
            try:
                profile_manager = ProfileManager.get_instance()
                await profile_manager.initialize()
                profile_context = await profile_manager.get_profile_context(user_id)
                span.set_attribute("oci.profile", profile_context.get("profile", "DEFAULT"))
            except Exception as e:
                logger.warning("Failed to get profile context", error=str(e))
                profile_context = {"profile": "DEFAULT", "needs_selection": False}

            try:
                # Try LangGraph coordinator first (if enabled)
                use_langgraph = os.getenv("USE_LANGGRAPH_COORDINATOR", "true").lower() == "true"

                if use_langgraph:
                    try:
                        result = await self._invoke_langgraph_coordinator(
                            text=text,
                            user_id=user_id,
                            thread_id=thread_ts,
                            on_thinking_update=on_thinking_update,
                            profile_context=profile_context,
                        )
                        if result and result.get("success"):
                            span.set_attribute("routing.type", result.get("routing_type", "langgraph"))
                            span.set_attribute("routing.method", "langgraph")
                            # Get the actual workflow/agent name for attribution
                            agent_id = (
                                result.get("selected_workflow")
                                or result.get("selected_agent")
                                or result.get("routing_type")
                                or "coordinator"
                            )
                            return {
                                "type": "agent_response",
                                "agent_id": agent_id,
                                "query": text,
                                "message": result.get("response", ""),
                                "sections": [],
                                "thinking_trace": result.get("thinking_trace"),
                                "thinking_summary": result.get("thinking_summary"),
                                "agent_candidates": result.get("agent_candidates", []),
                            }
                        elif result and result.get("error"):
                            # Check if it's an auth error - propagate immediately
                            error_msg = result.get("error", "")
                            if "authentication" in error_msg.lower() or "requires authentication" in error_msg.lower():
                                logger.info("OCA authentication required, returning auth_required response")
                                return {"type": "auth_required"}

                            # LangGraph returned error, fall through to keyword routing
                            logger.warning(
                                "LangGraph coordinator returned error, falling back to keyword routing",
                                error=result.get("error"),
                            )
                    except Exception as e:
                        # Check if it's an auth error - propagate immediately
                        error_str = str(e)
                        if "authentication" in error_str.lower() or "requires authentication" in error_str.lower():
                            logger.info("OCA authentication required (exception)", error=error_str)
                            return {"type": "auth_required"}

                        logger.warning(
                            "LangGraph coordinator failed, falling back to keyword routing",
                            error=error_str,
                        )

                # Fall back to keyword-based routing
                span.set_attribute("routing.method", "keyword")
                catalog = AgentCatalog.get_instance()
                if not catalog.list_all():
                    catalog.auto_discover()

                agent_response = await self._route_to_agent(
                    text=text,
                    catalog=catalog,
                    user_id=user_id,
                )

                return agent_response

            except Exception as e:
                # Check if it's an auth error - propagate immediately
                error_str = str(e)
                if "authentication" in error_str.lower() or "requires authentication" in error_str.lower():
                    logger.info("OCA authentication required (outer exception)", error=error_str)
                    return {"type": "auth_required"}

                logger.error("Coordinator invocation failed", error=error_str)
                span.set_attribute("error", True)
                return None

    async def _invoke_langgraph_coordinator(
        self,
        text: str,
        user_id: str,
        thread_id: str | None = None,
        on_thinking_update: Any | None = None,
        profile_context: dict | None = None,
    ) -> dict | None:
        """Invoke the LangGraph coordinator for workflow-first routing.

        The LangGraph coordinator provides:
        - Intent classification with confidence scoring
        - Workflow-first routing (70%+ requests go to deterministic workflows)
        - Agent delegation for complex queries
        - Parallel orchestration for cross-domain queries
        - Thinking trace for transparency
        - Profile-aware OCI operations (uses user's selected profile)

        Args:
            text: User message text
            user_id: User ID
            thread_id: Thread ID for conversation continuity
            on_thinking_update: Optional callback for real-time thinking updates
            profile_context: OCI profile context (profile name, region, etc.)

        Returns:
            Coordinator result dict or None
        """
        from src.agents.catalog import AgentCatalog
        from src.agents.coordinator.graph import LangGraphCoordinator
        from src.llm import get_llm
        from src.mcp.catalog import ToolCatalog
        from src.memory.manager import SharedMemoryManager

        with self._tracer.start_as_current_span("langgraph_coordinator") as span:
            span.set_attribute("query.text", text[:200])
            span.set_attribute("user.id", user_id)

            try:
                # Get or create coordinator instance
                if not hasattr(self, "_langgraph_coordinator") or self._langgraph_coordinator is None:
                    # Try to use pre-warmed coordinator from main.py (fastest path)
                    from src.main import get_coordinator
                    prewarm_coordinator = get_coordinator()

                    if prewarm_coordinator is not None and not _has_llm_channel_overrides("slack"):
                        self._langgraph_coordinator = prewarm_coordinator
                        logger.info("Using pre-warmed LangGraph coordinator")
                    else:
                        # Fallback: lazy initialization (should rarely happen)
                        if prewarm_coordinator is not None:
                            logger.info("Slack LLM overrides set; initializing Slack-specific coordinator")
                        else:
                            logger.warning("Pre-warmed coordinator not available, initializing lazily")

                        # Initialize coordinator components
                        llm = get_llm(channel_type="slack")

                        # Use MCPConnectionManager for persistent tool catalog connections
                        # This ensures we reuse the initialized MCP connections from main.py
                        from src.mcp.connection_manager import MCPConnectionManager
                        try:
                            mcp_manager = await MCPConnectionManager.get_instance()
                            tool_catalog = await mcp_manager.get_tool_catalog()
                            if tool_catalog:
                                logger.debug(
                                    "Using persistent MCP connections for LangGraph",
                                    tool_count=len(tool_catalog.list_tools()),
                                )
                        except Exception as e:
                            logger.warning(
                                "MCPConnectionManager failed, falling back to direct catalog",
                                error=str(e),
                            )
                            tool_catalog = ToolCatalog.get_instance()

                        agent_catalog = AgentCatalog.get_instance()

                        # Warn if no tools available (likely MCP connection issue)
                        if tool_catalog is None:
                            logger.warning(
                                "No tool catalog available for LangGraph coordinator",
                            )
                        elif len(tool_catalog.list_tools()) == 0:
                            logger.warning(
                                "Tool catalog is empty - MCP servers may not be connected",
                            )

                        # Initialize memory manager with Redis cache
                        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                        memory = SharedMemoryManager(redis_url=redis_url)

                        # Load pre-built workflows for fast deterministic routing
                        from src.agents.coordinator.workflows import (
                            get_workflow_registry,
                        )
                        workflow_registry = get_workflow_registry()

                        # Create coordinator with workflows
                        self._langgraph_coordinator = LangGraphCoordinator(
                            llm=llm,
                            tool_catalog=tool_catalog,
                            agent_catalog=agent_catalog,
                            memory=memory,
                            workflow_registry=workflow_registry,
                            max_iterations=10,
                        )

                        # Initialize the graph
                        await self._langgraph_coordinator.initialize()
                        logger.info(
                            "LangGraph coordinator initialized for Slack",
                            workflow_count=len(workflow_registry),
                        )

                # Invoke coordinator with thinking callback for real-time updates
                # Pass profile context for profile-aware OCI operations
                oci_profile = profile_context.get("profile", "DEFAULT") if profile_context else "DEFAULT"
                result = await self._langgraph_coordinator.invoke(
                    query=text,
                    thread_id=thread_id,
                    user_id=user_id,
                    on_thinking_update=on_thinking_update,
                    metadata={"oci_profile": oci_profile, "profile_context": profile_context},
                )

                span.set_attribute("response.success", result.get("success", False))
                span.set_attribute("response.routing_type", result.get("routing_type", "unknown"))
                span.set_attribute("response.iterations", result.get("iterations", 0))

                return result

            except Exception as e:
                logger.error("LangGraph coordinator invocation failed", error=str(e))
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                return {"success": False, "error": str(e)}

    async def _route_to_agent(
        self,
        text: str,
        catalog,
        user_id: str,
    ) -> dict | None:
        """Route request to appropriate agent based on content.

        Args:
            text: User message
            catalog: Agent catalog
            user_id: User ID

        Returns:
            Agent response
        """
        text_lower = text.lower()

        # Simple keyword-based routing
        # Infrastructure keywords should match first (most common)
        if any(kw in text_lower for kw in [
            "compute", "instance", "vm", "network", "vcn", "subnet",
            "compartment", "tenancy", "region", "infra", "infrastructure",
            "load balancer", "security list", "route table", "storage"
        ]):
            capability = "compute-management"
        elif any(kw in text_lower for kw in ["database", "db", "awr", "sql", "performance", "slow", "query", "autonomous", "fleet", "addm", "opsi", "capacity"]):
            capability = "database-analysis"
        elif any(kw in text_lower for kw in ["log", "error", "exception", "audit trail"]):
            capability = "log-search"
        elif any(kw in text_lower for kw in ["security", "threat", "vulnerability", "compliance", "cloud guard", "iam"]):
            capability = "threat-detection"
        elif any(kw in text_lower for kw in ["cost", "budget", "spending", "finops", "billing"]):
            capability = "cost-analysis"
        else:
            # Default to infrastructure for general OCI queries
            capability = "compute-management"

        # Find agent with this capability
        agents = catalog.get_by_capability(capability)

        if not agents:
            logger.warning("No agent found for capability", capability=capability)
            return {
                "type": "error",
                "message": f"No agent available for: {capability}",
            }

        # Get first matching agent definition
        agent_def = agents[0]
        logger.info(
            "Routing to agent",
            agent=agent_def.agent_id,
            capability=capability,
        )

        try:
            # Create and invoke the actual agent
            agent_response = await self._invoke_agent(agent_def, text, user_id)
            return agent_response

        except Exception as e:
            logger.error(
                "Agent invocation failed",
                agent=agent_def.agent_id,
                error=str(e),
                trace_id=get_trace_id(),
            )
            return {
                "type": "error",
                "message": f"Agent error: {str(e)[:200]}",
            }

    async def _invoke_agent(
        self,
        agent_def,
        query: str,
        user_id: str,
    ) -> dict:
        """Actually invoke an agent instance with MCP tools.

        Uses ReAct agent pattern to make real API calls via MCP tools.

        Args:
            agent_def: Agent definition from catalog
            query: User query
            user_id: User ID

        Returns:
            Agent response dict
        """
        from src.agents.react_agent import SpecializedReActAgent
        from src.cache.oci_resource_cache import OCIResourceCache
        from src.llm import get_llm

        # Get agent-specific tracer for APM
        agent_tracer = get_tracer(agent_def.agent_id)

        with agent_tracer.start_as_current_span(f"agent.invoke.{agent_def.agent_id}") as span:
            span.set_attribute("agent.id", agent_def.agent_id)
            span.set_attribute("agent.capabilities", ",".join(agent_def.capabilities))
            span.set_attribute("agent.skills", ",".join(agent_def.skills))
            span.set_attribute("query.text", query[:200])
            span.set_attribute("user.id", user_id)

            # Get LLM for agent
            llm = get_llm(channel_type="slack")

            # Get tool catalog for MCP tools using persistent connection manager.
            # The MCPConnectionManager maintains connections across messages,
            # avoiding the 2-5s reconnection overhead on every request.
            tool_catalog = None
            try:
                from src.mcp.connection_manager import MCPConnectionManager

                manager = await MCPConnectionManager.get_instance()
                tool_catalog = await manager.get_tool_catalog()

                if tool_catalog:
                    span.set_attribute("tools.available", len(tool_catalog.list_tools()))
                    span.set_attribute("mcp.persistent_connection", True)
                    status = manager.get_status()
                    span.set_attribute("mcp.connected_servers", len(status["connected_servers"]))
            except Exception as e:
                logger.warning("Tool catalog not available", error=str(e))

            # Get resource cache
            resource_cache = None
            try:
                resource_cache = OCIResourceCache.get_instance()
            except Exception as e:
                logger.warning("Resource cache not available", error=str(e))

            # Map agent to domain for specialized ReAct agent
            domain_map = {
                "db-troubleshoot-agent": "database",
                "infrastructure-agent": "infrastructure",
                "finops-agent": "finops",
                "security-threat-agent": "security",
                "log-analytics-agent": "database",  # Use database tools for logs
            }
            domain = domain_map.get(agent_def.agent_id, "infrastructure")

            # Use ReAct agent with MCP tools
            try:
                react_agent = SpecializedReActAgent(
                    domain=domain,
                    llm=llm,
                    tool_catalog=tool_catalog,
                    resource_cache=resource_cache,
                    max_iterations=5,
                )

                result = await react_agent.run(query, user_id)

                span.set_attribute("response.success", result.success)
                span.set_attribute("response.length", len(result.response))
                span.set_attribute("tool_calls.count", len(result.tool_calls))

                # Log tool calls for debugging
                if result.tool_calls:
                    logger.info(
                        "Agent made tool calls",
                        agent=agent_def.agent_id,
                        tool_count=len(result.tool_calls),
                        tools=[t["tool"] for t in result.tool_calls],
                    )

                if result.success:
                    return {
                        "type": "agent_response",
                        "agent_id": agent_def.agent_id,
                        "query": query,
                        "message": result.response,
                        "tool_calls": result.tool_calls,
                        "sections": [],
                    }
                else:
                    return {
                        "type": "error",
                        "agent_id": agent_def.agent_id,
                        "message": result.response,
                        "error": result.error,
                    }

            except Exception as e:
                import traceback
                error_msg = str(e) if str(e) else repr(e)
                error_trace = traceback.format_exc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", error_msg[:200])
                logger.error(
                    "ReAct agent failed",
                    agent=agent_def.agent_id,
                    error=error_msg,
                    traceback=error_trace[:500],
                )

                # Fallback to simple LLM response
                logger.info("Falling back to simple LLM response")
                try:
                    response = await self._invoke_agent_with_llm(llm, agent_def, query)
                    return {
                        "type": "agent_response",
                        "agent_id": agent_def.agent_id,
                        "query": query,
                        "message": response,
                        "sections": [],
                    }
                except Exception as fallback_error:
                    fallback_msg = str(fallback_error) if str(fallback_error) else repr(fallback_error)
                    logger.error(
                        "Fallback LLM also failed",
                        agent=agent_def.agent_id,
                        error=fallback_msg,
                        traceback=traceback.format_exc()[:500],
                    )
                    return {
                        "type": "error",
                        "agent_id": agent_def.agent_id,
                        "message": f"Agent {agent_def.agent_id} error: {error_msg[:200] or fallback_msg[:200] or 'Unknown error'}",
                    }

    def _format_response(self, response: dict) -> dict:
        """Format agent response for Slack.

        Uses ResponseParser to extract structured content from raw agent
        responses, then SlackFormatter to convert to Block Kit format.

        Args:
            response: Agent response with 'message' and optional 'agent_id'

        Returns:
            Slack Block Kit formatted response with 'blocks' and 'summary' keys.
            'summary' is a plain text description for notifications/accessibility.
        """
        if response.get("type") == "error":
            result = _slack_formatter.format_error(response.get("message", "Unknown error"))
            result["summary"] = f"Error: {response.get('message', 'Unknown error')[:150]}"
            return result

        # Extract agent info
        agent_id = response.get("agent_id", "Agent Response")
        message = response.get("message", "")

        # Use ResponseParser to convert raw message to StructuredResponse
        parse_result = _response_parser.parse(message, agent_name=agent_id)

        # Use SlackFormatter to convert StructuredResponse to Block Kit
        formatted = _slack_formatter.format_response(parse_result.response)

        # Generate plain text summary for notifications/accessibility
        summary = self._generate_summary(parse_result.response)

        # Add any extra sections from the original response (backwards compat)
        blocks = formatted.get("blocks", [])
        for sec in response.get("sections", []):
            if sec.get("title"):
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{sec['title']}*"},
                })
            if sec.get("content"):
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": sec["content"][:2900]},
                })

        return {"blocks": blocks, "summary": summary}

    def _generate_summary(self, response) -> str:
        """Generate plain text summary for Slack notifications.

        Extracts title and key metrics from StructuredResponse for use
        as fallback text in notifications and accessibility.

        Args:
            response: StructuredResponse object

        Returns:
            Plain text summary (max 200 chars)
        """
        parts = []

        # Start with header title
        if response.header and response.header.title:
            parts.append(response.header.title)

        # Add key metrics if available (first section with metrics)
        for section in response.sections:
            if section.metrics:
                metric_parts = []
                for metric in section.metrics[:3]:  # First 3 metrics
                    label = metric.label or ""
                    value = metric.value if metric.value is not None else ""
                    if label and value:
                        metric_parts.append(f"{label}: {value}")
                if metric_parts:
                    parts.append(" | ".join(metric_parts))
                break  # Only use first section with metrics

        summary = " - ".join(parts) if parts else "Response"
        return summary[:200]

    # NOTE: _extract_content_from_response and _infer_table_title removed
    # Parsing logic consolidated into src/formatting/parser.py (ResponseParser)

    async def _invoke_agent_with_llm(self, llm, agent_def, query: str) -> str:
        """Use OCA LLM for intelligent agent responses.

        Args:
            llm: LangChain LLM instance
            agent_def: Agent definition with capabilities and skills
            query: User query

        Returns:
            LLM-generated response
        """
        import time

        from langchain_core.messages import HumanMessage, SystemMessage

        # Get tracer for LLM calls
        llm_tracer = get_tracer(agent_def.agent_id)

        with llm_tracer.start_as_current_span(f"llm.invoke.{agent_def.agent_id}") as span:
            span.set_attribute("llm.model", "oca/gpt5")
            span.set_attribute("llm.agent", agent_def.agent_id)
            span.set_attribute("llm.query_length", len(query))

            start_time = time.time()

            # Agent-specific system prompts
            agent_prompts = {
            "db-troubleshoot-agent": """You are an Oracle Database Expert Agent. You help users analyze and troubleshoot Oracle databases including Autonomous Database (ATP/ADW), DB Systems, and Exadata.

When users ask about database issues, provide:
1. **Analysis**: Understand what they're asking about (blocking sessions, performance, wait events, etc.)
2. **SQL Queries**: Provide the exact SQL queries they can run to investigate
3. **Recommendations**: Give actionable steps to resolve issues

For blocking sessions specifically, provide:
- SQL to find blocking sessions:
```sql
SELECT s.blocking_session AS blocker_sid, s.sid AS blocked_sid,
       s.username, s.event, s.seconds_in_wait, s.sql_id
FROM v$session s WHERE s.blocking_session IS NOT NULL;
```
- How to identify the blocker's SQL
- Options to resolve (wait, kill session, etc.)""",

            "infrastructure-agent": """You are an OCI Infrastructure Expert Agent. You help users manage and analyze Oracle Cloud Infrastructure resources including:
- Compute instances (VMs, bare metal, shapes)
- Virtual Cloud Networks (VCN), subnets, security lists
- Block storage, object storage
- Load balancers and network security groups

Provide:
1. **Resource Analysis**: Inventory and status of resources
2. **OCI CLI Commands**: Exact commands to manage resources
3. **Best Practices**: Recommendations for architecture and security
4. **Cost Optimization**: Suggestions for rightsizing""",

            "security-threat-agent": """You are an OCI Security Expert Agent. You help users with security and compliance in Oracle Cloud Infrastructure:
- Cloud Guard findings and threats
- IAM policies and security best practices
- Network security (NSGs, security lists)
- MITRE ATT&CK framework mapping
- Compliance monitoring

Provide:
1. **Threat Analysis**: Identify and explain security risks
2. **Remediation Steps**: Exact steps to fix issues
3. **Policy Examples**: IAM policy examples for least privilege
4. **Compliance Guidance**: CIS benchmark recommendations""",

            "finops-agent": """You are an OCI FinOps Expert Agent. You help users analyze and optimize cloud spending:
- Cost analysis by service, compartment, resource
- Budget tracking and alerts
- Usage anomaly detection
- Rightsizing recommendations
- Reserved capacity planning

Provide:
1. **Cost Breakdown**: Analysis by service and resource
2. **Optimization Tips**: Specific savings opportunities
3. **Budget Recommendations**: Alert thresholds
4. **Forecasting**: Usage trends and projections""",

            "log-analytics-agent": """You are an OCI Log Analytics Expert Agent. You help users search, analyze, and correlate logs:
- Log search and pattern detection
- Error correlation across services
- Audit log analysis
- Anomaly detection
- Custom query building

Provide:
1. **Log Queries**: Examples using OCI Log Analytics syntax
2. **Pattern Analysis**: Common error patterns
3. **Correlation**: How to trace issues across services
4. **Audit Insights**: Security-relevant log findings""",
            }

            # Get agent-specific prompt or use generic
            agent_id = agent_def.agent_id
            base_prompt = agent_prompts.get(agent_id, f"""You are an OCI Expert Agent ({agent_id}).
You help users with Oracle Cloud Infrastructure tasks.
Capabilities: {', '.join(agent_def.capabilities)}
Skills: {', '.join(agent_def.skills)}

Provide helpful, accurate responses about OCI.""")

            system_prompt = f"""{base_prompt}

Always format your response with:
- Clear markdown formatting
- Specific commands or queries when applicable
- Actionable recommendations
- Keep responses focused and practical"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]

            try:
                response = await llm.ainvoke(messages)
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                span.set_attribute("llm.response_length", len(response.content))
                span.set_attribute("llm.success", True)
                return response.content
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                span.set_attribute("llm.error", True)
                error_str = str(e)
                span.set_attribute("llm.error_message", error_str[:200])

                # Handle timeout specifically
                if "timeout" in error_str.lower() or "ReadTimeout" in error_str:
                    logger.error(
                        "LLM request timed out",
                        agent=agent_id,
                        duration_ms=duration_ms,
                        error=error_str[:200],
                    )
                    # Return a helpful message instead of raising
                    return f"I apologize, but my response is taking longer than expected. The {agent_id.replace('-', ' ')} is processing your request. Please try again in a moment, or simplify your query."

                logger.error("LLM invocation failed", agent=agent_id, error=error_str)
                raise

    def _clean_message(self, text: str) -> str:
        """Remove bot mention and clean message text.

        Args:
            text: Raw message text

        Returns:
            Cleaned text
        """
        # Remove <@BOTID> mentions
        text = re.sub(r"<@[A-Z0-9]+>", "", text)
        return text.strip()

    async def _send_file_attachment(
        self,
        client: Any,
        channel: str,
        file_content: bytes | str,
        filename: str,
        thread_ts: str | None = None,
        title: str | None = None,
        comment: str | None = None,
    ) -> dict | None:
        """Upload a file attachment to Slack.

        Uses the files.upload API to send files (e.g., AWR HTML reports).

        Args:
            client: Slack WebClient
            channel: Channel ID to send to
            file_content: File content (bytes or string)
            filename: Filename for the upload
            thread_ts: Thread timestamp for threading
            title: Display title for the file
            comment: Initial comment with the file

        Returns:
            Response from Slack API or None on failure
        """
        try:
            # Convert string to bytes if needed
            if isinstance(file_content, str):
                file_content = file_content.encode("utf-8")

            # Use files_upload_v2 for better reliability
            # Use _safe_client_call to handle both sync and async clients
            response = await self._safe_client_call(
                client.files_upload_v2,
                file=file_content,
                filename=filename,
                channel=channel,
                thread_ts=thread_ts,
                title=title or filename,
                initial_comment=comment or f"📎 {title or filename}",
            )

            logger.info(
                "File uploaded to Slack",
                filename=filename,
                size_bytes=len(file_content),
                channel=channel,
            )

            return response

        except Exception as e:
            logger.error(
                "Failed to upload file to Slack",
                filename=filename,
                error=str(e),
            )
            return None

    async def _handle_drill_down(
        self,
        value: str,
        body: dict,
        client: Any,
    ) -> None:
        """Handle drill-down action."""
        channel = body.get("channel", {}).get("id")
        thread_ts = body.get("message", {}).get("ts")

        # Send follow-up message - use _safe_client_call for sync/async compatibility
        await self._safe_client_call(
            client.chat_postMessage,
            channel=channel,
            thread_ts=thread_ts,
            text=f"Drilling down into: {value}...",
        )

    async def _handle_refresh(
        self,
        value: str,
        body: dict,
        client: Any,
    ) -> None:
        """Handle refresh action."""
        channel = body.get("channel", {}).get("id")
        message_ts = body.get("message", {}).get("ts")

        # Update the original message - use _safe_client_call for sync/async compatibility
        await self._safe_client_call(
            client.chat_update,
            channel=channel,
            ts=message_ts,
            text=f"Refreshing: {value}...",
        )

    def start(self, port: int = 3000, socket_mode: bool = True) -> None:
        """Start the Slack handler in BLOCKING sync mode.

        WARNING: This uses the sync SocketModeHandler which creates its own
        internal event loop. Use start_async() when running alongside other
        async services (like the API server) to avoid BrokenPipeError.

        Args:
            port: Port for HTTP mode (if not using socket mode)
            socket_mode: Use Socket Mode (recommended)
        """
        if socket_mode and self.app_token:
            from slack_bolt.adapter.socket_mode import SocketModeHandler

            handler = SocketModeHandler(self.app, self.app_token)
            logger.info("Starting Slack handler in Socket Mode (blocking)")
            handler.start()
        else:
            logger.info("Starting Slack handler in HTTP mode", port=port)
            self.app.start(port=port)

    async def start_async(self) -> None:
        """Start the Slack handler in ASYNC mode.

        This uses AsyncSocketModeHandler with AsyncApp which properly integrates
        with the asyncio event loop. Use this when running alongside other
        async services (like the API server with uvicorn).

        This avoids the BrokenPipeError that occurs when sync SocketModeHandler
        runs in a background thread while uvicorn uses the main async loop.
        """
        if not self.app_token:
            logger.warning("SLACK_APP_TOKEN not set - cannot start async Socket Mode")
            return

        # Validate token format
        if not self.app_token.startswith("xapp-"):
            raise ValueError("SLACK_APP_TOKEN must start with 'xapp-'")
        if not self.bot_token.startswith("xoxb-"):
            raise ValueError("SLACK_BOT_TOKEN must start with 'xoxb-'")

        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        logger.info("Starting Slack handler in Async Socket Mode")

        # Create async handler with AsyncApp (not sync App!)
        # AsyncSocketModeHandler requires AsyncApp for proper async operation
        handler = AsyncSocketModeHandler(self.async_app, self.app_token)

        # Log when connected
        # Test the connection and log bot info
        try:
            from slack_sdk.web.async_client import AsyncWebClient
            test_client = AsyncWebClient(token=self.bot_token)
            auth_result = await test_client.auth_test()
            self._bot_user_id = auth_result.get("user_id")
            logger.info("Slack bot connected", user=auth_result.get("user"), team=auth_result.get("team"))
        except Exception as auth_err:
            logger.warning("Could not test Slack auth", error=str(auth_err))
        try:
            # Start and keep running - this will handle reconnections automatically
            # The start_async() method blocks until the connection is closed
            await handler.start_async()
        except asyncio.CancelledError:
            logger.info("Slack handler received cancellation, shutting down gracefully")
            await handler.close_async()
        except Exception as e:
            error_msg = str(e)
            logger.error("Async Socket Mode handler failed", error=error_msg)
            # Don't raise on BrokenPipeError during shutdown
            if "BrokenPipeError" not in error_msg and "Broken pipe" not in error_msg:
                raise


def create_slack_app(
    bot_token: str | None = None,
    app_token: str | None = None,
    signing_secret: str | None = None,
) -> SlackHandler:
    """Create a Slack handler with the coordinator.

    Args:
        bot_token: Slack bot token
        app_token: Slack app token for Socket Mode
        signing_secret: Slack signing secret

    Returns:
        Configured SlackHandler
    """
    return SlackHandler(
        bot_token=bot_token,
        app_token=app_token,
        signing_secret=signing_secret,
    )
