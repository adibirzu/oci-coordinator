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

import os
import re
from collections.abc import Callable
from typing import Any

import structlog
from opentelemetry import trace

from src.channels.async_runtime import run_async
from src.formatting.slack import SlackFormatter
from src.observability import get_trace_id, init_observability
from src.observability.tracing import get_tracer

logger = structlog.get_logger(__name__)


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

    # Save verifier for later use
    CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    verifier_path = CACHE_DIR / "verifier.txt"
    verifier_path.write_text(verifier)
    verifier_path.chmod(0o600)

    params = {
        "response_type": "code",
        "client_id": IDCS_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "openid offline_access",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }

    return f"{IDCS_OAUTH_URL}/oauth2/v1/authorize?{urlencode(params)}"

# Slack formatter for response conversion
_slack_formatter = SlackFormatter()


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
    """Build help message blocks using Block Kit."""
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
                "text": "I can help you with various OCI tasks. Just describe what you need!"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:database: Database Commands*"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "• `check database performance` - Analyze DB metrics\n• `show slow queries` - Find problematic queries\n• `analyze AWR report` - Deep dive into AWR data"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:cloud: Infrastructure Commands*"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "• `list instances` - Show compute instances\n• `check network` - Analyze VCN configuration\n• `show instance metrics` - View resource utilization"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*:moneybag: FinOps Commands*"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "• `show costs` - Current spending summary\n• `analyze budget` - Budget vs actual comparison\n• `cost optimization` - Get savings recommendations"
            }
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":zap: Powered by Oracle Code Assist (OCA)"
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
    content_chunks = [content[i:i+2900] for i in range(0, len(content), 2900)]
    for chunk in content_chunks[:10]:  # Max 10 sections
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": chunk
            }
        })

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

        self._app = None
        self._coordinator = None

        # Initialize observability FIRST - sets up tracer provider with OCI APM export
        # This MUST happen before getting the tracer to ensure spans are exported
        init_observability(agent_name="slack-handler")

        # Now get tracer from the properly configured provider
        self._tracer = trace.get_tracer("oci-slack-handler")

    @property
    def app(self):
        """Get the Slack Bolt app instance."""
        if self._app is None:
            self._app = self._create_app()
        return self._app

    def _create_app(self):
        """Create and configure the Slack Bolt app."""
        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler

            app = App(
                token=self.bot_token,
                signing_secret=self.signing_secret,
            )

            # Register event handlers
            self._register_handlers(app)

            logger.info("Slack app created successfully")
            return app

        except ImportError:
            logger.error(
                "slack_bolt not installed",
                help="Run: poetry add slack-bolt",
            )
            raise

    def _register_handlers(self, app) -> None:
        """Register Slack event handlers.

        Uses shared AsyncRuntime instead of asyncio.run() to maintain
        a single event loop across all handlers. This prevents MCP
        connection issues that occur when creating new event loops.
        """
        # Ensure AsyncRuntime is started before registering handlers
        from src.channels.async_runtime import AsyncRuntime
        import time

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
            print(f"[SLACK] Received mention from {event.get('user')}: {event.get('text', '')[:50]}", flush=True)
            # Use shared async runtime instead of asyncio.run()
            run_async(self._process_message(event, say, client))

        @app.event("message")
        def handle_message(event: dict, say: Callable, client: Any) -> None:
            """Handle direct messages and thread replies."""
            text = event.get("text", "")
            channel_type = event.get("channel_type")
            thread_ts = event.get("thread_ts")
            print(f"[SLACK] Received message type={channel_type}, thread={thread_ts is not None}: {text[:50]}", flush=True)

            # Skip bot's own messages and message_changed events
            if event.get("bot_id") or event.get("subtype"):
                print("[SLACK] Skipping bot/subtype message", flush=True)
                return

            # Handle DMs
            if channel_type == "im":
                print("[SLACK] Processing DM...", flush=True)
                run_async(self._process_message(event, say, client))
            # Handle thread replies (user continuing conversation without @mention)
            elif thread_ts:
                print("[SLACK] Processing thread reply...", flush=True)
                run_async(self._process_message(event, say, client))
            else:
                # Skip non-threaded channel messages - app_mention handles @mentions
                print("[SLACK] Skipping channel message (need @mention or thread reply)", flush=True)

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

        @app.command("/oci")
        def handle_command(ack: Callable, body: dict, respond: Callable) -> None:
            """Handle /oci slash command."""
            ack()
            run_async(self._process_command(body, respond))

        logger.info("Slack handlers registered with shared AsyncRuntime")

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
        print("[SLACK] Processing message...", flush=True)
        with self._tracer.start_as_current_span("slack_message") as span:
            user = event.get("user", "unknown")
            channel = event.get("channel")
            text = event.get("text", "")
            thread_ts = event.get("thread_ts") or event.get("ts")

            # Remove bot mention from text
            text = self._clean_message(text)
            text_lower = text.lower().strip()
            print(f"[SLACK] Cleaned text: {text[:100]}", flush=True)

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
                    print("[SLACK] Sending help message", flush=True)
                    say(
                        text="OCI Coordinator Help",
                        blocks=build_help_blocks(),
                        thread_ts=thread_ts,
                    )
                    print("[SLACK] Help message sent", flush=True)
                    return

                if text_lower in ("hello", "hi", "hey", "start", "welcome"):
                    print("[SLACK] Sending welcome message", flush=True)
                    say(
                        text="Welcome to OCI Coordinator!",
                        blocks=build_welcome_blocks(),
                        thread_ts=thread_ts,
                    )
                    print("[SLACK] Welcome message sent", flush=True)
                    return

                # Check OCA authentication status
                from src.llm.oca import is_oca_authenticated

                auth_status = is_oca_authenticated()
                print(f"[SLACK] OCA auth status: {auth_status}", flush=True)

                if not auth_status:
                    auth_url = get_oca_auth_url()
                    print("[SLACK] Sending auth required message", flush=True)
                    say(
                        text="Authentication required. Please log in with Oracle SSO.",
                        blocks=build_auth_required_blocks(auth_url),
                        thread_ts=thread_ts,
                    )
                    print("[SLACK] Auth message sent", flush=True)
                    return

                # 3-second ack pattern: Send immediate "thinking" message
                # This ensures Slack doesn't timeout while we process
                thinking_ts = None
                try:
                    thinking_response = client.chat_postMessage(
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
                    print(f"[SLACK] Sent thinking message: {thinking_ts}", flush=True)
                except Exception as e:
                    logger.warning("Failed to send thinking message", error=str(e))

                # Process with coordinator (may take time)
                response = None
                error_msg = None
                try:
                    response = await self._invoke_coordinator(
                        text=text,
                        user_id=user,
                        channel_id=channel,
                        thread_ts=thread_ts,
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.error("Coordinator failed", error=error_msg)

                # Delete the thinking message now that we have a response
                if thinking_ts:
                    try:
                        client.chat_delete(channel=channel, ts=thinking_ts)
                    except Exception:
                        pass  # Ignore if delete fails (e.g., already deleted)

                # Handle coordinator error
                if error_msg:
                    say(
                        text=f"Error processing request: {error_msg[:100]}",
                        blocks=build_error_blocks(f"Request failed: {error_msg}"),
                        thread_ts=thread_ts,
                    )
                    return

                # Format and send response
                print(f"[SLACK] Got response: {response}", flush=True)
                if response:
                    # Check if it's an auth error
                    if response.get("type") == "auth_required":
                        auth_url = get_oca_auth_url()
                        say(
                            text="Authentication required",
                            blocks=build_auth_required_blocks(auth_url),
                            thread_ts=thread_ts,
                        )
                    elif response.get("type") == "error":
                        error_msg = response.get("message", "Unknown error")
                        if "authentication" in error_msg.lower():
                            auth_url = get_oca_auth_url()
                            say(
                                text="Authentication required",
                                blocks=build_auth_required_blocks(auth_url),
                                thread_ts=thread_ts,
                            )
                        else:
                            say(
                                text=f"Error: {error_msg[:100]}",
                                blocks=build_error_blocks(error_msg),
                                thread_ts=thread_ts,
                            )
                    else:
                        # Format successful response
                        formatted = self._format_response(response)
                        msg_text = response.get("message", "")

                        # Handle empty or whitespace-only responses
                        if not msg_text or not msg_text.strip():
                            msg_text = (
                                "I processed your request but the response was empty. "
                                "Please try rephrasing your question."
                            )
                            print(f"[SLACK] Warning: Empty response received", flush=True)

                        print(f"[SLACK] Sending formatted response (length={len(msg_text)})", flush=True)
                        say(
                            text=msg_text[:200] if isinstance(msg_text, str) else "Response",
                            blocks=formatted.get("blocks", []) if formatted.get("blocks") else None,
                            thread_ts=thread_ts,
                        )
                        print("[SLACK] Response sent", flush=True)

                        # Send file attachments if present
                        attachments = response.get("attachments", [])
                        if attachments:
                            print(f"[SLACK] Sending {len(attachments)} file attachment(s)", flush=True)
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
                        client.reactions_add(
                            channel=channel,
                            timestamp=event.get("ts"),
                            name="white_check_mark",
                        )
                else:
                    say(
                        text="I couldn't process that request.",
                        blocks=build_error_blocks(
                            "I couldn't process that request.",
                            "Try rephrasing your question or type `help` for available commands."
                        ),
                        thread_ts=thread_ts,
                    )

                span.set_attribute("response.success", True)

            except ValueError as e:
                error_msg = str(e)
                print(f"[SLACK] ValueError: {error_msg}", flush=True)
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
                    say(
                        text="Authentication required",
                        blocks=build_auth_required_blocks(auth_url),
                        thread_ts=thread_ts,
                    )
                else:
                    say(
                        text=f"Error: {error_msg[:100]}",
                        blocks=build_error_blocks(error_msg[:200]),
                        thread_ts=thread_ts,
                    )

                # Add error reaction
                try:
                    client.reactions_remove(
                        channel=channel,
                        timestamp=event.get("ts"),
                        name="hourglass_flowing_sand",
                    )
                except Exception:
                    pass
                try:
                    client.reactions_add(
                        channel=channel,
                        timestamp=event.get("ts"),
                        name="x",
                    )
                except Exception:
                    pass

            except Exception as e:
                print(f"[SLACK] Exception: {e!s}", flush=True)
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
                say(
                    text=f"Error: {str(e)[:100]}",
                    blocks=build_error_blocks(str(e)[:200]),
                    thread_ts=thread_ts,
                )

                # Add error reaction
                try:
                    client.reactions_remove(
                        channel=channel,
                        timestamp=event.get("ts"),
                        name="hourglass_flowing_sand",
                    )
                except Exception:
                    pass
                try:
                    client.reactions_add(
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

                if not is_oca_authenticated():
                    auth_url = get_oca_auth_url()
                    respond(blocks=build_auth_required_blocks(auth_url))
                    return

                # Process with coordinator
                response = await self._invoke_coordinator(
                    text=text,
                    user_id=user,
                    channel_id=body.get("channel_id"),
                )

                if response:
                    # Check for auth errors
                    if response.get("type") == "auth_required":
                        auth_url = get_oca_auth_url()
                        respond(blocks=build_auth_required_blocks(auth_url))
                    elif response.get("type") == "error":
                        error_msg = response.get("message", "Unknown error")
                        if "authentication" in error_msg.lower():
                            auth_url = get_oca_auth_url()
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

    async def _invoke_coordinator(
        self,
        text: str,
        user_id: str,
        channel_id: str | None = None,
        thread_ts: str | None = None,
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

        Returns:
            Agent response or None
        """
        from src.agents.catalog import AgentCatalog

        with self._tracer.start_as_current_span("invoke_coordinator") as span:
            span.set_attribute("input.text", text[:100])
            span.set_attribute("user.id", user_id)

            try:
                # Try LangGraph coordinator first (if enabled)
                use_langgraph = os.getenv("USE_LANGGRAPH_COORDINATOR", "true").lower() == "true"

                if use_langgraph:
                    try:
                        result = await self._invoke_langgraph_coordinator(
                            text=text,
                            user_id=user_id,
                            thread_id=thread_ts,
                        )
                        if result and result.get("success"):
                            span.set_attribute("routing.type", result.get("routing_type", "langgraph"))
                            span.set_attribute("routing.method", "langgraph")
                            return {
                                "type": "agent_response",
                                "agent_id": result.get("routing_type", "coordinator"),
                                "query": text,
                                "message": result.get("response", ""),
                                "sections": [],
                            }
                        elif result and result.get("error"):
                            # LangGraph returned error, fall through to keyword routing
                            logger.warning(
                                "LangGraph coordinator returned error, falling back to keyword routing",
                                error=result.get("error"),
                            )
                    except Exception as e:
                        logger.warning(
                            "LangGraph coordinator failed, falling back to keyword routing",
                            error=str(e),
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
                logger.error("Coordinator invocation failed", error=str(e))
                span.set_attribute("error", True)
                return None

    async def _invoke_langgraph_coordinator(
        self,
        text: str,
        user_id: str,
        thread_id: str | None = None,
    ) -> dict | None:
        """Invoke the LangGraph coordinator for workflow-first routing.

        The LangGraph coordinator provides:
        - Intent classification with confidence scoring
        - Workflow-first routing (70%+ requests go to deterministic workflows)
        - Agent delegation for complex queries
        - Parallel orchestration for cross-domain queries

        Args:
            text: User message text
            user_id: User ID
            thread_id: Thread ID for conversation continuity

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
                    # Initialize coordinator components
                    llm = get_llm()
                    tool_catalog = ToolCatalog.get_instance()
                    agent_catalog = AgentCatalog.get_instance()

                    # Initialize memory manager
                    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                    memory = SharedMemoryManager(redis_url=redis_url)

                    # Load pre-built workflows for fast deterministic routing
                    from src.agents.coordinator.workflows import get_workflow_registry
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

                # Invoke coordinator
                result = await self._langgraph_coordinator.invoke(
                    query=text,
                    thread_id=thread_id,
                    user_id=user_id,
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
        elif any(kw in text_lower for kw in ["database", "db", "awr", "sql", "performance", "slow", "query", "autonomous"]):
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
            llm = get_llm()

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

        Handles:
        - Raw JSON responses from React agents
        - List/table data from tool calls
        - Plain text responses

        Args:
            response: Agent response

        Returns:
            Slack Block Kit formatted response
        """
        if response.get("type") == "error":
            return _slack_formatter.format_error(response.get("message", "Unknown error"))

        # Build blocks directly for simpler, more reliable formatting
        blocks = []

        # Header with agent name
        agent_id = response.get("agent_id", "Agent Response")
        agent_display = agent_id.replace("-", " ").replace("agent", "Agent").title()
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"✅ {agent_display}",
                "emoji": True,
            }
        })

        # Get the message content
        message = response.get("message", "")

        # Clean and parse the message if it contains React agent JSON
        cleaned_message, table_data = self._extract_content_from_response(message)

        # If we detected table data (list of items), format as table block
        if table_data and isinstance(table_data, list) and len(table_data) > 0:
            # Detect data type and configure table accordingly
            first_item = table_data[0] if table_data else {}
            keys = list(first_item.keys()) if first_item else []

            # Cost data: explicit columns for consistent ordering
            if "service" in keys and ("cost" in keys or "percent" in keys):
                columns = ["service", "cost", "percent"]
                title = ":bar_chart: *Top Services by Spend*"
                footer = f"Showing top {len(table_data)} services"
            else:
                columns = None  # Auto-detect
                title = self._infer_table_title(table_data)
                footer = f"Found {len(table_data)} items"

            # Use native Slack table block
            table_payload = _slack_formatter.format_table_from_list(
                items=table_data,
                columns=columns,
                title=title,
                footer=footer,
            )
            blocks.extend(table_payload.get("blocks", []))

            # Add any additional text that wasn't table data (e.g., summary header)
            if cleaned_message and not cleaned_message.startswith("["):
                text_part = cleaned_message.split("[")[0].strip()
                if text_part:
                    blocks.insert(1, {  # After header
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": text_part[:2900]},
                    })
        elif cleaned_message:
            # Regular text response - split if too long
            chunks = [cleaned_message[i:i+2900] for i in range(0, len(cleaned_message), 2900)]
            for chunk in chunks[:5]:  # Max 5 sections
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": chunk,
                    }
                })

        # Add sections if any
        for sec in response.get("sections", []):
            if sec.get("title"):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{sec['title']}*",
                    }
                })
            if sec.get("content"):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": sec["content"][:2900],
                    }
                })

        return {"blocks": blocks}

    def _extract_content_from_response(self, message: str) -> tuple[str, list | None]:
        """
        Extract clean content from agent response.

        Handles:
        - React agent JSON format: {"thought": "...", "final_answer": "..."}
        - Multiple concatenated JSON objects from iterations
        - Tool output JSON arrays: [{"name": "...", ...}, ...]
        - Plain text with embedded JSON

        Returns:
            Tuple of (cleaned_message, table_data_if_found)
        """
        import json as json_module
        import re

        if not message:
            return "", None

        table_data = None
        extracted_answer = None

        # Handle multiple concatenated JSON objects (React agent iterations)
        # Find ALL final_answer values and use the last one
        if '"thought"' in message or '"final_answer"' in message:
            # Try regex to find all final_answer values
            final_answers = re.findall(
                r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"|"final_answer"\s*:\s*"([^"]*)"',
                message,
                re.DOTALL
            )
            if final_answers:
                # Get the last final_answer (most complete)
                for match in final_answers:
                    answer = match[0] or match[1]
                    if answer:
                        extracted_answer = answer.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')

            # If no final_answer found, try to extract any meaningful text
            if not extracted_answer:
                # Try to find response field
                response_matches = re.findall(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', message, re.DOTALL)
                if response_matches:
                    extracted_answer = response_matches[-1].replace('\\"', '"').replace('\\n', '\n')

            # If still nothing, try parsing each JSON object
            if not extracted_answer:
                # Find all JSON objects
                json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', message)
                for json_str in reversed(json_objects):  # Start from last (most recent)
                    try:
                        parsed = json_module.loads(json_str)
                        if isinstance(parsed, dict):
                            if "final_answer" in parsed:
                                extracted_answer = parsed["final_answer"]
                                break
                            elif "response" in parsed:
                                extracted_answer = parsed["response"]
                                break
                    except json_module.JSONDecodeError:
                        continue

            if extracted_answer:
                message = extracted_answer

        # Check for structured JSON responses FIRST (cost_summary, etc.)
        # These have a "type" field and should be parsed specially
        if "{" in message and '"type"' in message:
            try:
                # Find JSON object in message
                start = message.find("{")
                end = message.rfind("}") + 1
                json_part = message[start:end]
                parsed = json_module.loads(json_part)

                if isinstance(parsed, dict) and "type" in parsed:
                    # Handle cost_summary type
                    if parsed.get("type") == "cost_summary":
                        if "error" in parsed:
                            message = f"⚠️ {parsed['error']}"
                        else:
                            summary = parsed.get("summary", {})
                            services = parsed.get("services", [])
                            days = summary.get("days", 30)
                            total = summary.get("total", "N/A")
                            period = summary.get("period", "N/A")

                            # Build formatted header
                            message = (
                                f":moneybag: *Tenancy Cost Summary*\n"
                                f":calendar: *Period:* {period} ({days} days)\n"
                                f":chart_with_upwards_trend: *Total Spend:* `{total}`"
                            )
                            if services:
                                table_data = services
                        return message.strip(), table_data
            except (json_module.JSONDecodeError, IndexError, KeyError):
                pass

        # Remove all raw JSON objects from the message (React agent artifacts)
        if '"thought"' in message or message.strip().startswith("{"):
            # Remove JSON blocks but keep any non-JSON text
            message = re.sub(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', message)

        # Check for JSON array (list data from tools)
        if "[" in message and "]" in message:
            try:
                # Find JSON array in message
                start = message.find("[")
                end = message.rfind("]") + 1
                json_part = message[start:end]
                parsed = json_module.loads(json_part)

                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    table_data = parsed
                    # Remove the JSON array from message
                    message = message[:start] + message[end:]
            except (json_module.JSONDecodeError, IndexError):
                pass

        # Clean up any remaining artifacts
        message = message.replace('```json', '').replace('```', '')
        message = re.sub(r'\s+', ' ', message)  # Normalize whitespace

        return message.strip(), table_data

    def _infer_table_title(self, data: list[dict]) -> str:
        """Infer a table title from the data structure."""
        if not data:
            return "Results"

        first_item = data[0]
        keys = list(first_item.keys())

        # Detect cost data (from cost_summary)
        if "service" in keys and ("cost" in keys or "percent" in keys):
            return "Cost by Service"

        # Detect common OCI resource types
        if "compartment_id" in keys or "compartment_ocid" in keys:
            return "OCI Resources"
        if "display_name" in keys and "lifecycle_state" in keys:
            if "cidr_block" in keys:
                return "VCNs"
            if "shape" in keys:
                return "Compute Instances"
            if "db_name" in keys:
                return "Databases"
            return "Resources"
        if "name" in keys and "id" in keys:
            if "tenancy" in str(first_item.get("id", "")):
                return "Compartments"
            return "Items"

        return "Results"

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

    def _send_file_attachment(
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
            response = client.files_upload_v2(
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

        # Send follow-up message
        client.chat_postMessage(
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

        # Update the original message
        client.chat_update(
            channel=channel,
            ts=message_ts,
            text=f"Refreshing: {value}...",
        )

    def start(self, port: int = 3000, socket_mode: bool = True) -> None:
        """Start the Slack handler.

        Args:
            port: Port for HTTP mode (if not using socket mode)
            socket_mode: Use Socket Mode (recommended)
        """
        if socket_mode and self.app_token:
            from slack_bolt.adapter.socket_mode import SocketModeHandler

            print("[SLACK] Starting Socket Mode handler...", flush=True)
            handler = SocketModeHandler(self.app, self.app_token)
            logger.info("Starting Slack handler in Socket Mode")
            handler.start()
        else:
            print(f"[SLACK] Starting HTTP mode on port {port}...", flush=True)
            logger.info("Starting Slack handler in HTTP mode", port=port)
            self.app.start(port=port)


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
