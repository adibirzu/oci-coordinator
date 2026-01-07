#!/usr/bin/env python3
"""
Diagnose Slack Bot Configuration Issues.

This script checks if your Slack app is properly configured to receive messages.

Usage:
    poetry run python scripts/diagnose_slack.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env.local FIRST - before any imports that might read env vars
from dotenv import load_dotenv

env_file = project_root / ".env.local"
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"✅ Loaded environment from {env_file}")
else:
    print(f"⚠️  No .env.local found at {env_file}")

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def diagnose_slack():
    """Run Slack configuration diagnostics."""
    print()
    print("=" * 60)
    print("🔍 Slack Bot Diagnostics")
    print("=" * 60)

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token:
        print("❌ SLACK_BOT_TOKEN not set in environment")
        return

    if not app_token:
        print("❌ SLACK_APP_TOKEN not set in environment")
        return

    print("✅ Environment variables found")
    print()

    client = WebClient(token=bot_token)

    # 1. Verify bot token
    print("1️⃣  Verifying bot token...")
    try:
        auth = client.auth_test()
        print(f"   ✅ Bot: {auth['user']} (ID: {auth['user_id']})")
        print(f"   ✅ Team: {auth['team']} (ID: {auth['team_id']})")
        print(f"   ✅ Bot ID: {auth['bot_id']}")
    except SlackApiError as e:
        error = e.response['error']
        print(f"   ❌ Auth failed: {error}")
        print()
        if error == "invalid_auth":
            print("   " + "=" * 55)
            print("   🚨 SLACK_BOT_TOKEN IS INVALID OR EXPIRED")
            print("   " + "=" * 55)
            print()
            print("   This is the root cause of why messages aren't working!")
            print()
            print("   TO FIX:")
            print("   1. Go to: https://api.slack.com/apps")
            print("   2. Select your Slack App")
            print("   3. Go to 'OAuth & Permissions' in the sidebar")
            print("   4. Find 'Bot User OAuth Token' (starts with xoxb-)")
            print("   5. Copy the token")
            print("   6. Update SLACK_BOT_TOKEN in your .env.local file")
            print("   7. Restart the coordinator: poetry run python -m src.main")
            print()
            print("   If the token field is empty:")
            print("   → You need to reinstall the app to your workspace")
            print("   → Click 'Install to Workspace' or 'Reinstall to Workspace'")
            print()
            # Show current token (masked)
            masked = bot_token[:10] + "..." + bot_token[-4:] if len(bot_token) > 14 else "***"
            print(f"   Current token in .env.local: {masked}")
            print()
        return
    print()

    # 2. Check OAuth scopes
    print("2️⃣  Checking OAuth scopes...")
    required_scopes = [
        "app_mentions:read",
        "chat:write",
        "im:history",
        "im:write",
        "channels:history",
    ]
    print("   ℹ️  Scopes should include:")
    for scope in required_scopes:
        print(f"      - {scope}")
    print("   ℹ️  Check your Slack App → OAuth & Permissions → Scopes")
    print()

    # 3. Check Event Subscriptions (this is critical!)
    print("3️⃣  Event Subscriptions (CRITICAL for receiving messages)...")
    print("   ⚠️  Cannot verify via API - must check manually in Slack App settings:")
    print()
    print("   📋 Go to: https://api.slack.com/apps → Your App → Event Subscriptions")
    print()
    print("   Required Bot Events for Socket Mode:")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │  ☐ app_mention      - Receive @mentions                    │")
    print("   │  ☐ message.im       - Receive DMs (REQUIRED!)              │")
    print("   │  ☐ message.channels - Receive channel messages             │")
    print("   │  ☐ message.groups   - Receive private channel messages     │")
    print("   │  ☐ message.mpim     - Receive group DM messages            │")
    print("   └─────────────────────────────────────────────────────────────┘")
    print()
    print("   ❗ If these events are NOT subscribed, the bot will NOT")
    print("   ❗ receive any messages - only PING/PONG heartbeats!")
    print()

    # 4. Check Socket Mode
    print("4️⃣  Socket Mode Configuration...")
    if app_token and app_token.startswith("xapp-"):
        print("   ✅ App-level token format is correct (xapp-...)")
        print("   ℹ️  Verify Socket Mode is enabled:")
        print("      → Slack App → Socket Mode → Toggle ON")
    else:
        print("   ❌ App token should start with 'xapp-'")
    print()

    # 5. Check bot's channel memberships
    print("5️⃣  Bot's Channel Memberships...")
    try:
        conversations = client.conversations_list(
            types="public_channel,private_channel,im,mpim",
            limit=100
        )

        channels = conversations.get("channels", [])
        public = [c for c in channels if c.get("is_channel") and not c.get("is_private")]
        private = [c for c in channels if c.get("is_channel") and c.get("is_private")]
        dms = [c for c in channels if c.get("is_im")]

        print(f"   📢 Public channels: {len(public)}")
        for c in public[:5]:
            member = "✅" if c.get("is_member") else "❌"
            print(f"      {member} #{c.get('name', 'unknown')}")
        if len(public) > 5:
            print(f"      ... and {len(public) - 5} more")

        print(f"   🔒 Private channels: {len(private)}")
        print(f"   💬 Direct messages: {len(dms)}")

        if len(public) == 0 and len(dms) == 0:
            print()
            print("   ⚠️  Bot is not in any channels!")
            print("   ⚠️  Invite the bot to a channel: /invite @oracle_oci_agent")

    except SlackApiError as e:
        print(f"   ⚠️  Could not list conversations: {e.response['error']}")
    print()

    # 6. Summary
    print("=" * 60)
    print("📋 TROUBLESHOOTING CHECKLIST")
    print("=" * 60)
    print()
    print("If you're not receiving messages, check:")
    print()
    print("1. Event Subscriptions are enabled in Slack App settings")
    print("   → Enable: message.im, app_mention, message.channels")
    print()
    print("2. Socket Mode is enabled")
    print("   → Slack App → Socket Mode → Toggle ON")
    print()
    print("3. For channel messages:")
    print("   → Bot must be invited to the channel")
    print("   → Use @mention to trigger the bot (or reply in a thread)")
    print()
    print("4. For DMs:")
    print("   → Just message the bot directly")
    print("   → message.im event must be subscribed")
    print()
    print("After making changes, restart the coordinator:")
    print("   poetry run python -m src.main")
    print()


def test_socket_mode_connection():
    """Test Socket Mode connection to verify events can be received."""
    import asyncio

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token or not app_token:
        print("⚠️  Missing tokens, skipping Socket Mode test")
        return False

    print()
    print("=" * 60)
    print("🔌 Testing Socket Mode Connection")
    print("=" * 60)
    print()

    async def test_connection():
        try:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
            from slack_bolt.async_app import AsyncApp
            from slack_sdk.web.async_client import AsyncWebClient

            print("1️⃣  Creating AsyncApp...")
            app = AsyncApp(token=bot_token)
            print("   ✅ AsyncApp created")

            # Register a test event handler
            events_received = []

            @app.event("app_mention")
            async def test_mention(event, say):
                events_received.append(event)
                print("   📨 Received app_mention event!")

            @app.event("message")
            async def test_message(event, say):
                if not event.get("bot_id") and not event.get("subtype"):
                    events_received.append(event)
                    print("   📨 Received message event!")

            print("2️⃣  Testing bot token with API...")
            client = AsyncWebClient(token=bot_token)
            auth = await client.auth_test()
            print(f"   ✅ Authenticated as: {auth['user']} (bot_id: {auth['bot_id']})")

            print("3️⃣  Connecting to Socket Mode...")
            handler = AsyncSocketModeHandler(app, app_token)

            # Try to connect for a few seconds to verify connection works
            print("   🔄 Establishing WebSocket connection...")

            # Create a connection task
            connection_task = asyncio.create_task(handler.connect_async())

            # Wait a bit for connection
            await asyncio.sleep(3)

            # Check if connected (async client may return a coroutine)
            is_connected = False
            if handler.client:
                result = handler.client.is_connected()
                if asyncio.iscoroutine(result):
                    is_connected = await result
                else:
                    is_connected = bool(result)

            if is_connected:
                print("   ✅ Socket Mode connected successfully!")
                print()
                print("   🎉 Connection test PASSED!")
                print()
                print("   The bot should now receive events.")
                print("   Try sending a DM or @mentioning the bot in a channel.")
                print()

                # Keep connection alive for a bit to receive test event
                print("   ⏳ Waiting 10 seconds for a test message...")
                print("      (Send a DM to the bot NOW to test)")
                await asyncio.sleep(10)

                if events_received:
                    print(f"   ✅ Received {len(events_received)} event(s)!")
                else:
                    print("   ℹ️  No events received (this is normal if you didn't send a message)")
            else:
                print("   ❌ Socket Mode connection failed!")

            # Clean up
            await handler.close_async()
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    return asyncio.run(test_connection())


def test_full_slack_handler():
    """Test the actual SlackHandler class from the codebase."""
    import asyncio

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")

    if not bot_token or not app_token:
        print("⚠️  Missing tokens, skipping full handler test")
        return False

    print()
    print("=" * 60)
    print("🧪 Testing Full SlackHandler Integration")
    print("=" * 60)
    print()

    async def test_handler():
        try:
            from src.channels.slack import SlackHandler

            print("1️⃣  Creating SlackHandler...")
            handler = SlackHandler(
                bot_token=bot_token,
                app_token=app_token,
            )
            print("   ✅ SlackHandler created")
            print(f"      bot_token: {handler.bot_token[:15]}...{handler.bot_token[-4:]}")
            print(f"      app_token: {handler.app_token[:15]}...{handler.app_token[-4:]}")

            print("2️⃣  Getting async_app...")
            async_app = handler.async_app
            print(f"   ✅ AsyncApp created: {type(async_app)}")

            print("3️⃣  Verifying token in app...")
            # The app should have the token configured
            from slack_sdk.web.async_client import AsyncWebClient
            test_client = AsyncWebClient(token=handler.bot_token)
            auth = await test_client.auth_test()
            print(f"   ✅ Token valid: {auth['user']} (team: {auth['team']})")

            print("4️⃣  Testing coordinator availability...")
            if hasattr(handler, "_invoke_coordinator"):
                print("   ✅ Coordinator entrypoint available (_invoke_coordinator)")
            else:
                print("   ⚠️  Coordinator entrypoint not found on SlackHandler")
                print("      (This is OK - coordinator wiring is initialized at runtime)")

            print()
            print("   🎉 SlackHandler integration test PASSED!")
            print()
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    return asyncio.run(test_handler())


if __name__ == "__main__":
    diagnose_slack()

    # Ask if user wants to run additional tests
    print()
    response = input("Run Socket Mode connection test? (y/n): ").strip().lower()
    if response == 'y':
        test_socket_mode_connection()

    print()
    response = input("Run full SlackHandler integration test? (y/n): ").strip().lower()
    if response == 'y':
        test_full_slack_handler()
