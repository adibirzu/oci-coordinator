#!/usr/bin/env python3
"""Slack Token Verification Script.

Verifies Slack credentials without starting the full application.
Run this after updating tokens in .env.local to confirm they work.

Usage:
    poetry run python scripts/verify_slack_tokens.py
"""

import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file, override=True)
    print(f"✓ Loaded environment from {env_file}")
else:
    print(f"✗ No .env.local found at {env_file}")
    sys.exit(1)

import httpx


def mask_token(token: str | None) -> str:
    """Mask token for safe display."""
    if not token:
        return "(not set)"
    if len(token) < 20:
        return f"{token[:4]}...{token[-4:]}"
    return f"{token[:10]}...{token[-6:]}"


def check_bot_token() -> bool:
    """Verify the bot token via auth.test API."""
    token = os.getenv("SLACK_BOT_TOKEN")
    print(f"\n📋 Bot Token: {mask_token(token)}")

    if not token:
        print("  ✗ SLACK_BOT_TOKEN not set")
        return False

    if not token.startswith("xoxb-"):
        print(f"  ✗ Invalid format - should start with 'xoxb-', got '{token[:5]}...'")
        return False

    print("  ✓ Format valid (xoxb-...)")

    # Test with Slack API
    try:
        response = httpx.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        data = response.json()

        if data.get("ok"):
            print("  ✓ Token valid!")
            print(f"    Team: {data.get('team', 'unknown')}")
            print(f"    Bot: {data.get('user', 'unknown')}")
            print(f"    Bot ID: {data.get('user_id', 'unknown')}")
            return True
        else:
            error = data.get("error", "unknown")
            print(f"  ✗ Token rejected: {error}")
            if error == "invalid_auth":
                print("    → Token is expired, revoked, or from a different app")
                print("    → Regenerate at: https://api.slack.com/apps → OAuth & Permissions")
            elif error == "token_revoked":
                print("    → Token was explicitly revoked")
            elif error == "account_inactive":
                print("    → The workspace or bot user is deactivated")
            return False
    except Exception as e:
        print(f"  ✗ API call failed: {e}")
        return False


def check_app_token() -> bool:
    """Verify the app-level token for Socket Mode."""
    token = os.getenv("SLACK_APP_TOKEN")
    print(f"\n📋 App Token: {mask_token(token)}")

    if not token:
        print("  ✗ SLACK_APP_TOKEN not set")
        return False

    if not token.startswith("xapp-"):
        print(f"  ✗ Invalid format - should start with 'xapp-', got '{token[:5]}...'")
        return False

    print("  ✓ Format valid (xapp-...)")

    # Test with Slack API (apps.connections.open)
    try:
        response = httpx.post(
            "https://slack.com/api/apps.connections.open",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        data = response.json()

        if data.get("ok"):
            print("  ✓ App token valid for Socket Mode!")
            return True
        else:
            error = data.get("error", "unknown")
            print(f"  ✗ App token rejected: {error}")
            if error == "invalid_auth":
                print("    → App token is expired or revoked")
                print("    → Regenerate at: https://api.slack.com/apps → Basic Information → App-Level Tokens")
            elif error == "missing_scope":
                print("    → App token missing 'connections:write' scope")
                print("    → Create new token with connections:write scope")
            return False
    except Exception as e:
        print(f"  ✗ API call failed: {e}")
        return False


def check_signing_secret() -> bool:
    """Check if signing secret is set (can't verify without a request)."""
    secret = os.getenv("SLACK_SIGNING_SECRET")
    print(f"\n📋 Signing Secret: {mask_token(secret)}")

    if not secret:
        print("  ✗ SLACK_SIGNING_SECRET not set")
        return False

    if len(secret) != 32:
        print(f"  ⚠ Unusual length ({len(secret)} chars, expected 32)")
    else:
        print("  ✓ Length valid (32 chars)")

    print("  ℹ Cannot verify signing secret without an actual Slack request")
    return True


def main():
    print("=" * 60)
    print("Slack Token Verification")
    print("=" * 60)

    results = {
        "bot_token": check_bot_token(),
        "app_token": check_app_token(),
        "signing_secret": check_signing_secret(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_ok = all(results.values())

    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name.replace('_', ' ').title()}")

    if all_ok:
        print("\n✅ All tokens valid! You can start the Slack bot.")
    else:
        print("\n❌ Some tokens are invalid. Please update .env.local with new tokens.")
        print("\nTo regenerate tokens:")
        print("1. Go to https://api.slack.com/apps")
        print("2. Select your app")
        print("3. Bot Token: OAuth & Permissions → Bot User OAuth Token → Reinstall")
        print("4. App Token: Basic Information → App-Level Tokens → Generate")
        print("5. Signing Secret: Basic Information → App Credentials → Signing Secret")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
