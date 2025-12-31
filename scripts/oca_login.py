#!/usr/bin/env python
"""
OCA (Oracle Code Assist) OAuth Login Helper.

This script initiates the PKCE OAuth flow for OCA authentication.
It opens a browser for authentication and handles the callback to cache the token.

Usage:
    poetry run python scripts/oca_login.py
    poetry run python scripts/oca_login.py --status  # Check current token status
    poetry run python scripts/oca_login.py --clear   # Clear cached token
"""

import argparse
import asyncio
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

import httpx


class OCALoginHandler:
    """Handles OCA OAuth PKCE login flow."""

    # IDCS OAuth configuration
    IDCS_CLIENT_ID = "a8331954c0cf48ba99b5dd223a14c6ea"
    IDCS_OAUTH_URL = "https://idcs-9dc693e80d9b469480d7afe00e743931.identity.oraclecloud.com"
    SCOPES = "openid offline_access"  # offline_access for refresh token

    # Local callback server - must match IDCS registered redirect URI
    CALLBACK_HOST = "127.0.0.1"
    CALLBACK_PORT = int(os.getenv("OCA_CALLBACK_PORT", "48801"))
    CALLBACK_PATH = "/auth/oca"  # Must match IDCS OAuth client configuration

    # Token cache
    CACHE_DIR = Path(os.getenv("OCA_CACHE_DIR", Path.home() / ".oca"))

    def __init__(self):
        self.auth_code = None
        self.code_verifier = None
        self.code_challenge = None
        self.state = None
        self.server_ready = threading.Event()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if needed."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _generate_pkce(self) -> None:
        """Generate PKCE code verifier and challenge."""
        # Generate a 43-128 character code verifier
        self.code_verifier = secrets.token_urlsafe(32)

        # Create SHA256 hash for challenge
        verifier_bytes = self.code_verifier.encode("ascii")
        digest = hashlib.sha256(verifier_bytes).digest()
        self.code_challenge = (
            base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        )

        # Generate random state
        self.state = secrets.token_urlsafe(16)

    @property
    def redirect_uri(self) -> str:
        """Get the redirect URI for OAuth callback."""
        return f"http://{self.CALLBACK_HOST}:{self.CALLBACK_PORT}{self.CALLBACK_PATH}"

    def _get_auth_url(self) -> str:
        """Build the IDCS authorization URL."""
        params = {
            "response_type": "code",
            "client_id": self.IDCS_CLIENT_ID,
            "redirect_uri": self.redirect_uri,
            "scope": self.SCOPES,
            "state": self.state,
            "code_challenge": self.code_challenge,
            "code_challenge_method": "S256",
        }

        return f"{self.IDCS_OAUTH_URL}/oauth2/v1/authorize?{urlencode(params)}"

    def _create_callback_handler(self):
        """Create the HTTP request handler for OAuth callback."""
        login_handler = self

        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Suppress default logging
                pass

            def do_GET(self):
                parsed = urlparse(self.path)

                if parsed.path == login_handler.CALLBACK_PATH:
                    query = parse_qs(parsed.query)

                    # Check state
                    if query.get("state", [None])[0] != login_handler.state:
                        self.send_error(400, "Invalid state")
                        return

                    # Get authorization code
                    if "code" in query:
                        login_handler.auth_code = query["code"][0]

                        self.send_response(200)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()

                        html = """
                        <html>
                        <head><title>OCA Login Success</title></head>
                        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                            <h1 style="color: green;">✓ OCA Login Successful!</h1>
                            <p>You can close this browser tab.</p>
                            <p>Return to your terminal to continue.</p>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode())

                    elif "error" in query:
                        error = query.get("error", ["unknown"])[0]
                        error_desc = query.get("error_description", [""])[0]

                        self.send_response(400)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()

                        html = f"""
                        <html>
                        <head><title>OCA Login Failed</title></head>
                        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                            <h1 style="color: red;">✗ OCA Login Failed</h1>
                            <p>Error: {error}</p>
                            <p>{error_desc}</p>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode())

                else:
                    self.send_error(404)

        return CallbackHandler

    async def _exchange_code_for_token(self) -> dict:
        """Exchange authorization code for access token."""
        token_url = f"{self.IDCS_OAUTH_URL}/oauth2/v1/token"

        data = {
            "grant_type": "authorization_code",
            "code": self.auth_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.IDCS_CLIENT_ID,
            "code_verifier": self.code_verifier,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                raise Exception(
                    f"Token exchange failed: {response.status_code} - {response.text}"
                )

            return response.json()

    def _cache_token(self, token: dict) -> None:
        """Save token to cache file."""
        self._ensure_cache_dir()

        # Add cache metadata
        now = time.time()
        token["_cached_at"] = now
        token["_expires_at"] = now + token.get("expires_in", 3600)
        token["_refresh_expires_at"] = now + 8 * 60 * 60  # 8 hours

        token_file = self.CACHE_DIR / "token.json"
        token_file.write_text(json.dumps(token, indent=2))
        token_file.chmod(0o600)

        print(f"\n  Token cached to: {token_file}")

    def get_token_status(self) -> dict:
        """Get current token status."""
        token_file = self.CACHE_DIR / "token.json"

        if not token_file.exists():
            return {"has_token": False, "is_valid": False, "can_refresh": False}

        try:
            token = json.loads(token_file.read_text())
            now = time.time()

            expires_at = token.get("_expires_at", 0)
            refresh_expires_at = token.get("_refresh_expires_at", 0)

            # 3 minute buffer for expiry
            is_valid = now < expires_at - 180
            can_refresh = (
                bool(token.get("refresh_token")) and now < refresh_expires_at
            )

            return {
                "has_token": True,
                "is_valid": is_valid,
                "can_refresh": can_refresh,
                "expires_in_seconds": max(0, expires_at - now),
                "refresh_expires_in_seconds": max(0, refresh_expires_at - now),
                "cached_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(token.get("_cached_at", 0)),
                ),
            }

        except Exception as e:
            return {"has_token": False, "error": str(e)}

    def clear_token(self) -> None:
        """Clear cached token."""
        token_file = self.CACHE_DIR / "token.json"
        if token_file.exists():
            token_file.unlink()
            print(f"  Token cleared: {token_file}")
        else:
            print("  No token to clear")

    async def login(self) -> bool:
        """Perform full OAuth login flow."""
        print("\n" + "=" * 60)
        print("  OCA OAuth Login")
        print("=" * 60)

        # Generate PKCE parameters
        self._generate_pkce()
        print(f"\n  Generated PKCE challenge (method: S256)")

        # Start local callback server
        handler = self._create_callback_handler()
        server = http.server.HTTPServer((self.CALLBACK_HOST, self.CALLBACK_PORT), handler)
        server.timeout = 120  # 2 minute timeout

        def run_server():
            self.server_ready.set()
            while self.auth_code is None:
                server.handle_request()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready
        self.server_ready.wait()
        print(f"  Callback server listening on port {self.CALLBACK_PORT}")

        # Open browser for authentication
        auth_url = self._get_auth_url()
        print(f"\n  Opening browser for authentication...")
        print(f"  If browser doesn't open, visit:\n  {auth_url[:80]}...")

        webbrowser.open(auth_url)

        # Wait for callback
        print("\n  Waiting for authentication (2 minute timeout)...")
        server_thread.join(timeout=120)

        if self.auth_code is None:
            print("\n  ERROR: Authentication timed out or failed")
            return False

        print("  Received authorization code")

        # Exchange code for token
        print("  Exchanging code for access token...")
        try:
            token = await self._exchange_code_for_token()
            self._cache_token(token)

            print("\n" + "-" * 60)
            print("  SUCCESS: OCA authentication complete!")
            print("-" * 60)

            # Show token info
            expires_in = token.get("expires_in", 0)
            print(f"\n  Access token expires in: {expires_in // 60} minutes")

            if token.get("refresh_token"):
                print("  Refresh token: Available")

            return True

        except Exception as e:
            print(f"\n  ERROR: Token exchange failed: {e}")
            return False


def print_status():
    """Print current token status."""
    handler = OCALoginHandler()
    status = handler.get_token_status()

    print("\n" + "=" * 60)
    print("  OCA Token Status")
    print("=" * 60)

    if not status.get("has_token"):
        print("\n  No token cached")
        if "error" in status:
            print(f"  Error: {status['error']}")
    else:
        print(f"\n  Has Token: {status['has_token']}")
        print(f"  Is Valid: {status['is_valid']}")
        print(f"  Can Refresh: {status['can_refresh']}")

        if status.get("expires_in_seconds"):
            mins = status["expires_in_seconds"] / 60
            print(f"  Expires In: {mins:.1f} minutes")

        if status.get("refresh_expires_in_seconds"):
            hrs = status["refresh_expires_in_seconds"] / 3600
            print(f"  Refresh Expires In: {hrs:.1f} hours")

        if status.get("cached_at"):
            print(f"  Cached At: {status['cached_at']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="OCA OAuth Login Helper")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current token status",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear cached token",
    )

    args = parser.parse_args()

    handler = OCALoginHandler()

    if args.status:
        print_status()
    elif args.clear:
        handler.clear_token()
    else:
        # Perform login
        success = asyncio.run(handler.login())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
