#!/usr/bin/env python3
"""OCA Authentication Helper.

Runs the OAuth PKCE flow to obtain OCA access tokens.
Opens a browser for SSO login and handles the callback.

Usage:
    # Standalone login (generates PKCE and opens browser)
    poetry run python scripts/oca_auth.py

    # Callback-only mode (for Slack-initiated login)
    poetry run python scripts/oca_auth.py --callback-only

    # Check token status
    poetry run python scripts/oca_auth.py --status

For Slack-initiated logins:
    1. Run: poetry run python scripts/oca_auth.py --callback-only
    2. Click the 'Login with Oracle SSO' button in Slack
    3. Complete SSO in browser - callback server will handle the rest
"""

import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

# OCA Configuration (matches TypeScript client)
IDCS_CLIENT_ID = "a8331954c0cf48ba99b5dd223a14c6ea"
IDCS_OAUTH_URL = "https://idcs-9dc693e80d9b469480d7afe00e743931.identity.oraclecloud.com"
TOKEN_ENDPOINT = f"{IDCS_OAUTH_URL}/oauth2/v1/token"
AUTHZ_ENDPOINT = f"{IDCS_OAUTH_URL}/oauth2/v1/authorize"

CALLBACK_HOST = os.getenv("OCA_CALLBACK_HOST", "127.0.0.1")
CALLBACK_PORT = int(os.getenv("OCA_CALLBACK_PORT", "48801"))
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/auth/oca"

CACHE_DIR = Path(os.getenv("OCA_CACHE_DIR", Path.home() / ".oca"))
TOKEN_CACHE_PATH = CACHE_DIR / "token.json"
VERIFIER_CACHE_PATH = CACHE_DIR / "verifier.txt"


def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)


def generate_pkce_pair():
    """Generate PKCE verifier and challenge."""
    verifier = secrets.token_urlsafe(40)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    return verifier, challenge


def save_verifier(verifier: str):
    """Save verifier for later use."""
    ensure_cache_dir()
    VERIFIER_CACHE_PATH.write_text(verifier)
    VERIFIER_CACHE_PATH.chmod(0o600)


def load_verifier() -> str | None:
    """Load saved verifier."""
    try:
        if VERIFIER_CACHE_PATH.exists():
            return VERIFIER_CACHE_PATH.read_text().strip()
    except Exception:
        pass
    return None


def clear_verifier():
    """Clear saved verifier."""
    if VERIFIER_CACHE_PATH.exists():
        VERIFIER_CACHE_PATH.unlink()


def save_token(token: dict):
    """Save token to cache."""
    import time
    now = time.time()
    token["_cached_at"] = now
    token["_expires_at"] = now + token.get("expires_in", 3600)
    token["_refresh_expires_at"] = now + 8 * 60 * 60  # 8 hours

    ensure_cache_dir()
    TOKEN_CACHE_PATH.write_text(json.dumps(token, indent=2))
    TOKEN_CACHE_PATH.chmod(0o600)
    print(f"[OCA Auth] Token saved to {TOKEN_CACHE_PATH}")


def get_auth_url(verifier: str, challenge: str) -> str:
    """Generate OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": IDCS_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "openid offline_access",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTHZ_ENDPOINT}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(code: str, verifier: str) -> dict | None:
    """Exchange authorization code for token."""
    try:
        print("[OCA Auth] Exchanging code for token...")
        response = httpx.post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": IDCS_CLIENT_ID,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )

        if response.status_code != 200:
            print(f"[OCA Auth] Token exchange failed: {response.status_code}")
            print(response.text[:500])
            return None

        return response.json()
    except Exception as e:
        print(f"[OCA Auth] Token exchange error: {e}")
        return None


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handle OAuth callback."""

    token = None
    error = None

    def log_message(self, format, *args):
        pass  # Suppress logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/auth/oca":
            if "error" in query:
                CallbackHandler.error = query.get("error", ["Unknown"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                    <html>
                    <body style="font-family: system-ui; text-align: center; padding: 50px;">
                        <h1 style="color: #e74c3c;">Authentication Failed</h1>
                        <p>{CallbackHandler.error}</p>
                        <p>You can close this window.</p>
                    </body>
                    </html>
                """.encode())
                return

            code = query.get("code", [None])[0]
            if not code:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"Missing authorization code")
                return

            # Exchange code for token
            verifier = load_verifier()
            if not verifier:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"No verifier found")
                return

            token = exchange_code_for_token(code, verifier)
            if token:
                save_token(token)
                clear_verifier()
                CallbackHandler.token = token

                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write("""
                    <html>
                    <body style="font-family: system-ui; text-align: center; padding: 50px;">
                        <h1 style="color: #27ae60;">Authentication Successful!</h1>
                        <p>You can close this window and return to your terminal.</p>
                    </body>
                    </html>
                """.encode())
            else:
                CallbackHandler.error = "Token exchange failed"
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"Token exchange failed")
        else:
            self.send_response(404)
            self.end_headers()


def run_auth_flow():
    """Run the complete OAuth flow."""
    print("[OCA Auth] Starting authentication flow...")
    print(f"[OCA Auth] Callback server: http://{CALLBACK_HOST}:{CALLBACK_PORT}")

    # Generate PKCE
    verifier, challenge = generate_pkce_pair()
    save_verifier(verifier)

    # Get auth URL
    auth_url = get_auth_url(verifier, challenge)

    # Start callback server
    server = http.server.HTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    server.timeout = 300  # 5 minute timeout

    print(f"\n[OCA Auth] Opening browser for authentication...")
    print(f"[OCA Auth] If browser doesn't open, visit:")
    print(f"    {auth_url}\n")

    # Open browser
    webbrowser.open(auth_url)

    print("[OCA Auth] Waiting for authentication (timeout: 5 minutes)...")

    # Handle one request
    try:
        while CallbackHandler.token is None and CallbackHandler.error is None:
            server.handle_request()
    except KeyboardInterrupt:
        print("\n[OCA Auth] Cancelled by user")
        clear_verifier()
        return False
    finally:
        server.server_close()

    if CallbackHandler.error:
        print(f"[OCA Auth] Authentication failed: {CallbackHandler.error}")
        return False

    if CallbackHandler.token:
        print("\n[OCA Auth] Authentication successful!")
        print(f"[OCA Auth] Token expires in: {CallbackHandler.token.get('expires_in', 0)} seconds")
        return True

    return False


def check_token_status():
    """Check current token status."""
    import time

    if not TOKEN_CACHE_PATH.exists():
        print("[OCA Auth] No token found")
        return False

    try:
        token = json.loads(TOKEN_CACHE_PATH.read_text())
        now = time.time()
        expires_at = token.get("_expires_at", 0)
        refresh_expires_at = token.get("_refresh_expires_at", 0)

        print(f"[OCA Auth] Token status:")
        print(f"  - Expires in: {max(0, expires_at - now):.0f} seconds")
        print(f"  - Refresh expires in: {max(0, refresh_expires_at - now):.0f} seconds")
        print(f"  - Valid: {now < expires_at - 180}")  # 3 min buffer
        print(f"  - Can refresh: {now < refresh_expires_at}")

        return now < expires_at - 180 or now < refresh_expires_at
    except Exception as e:
        print(f"[OCA Auth] Error checking token: {e}")
        return False


def run_callback_only():
    """Run callback server only (for Slack-initiated flows).

    When user clicks login button in Slack, the Slack bot generates PKCE
    and opens the browser. This mode runs ONLY the callback server to
    handle the OAuth redirect, using the verifier already saved by Slack.
    """
    print("[OCA Auth] Starting callback-only mode (for Slack-initiated login)...")
    print(f"[OCA Auth] Callback server: http://{CALLBACK_HOST}:{CALLBACK_PORT}")

    # Check if verifier exists (should be saved by Slack bot)
    verifier = load_verifier()
    if not verifier:
        print("\n[OCA Auth] ERROR: No PKCE verifier found.")
        print("[OCA Auth] This mode requires a Slack-initiated login first.")
        print("[OCA Auth] Click the 'Login with Oracle SSO' button in Slack,")
        print("[OCA Auth] then run this command before completing auth in browser.")
        return False

    print("[OCA Auth] Found existing PKCE verifier (from Slack)")

    # Start callback server
    server = http.server.HTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    server.timeout = 300  # 5 minute timeout

    print("\n[OCA Auth] Waiting for OAuth callback (timeout: 5 minutes)...")
    print("[OCA Auth] Complete authentication in your browser, then the callback will be handled.")

    # Handle requests until we get a token or error
    try:
        while CallbackHandler.token is None and CallbackHandler.error is None:
            server.handle_request()
    except KeyboardInterrupt:
        print("\n[OCA Auth] Cancelled by user")
        return False
    finally:
        server.server_close()

    if CallbackHandler.error:
        print(f"[OCA Auth] Authentication failed: {CallbackHandler.error}")
        return False

    if CallbackHandler.token:
        print("\n[OCA Auth] Authentication successful!")
        print(f"[OCA Auth] Token expires in: {CallbackHandler.token.get('expires_in', 0)} seconds")
        return True

    return False


if __name__ == "__main__":
    print("=" * 60)
    print("OCA Authentication Helper")
    print("=" * 60)

    if "--status" in sys.argv:
        check_token_status()
        sys.exit(0)

    if "--callback-only" in sys.argv:
        # Callback-only mode for Slack-initiated logins
        print("\n[Mode: Callback Only - for Slack-initiated login]")
        success = run_callback_only()
        sys.exit(0 if success else 1)

    print("\nCurrent token status:")
    if check_token_status():
        print("\nToken is still valid or can be refreshed.")
        answer = input("Do you want to re-authenticate anyway? [y/N]: ")
        if answer.lower() != "y":
            print("Exiting.")
            sys.exit(0)

    print()
    success = run_auth_flow()
    sys.exit(0 if success else 1)
