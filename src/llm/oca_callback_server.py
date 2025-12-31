"""OCA OAuth Callback Server.

Runs a background HTTP server to handle OAuth callbacks for OCA authentication.
This server starts automatically with the Slack bot and handles the redirect
after SSO login.

Usage:
    # Automatically started by main.py when running Slack mode
    # Or can be started manually:
    from src.llm.oca_callback_server import start_callback_server
    await start_callback_server()
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import structlog

logger = structlog.get_logger(__name__)

# OCA Configuration (matches oca.py)
IDCS_CLIENT_ID = "a8331954c0cf48ba99b5dd223a14c6ea"
IDCS_OAUTH_URL = "https://idcs-9dc693e80d9b469480d7afe00e743931.identity.oraclecloud.com"
TOKEN_ENDPOINT = f"{IDCS_OAUTH_URL}/oauth2/v1/token"

CALLBACK_HOST = os.getenv("OCA_CALLBACK_HOST", "127.0.0.1")
CALLBACK_PORT = int(os.getenv("OCA_CALLBACK_PORT", "48801"))
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/auth/oca"

CACHE_DIR = Path(os.getenv("OCA_CACHE_DIR", Path.home() / ".oca"))
TOKEN_CACHE_PATH = CACHE_DIR / "token.json"
VERIFIER_CACHE_PATH = CACHE_DIR / "verifier.txt"


def ensure_cache_dir() -> None:
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)


def load_verifier() -> Optional[str]:
    """Load saved PKCE verifier."""
    try:
        if VERIFIER_CACHE_PATH.exists():
            return VERIFIER_CACHE_PATH.read_text().strip()
    except Exception as e:
        logger.warning("Failed to load verifier", error=str(e))
    return None


def clear_verifier() -> None:
    """Clear saved verifier."""
    if VERIFIER_CACHE_PATH.exists():
        VERIFIER_CACHE_PATH.unlink()


def save_token(token: dict) -> None:
    """Save token to cache."""
    import time

    now = time.time()
    token["_cached_at"] = now
    token["_expires_at"] = now + token.get("expires_in", 3600)
    token["_refresh_expires_at"] = now + 8 * 60 * 60  # 8 hours

    ensure_cache_dir()
    TOKEN_CACHE_PATH.write_text(json.dumps(token, indent=2))
    TOKEN_CACHE_PATH.chmod(0o600)
    logger.info("OCA token saved", path=str(TOKEN_CACHE_PATH))


def exchange_code_for_token(code: str, verifier: str) -> Optional[dict]:
    """Exchange authorization code for token."""
    try:
        logger.info("Exchanging OAuth code for token")
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
            logger.error(
                "Token exchange failed",
                status=response.status_code,
                response=response.text[:200],
            )
            return None

        return response.json()
    except Exception as e:
        logger.error("Token exchange error", error=str(e))
        return None


class OCACallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callbacks."""

    # Class-level callback for token received
    on_token_received: Optional[Callable[[dict], None]] = None

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def do_GET(self):
        """Handle GET requests (OAuth callback)."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/auth/oca":
            self._handle_oauth_callback(query)
        elif parsed.path == "/health":
            self._handle_health_check()
        else:
            self.send_error(404)

    def _handle_oauth_callback(self, query: dict) -> None:
        """Handle OAuth callback from IDCS."""
        # Check for error
        if "error" in query:
            error = query.get("error", ["Unknown"])[0]
            error_desc = query.get("error_description", [""])[0]
            logger.error("OAuth error", error=error, description=error_desc)

            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"""
                <html>
                <body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1 style="color: #e74c3c;">Authentication Failed</h1>
                    <p>Error: {error}</p>
                    <p>{error_desc}</p>
                    <p>You can close this window.</p>
                </body>
                </html>
            """.encode())
            return

        # Get authorization code
        code = query.get("code", [None])[0]
        if not code:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            return

        # Load PKCE verifier
        verifier = load_verifier()
        if not verifier:
            logger.error("No PKCE verifier found for callback")
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1 style="color: #e74c3c;">Authentication Failed</h1>
                    <p>No PKCE verifier found. Please try logging in again from Slack.</p>
                </body>
                </html>
            """)
            return

        # Exchange code for token
        token = exchange_code_for_token(code, verifier)
        if token:
            save_token(token)
            clear_verifier()

            # Notify callback if registered
            if OCACallbackHandler.on_token_received:
                try:
                    OCACallbackHandler.on_token_received(token)
                except Exception as e:
                    logger.warning("Token callback failed", error=str(e))

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write("""
                <html>
                <body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1 style="color: #27ae60;">Authentication Successful!</h1>
                    <p>You can close this window and return to Slack.</p>
                    <p>Your session is now active.</p>
                </body>
                </html>
            """.encode())

            logger.info(
                "OCA authentication successful",
                expires_in=token.get("expires_in", 0),
            )
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="font-family: system-ui; text-align: center; padding: 50px;">
                    <h1 style="color: #e74c3c;">Authentication Failed</h1>
                    <p>Token exchange failed. Please try again.</p>
                </body>
                </html>
            """)

    def _handle_health_check(self) -> None:
        """Handle health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "service": "oca-callback"}).encode())


class OCACallbackServer:
    """Background OCA callback server manager."""

    _instance: Optional["OCACallbackServer"] = None
    _server: Optional[HTTPServer] = None
    _thread: Optional[threading.Thread] = None
    _running: bool = False

    @classmethod
    def get_instance(cls) -> "OCACallbackServer":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, on_token_received: Optional[Callable[[dict], None]] = None) -> bool:
        """Start the callback server in a background thread.

        Args:
            on_token_received: Optional callback when token is received

        Returns:
            True if started successfully
        """
        if self._running:
            logger.debug("OCA callback server already running")
            return True

        try:
            # Set callback
            if on_token_received:
                OCACallbackHandler.on_token_received = on_token_received

            # Create server
            self._server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), OCACallbackHandler)
            self._server.timeout = 1  # Short timeout for clean shutdown

            # Start server thread
            self._thread = threading.Thread(
                target=self._run_server,
                name="oca-callback-server",
                daemon=True,
            )
            self._running = True
            self._thread.start()

            logger.info(
                "OCA callback server started",
                host=CALLBACK_HOST,
                port=CALLBACK_PORT,
                url=f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/auth/oca",
            )
            return True

        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(
                    "OCA callback server port in use (another instance may be running)",
                    port=CALLBACK_PORT,
                )
                # Port in use is OK - means another server is handling callbacks
                return True
            logger.error("Failed to start OCA callback server", error=str(e))
            return False
        except Exception as e:
            logger.error("Failed to start OCA callback server", error=str(e))
            return False

    def _run_server(self) -> None:
        """Server loop running in background thread."""
        while self._running:
            try:
                self._server.handle_request()
            except Exception as e:
                if self._running:
                    logger.warning("Callback server error", error=str(e))

    def stop(self) -> None:
        """Stop the callback server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

        logger.info("OCA callback server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def url(self) -> str:
        """Get the callback URL."""
        return f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/auth/oca"


async def start_callback_server(
    on_token_received: Optional[Callable[[dict], None]] = None
) -> bool:
    """Start the OCA callback server.

    This is the main entry point for starting the callback server.
    It runs in a background thread and handles OAuth redirects.

    Args:
        on_token_received: Optional callback when token is received

    Returns:
        True if started successfully
    """
    server = OCACallbackServer.get_instance()
    return server.start(on_token_received)


def stop_callback_server() -> None:
    """Stop the OCA callback server."""
    server = OCACallbackServer.get_instance()
    server.stop()


def is_callback_server_running() -> bool:
    """Check if callback server is running."""
    server = OCACallbackServer.get_instance()
    return server.is_running
