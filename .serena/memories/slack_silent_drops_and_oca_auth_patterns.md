# Slack Silent Message Drops & OCA Auth Issues — Troubleshooting Patterns

## Issue 1: No Response in Slack at All

### Most Likely: Bot Not Running
- **Check first**: `ps aux | grep "python.*src.main" | grep -v grep`
- If no process, start with: `nohup poetry run python -m src.main > /tmp/oci-coordinator.log 2>&1 &`
- Don't use Claude Code's `run_in_background` for long-running servers — the 2-min timeout kills them
- **Logs location**: `/tmp/oci-coordinator.log` (when started with nohup)

### Secondary: Multiple Instances Running
- `lsof -i :48803` and `lsof -i :3001` to check for port conflicts
- Kill old instances before starting new ones: `kill <old_pid>`
- The callback server silently returns `True` on "port in use" but never creates `self._server`

### Error Boundaries in `src/channels/slack.py`
Three layers of error protection were added (Feb 2026):
1. **Sync handler level**: `_sync_error_reply()` helper wraps `run_async()` in `handle_mention` and `handle_message`
2. **`_process_message` level**: `try:` moved up to cover dedup/clean code before inner try. `except Exception` handler wraps its error-sending in try/catch with last-resort `chat_postMessage`
3. **`_safe_say_with_fallback` level**: 3-tier fallback: Block Kit → plain text → minimal `chat_postMessage` → `logger.critical`

## Issue 2: OCA SSO Page Blocked / No Success Message

### Root Cause: Missing HTTP Response Headers
- `BaseHTTPRequestHandler` defaults to HTTP/1.1 (keep-alive connections)
- Without `Content-Length` and `Connection: close`, the browser waits for more data indefinitely
- **Fix (Feb 2026)**: Added `_send_html()` helper in `OCACallbackHandler` that sends:
  - `Content-Length` header (exact byte count)
  - `Connection: close` header
  - `wfile.flush()` after writing
- **File**: `src/llm/oca_callback_server.py`

### Debugging OCA Auth Flow
1. Check callback server: `curl -s http://127.0.0.1:48803/health` (should return 200)
2. Check token freshness: `ls -la ~/.oca/token.json` (modification time)
3. Check verifier/state: `ls ~/.oca/verifier.txt ~/.oca/state.txt` (cleared after successful auth)
4. Port is configured in `.env.local`: `OCA_CALLBACK_PORT=48803`

### OCA Token Expired
- Startup log: `OCA token expired and cannot be refreshed. Please re-authenticate.`
- The bot will show "Authentication required" button in Slack when token is expired
- After auth, `token.json` is updated and `verifier.txt`/`state.txt` are deleted
- No fallback LLMs configured (lm_studio, ollama offline; anthropic, openai keys not set)

## Key Files
| File | Purpose |
|------|---------|
| `src/channels/slack.py` | Slack event handlers, error boundaries, auth URL generation |
| `src/llm/oca_callback_server.py` | OAuth callback HTTP server, token exchange |
| `src/llm/oca.py` | Token management, auth status checks |
| `src/channels/async_runtime.py` | `run_async()` bridge from sync→async |

## Startup Checklist
1. Kill any existing instances: `pkill -f "python.*src.main"`
2. Start: `nohup poetry run python -m src.main > /tmp/oci-coordinator.log 2>&1 &`
3. Verify: `grep "Slack async handlers registered" /tmp/oci-coordinator.log`
4. Check health: `curl -s http://127.0.0.1:48803/health` and `curl -s http://localhost:3001/api/health`
