# Troubleshooting Wiki - OCI AI Agent Coordinator

This wiki documents solved issues and their solutions. Use this as a baseline when troubleshooting new issues. If a problem seems similar to a documented issue, try the corresponding solution first.

**Last Updated**: 2026-01-04

---

## Table of Contents

1. [Slack Integration Issues](#slack-integration-issues)
2. [OCA OAuth Issues](#oca-oauth-issues)
3. [MCP Server Issues](#mcp-server-issues)
4. [Environment Configuration](#environment-configuration)

---

## Slack Integration Issues

### Issue: `invalid_auth` Error on Slack Connection

**Symptoms:**
- Slack bot fails to start
- Error message: `invalid_auth` or `The server responded with: {'ok': False, 'error': 'invalid_auth'}`
- Bot token appears valid but connection fails

**Root Causes (Check in order):**

1. **Token Typo in `.env.local`**
   - Check if token has typo like `xxoxb-` instead of `xoxb-`
   - Token must start with exactly `xoxb-` (not `xxoxb-`, `xoxb -`, etc.)

2. **Shell Environment Override**
   - Shell profile (`~/.zshrc`, `~/.bashrc`) may export placeholder tokens
   - Run `echo $SLACK_BOT_TOKEN` to see actual value in shell
   - If shows placeholder like `xoxb-your-slack-bot-token`, the shell is overriding `.env.local`

**Solutions:**

1. **Fix token in `.env.local`:**
   ```bash
   # Verify token format
   grep SLACK_BOT_TOKEN .env.local
   # Should show: SLACK_BOT_TOKEN=xoxb-... (real token)
   ```

2. **Fix shell environment override:**
   ```bash
   # Check if shell exports placeholder
   grep SLACK ~/.zshrc

   # If found, comment out or remove:
   # export SLACK_BOT_TOKEN="xoxb-your-slack..."  # REMOVE THIS

   # Then reload or unset in current session
   unset SLACK_BOT_TOKEN SLACK_TEAM_ID
   ```

3. **Verify tokens work:**
   ```bash
   poetry run python scripts/verify_slack_tokens.py
   ```

**Prevention:**
- Never export placeholder tokens in shell profiles
- Keep real tokens only in `.env.local` (git-ignored)
- Use `verify_slack_tokens.py` script after any token changes

---

### Issue: Slack Bot Connects but Stops Responding

**Symptoms:**
- Bot initially works, then stops responding to messages
- No error messages in logs
- Process is still running but appears hung

**Root Causes:**

1. **Stale WebSocket Connection**
   - Network changes (suspend/resume, VPN) can break Socket Mode connection
   - Connection appears alive but messages don't flow

2. **Event Loop Issues**
   - Multiple async event loops can cause deadlocks
   - MCP connections may conflict with Slack's event loop

**Solutions:**

1. **Restart the bot:**
   ```bash
   # Find and kill the process
   pgrep -f "src.main" | xargs kill -9

   # Restart with fresh environment
   unset SLACK_BOT_TOKEN SLACK_TEAM_ID
   poetry run python -m src.main --mode slack
   ```

2. **Check Socket Mode connection:**
   - Look for `Starting to receive messages from a new connection` in logs
   - Should see periodic `hello` messages from Slack

**Prevention:**
- The system now uses `AsyncRuntime` to share a single event loop
- Restart bot after laptop sleep/resume if issues occur

---

## OCA OAuth Issues

### Issue: OAuth Redirect Not Working (callback server not responding)

**Symptoms:**
- User clicks "Login with Oracle SSO" in Slack
- SSO login completes successfully in browser
- Browser shows error or hangs at `http://127.0.0.1:48801/auth/oca`
- No "Authentication Successful" page shown

**Root Causes:**

1. **Callback Server Hung on Stale Connections**
   - After system suspend/resume, server may have stale socket connections
   - Server is listening but not processing requests

2. **Callback Server Not Running**
   - Port 48801 may not be bound
   - Process crashed or was never started

**Diagnosis:**

```bash
# Check if port is listening
lsof -i :48801

# Should show Python process LISTENING:
# Python  74828 user  6u  IPv4 ...  TCP *:48801 (LISTEN)

# Check if server responds to health check
curl -s --max-time 3 http://127.0.0.1:48801/health
# Should return: {"status": "ok", "service": "oca-callback"}
```

**Solutions:**

1. **If health check times out but port is listening:**
   ```bash
   # Server is stuck - kill and restart
   pgrep -f "src.main" | xargs kill -9
   sleep 2

   # Restart bot (callback server starts automatically)
   unset SLACK_BOT_TOKEN SLACK_TEAM_ID
   poetry run python -m src.main --mode slack
   ```

2. **If port is not listening:**
   ```bash
   # Callback server didn't start - check logs for errors
   # Then restart the bot
   poetry run python -m src.main --mode slack
   ```

3. **If running standalone auth:**
   ```bash
   # Use the auth script
   poetry run python scripts/oca_auth.py
   ```

**Prevention:**
- Callback server now has socket timeouts (10s) to prevent hangs
- `RobustHTTPServer` with error recovery handles stale connections
- After system resume, restart the bot for fresh connections

---

### Issue: OCA Token Expired / Authentication Loop

**Symptoms:**
- Repeated "Authentication required" prompts
- User authenticates but immediately asked again
- Token appears to be refreshed but not used

**Root Causes:**

1. **Refresh Token Expired**
   - Refresh tokens are valid for 8 hours
   - After 8 hours, full re-authentication is required

2. **Token Cache Permissions**
   - `~/.oca/token.json` may have wrong permissions
   - Other processes may be writing invalid tokens

**Solutions:**

1. **Check token status:**
   ```bash
   poetry run python scripts/oca_auth.py --status
   ```

2. **Clear and re-authenticate:**
   ```bash
   rm ~/.oca/token.json
   poetry run python scripts/oca_auth.py
   ```

3. **Check file permissions:**
   ```bash
   ls -la ~/.oca/
   # token.json should be -rw------- (600)
   ```

---

## MCP Server Issues

### Issue: MCP Server Connection Failed

**Symptoms:**
- `MCP client connection failed` in logs
- Specific tools unavailable
- `Server not connected` errors

**Root Causes:**

1. **Server Process Not Found**
   - Working directory path incorrect
   - Python/command not found

2. **Server Timeout**
   - Server takes too long to start
   - Resource-heavy initialization

**Solutions:**

1. **Verify server configuration:**
   ```bash
   # Check config/mcp_servers.yaml
   cat config/mcp_servers.yaml | grep -A 5 "database-observatory"

   # Verify working_dir exists
   ls -la /path/to/server
   ```

2. **Test server manually:**
   ```bash
   cd /path/to/server
   poetry run python -m mcp_server
   ```

3. **Check server logs:**
   - Look for initialization errors in stderr
   - Verify required env vars are set

---

## Environment Configuration

### Issue: Environment Variables Not Loading

**Symptoms:**
- Values from `.env.local` not being used
- Shell environment overriding application config
- Different behavior between terminal sessions

**Root Causes:**

1. **Shell Profile Exports**
   - `~/.zshrc` or `~/.bashrc` exporting variables
   - These take precedence over `.env.local`

2. **File Not Found**
   - `.env.local` in wrong location
   - Dotenv not loading file

**Solutions:**

1. **Check shell exports:**
   ```bash
   grep -E "^export" ~/.zshrc | grep -E "(SLACK|OCA|OCI)"

   # Comment out any placeholder exports
   ```

2. **Verify dotenv loading:**
   ```bash
   # Check .env.local exists
   ls -la .env.local

   # Verify it's being loaded
   python -c "from dotenv import load_dotenv; load_dotenv('.env.local'); import os; print(os.getenv('SLACK_BOT_TOKEN', 'NOT SET')[:10])"
   ```

3. **Unset in current session:**
   ```bash
   unset SLACK_BOT_TOKEN SLACK_APP_TOKEN SLACK_SIGNING_SECRET SLACK_TEAM_ID
   ```

---

## Quick Diagnostics

### Slack Connection

```bash
# Verify tokens
poetry run python scripts/verify_slack_tokens.py

# Check for shell overrides
echo "SLACK_BOT_TOKEN: ${SLACK_BOT_TOKEN:0:15}..."
```

### OCA Authentication

```bash
# Check callback server
curl -s --max-time 3 http://127.0.0.1:48801/health

# Check token status
poetry run python scripts/oca_auth.py --status
```

### MCP Servers

```bash
# Check server connectivity via API
curl -s http://localhost:3001/mcp/servers | jq '.servers[] | {name, status}'
```

---

## Adding New Issues

When you solve a new issue, add it to this wiki:

1. **Document the symptoms** - What did you observe?
2. **Document root causes** - What caused it? (list in order of likelihood)
3. **Document solutions** - Step-by-step fix
4. **Add prevention** - How to avoid this in the future

This helps the AI assistant (and humans) quickly diagnose and fix recurring issues.
