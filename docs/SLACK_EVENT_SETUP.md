# Slack Event Subscriptions Setup

## Current Status: Socket Mode Connected, Events Not Configured

The Slack bot is successfully connected via Socket Mode (PING/PONG heartbeats working),
but **Event Subscriptions are not enabled**, causing the bot to not receive any user messages.

## Required Configuration

### 1. Go to Slack App Settings
Navigate to: https://api.slack.com/apps → Select your app

### 2. Enable Event Subscriptions
**Settings → Event Subscriptions**

1. Toggle **"Enable Events"** to ON
2. Under **"Subscribe to bot events"**, add:
   - `app_mention` - Receive @mentions
   - `message.im` - Receive direct messages (REQUIRED)
   - `message.channels` - Receive channel messages
   - `message.groups` - Receive private channel messages
   - `message.mpim` - Receive group DM messages

3. Click **"Save Changes"**

### 3. Verify Socket Mode
**Settings → Socket Mode**

1. Ensure Socket Mode is **enabled** (toggle ON)
2. App-level token should start with `xapp-`

### 4. Reinstall App (if needed)
After adding events, you may need to reinstall:
**Settings → Install App → Reinstall to Workspace**

## Verification

After configuration, restart the coordinator and check logs for:

```
✅ "Received message (type: events_api ..." - Events are working
❌ Only "PING/PONG" messages - Events not configured
```

## Current OAuth Scopes (Verified)

The bot has these scopes (sufficient for messaging):
- `app_mentions:read` - Receive @mentions ✅
- `im:history` - Read DM history ✅
- `im:write` - Send DMs ✅
- `channels:history` - Read channel messages ✅
- `chat:write` - Send messages ✅

## Troubleshooting Commands

```bash
# Run diagnostics
poetry run python scripts/diagnose_slack.py

# Check event subscriptions in logs
grep -E "events_api|app_mention|message.im" /path/to/coordinator/logs
```
