"""
Thinking Trace Formatter for different output channels.

Formats the coordinator's thinking process for display in Slack, Teams, Markdown, etc.
"""

from __future__ import annotations

from typing import Any

from src.agents.coordinator.transparency import ThinkingPhase, ThinkingTrace


def format_thinking_for_slack(thinking_trace: ThinkingTrace | None) -> list[dict]:
    """
    Format thinking trace as Slack Block Kit blocks.

    Args:
        thinking_trace: The thinking trace from coordinator

    Returns:
        List of Slack block dictionaries
    """
    if not thinking_trace or not thinking_trace.steps:
        return []

    blocks = []

    # Header with collapsible indicator
    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": ":brain: *Thinking Process*",
        }]
    })

    # Build step summary
    step_texts = []
    for step in thinking_trace.steps[-6:]:  # Last 6 steps max
        emoji = _get_phase_emoji(step.phase)
        step_texts.append(f"{emoji} {step.message}")

    if step_texts:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(step_texts),
            }
        })

    # Show selected agent if any
    selected = [c for c in thinking_trace.agent_candidates if c.selected]
    if selected:
        agent = selected[0]
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": (
                    f":robot_face: *Selected:* {agent.agent_role} "
                    f"({agent.confidence:.0%} confidence)"
                ),
            }]
        })

    return blocks


def format_thinking_for_markdown(thinking_trace: ThinkingTrace | None) -> str:
    """
    Format thinking trace as Markdown for API/CLI output.

    Args:
        thinking_trace: The thinking trace from coordinator

    Returns:
        Markdown formatted string
    """
    if not thinking_trace or not thinking_trace.steps:
        return ""

    lines = ["### Thinking Process", ""]

    for step in thinking_trace.steps:
        emoji = _get_phase_emoji(step.phase)
        lines.append(f"- {emoji} {step.message}")

    # Show agent candidates if any
    if thinking_trace.agent_candidates:
        lines.append("")
        lines.append("**Agent Candidates:**")
        for candidate in thinking_trace.agent_candidates[:3]:
            selected = " (selected)" if candidate.selected else ""
            lines.append(
                f"- {candidate.agent_role}: {candidate.confidence:.0%}{selected}"
            )

    return "\n".join(lines)


def format_thinking_summary(thinking_trace: ThinkingTrace | None) -> str:
    """
    Get a compact one-line summary of the thinking process.

    Args:
        thinking_trace: The thinking trace from coordinator

    Returns:
        Short summary string
    """
    if thinking_trace and hasattr(thinking_trace, "to_compact_summary"):
        return thinking_trace.to_compact_summary()
    return ""


def format_response_with_thinking(
    response: str,
    thinking_trace: ThinkingTrace | None,
    output_format: str = "slack",
    include_thinking: bool = True,
) -> dict[str, Any]:
    """
    Format the final response with thinking trace.

    Args:
        response: The main response text
        thinking_trace: The thinking trace from coordinator
        output_format: Output format (slack, markdown, plain)
        include_thinking: Whether to include thinking trace

    Returns:
        Formatted response dict with 'text', 'blocks' (for Slack), etc.
    """
    result = {
        "text": response,
        "blocks": [],
        "thinking_summary": format_thinking_summary(thinking_trace),
    }

    if output_format == "slack":
        blocks = []

        # Add thinking trace if enabled
        if include_thinking and thinking_trace:
            thinking_blocks = format_thinking_for_slack(thinking_trace)
            blocks.extend(thinking_blocks)
            if thinking_blocks:
                blocks.append({"type": "divider"})

        # Main response content
        # Split response into chunks for Slack's 3000 char limit
        chunks = [response[i:i+2900] for i in range(0, len(response), 2900)]
        for chunk in chunks[:10]:  # Max 10 sections
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": chunk,
                }
            })

        result["blocks"] = blocks

    elif output_format == "markdown":
        if include_thinking and thinking_trace:
            thinking_md = format_thinking_for_markdown(thinking_trace)
            result["text"] = f"{thinking_md}\n\n---\n\n{response}"

    return result


def _get_phase_emoji(phase: ThinkingPhase) -> str:
    """Get emoji for a thinking phase."""
    emoji_map = {
        ThinkingPhase.RECEIVED: ":inbox_tray:",
        ThinkingPhase.ENHANCING: ":brain:",
        ThinkingPhase.ENHANCED: ":white_check_mark:",
        ThinkingPhase.CLASSIFYING: ":mag:",
        ThinkingPhase.CLASSIFIED: ":dart:",
        ThinkingPhase.DISCOVERING: ":busts_in_silhouette:",
        ThinkingPhase.DISCOVERED: ":raising_hand:",
        ThinkingPhase.ROUTING: ":railway_track:",
        ThinkingPhase.ROUTED: ":round_pushpin:",
        ThinkingPhase.DELEGATING: ":handshake:",
        ThinkingPhase.EXECUTING: ":gear:",
        ThinkingPhase.TOOL_CALL: ":wrench:",
        ThinkingPhase.TOOL_RESULT: ":package:",
        ThinkingPhase.SYNTHESIZING: ":sparkles:",
        ThinkingPhase.COMPLETE: ":white_check_mark:",
        ThinkingPhase.ERROR: ":x:",
    }
    return emoji_map.get(phase, ":gear:")
