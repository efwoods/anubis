"""Reference (research-only) Slack adapter that fronts the deep agent.

Not imported by anything in src/. Demonstrates the three Slack entry points
described in 03_slack_interface.md.

Run in dev (Socket Mode):

    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    export SLACK_SIGNING_SECRET=...
    python -m research.data_analysis_slack_agent.reference_code.slack_bolt_app
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler


# Stub agent so this file is runnable without the full deepagents stack present.
# Production wiring imports `data_analysis_graph` from data_analysis_graph.py.
async def _agent_invoke(channel: str, user: str, prompt: str) -> str:
    return (
        f":dna: anubis received your request from <@{user}> in <#{channel}>.\n"
        f"prompt: {prompt}\n"
        "Production build will run the deep agent here and post the AAR + chart."
    )


bolt_app = AsyncApp(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)


async def _run(channel: str, user: str, prompt: str):
    """Long-running agent invocation; isolated from the 3-second ack."""
    interim = await bolt_app.client.chat_postMessage(
        channel=channel,
        text=f":hourglass_flowing_sand: working on this for <@{user}>: `{prompt[:120]}`",
    )
    try:
        result = await _agent_invoke(channel=channel, user=user, prompt=prompt)
    except Exception as exc:
        await bolt_app.client.chat_postMessage(
            channel=channel,
            thread_ts=interim["ts"],
            text=f":warning: AAR failed: `{exc}`",
        )
        return
    await bolt_app.client.chat_postMessage(
        channel=channel,
        thread_ts=interim["ts"],
        text=result,
    )


@bolt_app.command("/aar")
async def handle_aar(ack, command, respond):
    await ack()
    window = (command.get("text") or "last-24h").strip()
    asyncio.create_task(
        _run(
            channel=command["channel_id"],
            user=command["user_id"],
            prompt=f"Produce an AAR for window={window}.",
        )
    )


@bolt_app.event("app_mention")
async def handle_mention(event, say):
    text = event["text"].split(">", 1)[-1].strip()
    if any(text.lower().startswith(prefix) for prefix in ("status", "snapshot", "how are things")):
        prompt = "Give a one-message live overview: company direction + sleep quality. No chart."
    else:
        prompt = text
    asyncio.create_task(_run(channel=event["channel"], user=event["user"], prompt=prompt))


@bolt_app.event("message")
async def handle_dm(event, say):
    if event.get("channel_type") != "im" or event.get("bot_id"):
        return
    asyncio.create_task(_run(channel=event["channel"], user=event["user"], prompt=event["text"]))


# ---------------------------------------------------------------------------
# Two run modes — Socket Mode for dev, FastAPI mount for prod.
# ---------------------------------------------------------------------------


async def main_socket():
    """Dev: connect via app-level token, no public URL needed."""
    handler = AsyncSocketModeHandler(bolt_app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()


def fastapi_handler():
    """Prod: mount this in src/api/webapp.py.

    Example:
        from src.api.slack_routes import slack_handler

        @slack_router.post("/slack/events")
        async def slack_events(req: Request):
            return await slack_handler.handle(req)
    """
    from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

    return AsyncSlackRequestHandler(bolt_app)


if __name__ == "__main__":
    asyncio.run(main_socket())
