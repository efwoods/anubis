# Slack Interface (Bolt-Python + LangGraph)

The agent fronts Slack via [`slack-bolt`](https://docs.slack.dev/tools/bolt-python/getting-started/), running under the existing FastAPI app in `src/api/webapp.py` so deployment topology stays unchanged.

## 1. Two deployment shapes

| Shape | Use case | How |
| --- | --- | --- |
| **HTTP / Events API** (prod) | Public domain, TLS, Slack delivers POSTs. | `AsyncSlackRequestHandler(app).handle(request)` from a FastAPI route. Requires `SLACK_SIGNING_SECRET`. |
| **Socket Mode** (dev) | No public URL needed; Slack opens a websocket back. | `AsyncSocketModeHandler(app, app_token).start_async()`. Requires `SLACK_APP_TOKEN` (`xapp-...`). |

Both paths use the same `AsyncApp` instance and the same handlers.

## 2. Slack app manifest (suggested)

Save as `slack_app_manifest.yaml` outside the repo. Bot scopes that cover everything in `01_data_source_matrix.md`:

```yaml
display_information:
  name: anubis-aar
  description: After-action reports across sleep, finance, code, and cloud.
features:
  app_home:
    home_tab_enabled: true
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false
  bot_user:
    display_name: anubis
  slash_commands:
    - command: /aar
      url: https://YOUR_DOMAIN/slack/events
      description: After-action report for a window
      usage_hint: "[today|yesterday|this-week|last-week|last-30d|YYYY-MM-DD..YYYY-MM-DD]"
      should_escape: false
oauth_config:
  scopes:
    bot:
      - app_mentions:read
      - channels:history
      - chat:write
      - commands
      - files:write          # for files_upload_v2
      - groups:history
      - im:history
      - im:read
      - im:write
      - mpim:history
      - users:read
      - users:read.email
settings:
  event_subscriptions:
    request_url: https://YOUR_DOMAIN/slack/events
    bot_events:
      - app_mention
      - message.im
  interactivity:
    is_enabled: true
    request_url: https://YOUR_DOMAIN/slack/events
  socket_mode_enabled: false   # flip true for dev
  token_rotation_enabled: false
```

## 3. Three handlers, one agent invocation

```python
# src/api/slack_routes.py  (proposed)
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

bolt_app = AsyncApp(
    token=ctx.slack_bot_token,
    signing_secret=ctx.slack_signing_secret,
)

@bolt_app.command("/aar")
async def handle_aar(ack, command, respond):
    await ack()  # < 3 s
    window = command.get("text", "").strip() or "last-24h"
    await _run_agent_in_background(
        channel=command["channel_id"],
        user=command["user_id"],
        prompt=f"Produce an AAR for window={window} and post the summary + chart to <#{command['channel_id']}>.",
    )

@bolt_app.event("app_mention")
async def handle_mention(event, say):
    text = event["text"].split(">", 1)[-1].strip()
    if text.lower().startswith("status"):
        prompt = "Give a one-message live overview: company direction + sleep quality. No chart."
    else:
        prompt = text
    await _run_agent_in_background(channel=event["channel"], user=event["user"], prompt=prompt)

@bolt_app.event("message")
async def handle_dm(event, say):
    if event.get("channel_type") != "im" or event.get("bot_id"):
        return
    await _run_agent_in_background(channel=event["channel"], user=event["user"], prompt=event["text"])

slack_handler = AsyncSlackRequestHandler(bolt_app)
```

Then mount on the existing FastAPI `app` in `webapp.py`:

```python
# src/api/webapp.py (one new mount, no other changes)
from fastapi import APIRouter, Request
from src.api.slack_routes import slack_handler

slack_router = APIRouter()

@slack_router.post("/slack/events")
async def slack_events(req: Request):
    return await slack_handler.handle(req)

app.include_router(slack_router)
```

`AsyncSlackRequestHandler` performs Slack signature verification and dispatches to the right handler (slash command, event, interaction).

## 4. The "background" agent invocation

Slack expects a 200 within 3 s. The 3-s ack is independent of the long agent run. Use FastAPI `BackgroundTasks` *or* the existing LangGraph SDK client (`langgraph_sdk.get_client`) which `webapp.py` already uses:

```python
from langgraph_sdk import get_client

LG = get_client(url=ctx.langgraph_api_url)

async def _run_agent_in_background(channel: str, user: str, prompt: str):
    thread = await LG.threads.create()
    async for chunk in LG.runs.stream(
        thread["thread_id"],
        assistant_id="data_analysis_graph",   # registered in langgraph.json
        input={"messages":[{"role":"user","content":prompt}]},
        context={"slack_channel": channel, "slack_user": user},
        stream_mode="updates",
    ):
        # incremental Slack updates if you want a live progress thread
        ...
```

The agent itself owns the final `chat.postMessage` / `files.upload_v2` call via the `slack_send_message` tool documented in the LangChain tutorial. Streaming intermediate steps is optional; the simplest first pass is "agent runs to completion, then posts once".

## 5. The `slack_send_message` tool

Direct adaptation of the doc, parameterized for any channel and with optional file upload:

```python
# src/anubis/utils/tools/slack/slack_tools.py  (proposed full body)
from langchain.tools import tool
from slack_sdk.web.async_client import AsyncWebClient
from src.anubis.utils.context import GlobalContext

@tool(parse_docstring=True)
async def slack_send_message(
    text: str,
    channel: str | None = None,
    file_path: str | None = None,
) -> str:
    """Post a message to Slack, optionally attaching a file from the sandbox FS.

    Args:
        text: Message body (Slack mrkdwn).
        channel: Channel id; defaults to SLACK_DEFAULT_CHANNEL.
        file_path: Absolute path inside the sandbox FS to attach (e.g. /artifacts/run/dashboard.png).
    """
    context = GlobalContext()
    client = AsyncWebClient(token=context.slack_bot_token)
    target = channel or context.slack_default_channel
    if file_path is None:
        await client.chat_postMessage(channel=target, text=text)
        return "ok"
    # backend is captured by the deep-agent runtime; passed via partial
    fp = (await BACKEND.download_files([file_path]))[0]
    await client.files_upload_v2(channel=target, content=fp.content, initial_comment=text)
    return "ok"
```

Wiring `BACKEND` (the deep-agent backend) into the tool is done with `functools.partial` at agent-creation time so the tool can talk to whichever sandbox the agent is running.

## 6. Additional "history" tool (Slack as data source)

Distinct from `slack_send_message`. Lives in the same module:

```python
@tool(parse_docstring=True)
async def ingest_slack_history(
    channels: list[str] | None = None,
    days: int = 7,
) -> str:
    """Pull Slack messages into the sandbox FS as parquet.

    Args:
        channels: Channel ids to ingest. None means all the bot is in.
        days: Lookback window.
    Returns:
        Sandbox path to the written parquet.
    """
    context = GlobalContext()
    client = AsyncWebClient(token=context.slack_user_token or context.slack_bot_token)
    ...
    # writes /data/slack/clean/<run_id>.parquet
    return "/data/slack/clean/<run_id>.parquet"
```

Reading bot-only channels needs `channels:history` + the bot to be invited; cross-workspace history needs the **user** token.

## 7. Failure & rate-limit handling

- Wrap every Slack call with a Tier-3 retry (`Retry-After` header). `slack_sdk` already exposes `slack_sdk.errors.SlackApiError`.
- Always `ack()` first inside `/aar` and event handlers. Use `respond` ephemeral if the channel-public reply would be spammy.
- For long AAR runs, the agent should `chat.postMessage` an interim "Working on AAR for <window>..." with a progress emoji, then update via `chat.update` or post the final result in a thread.

## 8. Testing recipe (Socket Mode, no public URL)

```bash
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
export SLACK_SIGNING_SECRET=...
python -m research.data_analysis_slack_agent.reference_code.slack_bolt_app
```

In Slack, DM the bot "give me an AAR for the last 24 hours". The reference script under `reference_code/` demonstrates the full loop with a stub agent.
