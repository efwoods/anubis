# Proposed File Layout (when implementation lands)

> Reminder: this research does not modify `src/`. The tree below is the *proposed* layout for when implementation begins.

```
src/
  anubis/
    utils/
      context.py                           [extend in-place — fields from 02_context_additions.md]
      tools/
        slack/
          slack_tools.py                   [fill in: slack_send_message, ingest_slack_history]
        git/
          __init__.py
          git_tools.py                     [ingest_git_local, ingest_github_org]
        cursor/
          cursor_tools.py                  [ingest_cursor_local, ingest_cursor_cloud]
        google/
          meet_tools.py                    [ingest_meet_transcripts, ingest_meet_audio_via_process_media_graph]
          drive_tools.py                   [drive helpers shared by meet]
          oauth.py                         [google OAuth helper]
        health/
          sleep_tools.py                   [ingest_apple_health_zip, ingest_oura, ingest_whoop]
        boa/
          plaid_tools.py                   [ingest_boa, plaid_link_token, plaid_exchange]
        aws/
          cost_tools.py                    [ingest_aws_cost]
        azure/
          cost_tools.py                    [ingest_azure_cost]
        openai_usage/
          chatgpt_tools.py                 [ingest_chatgpt_usage, ingest_chatgpt_costs, ingest_chatgpt_export_zip]
      sandbox.py                           [factory: build_backend(context) -> BackendProtocol]

  subgraphs/
    data-analysis/                          [folder already exists — populate]
      __init__.py
      data_analysis_graph.py               [create_deep_agent(...) entrypoint; exports `data_analysis_graph`]
      schemas.py                           [AfterActionReport + sub-models]
      prompts.py                           [AAR_SYSTEM_PROMPT, STATUS_SYSTEM_PROMPT]
      skills/
        aar/SKILL.md                       [progressive-disclosure SOP]
        sleep_analysis/SKILL.md
        cloud_cost/SKILL.md
        finance/SKILL.md
        engineering_velocity/SKILL.md
        meeting_summary/SKILL.md
      subagents.py                         [declarative subagent dicts: sleep_analyst, finance_analyst, ...]
      tests/
        test_aar_schema.py
        test_connectors.py

  api/
    webapp.py                              [+ one router include for /slack/events; no other changes]
    slack_routes.py                        [new: bolt App, request handler, /aar slash, app_mention, message.im]

langgraph.json                              [+ register `data_analysis_graph` next to `Anubis`]
.env / .env.dev                             [+ keys from 02_context_additions.md]
```

## Why this shape

- **Tools belong in `src/anubis/utils/tools/<source>/`** — the existing convention (`slack/`, `identity/` already live there). One folder per data domain.
- **Subgraphs belong in `src/subgraphs/<name>/`** — same convention used for `process_media_graph`, `vector_store_graph`, `email`. The empty `data-analysis/` folder is already a placeholder.
- **`data_analysis_graph.py` is a top-level module export** mirroring how `process_media_graph_api_endpoint.py` exports `process_media_graph_api_endpoint`. Register in `langgraph.json` so it can be served alongside the avatar graph.
- **Prompts go in `subgraphs/data-analysis/prompts.py`**, not in the global `src/anubis/utils/prompts/`, because they are local to this graph and shouldn't be loaded by the avatar.
- **Skills are markdown** and the `skills=` argument to `create_deep_agent` points at `subgraphs/data-analysis/skills/`. Per the docs, the skill loader caches frontmatter and reads bodies on demand.
- **Slack adapter goes in `src/api/slack_routes.py`** — `webapp.py` already imports many `APIRouter`s. One additional `app.include_router(slack_router)` is the only edit there.
- **`sandbox.py` lives once at `src/anubis/utils/sandbox.py`** so the avatar graph can also adopt sandboxed analysis later without duplication.

## Imports the agent module will need (none new at the framework level beyond `deepagents`)

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, StateBackend, StoreBackend
from langchain_openai import ChatOpenAI                       # already in repo
from langchain_core.messages import SystemMessage             # already in repo
from langgraph.checkpoint.memory import InMemorySaver         # already in repo
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # already in repo
from langgraph.store.postgres import AsyncPostgresStore       # already in repo
```

New runtime dependencies to add to `pyproject.toml` when implementation begins:

```toml
"deepagents>=1.0.0",                         # core
"slack-bolt>=1.20",                          # slack adapter
"plaid-python>=29",                          # BoA via Plaid
"PyGithub>=2.4",                             # github org commits
"google-api-python-client>=2.150",
"google-auth-oauthlib>=1.2",
"google-auth-httplib2>=0.2",
"azure-identity>=1.18",
"azure-mgmt-costmanagement>=4.0",
"boto3>=1.35",
"matplotlib>=3.10",                          # already in dev group; promote to runtime
"seaborn>=0.13",
```

Do **not** pin a deepagents version yet — pull `latest` during the spike, then pin once stable.

## Operational notes

- `langgraph.json` will gain another assistant id (e.g. `"data_analysis_graph": "src/subgraphs/data-analysis/data_analysis_graph.py:data_analysis_graph"`).
- Postgres namespace conventions:
  - AAR persistence: `("aar", user_id, run_id)` → JSON of `AfterActionReport`.
  - Per-source cursors (Plaid `transactions_sync`, Slack `cursor`, GitHub etag): `("ingest_cursor", user_id, source)`.
- Daily 07:00 cron: a tiny LangGraph cron node invokes the same graph with `prompt="Daily AAR for the prior 24h, post to <#default>"`.
- Cost guardrails: each ingest tool accepts `days` and clamps it (`days = min(days, 30)`) to bound API cost. Cost Explorer charges per-call; cache per-day result hashes in Postgres before re-querying.
