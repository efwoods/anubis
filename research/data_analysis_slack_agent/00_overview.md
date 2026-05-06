# Data Analysis Agent (Slack-Native) — Research

> Research only. No code in `src/anubis` is changed. All deliverables live under `/home/user/gh/anubis/research/data_analysis_slack_agent/`.

## 1. Goal

Build a **personal + company "state-of-the-business" agent** that:

| Capability | What it produces |
| --- | --- |
| Pulls multi-source signal | sleep, Bank of America (BoA), Google Meets transcripts, Slack messages, git commits, Cursor activity, ChatGPT token usage, Azure cloud, AWS |
| Runs analysis on demand | tabular EDA + matplotlib/seaborn plots in a sandbox, just like the LangChain `deepagents` data-analysis tutorial |
| Returns an After-Action Report (AAR) | What happened. What worked. What did not. New standards for excellence. |
| Returns a live "snapshot" | Current company overview + direction of the company at any minute, current personal sleep quality at any minute |
| Is queryable from Slack | Slash command, mention, or DM in Slack drives the same agent and replies in-channel with text + attached plot |

The framework comes from [LangChain Deep Agents → Build a data analysis agent](https://docs.langchain.com/oss/python/deepagents/data-analysis). Anubis already pins `langchain>=1.2.10`, `langgraph>=1.0.10`, `langchain-openai`, `slack-sdk>=3.41.0`, `pandas`, `tiktoken`, etc., so the runtime fit is direct.

## 2. Why deep agents (and not a single LangGraph agent)

The deep-agent harness gives us, out of the box, the four primitives this product needs:

1. **Planner (`write_todos`)** — the agent decomposes "give me an AAR for this week" into per-source steps and tracks them.
2. **Virtual filesystem (`ls` / `read_file` / `write_file` / `edit_file`)** — every connector dumps its CSV/JSON to `/data/<source>/...`, every analysis script lives at `/scripts/...`, every plot at `/artifacts/...`. This is exactly the pattern the docs use with `backend.upload_files([...])`.
3. **`execute` shell** (sandbox backend) — runs `python analyze_*.py` to produce charts and tables. Anubis already has `pandas`, `matplotlib` (in `dev` group), `nltk`, `scipy`, `sentence-transformers` available.
4. **Subagents** — one subagent per data source (`sleep-analyst`, `finance-analyst`, `cloud-cost-analyst`, `engineering-velocity-analyst`, `meeting-summarizer`, ...) keeps the main agent context clean. Each subagent has a narrow tool set and its own system prompt.

The existing anubis super-graph (`src/anubis/graph.py`) is a `create_agent`-style avatar/identity loop. Deep agents are complementary: the avatar graph drives a person's identity; the data-analysis graph is a **separate, parallel subgraph** registered alongside `vector_store_graph` and `process_media_graph`, exposed through `webapp.py` and Slack.

## 3. Mapping to existing anubis architecture

| Deep-agent concept | Anubis location it maps to |
| --- | --- |
| `create_deep_agent(...)` agent module | `src/subgraphs/data-analysis/data_analysis_graph.py` (folder already exists, currently empty) |
| Custom tools | `src/anubis/utils/tools/<source>/...` (`slack/`, `identity/` already exist; add `git/`, `cursor/`, `meet/`, `health/`, `boa/`, `aws/`, `azure/`, `chatgpt/`) |
| Backend/Sandbox config | New module `src/anubis/utils/sandbox.py` (factory that returns `Daytona`, `Modal`, `Runloop`, or `LocalShellBackend` based on env) |
| Env vars | Added to `src/anubis/utils/context.py` `GlobalContext` per the workspace rule (uppercase env names, lowercase fields with descriptions) |
| Skills (Markdown SOPs) | `src/subgraphs/data-analysis/skills/<name>/SKILL.md` |
| Slack chat surface | New FastAPI router in `src/api/slack_routes.py` mounted on the existing `webapp.py` FastAPI app, fronting `slack_bolt.AsyncApp` |
| Long-term memory of past AARs | Reuse the existing `AsyncPostgresStore` already wired in `webapp.py` (namespace `("aar", user_id, ...)`)  |
| LangGraph deployment entry | Register a new graph id in `langgraph.json` next to the current `Anubis` graph |

Nothing in `src/anubis/graph.py`, `src/anubis/utils/state.py`, `src/anubis/utils/context.py`, `src/api/webapp.py`, or any existing subgraph needs to change for this work. The data-analysis agent is additive.

## 4. End-to-end runtime topology

```
Slack (slash cmd / mention / DM)
        │
        ▼
Slack Events API → FastAPI (src/api/slack_routes.py)
        │            └─ HMAC signature verification (slack_sdk)
        ▼
slack_bolt.AsyncApp request_handler
        │
        ▼
LangGraph Server (langgraph.json registers data_analysis_graph)
        │
        ▼
data_analysis_graph (deepagents.create_deep_agent)
   ├── tools:  ingest_<source>, plot_to_slack, post_to_slack
   ├── skills: aar_generation, sleep_analysis, finance_analysis, ...
   ├── subagents: sleep-analyst, finance-analyst, cloud-cost-analyst, ...
   └── backend: DaytonaSandbox  (or LocalShellBackend in dev)
        │
        ▼
Sandbox FS:
   /data/<source>/raw/{...}.csv          ← connectors write here
   /data/<source>/clean/{...}.parquet    ← agent normalizes
   /scripts/analysis/<run>.py            ← agent authors
   /artifacts/<run>/<plot>.png           ← agent emits
        │
        ▼
slack_send_message tool → Slack (text + file attachment)
        │
        ▼
AsyncPostgresStore  (namespace=("aar", user_id, run_id))
   ← agent persists JSON AAR for trend analysis across runs
```

Streaming back to Slack uses LangGraph's `astream_events` `updates` mode so partial answers can be posted as `chat.postMessage` updates (mirroring the `_latest_ai_from_stream_update` helper already used in `webapp.py`).

## 5. The "After-Action Report" output contract

The AAR is the canonical artifact. Pin it as a Pydantic schema so every connector writes through the same shape and trends are queryable historically. Proposed shape (lives at `src/subgraphs/data-analysis/schemas.py`):

```python
class AfterActionReport(BaseModel):
    window_start: datetime
    window_end: datetime
    headline: str
    what_happened: list[str]          # observed, sourced, dated
    what_worked: list[str]            # positive deltas vs prior window
    what_did_not_work: list[str]      # regressions / missed targets
    new_standards_for_excellence: list[str]  # forward-looking commitments
    company_overview: CompanyOverview        # current state + 1-paragraph direction
    sleep_overview: SleepOverview            # quality scores + readiness
    sources: list[SourceCitation]            # path in sandbox FS, row count, hash
    plots: list[ArtifactRef]                 # /artifacts/...
```

The deep agent is given this schema as `response_format=AfterActionReport`. That maps to the `response_format=` arg on `create_deep_agent` (see customization doc). When the run finishes, the structured AAR is what is persisted to Postgres; the prose + plot is what is posted to Slack.

## 6. Slack interface, three entry points

All three feed the same `data_analysis_graph`:

1. **`/aar [window]`** slash command — `window` ∈ `today | yesterday | this-week | last-week | last-30d | YYYY-MM-DD..YYYY-MM-DD`. Defaults to last 24 h.
2. **`@anubis status`** mention — emits the live company + sleep overview (no full AAR).
3. **DM "ask"** — free text. Invokes the agent with the user's question; the planner picks which connector subagents to wake.

Bolt-Python's async `App.command`, `App.event("app_mention")`, and `App.event("message")` handlers ack within 3 s (Slack requirement) and then push real work to a background task that streams to the channel.

## 7. Security / secrets posture

Following the tutorial's practice:

- All API tokens (Slack bot token, Slack signing secret, Plaid secret, Google service-account JSON, GitHub PAT, Cursor token, OpenAI key, Azure SP, AWS keys, BoA / Plaid creds) are kept **outside** the sandbox.
- Tools accept already-fetched payloads (`bytes` / DataFrames / paths) and pass them in via `backend.upload_files(...)`.
- Sandbox env strips PATH to `/usr/bin:/bin` for `LocalShellBackend` dev mode.
- Per the workspace rule, every secret is lifted onto `GlobalContext` (uppercase env, lowercase field with description). See `02_context_additions.md`.

## 8. Roadmap

| Phase | Outcome |
| --- | --- |
| 0. Spike (this research) | Architecture doc, connector matrix, reference skeletons. |
| 1. Local AAR | Slack DM → deep agent on `LocalShellBackend` → AAR for one source (git commits) → Slack reply with chart. |
| 2. Connector breadth | Add Slack history, Cursor, ChatGPT exports, Google Meets, GitHub org, AWS Cost Explorer, Azure Cost Mgmt, BoA via Plaid, Apple Health/Oura. |
| 3. Sandboxing + scheduling | Switch backend to Daytona/Modal. Add a LangGraph cron node that produces a 7am AAR daily. |
| 4. Persistence + trend | Persist AAR JSON to `AsyncPostgresStore`. Add a `compare_to_last_week` tool. |
| 5. HITL + permissions | Wire `interrupt_on={"slack_send_message": True}` for sensitive replies, and `permissions=` for write-restricted dirs. |

## 9. Files in this research bundle

- `00_overview.md` — this file
- `01_data_source_matrix.md` — per-source connector recipe (lib, auth, schema, frequency)
- `02_context_additions.md` — exact `GlobalContext` field additions following `.cursorrules`
- `03_slack_interface.md` — Bolt-Python entrypoints + LangGraph wiring
- `04_aar_prompting.md` — system-prompt structure for "what worked / what didn't / new standards"
- `05_proposed_layout.md` — file tree the implementation should land at, in `src/`
- `reference_code/data_analysis_graph.py` — runnable skeleton (deep agent + tools)
- `reference_code/slack_bolt_app.py` — runnable Slack bridge skeleton
- `reference_code/connectors/` — per-source connector function stubs
- `skills/aar/SKILL.md` — example skill: how to produce an AAR
- `skills/sleep_analysis/SKILL.md` — example skill: sleep quality scoring
