# After-Action Report — Prompt + Schema Design

The agent's value is the AAR. The prompt is therefore the contract. It follows GPT-5 prompting guidance (per workspace rule): explicit role, scoped instructions, structured output, and *no ambiguity about which tool to call when*.

## 1. Top-level system prompt (passed as `system_prompt=` to `create_deep_agent`)

```text
<ROLE>
You are Anubis-AAR, the user's chief-of-staff data analyst. You produce After-Action
Reports that are precise, sourced, and decision-grade. You also answer ad-hoc
questions about the company and the user's personal sleep when asked for a "status".
</ROLE>

<OBJECTIVE>
For every user request, decide between two output modes:
  1. AAR mode — a full After-Action Report with chart + structured response.
  2. STATUS mode — a single Slack-friendly message with current company direction and
     sleep quality, no chart, no structured response.

Default to AAR mode unless the user message starts with "status", "snapshot", or
"how are things". For STATUS mode, skip the planner and the chart and answer directly
in <= 200 words.
</OBJECTIVE>

<TOOLS_POLICY>
- write_todos before any ingestion call. The plan is the contract.
- For each data source the user mentions or implies, call exactly ONE ingest_<source>
  tool. Do not re-ingest the same source within a single run.
- Save every intermediate dataframe to /data/<source>/clean/<run_id>.parquet.
- Save every chart to /artifacts/<run_id>/<name>.png at 300 dpi.
- Use the execute tool to run a single analysis script per run, located at
  /scripts/analysis/<run_id>.py.
- Call slack_send_message exactly once at the end (with file_path for the chart).
- Never emit credentials, tokens, or PII into messages or files. Connector tools
  already strip them.
</TOOLS_POLICY>

<AAR_STRUCTURE>
The AAR (when in AAR mode) MUST contain, in this order:
  1. headline — one sentence, present tense, what changed in the window.
  2. what_happened — bullet list, 3–8 items. Each item ends with a bracketed
     citation [source:rows] e.g. [git:182] or [aws:31].
  3. what_worked — bullet list, ≥1 item. Quantified delta vs prior window.
  4. what_did_not_work — bullet list, ≥1 item. Quantified delta vs prior window.
  5. new_standards_for_excellence — bullet list, 3 items. Each item is forward-
     looking, specific, measurable, and achievable in the next window.
  6. company_overview — 2 short paragraphs:
        (a) Current state: revenue/burn (BoA + AWS + Azure + ChatGPT $),
            engineering velocity (commits/PRs), meeting load, decision throughput.
        (b) Direction: where the company is pointing right now and the single
            most leveraged next bet given the data.
  7. sleep_overview — 1 paragraph:
        Last-night composite score, 7-day rolling avg, HRV trend, recommendation
        for tonight's wind-down based on tomorrow's calendar load.
  8. sources — list of every parquet path read, with row count.
  9. plots — list of every artifact path written.
</AAR_STRUCTURE>

<RIGOR>
- Aim for excellence. Treat "what worked" as evidence-backed, not aspirational.
- If a data source is missing or stale, list it under what_did_not_work and call out
  the new standard "wire up X by EOW".
- Never speculate beyond the data. If you do not know, say so.
- Time-window math is in the user's local timezone unless told otherwise.
</RIGOR>

<OUTPUT>
When in AAR mode, your final assistant message MUST be the AfterActionReport
schema. The Slack post is your tool call's text argument and is human-prose,
mirroring the headline + a compressed bullet summary, with the chart as
file_path.
</OUTPUT>
```

## 2. Pydantic response schema (mirrors `<AAR_STRUCTURE>`)

```python
# src/subgraphs/data-analysis/schemas.py  (proposed)
from datetime import datetime
from pydantic import BaseModel, Field

class SourceCitation(BaseModel):
    source: str             # "git" | "slack" | "aws" | ...
    sandbox_path: str       # /data/<source>/clean/...
    row_count: int
    sha256: str | None = None

class ArtifactRef(BaseModel):
    sandbox_path: str       # /artifacts/<run_id>/<name>.png
    caption: str

class CompanyOverview(BaseModel):
    state: str              # paragraph (a)
    direction: str          # paragraph (b)
    burn_rate_usd_30d: float | None = None
    revenue_30d: float | None = None
    velocity_commits_7d: int | None = None

class SleepOverview(BaseModel):
    last_night_score: int | None = None
    seven_day_avg_score: float | None = None
    hrv_trend_pct: float | None = None
    recommendation: str

class AfterActionReport(BaseModel):
    window_start: datetime
    window_end: datetime
    headline: str
    what_happened: list[str] = Field(min_length=3, max_length=8)
    what_worked: list[str] = Field(min_length=1)
    what_did_not_work: list[str] = Field(min_length=1)
    new_standards_for_excellence: list[str] = Field(min_length=3, max_length=3)
    company_overview: CompanyOverview
    sleep_overview: SleepOverview
    sources: list[SourceCitation]
    plots: list[ArtifactRef]
```

Pass to deep agent:

```python
agent = create_deep_agent(
    model=...,
    tools=[*ingest_tools, slack_send_message],
    backend=backend,
    system_prompt=AAR_SYSTEM_PROMPT,
    response_format=AfterActionReport,   # final assistant turn parses to this
    skills=["/skills/aar/", "/skills/sleep_analysis/", "/skills/cloud_cost/"],
    checkpointer=InMemorySaver(),        # swap for Postgres in prod
)
```

## 3. STATUS mode (the "anytime overview")

When the user asks `@anubis status`, the agent skips the AAR scaffold and emits a single Slack message under 200 words:

```
:dna: ANUBIS — live snapshot, {now}

Company direction: {one sentence based on most recent commit pace,
  burn vs revenue, and meeting topics}.
Sleep last night: {score}/100 ({trend arrow} vs 7d avg). HRV {hrv} ({trend}).
Recommended focus today: {one sentence}.
```

The agent decides STATUS mode by reading the routing hint in the prompt the Slack
adapter injects: `mode=status` is set when the user mention starts with `status`/`snapshot`/`how are things`.

## 4. "What worked / what didn't" — how the agent computes deltas

In `/scripts/analysis/<run_id>.py` the agent always:

1. Computes summary stats for *current window* and *prior window of equal length*.
2. Joins on the natural date key (UTC date for cloud, local date for sleep).
3. Emits `delta_pct` and `delta_abs` columns.
4. Marks any KPI moving by `>=10%` or `>=2σ` as a `what_worked`/`what_did_not_work` candidate; the LLM then writes the prose.

KPI catalog the prompt assumes:

| Pillar | KPI | Source |
| --- | --- | --- |
| Velocity | Commits, lines added, PRs merged | git/github |
| Velocity | Cursor agent runs, success rate | cursor |
| Cost | AWS $/day, Azure $/day, ChatGPT token cost | aws / azure / chatgpt |
| Cost | Total burn 30d (sum cloud + ChatGPT - revenue) | derived |
| Revenue / Cash | BoA inflow 7d, balance | plaid/boa |
| Communication | Decisions captured, action items completed | meet |
| Communication | Slack thread response time, P50/P95 | slack |
| Health | Sleep score, HRV, deep+REM minutes | sleep |

## 5. Skills (progressive disclosure)

Each pillar gets a thin `SKILL.md` that the agent will pull in only when the prompt mentions it. Reference skill skeleton in `skills/aar/SKILL.md`. The `skills=[...]` arg points the agent at `/skills/`. The `SkillsMiddleware` is enabled automatically when `skills=` is set (per the customization doc).

## 6. Why this prompt structure works

- **Role + objective + tools_policy + rigor + output** is the GPT-5 "five-block" recommendation.
- The `AAR_STRUCTURE` section is the contract; pinning it as a Pydantic `response_format` makes downstream persistence and trend analysis trivial — every AAR can be diffed against the previous AAR by JSON path.
- Citations like `[git:182]` are cheap for the model and make every claim traceable to a parquet row count, which is the simplest "did the agent hallucinate?" check.
- "Aim for excellence" is operationalized as a `new_standards_for_excellence` list of 3 forward bets — concrete, not platitudinal.
