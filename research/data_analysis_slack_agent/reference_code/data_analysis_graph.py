"""Reference (research-only) skeleton for the Slack-native data analysis deep agent.

This file is illustrative. It is NOT imported by anything in src/. It shows the
exact wiring described in the research docs:
- create_deep_agent with structured AfterActionReport response
- backend factory (local | daytona | modal | runloop)
- pluggable list of ingest tools
- the slack_send_message tool exactly as in the LangChain tutorial
- a checkpointer for multi-turn conversations
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import partial
from typing import Literal
from uuid import uuid4

from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from slack_sdk.web.async_client import AsyncWebClient


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def build_backend(provider: str = "local"):
    """Return a deepagents backend keyed by SANDBOX_PROVIDER.

    Real implementation reads from `GlobalContext()` at the top of the call.
    """
    from deepagents.backends import LocalShellBackend

    if provider == "local":
        return LocalShellBackend(
            root_dir=os.environ.get("SANDBOX_ROOT_DIR", "/tmp/anubis_sandbox"),
            env={"PATH": "/usr/bin:/bin"},
        )

    if provider == "daytona":
        from daytona import Daytona
        from langchain_daytona import DaytonaSandbox

        sandbox = Daytona().create()
        return DaytonaSandbox(sandbox=sandbox)

    if provider == "modal":
        import modal
        from langchain_modal import ModalSandbox

        app = modal.App.lookup("anubis-aar")
        return ModalSandbox(sandbox=modal.Sandbox.create(app=app))

    if provider == "runloop":
        from langchain_runloop import RunloopSandbox
        from runloop_api_client import RunloopSDK

        client = RunloopSDK(bearer_token=os.environ["RUNLOOP_API_KEY"])
        return RunloopSandbox(devbox=client.devbox.create())

    raise ValueError(f"Unknown sandbox provider: {provider}")


# ---------------------------------------------------------------------------
# Structured response (AAR)
# ---------------------------------------------------------------------------


class SourceCitation(BaseModel):
    source: str
    sandbox_path: str
    row_count: int
    sha256: str | None = None


class ArtifactRef(BaseModel):
    sandbox_path: str
    caption: str


class CompanyOverview(BaseModel):
    state: str
    direction: str
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


# ---------------------------------------------------------------------------
# slack_send_message — tool the agent uses to *post* (history ingest is separate)
# ---------------------------------------------------------------------------


def make_slack_send_message(backend):
    """Return a tool bound to a specific deepagents backend."""

    @tool(parse_docstring=True)
    async def slack_send_message(
        text: str,
        channel: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """Post a message to Slack, optionally attaching a sandbox-FS file.

        Args:
            text: Message body in Slack mrkdwn.
            channel: Channel id (C012...). Defaults to SLACK_DEFAULT_CHANNEL.
            file_path: Absolute sandbox-FS path to attach. None for text-only.
        """
        client = AsyncWebClient(token=os.environ["SLACK_BOT_TOKEN"])
        target = channel or os.environ.get("SLACK_DEFAULT_CHANNEL")
        if file_path is None:
            await client.chat_postMessage(channel=target, text=text)
            return "ok"
        downloaded = await backend.download_files([file_path])
        await client.files_upload_v2(
            channel=target,
            content=downloaded[0].content,
            initial_comment=text,
        )
        return "ok"

    return slack_send_message


# ---------------------------------------------------------------------------
# Ingest tool stubs — real bodies live in src/anubis/utils/tools/<source>/
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
async def ingest_git(scope: Literal["local", "github_org"] = "local", days: int = 7) -> str:
    """Pull commits to /data/git/clean/<run>.parquet.

    Args:
        scope: 'local' walks GIT_LOCAL_REPO_ROOT; 'github_org' uses GITHUB_PAT + GITHUB_ORG.
        days: Lookback window in days.
    """
    return "/data/git/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_slack_history(channels: list[str] | None = None, days: int = 7) -> str:
    """Pull Slack messages to /data/slack/clean/<run>.parquet.

    Args:
        channels: List of channel ids; None = all channels the bot is in.
        days: Lookback window in days.
    """
    return "/data/slack/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_sleep(
    provider: Literal["apple", "oura", "whoop", "fitbit"] = "oura",
    days: int = 7,
) -> str:
    """Pull sleep records to /data/sleep/clean/<run>.parquet.

    Args:
        provider: Source for sleep data.
        days: Lookback window in days.
    """
    return "/data/sleep/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_boa(
    method: Literal["plaid", "csv"] = "plaid",
    days: int = 30,
) -> str:
    """Pull BoA transactions to /data/boa/clean/<run>.parquet.

    Args:
        method: 'plaid' uses PLAID_BOA_ACCESS_TOKEN; 'csv' expects an uploaded csv.
        days: Lookback window in days.
    """
    return "/data/boa/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_meet(days: int = 7) -> str:
    """Pull Google Meet transcripts to /data/meet/clean/<run>.parquet.

    Args:
        days: Lookback window in days.
    """
    return "/data/meet/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_cursor(scope: Literal["local", "cloud"] = "local", days: int = 7) -> str:
    """Pull Cursor activity to /data/cursor/clean/<run>.parquet.

    Args:
        scope: 'local' walks ~/.config/Cursor/User; 'cloud' uses CURSOR_API_KEY.
        days: Lookback window in days.
    """
    return "/data/cursor/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_chatgpt_usage(days: int = 7) -> str:
    """Pull OpenAI org usage + costs to /data/chatgpt/clean/<run>.parquet.

    Args:
        days: Lookback window in days.
    """
    return "/data/chatgpt/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_aws_cost(days: int = 30) -> str:
    """Pull AWS Cost Explorer data to /data/aws/clean/<run>.parquet.

    Args:
        days: Lookback window in days.
    """
    return "/data/aws/clean/<run>.parquet"


@tool(parse_docstring=True)
async def ingest_azure_cost(days: int = 30) -> str:
    """Pull Azure Cost Management data to /data/azure/clean/<run>.parquet.

    Args:
        days: Lookback window in days.
    """
    return "/data/azure/clean/<run>.parquet"


# ---------------------------------------------------------------------------
# Subagent declarations (one per data pillar)
# ---------------------------------------------------------------------------


SUBAGENTS = [
    {
        "name": "engineering-velocity-analyst",
        "description": "Use for git/cursor velocity questions and 'what shipped' summaries.",
        "system_prompt": "You analyze engineering velocity from git+cursor parquet files. "
        "Always quantify: commits, LOC, PR throughput, agent run success rate. "
        "Compare current window vs prior window of equal length.",
        "tools": [ingest_git, ingest_cursor],
    },
    {
        "name": "cloud-cost-analyst",
        "description": "Use for AWS+Azure+ChatGPT cost questions and burn analysis.",
        "system_prompt": "You analyze cloud + LLM cost. Always include: $/day, top 3 services, "
        "delta vs prior window, and one optimization recommendation if delta > +10%.",
        "tools": [ingest_aws_cost, ingest_azure_cost, ingest_chatgpt_usage],
    },
    {
        "name": "finance-analyst",
        "description": "Use for BoA balance/cashflow questions and runway calculations.",
        "system_prompt": "You analyze inflows, outflows, and runway from BoA + cloud cost.",
        "tools": [ingest_boa, ingest_aws_cost, ingest_azure_cost, ingest_chatgpt_usage],
    },
    {
        "name": "meeting-summarizer",
        "description": "Use for 'what was decided' and action-item rollups.",
        "system_prompt": "You summarize Google Meet transcripts: decisions, action items, "
        "owners, due dates, sentiment.",
        "tools": [ingest_meet],
    },
    {
        "name": "sleep-analyst",
        "description": "Use for sleep, HRV, and 'how am I' questions.",
        "system_prompt": "You analyze sleep parquet files. Compute composite scores, HRV trend, "
        "deep + REM % of total. Output one tonight-recommendation grounded in the data.",
        "tools": [ingest_sleep],
    },
    {
        "name": "communication-analyst",
        "description": "Use for Slack response-time and topic-load questions.",
        "system_prompt": "You analyze Slack history: response times, decision throughput, "
        "channel topic distribution.",
        "tools": [ingest_slack_history],
    },
]


# ---------------------------------------------------------------------------
# System prompt (excerpt — full prompt lives at 04_aar_prompting.md)
# ---------------------------------------------------------------------------


AAR_SYSTEM_PROMPT = """\
<ROLE>
You are Anubis-AAR, the user's chief-of-staff data analyst.
</ROLE>

<OBJECTIVE>
Decide between AAR mode (full After-Action Report + chart) and STATUS mode
(a single Slack-friendly snapshot, <= 200 words). Default to AAR.
</OBJECTIVE>

<TOOLS_POLICY>
- Use write_todos before any ingestion call.
- Call exactly one ingest_<source> per source in scope.
- Save dataframes to /data/<source>/clean/<run>.parquet.
- Save charts to /artifacts/<run>/<name>.png at 300 dpi.
- Run a single analysis script per run at /scripts/analysis/<run>.py via execute.
- Call slack_send_message exactly once at the end with file_path for the chart.
- Never emit credentials, tokens, or PII.
</TOOLS_POLICY>

<AAR_STRUCTURE>
1. headline
2. what_happened (3-8 cited bullets)
3. what_worked (>=1 quantified)
4. what_did_not_work (>=1 quantified)
5. new_standards_for_excellence (exactly 3 SMART items)
6. company_overview (state + direction)
7. sleep_overview (last night, 7d avg, HRV, tonight rec)
8. sources (parquet paths + row counts)
9. plots (artifact paths)
</AAR_STRUCTURE>

<RIGOR>
Aim for excellence. Quantified deltas only. No speculation beyond data.
Missing source -> goes under what_did_not_work AND new_standards.
</RIGOR>
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent(model: str = "openai:gpt-5.4", provider: str = "local"):
    """Build the deep agent with all wiring."""
    from deepagents import create_deep_agent

    backend = build_backend(provider)
    slack_send_message = make_slack_send_message(backend)

    ingest_tools = [
        ingest_git,
        ingest_slack_history,
        ingest_sleep,
        ingest_boa,
        ingest_meet,
        ingest_cursor,
        ingest_chatgpt_usage,
        ingest_aws_cost,
        ingest_azure_cost,
    ]

    return create_deep_agent(
        model=model,
        tools=[*ingest_tools, slack_send_message],
        backend=backend,
        system_prompt=AAR_SYSTEM_PROMPT,
        response_format=AfterActionReport,
        subagents=SUBAGENTS,
        skills=["/skills/"],
        checkpointer=InMemorySaver(),
    )


data_analysis_graph = None  # populated by build_agent() at startup; export hook for langgraph.json


def main():
    global data_analysis_graph
    data_analysis_graph = build_agent(
        model=os.environ.get("ANALYSIS_MODEL", "openai:gpt-5.4"),
        provider=os.environ.get("SANDBOX_PROVIDER", "local"),
    )
    return data_analysis_graph


if __name__ == "__main__":
    agent = main()
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Produce an AAR for the last 24 hours and post to the default channel.",
                }
            ]
        },
        config,
    )
    print(result)
