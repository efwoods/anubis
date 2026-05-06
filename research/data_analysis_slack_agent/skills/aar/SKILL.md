---
name: aar
description: After-Action Report SOP. Use whenever the user asks for an AAR, retrospective, weekly review, daily review, or "what happened in the last <window>". Drives the agent's planner toward the canonical AfterActionReport schema.
license: MIT
metadata:
  author: anubis
  version: "0.1"
allowed-tools: write_todos, ls, read_file, write_file, edit_file, execute, ingest_git, ingest_slack_history, ingest_sleep, ingest_boa, ingest_meet, ingest_cursor, ingest_chatgpt_usage, ingest_aws_cost, ingest_azure_cost, slack_send_message
---

# After-Action Report SOP

## When to use

The user asked for any of: AAR, retrospective, weekly/daily review, "what happened", "summary of the last X", "give me the report".

## What to produce

A single Slack post with: the headline, a 4-bullet condensed body (what happened / worked / didn't / new standards), and an attached chart. Plus a structured `AfterActionReport` final assistant message.

## Steps

### 1. Plan

Call `write_todos` with one item per data source in scope. Always include git, sleep, and at least one cost source (aws, azure, or chatgpt). Add slack and meet when the user mentions "communication", "decisions", "meetings", or "team".

### 2. Resolve window

Translate the user's window phrase to UTC `[start, end]`:

- "today" -> midnight today, now
- "yesterday" -> midnight yesterday, midnight today
- "this-week" -> this Monday 00:00, now
- "last-week" -> previous Monday 00:00, this Monday 00:00
- "last-30d" -> now - 30d, now
- `YYYY-MM-DD..YYYY-MM-DD` -> as written

Also compute the *prior window of equal length* for delta math.

### 3. Ingest in parallel via subagents

Dispatch each source to its specialist subagent (engineering-velocity-analyst, cloud-cost-analyst, finance-analyst, meeting-summarizer, sleep-analyst, communication-analyst). Each one writes `/data/<source>/clean/<run>.parquet` and returns the path.

### 4. Author one analysis script

Write `/scripts/analysis/<run>.py` that:

1. Reads every parquet path returned by step 3.
2. Joins on UTC date (or local date for sleep).
3. Computes per-source KPIs and `delta_pct`/`delta_abs` vs prior window.
4. Renders a single 16x10 dashboard PNG to `/artifacts/<run>/dashboard.png`.
5. Prints a JSON of computed KPIs to stdout (the LLM reads this for the prose).

### 5. Run it

`execute` the script. Capture stdout. If the script fails, edit and retry once before reporting failure under `what_did_not_work`.

### 6. Compose the AAR

Fill the `AfterActionReport` schema:

- `headline` is one sentence in present tense, anchored on the largest absolute delta.
- `what_happened` cites every source with row counts: e.g. "Shipped 41 commits across 3 repos [git:41]; AWS spend rose to $187/d [aws:43]".
- `what_worked` and `what_did_not_work` only include items with `|delta_pct| >= 10` or `|z| >= 2`.
- `new_standards_for_excellence` MUST be three SMART items aimed at the next window.
- `company_overview.state` is two short paragraphs grounded in numbers; `direction` names the single most leveraged next bet.
- `sleep_overview.recommendation` is one sentence specific to tomorrow's calendar.

### 7. Post to Slack

Call `slack_send_message` exactly once with `file_path="/artifacts/<run>/dashboard.png"` and a Slack-mrkdwn body of:

```
*{headline}*

*What worked*  •  {1-2 most quantified items}
*What didn't*  •  {1-2 items}
*New standards*  •  {3 SMART items}

_Full AAR persisted; reply to this thread to drill in._
```

## Failure modes

- A connector raises -> mark that source under `what_did_not_work` AND append a `new_standard` of "wire up <source> by EOW". Continue without it.
- The LLM cannot fill three `new_standards`. Reject and retry once. If it still cannot, default to: (1) instrument the missing source, (2) close the largest regression, (3) double down on the largest improvement.
- The chart fails to render. Retry once with smaller fig size. If still failing, post the AAR text without an attachment.
