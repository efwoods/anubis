# Data Source Connector Matrix

For each data source: the recommended Python library, auth flow, what to fetch, normalized schema, refresh frequency, and the LangChain tool that wraps it. Every connector writes a CSV/Parquet/JSON to a deterministic path in the sandbox FS so the agent can `read_file` them without guessing.

Convention used below:

```
sandbox FS layout
/data/<source>/raw/<YYYY-MM-DD>__<run_id>.{csv|json|parquet}
/data/<source>/clean/<YYYY-MM-DD>__<run_id>.parquet
```

The agent never calls these APIs itself. A LangChain `@tool` (registered with the deep agent) calls the API outside the sandbox, then `backend.upload_files([(path, bytes), ...])` pushes the bytes in. This matches the doc's warning: *"It is generally good practice to avoid adding credentials and other secrets to the sandbox. Here we manage the Slack token outside the sandbox in a tool."*

---

## 1. Sleep

Two practical pipelines:

**A. Apple Health (recommended if iPhone+Watch).** Apple Health does not have a public read API for third parties. The supported workflow is:

1. iOS Health app → "Export All Health Data" → produces `export.zip` containing `export.xml` (records) and a `workout-routes/` folder.
2. User uploads `export.zip` to Slack DM (or to an S3 bucket the agent watches).
3. Tool `ingest_apple_health(zip_bytes)` parses `export.xml` with `xml.etree.ElementTree` (already in stdlib), filters `HKCategoryTypeIdentifierSleepAnalysis` and `HKQuantityTypeIdentifierHeartRateVariabilitySDNN`, writes `/data/sleep/raw/<date>.parquet`.
4. The `ZipUploadProcessor` already in `src/anubis/utils/classes/ZipUploadProcessor.py` is a precedent for the unzip-and-route step.

**B. Oura / Whoop / Fitbit (API).** All three have OAuth + REST.
- Oura v2: `GET /v2/usercollection/sleep` — returns `score`, `total_sleep_duration`, `efficiency`, `deep_sleep_duration`, `rem_sleep_duration`, `restless_periods`, `average_hrv`. Token via OAuth2 Authorization Code; store refresh token in Postgres.
- Whoop: `GET /v1/cycle/{id}/sleep`.
- Fitbit: `GET /1.2/user/-/sleep/date/{date}.json`.

Pick one of the three. Code reference uses Oura because of HRV inclusion.

| Field | Type | Purpose |
| --- | --- | --- |
| `date` | date | partition key |
| `total_sleep_min` | int |  |
| `efficiency_pct` | float |  |
| `rem_min` | int |  |
| `deep_min` | int |  |
| `awake_min` | int |  |
| `hrv_ms` | float | autonomic recovery proxy |
| `score` | int 0–100 | provider's composite |

Refresh: nightly cron at 07:00 local.
Tool: `ingest_sleep(provider: Literal["oura","apple","whoop"], days: int=7) -> str` (returns sandbox path).

---

## 2. Bank of America finances

BoA does not publish a developer API. Three viable routes:

**A. Plaid (production-ready).** `pip install plaid-python`. The flow:
1. Frontend (or one-time CLI) creates a Link token, user logs into BoA inside Plaid Link, you receive a public token, exchange it for an `access_token` server-side, persist encrypted.
2. `client.transactions_sync(access_token, cursor)` returns added/modified/removed deltas. Use the cursor pattern for incremental sync.
3. `client.accounts_balance_get(access_token)` for balances.

**B. SimpleFIN Bridge (cheap alternative).** A bring-your-own-credentials proxy.

**C. CSV upload.** BoA exports CSV from Online Banking → "Download transactions". Tool accepts the CSV via Slack file upload.

Normalized schema:

| `txn_id` | `date` | `amount` | `currency` | `category` | `merchant` | `account_id` | `description` |

Refresh: daily.
Tool: `ingest_boa(method: Literal["plaid","csv"], days: int=30, csv_bytes: bytes|None=None)`.

---

## 3. Google Meets transcription

Google now stores Meet recordings in the meeting host's Drive. Pull transcripts via either:

**A. Meet REST API (`meet.googleapis.com`).** `conferenceRecords.transcripts.list` and `transcripts.entries.list` return structured speaker turns *if Gemini transcription was on*. Requires OAuth scope `https://www.googleapis.com/auth/meetings.space.readonly` and `meetings.space.created`.

**B. Drive + Docs API fallback.** `drive.files.list?q="mimeType='application/vnd.google-apps.document' and name contains 'Transcript'"`, then `docs.documents.get` for the body. Scope `drive.readonly`, `documents.readonly`.

Anubis already has Whisper-based transcription (`audio_transcription_model` in `GlobalContext`). If Meet recordings exist in Drive as `.mp4`, the `process_media_graph` already handles audio extraction → Whisper (`AsyncLlamaAPIClient` or OpenAI) → diarization. Reuse it: tool `ingest_meet(mode: Literal["api","drive_audio"]=...)` either pulls structured transcripts or routes the audio through `process_media_graph`.

Normalized schema (per turn):

| `meet_id` | `start_ts` | `end_ts` | `speaker` | `text` |

Plus a meeting-level summary row built by the agent (subagent `meeting-summarizer`):

| `meet_id` | `title` | `attendees` | `decisions` | `action_items[]` | `topics[]` | `sentiment` |

---

## 4. Slack messages

Use `slack_sdk.WebClient` (already in `requirements.txt`):

- `conversations.list(types="public_channel,private_channel,mpim,im", limit=1000)` for channel inventory.
- `conversations.history(channel=..., oldest=..., limit=1000)` for messages.
- `conversations.replies(channel=..., ts=...)` for thread expansion.
- `users.info(user=...)` for handle/email enrichment.

Required scopes: `channels:history`, `groups:history`, `im:history`, `mpim:history`, `users:read`. For DMs to the bot, `im:history` is enough.

Normalized schema:

| `channel_id` | `channel_name` | `ts` | `thread_ts` | `user` | `text` | `reaction_count` | `reply_count` | `is_thread_root` |

Refresh: every 30 min via a LangGraph cron node, or on-demand via the agent's `ingest_slack(channels: list[str], days: int)` tool.

Note: `conversations.history` is paginated by `cursor`. Implement with `slack_sdk.errors.SlackApiError` catch + sleep on `Retry-After` per Slack rate limits (Tier 3 = 50/min).

---

## 5. Git commits

Two layers:

**A. Local repos.** `pip install gitpython` already viable, or just shell out (`git log --since=... --pretty=format:'%H|%an|%ae|%at|%s' --shortstat`). Run inside the sandbox via `execute` — perfect fit for the deep-agent `execute` tool. The local-shell backend in this repo can read `~/gh` (anubis lives at `/home/user/gh/anubis`).

**B. GitHub org-wide.** `pip install PyGithub` or use REST. Useful when work spans repos outside `~/gh`. `GET /orgs/{org}/repos`, `GET /repos/{owner}/{repo}/commits?since=...&author=...`, `GET /search/commits?q=author:...`. PAT scope `repo`, `read:org`.

Normalized schema:

| `sha` | `repo` | `branch` | `author_name` | `author_email` | `committed_at` | `message` | `files_changed` | `insertions` | `deletions` |

Tool: `ingest_git(scope: Literal["local","github_org"], days: int=7)`.

---

## 6. Cursor progress

Cursor exposes:
- **Local SQLite history** at `~/.config/Cursor/User/History/` (workspace-tagged JSON entries) and `~/.config/Cursor/User/globalStorage/state.vscdb` (SQLite). Read-only is safe.
- **Cloud "Background agents" / Cursor Cloud Agent API** (newer): `https://api.cursor.com/v0/agents` — token from Cursor settings → "Create API Key". Returns runs, status, durations, and diff stats per agent run. (See the `sdk` Cursor skill if needed.)
- **Cursor Stats** dashboard exposes monthly token spend; same API key.

Pragmatic pipeline:

1. Tool `ingest_cursor_local(home_dir: str)` shells `find ~/.config/Cursor/User/History -type f -name '*.json'` and parses the small JSON envelopes — gives per-file edit counts and timestamps. The terminal text already in `terminals/*.txt` and `progress_1_hour.txt` in this repo are precedent that Anubis already harvests local activity.
2. Tool `ingest_cursor_cloud(api_key, days)` paginates `/v0/agents` and `/v0/agents/{id}/runs`.

Normalized schema:

| `ts` | `kind` (`edit` / `agent_run`) | `repo` | `file` | `agent_id` | `prompt_chars` | `tokens_in` | `tokens_out` | `duration_ms` |

---

## 7. ChatGPT token usage

Two paths:

**A. OpenAI API admin reporting.** `GET https://api.openai.com/v1/organization/usage/completions?start_time=...&group_by=user_id,model`. Requires an **admin** key (`sk-admin-...`), not a regular API key. Returns per-user, per-day, per-model token counts — perfect for the AAR.

**B. ChatGPT Data Export.** For ChatGPT Plus / Team accounts, Settings → Data Controls → Export. Yields a `conversations.json` zip. The tool ingests it for *content* analysis (subjects discussed) on top of token counts. Schema is documented at https://help.openai.com/en/articles/7260999.

Normalized schema (path A):

| `date` | `org_id` | `project_id` | `user_id` | `model` | `n_requests` | `prompt_tokens` | `completion_tokens` | `cached_tokens` |

Refresh: daily.
Tool: `ingest_chatgpt_usage(days: int=7)` — uses `OPENAI_ADMIN_API_KEY` env var.

(There is also `costs` endpoint at `/v1/organization/costs` that returns dollar amounts per day grouped the same way — fold it in for finance overlap.)

---

## 8. Azure cloud usage

The first-party path is **Microsoft Cost Management — Query API**:

`POST https://management.azure.com/{scope}/providers/Microsoft.CostManagement/query?api-version=2023-11-01`

Where `{scope}` is one of `/subscriptions/{sub_id}` or `/providers/Microsoft.Billing/billingAccounts/{ba_id}`. Body example:

```json
{
  "type": "ActualCost",
  "timeframe": "MonthToDate",
  "dataset": {
    "granularity": "Daily",
    "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
    "grouping": [
      {"type": "Dimension", "name": "ServiceName"},
      {"type": "Dimension", "name": "ResourceGroupName"}
    ]
  }
}
```

Auth: Service Principal (Azure AD app registration) → `client_credentials` flow → token for resource `https://management.azure.com/`. Use `azure-identity` + `azure-mgmt-costmanagement` (`CostManagementClient.query.usage(...)`).

Schema:

| `date` | `subscription_id` | `service_name` | `resource_group` | `cost_usd` | `currency` |

Tool: `ingest_azure_cost(scope: str, timeframe: Literal["MonthToDate","BillingMonthToDate","TheLastMonth","Custom"], custom_start, custom_end)`.

---

## 9. AWS usage

**A. AWS Cost Explorer (best for $).** `boto3.client("ce", region_name="us-east-1").get_cost_and_usage(...)`:

```python
ce.get_cost_and_usage(
    TimePeriod={"Start": "2026-04-01", "End": "2026-05-01"},
    Granularity="DAILY",
    Metrics=["UnblendedCost", "UsageQuantity"],
    GroupBy=[{"Type":"DIMENSION","Key":"SERVICE"},{"Type":"DIMENSION","Key":"USAGE_TYPE"}],
)
```

Auth: IAM user/role with `ce:GetCostAndUsage`, `ce:GetCostForecast`. Cost Explorer has a $0.01/request charge — cache aggressively in Postgres.

**B. AWS Cost & Usage Reports (CUR) → S3 → Athena.** Hourly granularity, free per-call. Use only if you already have CUR set up.

Schema:

| `date` | `account_id` | `service` | `usage_type` | `usage_qty` | `cost_usd` | `region` |

Tool: `ingest_aws_cost(profile: str|None, days: int=30)`.

---

## 10. Cross-source: how the agent reasons

The agent treats each `/data/<source>/clean/...parquet` as an analysis input. Typical chain inside the deep-agent loop:

1. **Plan** — `write_todos`: "(1) ingest sleep last 7d, (2) ingest git last 7d, (3) ingest aws+azure+chatgpt last 7d, (4) join on date, (5) compute weekly delta vs last week, (6) generate plot, (7) draft AAR, (8) post to slack".
2. **Ingest** in parallel where possible (subagents).
3. **Read** each parquet via `read_file` (the deep-agent FS tool turns small files into context-window-friendly content; bigger files stay on disk and are touched only via `execute` running pandas).
4. **Author** `/scripts/analysis/aar_<run_id>.py` that loads, joins, computes, plots.
5. **Execute** it. Capture `stdout` for facts, `/artifacts/<run>/dashboard.png` for the chart.
6. **Structured-output** the `AfterActionReport`.
7. **`slack_send_message`** the AAR + chart back to the channel.
