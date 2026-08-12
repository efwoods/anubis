# Personal Avatar: auto-provisioning, mailbox, social accounts, multi-device MCP

## Context

Neural Nexus distinguishes one avatar per user as the `PERSONAL_AVATAR_OF_THE_CREATOR`
— the only avatar allowed to reach the creator's private capabilities (desktop MCP data
servers, mailbox, personal/business analytics). The flag exists
(`is_personal_avatar_of_creator` on the LangGraph assistant metadata) and the MCP
capability already gates on it, but three things are missing:

1. **Nothing creates the personal avatar.** `assistants.create` is called in exactly one
   place — the `/create_avatar` route (`src/api/webapp.py:2453`) — where
   `is_personal_avatar_of_creator` defaults to `False`. A verified user therefore has no
   personal avatar until they manually create one and manually flag it. The stated product
   invariant is that every signed-up user always has exactly one, so no feature ever has to
   answer "create a personal avatar first."
2. **The personal avatar has no private capabilities beyond MCP filesystem access.** The
   creator needs the avatar to read and act on email (including following links in a message
   to reschedule a meeting), and to connect social accounts for identity verification and
   future data pulls.
3. **MCP is architecturally singleton-per-user.** One registration record, one connection
   record, one relay device per user. The creator runs several machines (macOS, Ubuntu
   desktop, mobile) and needs all of them connected and listable. The singleton design is
   also the direct cause of a recorded production incident: stopping the dev daemon deleted
   production's registration record, because both write the same key.

Priority order set by the owner: **(1) auto-provisioning → (2) IMAP + scheduling →
(3) social account connection → (4) multi-device MCP.** Each stage lands and is reviewed
before the next begins.

---

## Stage 1 — Auto-provision the personal avatar after email verification

### Where it hooks

Mirror the existing post-verification enrollment. `ensure_initial_subscription_after_verification`
(`src/security/auth.py:383`) is called from `get_user_with_api_key` (`src/security/auth.py:1077`)
the first time a verified account is seen; the personal avatar gets the same treatment
immediately after, guarded by an `app_metadata` marker so the work happens once per account.

### New module: `src/anubis/utils/personal_avatar.py`

Kept out of `auth.py` and out of `webapp.py` because `webapp.py` imports `auth.py` — a
provisioning helper in `webapp.py` would be a circular import. `auth.py` imports this module
lazily inside the function, matching the existing lazy-import convention.

- `PERSONAL_AVATAR_METADATA_FLAG = "is_personal_avatar_of_creator"`
- `async def find_personal_avatar(client, user_id) -> dict | None` — reuses
  `search_all_avatars_for_user` (`src/anubis/utils/avatar_deletion.py:112`, already paged;
  `assistants.search` defaults to `limit=10` and silently hides later avatars).
- `async def ensure_personal_avatar_for_user(request, user, api_key) -> dict | None`:
  1. Return immediately if `app_metadata.personal_avatar_provisioned` is set.
  2. Search the user's avatars; if one already carries the flag, write the marker and return
     (covers users who created theirs manually before this shipped).
  3. Otherwise `client.assistants.create(graph_id="Anubis", ...)` with metadata
     `{"user_id": ..., "is_public": False, "is_personal_avatar_of_creator": True}`, then
     `client.store.put_item((assistant_id, "creator_id"), key="creator_id", ...)` — the same
     two calls `/create_avatar` makes (`webapp.py:2493-2506`), so downstream code that reads
     `creator_id` behaves identically.
  4. Call `_demote_other_personal_avatars` (`webapp.py:2410`) to preserve the one-per-user
     invariant. Move that helper into this module and have `webapp.py` import it, so
     `/create_avatar`, `/modify_avatar`, and provisioning share one implementation.
  5. Write `personal_avatar_provisioned: True` and `personal_avatar_id` into Auth0
     `app_metadata` via `update_user_app_metadata_fields`.
- Default name: the Auth0 `name`, falling back to the email local part. Description states
  plainly that this is the user's own personal avatar.
- Best-effort and idempotent, exactly like the subscription enrollment: any failure logs and
  returns **without** writing the marker, so the next cache-miss retries. Never blocks auth.

### Re-entrancy — the one real hazard

`assistants.create` goes over HTTP to the LangGraph server, whose `@auth.authenticate`
handler (`src/security/auth.py:2054`) calls `get_user_with_api_key` again. If provisioning
runs before the API-key cache is populated, that nested call misses the cache and re-enters
provisioning — unbounded recursion. `ensure_initial_subscription_after_verification` is
unaffected because it only talks to Stripe and Auth0, never back to this API.

Two defenses, both required:

- **Populate `_api_key_cache` before provisioning.** Move the provisioning call after the
  `async with _cache_lock: _api_key_cache[cache_key] = user` block (`auth.py:1088-1091`) so
  the nested authenticate hits a warm cache and returns without side effects.
- **A module-level re-entrancy guard** in `personal_avatar.py`: a `set[str]` of user ids
  currently provisioning, added/removed in a `try/finally`, so a cache eviction mid-flight
  still cannot recurse.

### Known gap to state in the docstring

The customer portal authenticates with email + password and never holds an API key, so the
`/login` route cannot create an assistant (the SDK needs the user's key; only its hash is
stored). Provisioning therefore happens on the account's first API-key request after
verification. This mirrors the "Known gap" already documented on `/login` (`auth.py:1719`).

### Capability registry + endpoint

Add `PERSONAL_AVATAR_CAPABILITIES` to `personal_avatar.py` — the canonical list the owner
asked for, which later stages extend: `desktop_mcp_data_servers`, `mailbox`,
`social_accounts`, `browser_analytics`, `adapter_training_from_all_conversations`.

New `GET /personal_avatar` in `webapp.py` returns the resolved personal avatar (provisioning
it if absent — self-healing, never an error telling the user to create one) plus each
capability with its live status. Later stages fill in the status resolvers.

### Files

`src/anubis/utils/personal_avatar.py` (new), `src/security/auth.py` (hook + cache reorder),
`src/api/webapp.py` (`GET /personal_avatar`, import the moved demote helper).
Tests: new `tests/unit_tests/test_personal_avatar_provisioning.py`, modelled on
`tests/unit_tests/test_initial_subscription_provisioning.py` — cover first-verified-request
creation, idempotence via the marker, adoption of a pre-existing flagged avatar, demotion of
a second flagged avatar, failure leaving the marker unwritten, and no recursion when the
nested authenticate fires.

---

## Stage 2 — Mailbox connection (IMAP) and agentic scheduling

The owner's target flow: a notification email arrives containing an embedded conversation
link; the avatar reads the message, follows the link, reads the thread's context, follows the
meeting link it finds there (e.g. a Calendly page), selects a new time, and confirms. So
"scheduling" here is **browser-driven link following**, not a calendar API.

### 2a. Credentials — IMAP app password, encrypted at rest

- `POST /connect_mailbox` (host, port, username, app_password, use_ssl): verifies the
  credentials by opening a real IMAP connection before saving, encrypts the password with
  `cryptography.fernet` keyed by a new `MAILBOX_CREDENTIAL_ENCRYPTION_KEY`, and writes the
  record to store namespace `(user_id, "mailbox_connection")`, keyed by mailbox address so a
  user may connect more than one. Follows the store-access pattern of `/disconnect_mcp`
  (`webapp.py:2660`), which uses the SDK `StoreClient` rather than the in-process store.
- `GET /list_mailboxes` returns host / username / status / last polled — **never** the
  password or the ciphertext. `DELETE /disconnect_mailbox`.
- All three are personal-avatar-gated: resolve the caller's personal avatar and reject a
  mailbox bound to any other avatar.
- New env vars in `.env`, `.env.dev`, `.env.example` (empty in the example) plus matching
  lowercase `GlobalContext` fields: `MAILBOX_CREDENTIAL_ENCRYPTION_KEY`,
  `MAILBOX_POLL_INTERVAL_SECONDS`, `MAILBOX_FETCH_MAX_MESSAGES`.

**Use stdlib `imaplib` + `email`, not `langchain_imap`.** The feature notes
(`features/capabilities/email_features.md`) reference `langchain_imap`, but the runtime is
Python 3.11 on wolfi where every native-dependent package needs a cp311 wheel, and the
standard library covers fetch/parse completely. Adding a third-party retriever here buys
nothing and risks the image build.

New `src/anubis/utils/tools/email/imap_client.py`: `fetch_unseen_messages(credentials, limit)`
returning normalized `{message_id, sender, recipients, subject, sent_at, body_text, links}`.
HTML bodies go through the existing `_extract_text_from_html_bytes` (`webapp.py:5699`).

### 2b. Browser form-interaction tools — the missing piece

The wired toolkit is LangChain's `PlayWrightBrowserToolkit`
(`src/anubis/utils/tools/browser/browser_tools.py:265`), which supplies navigate,
navigate_back, click by CSS selector, extract_text, extract_hyperlinks, get_elements, and
current_webpage. It has **no way to type into a field, choose a listed option, submit a form,
or see the page**. Booking a meeting slot needs all four.

Add `src/anubis/utils/tools/browser/form_tools.py` with `fill_form_field`, `click_by_text`
(time slots are labelled buttons, not stable selectors), `select_option`, `press_key`, and
`capture_page_screenshot`. Build them on the same per-conversation browser registry and the
toolkit's own `aget_current_page`, so they act on exactly the page the navigate/click tools
left behind. Return them from `get_browser_toolkit_tools` alongside the toolkit's tools, under
the same `BROWSER_TOOLS_ENABLED` gate and the same turn lease.

**Security change in the same edit:** browser tools currently attach to *every* avatar
(`graph.py:801`, env gate only). Once the browser follows authenticated links and — in Stage 3
— carries the creator's logged-in session, that is a data-exposure path. Gate the browser
tools on `_user_personal_avatar` (`graph.py:1103`) in addition to the env switch.

### 2c. Email triage subgraph

Flesh out the commented stub at `src/subgraphs/email/utils/graph.py` (its intended node
sequence is already written there):

```
START → accept_email → classify_next_action → {ignore | notify | respond | schedule}
respond  → load_consciousness → write_response → approve (interrupt) → send_reply → END
schedule → load_consciousness → follow_links_and_schedule → approve (interrupt) → confirm → END
```

- `EmailTriageClassification` — a Pydantic structured-output model over
  `ignore / notify / respond / schedule`, per the OpenAI GPT-5 prompting conventions and the
  explicit-naming rule (no acronyms, no bare "it").
- Reuse `load_consciousness` (`src/anubis/utils/nodes.py`) so replies carry the target's voice.
- `follow_links_and_schedule` runs the deep agent with the Stage 2b browser tools, given the
  email body and its extracted links: open the conversation link, read context, find and open
  the meeting link, choose a time, fill the form, screenshot before confirming.
- **Nothing is sent or booked without approval.** Raise a LangGraph `interrupt` carrying the
  draft reply or the proposed booking plus the confirmation screenshot, exactly as
  `mcp_discovery` does (`graph.py:1166`); resolve through the existing
  `POST /message/{assistant_id}/resume` (`webapp.py:3879`).

### 2d. Polling schedule

Use LangGraph's built-in crons — the routes are already whitelisted in
`_is_auth_catch_all_target` (`webapp.py:1120-1126`), so no new scheduler dependency and no
lifespan background task. `POST /schedule_mailbox_poll` creates a cron over the email graph
for the caller's mailbox at `MAILBOX_POLL_INTERVAL_SECONDS`; `DELETE /schedule_mailbox_poll`
removes it. Register the new graph in `langgraph.json` alongside `Anubis` and `process_media`.

### 2e. Prompt

Add `EMAIL_CAPABILITY_PROMPT` to `src/anubis/utils/prompts/system_prompts.py` and append it in
`load_consciousness` next to `DATA_ANALYSIS_CAPABILITY_PROMPT` (`nodes.py:704-714`), under the
same `is_personal_avatar` condition — the prompt must never claim a capability the tool gate
will withhold. Never let the avatar echo mailbox credentials, session cookies, or full
one-click login URLs into a reply.

---

## Stage 3 — Social media account connection

Two mechanisms, because the four target platforms do not share one path.

**Auth0 social identity linking** (verification, and the basis of future data pulls) for
Twitch, Instagram, X/Twitter, and YouTube via Google:
- `GET /social_connect_url/{provider}` builds the Auth0 `/authorize` URL with
  `connection=<provider>` and a signed state bound to the caller.
- `GET /social_connect_callback` exchanges the code and links the secondary identity to the
  primary account with Management API `POST /api/v2/users/{id}/identities`, then records the
  provider, handle, and verification timestamp in store namespace
  `(user_id, "social_account")` keyed by provider, mirrored into `app_metadata.social_accounts`.
- `GET /list_social_accounts`, `DELETE /disconnect_social_account`.

**Playwright session login** for platforms whose API access is impractical: persist an
encrypted Playwright `storage_state` per user and provider (new
`BROWSER_SESSION_ENCRYPTION_KEY`) so the personal avatar's browser reuses a logged-in session.
Note the constraint already documented in the browser module's docstring — the toolkit
hard-codes `browser.contexts[0]`, so the session must be loaded into that first context at
launch rather than through an extra `new_context()`.

Verified linkage feeds the existing share gate: only an owner with a verified matching social
identity may make an avatar of their own likeness public. **Initial data pull and
subscribe-to-new-posts stay out of scope** — `features/login_and_pull_data_and_subscribe_from_social_media.md`
records them as the follow-on feature.

---

## Stage 4 — Multiple MCP servers (macOS, Ubuntu desktop, mobile) and listing

Today `mcp_registration_namespace(user_id)` and `mcp_connection_namespace(user_id)`
(`src/anubis/utils/tools/data_analysis/backend.py:68-89`) each hold **one** record under a
constant key, and `relay._device_id_by_user` maps a user to **one** device. Change the unit
of identity from the user to the device.

- **`backend.py`** — keep the namespaces, key each record by `device_id` instead of the
  constant `REGISTRATION_KEY` / `CONNECTION_KEY`. Records gain `device_label`
  ("macOS", "Ubuntu desktop", "mobile") and `platform`, supplied by the daemon's register body.
- **`relay.py`** — `_device_id_by_user: dict[str, str]` becomes `dict[str, set[str]]`;
  `session_for_user` becomes `sessions_for_user(user_id) -> list[RelaySession]`; `get_session`
  and `is_online` are already device-keyed and stay.
- **`discovery.py`** — pluralize `read_user_registration`, `read_user_connection`,
  `resolve_available_connection`, and `bound_connection_for` into list-returning equivalents;
  `save_user_connection` takes the `device_id`. On read, migrate a legacy record found under
  the old constant key by re-keying it under its own `device_id`, so existing installs keep
  working without a manual step.
- **`graph.py` `think`** (`graph.py:761-792`) — build tools across every bound connection.
  Give each MCP tool a `device_label` argument resolved against the connection list, defaulting
  to the sole connection when only one exists; this keeps the tool set and prompt small rather
  than emitting one tool per device. `mcp_discovery` offers each newly registered, unconnected
  device rather than returning early once any connection exists.
- **Endpoints** — new `GET /list_mcp_connections` (label, platform, device id, online, bound
  avatar, connected at); `/disconnect_mcp` and `/mcp/unregister` take `device_id` and delete
  only that device's record. **This closes the recorded production incident** where stopping
  the dev daemon deleted production's registration, since both wrote the same singleton key.
- **Out of this repo:** the macOS and mobile daemons themselves are separate deliverables. The
  existing `anubis-mcp-server-ubuntu` daemon must be updated to send `device_label` and
  `platform` on register and heartbeat.

Existing tests to update: `tests/unit_tests/test_mcp_relay.py`,
`test_mcp_discovery_flow.py`, `test_data_analysis_capability.py`.

---

## Verification

Run per stage; do not advance until the stage's checks pass.

**Unit tests.** `make test TEST_FILE=tests/unit_tests/<file>.py` per new/changed file. Do not
use a bare `make test` as the gate: the suite currently hangs on
`tests/unit_tests/test_think_interrupt_flow.py` and seven unrelated tests fail at baseline —
confirm any failure predates the change before treating it as a regression.
`make lint_diff` and `make format_diff` on the changed files.

**Stage 1, end to end.** `docker compose up`. Sign up a fresh account, verify the email, then
make one authenticated request with the returned API key. `GET /personal_avatar` and
`GET /list_user_avatars` must both show exactly one avatar carrying
`is_personal_avatar_of_creator: true`. Repeat the request and confirm via logs that no second
avatar is created and no second Auth0 write occurs. Manually flag a second avatar through
`/modify_avatar` and confirm the first is demoted.

**Stage 2, end to end.** Connect a real mailbox with an app password via `/connect_mailbox`;
confirm `/list_mailboxes` never returns the password. Send that mailbox a message containing a
conversation link that leads to a meeting-booking link. Trigger the poll, then confirm: the
message is classified `schedule`, the avatar opens both links, reaches the booking page,
proposes a time, and **stops at the interrupt**. Approve through
`POST /message/{assistant_id}/resume` and verify the booking completed from the returned
screenshot and the confirmation email. Separately confirm a non-personal avatar of the same
user has no email tools and no browser tools.

**Stage 3.** Link each provider through `/social_connect_url` → callback; confirm the identity
appears in the Auth0 user's `identities` array and in `/list_social_accounts`, and that
disconnecting removes both.

**Stage 4.** Run two daemons with distinct device ids (dev + prod, or two hosts). Confirm
`/list_mcp_connections` shows both as online, that a tool call routes to the requested
`device_label`, and — the regression that motivated this — that stopping one daemon leaves the
other's registration and live connection intact.
----

EMAIL INBOX
# Stage 2 — Mailbox connection, email-triage graph, Agent Inbox

## Immediate deliverable (owner's direction, 2026-08-12)

**Documentation only — no implementation yet.** Attach the plan below as a new section in
`_WORK_IN_PROGRESS.md` so it is ready for later implementation. Concretely:

1. Add a new section "Stage 2 (revised) — Mailbox, email-triage graph, Agent Inbox" to
   `_WORK_IN_PROGRESS.md` carrying the architecture, landing order, and verification steps
   below.
2. In the existing Stage 2 section, add a short pointer noting it is superseded by the
   revised section (keep the original text for history rather than deleting it).
3. Fold in the verification corrections so the doc stops citing dead references: the
   removed `mcp_discovery` interrupt (now `mcp_auto_adopt`, no interrupt; the live pattern
   is `graph.py:979` + `graph_interrupts.py`), the never-stubbed `/handle_email`, the
   broken import in the email stub, and the drifted line numbers
   (`_extract_text_from_html_bytes` → webapp.py:6014, resume → 4206, `/disconnect_mcp` →
   2760, capability-prompt append → nodes.py:686-725).
4. Mark Stage 4 as ✅ built (uncommitted) in the doc's priority list, since the doc still
   lists it as pending.

No source files change in this pass.

## Context

Stage 4 (multi-device MCP) is built; Stage 1 (auto-provisioning) is committed. The next
phase is Stage 2 of `_WORK_IN_PROGRESS.md`, **re-architected by the owner** on 2026-08-12:

1. **Mailbox connects like MCP does** — per-mailbox records in the store, connect/list/
   disconnect endpoints, status surfaced through the already-declared `mailbox` capability
   (`personal_avatar.py:78`, `status_key="connected_mailboxes"`). No consent friction.
2. **The email-triage process is a graph, callable as a tool.** When an email arrives, the
   triage graph runs for that email and classifies it **respond / ignore / notify**.
3. **Notify lands in Agent Inbox** (`/home/user/gh/anubis-project/agent-inbox` — LangChain's
   Next.js inbox UI). Notifications are ALSO queryable and respondable **from chat**: the
   avatar alerts the owner when notifications are pending and can resolve them in
   conversation.
4. **The avatar learns the owner's triage preferences** (which senders/kinds get responded
   to, ignored, escalated) from the owner's decisions, and feeds them back into triage.
5. **Responses are generated by the Anubis graph** — the owner's tone of voice via
   `load_consciousness`, plus tool calls to complete actions (e.g. scheduling via browser
   tools).

### Verification against code (2026-08-12)

Claims from the WIP doc checked before planning:

- **Accurate**: no mailbox/IMAP code exists (greenfield); the Playwright toolkit has exactly
  7 tools and cannot type/select/submit/screenshot; browser tools attach to every avatar
  (env gate only, `graph.py:803`) — a live exposure path; cron passthrough routes are
  whitelisted (`webapp.py:1122-1127`) but **never exercised anywhere**; `cryptography.fernet`
  is in `uv.lock`; `langchain_imap` is NOT installed → stdlib `imaplib` is right.
- **Stale/wrong**: `/handle_email` was never stubbed (no such route); the email subgraph
  stub (`src/subgraphs/email/utils/graph.py`) has a broken import
  (`from langgraph.graph import ... State`); the `mcp_discovery` interrupt the doc cites was
  removed by Stage 4 — the one remaining interrupt site is the deep-agent forwarding at
  `graph.py:979` using `src/anubis/utils/graph_interrupts.py`.
- **Agent Inbox contract** (verified in the repo): talks **directly to the LangGraph
  deployment** via SDK — `client.threads.search({status:"interrupted"})`; expects interrupt
  values shaped as `HumanInterrupt` (`action_request:{action,args}`,
  `config:{allow_ignore,allow_respond,allow_edit,allow_accept}`, `description`); resumes
  with `Command(resume=[{type:"accept"|"ignore"|"response"|"edit", args}])`. It sends auth
  as **`x-api-key`** (`agent-inbox/src/lib/client.ts:13`) while our deployment authenticates
  the `API-KEY` header — one-line fork patch in `client.ts`. Its thread-values type already
  includes an `Email` shape (`types.ts:57`) matching LangChain's email-assistant reference.

### Architectural consequence (replaces the old open question)

For interrupts to appear in Agent Inbox, each email's triage must run **on its own thread of
a registered `email` graph** — Agent Inbox lists interrupted threads per graph. So "called
as a tool" means the tool is a thin SDK wrapper that creates/queries **runs on the email
graph's threads** (one thread per email, thread values carrying the `Email` shape), not an
in-process subgraph call inside `think` (whose interrupts would surface on the conversation
thread instead). This also dissolves the earlier `/message/{assistant_id}/resume` mismatch:
email approvals resume through Agent Inbox or through the chat-side tool, never through the
message-resume endpoint.

### Owner decisions on this plan (2026-08-12)

- **Landing order**: Landing 1 = mailbox connection + secret store + triage graph +
  HumanInterrupt schema + cron fan-out + agent-inbox auth patch (provable end-to-end:
  email in → inbox item → approved reply out). Landing 2 = chat tools + alert block +
  preference learning. Landing 3 = browser form tools + personal-avatar browser gate.
- **Assumption (question not answered — default taken)**: every drafted reply requires an
  approval (accept/edit) before sending; preference learning improves classification and
  draft quality but never bypasses the interrupt. This matches the original WIP rule
  "nothing is sent or booked without approval" and can be relaxed later behind a
  confidence gate without reshaping the graph.

---

## Implementation

### 1. Mailbox connection — like MCP (`webapp.py`, new `imap_client.py`)

- Namespace `(user_id, "mailbox_connection")`, keyed by mailbox address — mirror of
  `mcp_connection_namespace` (`data_analysis/backend.py:68`). Read via the Stage 4
  multi-record helper `_search_device_records` (`webapp.py:2490`) — rename it to a neutral
  `_search_namespace_records` since it is not device-specific.
- `POST /connect_mailbox` (host, port, username, app_password, use_ssl): verify with a real
  `imaplib.IMAP4_SSL` login before saving; encrypt the password with `cryptography.fernet`
  keyed by `MAILBOX_CREDENTIAL_ENCRYPTION_KEY`. Build the Fernet helper generically
  (`src/anubis/utils/secret_store.py`) — first encryption-at-rest in the codebase, and
  Stage 3 needs the same for `BROWSER_SESSION_ENCRYPTION_KEY`.
- `GET /list_mailboxes` (never the password/ciphertext), `DELETE /disconnect_mailbox`.
  All three personal-avatar-gated via `resolve_personal_avatar` (`personal_avatar.py:356`).
- Fill the `connected_mailboxes` status in `GET /personal_avatar`
  (`_resolve_personal_avatar_capability_statuses`, `webapp.py`), replacing `not_configured`.
- New `src/anubis/utils/tools/email/imap_client.py` — stdlib `imaplib` + `email`:
  `fetch_unseen_messages(credentials, limit)` → normalized
  `{message_id, sender, recipients, subject, sent_at, body_text, links}`; HTML bodies
  through `_extract_text_from_html_bytes` (`webapp.py:6014`).

### 2. Email triage graph (`src/subgraphs/email/`) — registered in `langgraph.json`

Rewrite the broken stub. Register as `"email"` alongside `Anubis` / `process_media`.

```
START → accept_email → recall_triage_preferences → classify_next_action
  ignore  → record_outcome → END
  notify  → interrupt(HumanInterrupt) → apply_owner_decision → record_outcome → END
  respond → draft_with_anubis → interrupt(HumanInterrupt draft preview) →
            apply_owner_decision → send_reply → record_outcome → END
```

- `EmailTriageClassification` — Pydantic structured output over
  `ignore / notify / respond`, GPT-5 prompting conventions, explicit naming (no acronyms,
  no bare "it").
- **Interrupt payloads use the Agent Inbox `HumanInterrupt` schema exactly** — notify:
  `action` `"notify_owner"`, `config {allow_ignore:true, allow_respond:true, allow_edit:false,
  allow_accept:false}`; respond-draft: `action` `"send_email_reply"`, args carrying the
  draft, `config {allow_accept:true, allow_edit:true, allow_ignore:true, allow_respond:true}`.
  `apply_owner_decision` handles the `HumanResponse` list shape
  (`accept`/`edit`/`response`/`ignore`).
- Thread values carry the inbox `Email` shape (`types.ts:57`) so the inbox list renders
  subject/sender natively.
- `draft_with_anubis`: reuse `load_consciousness` (`src/anubis/utils/nodes.py`) +
  `init_model` so the draft is in the owner's voice; scheduling-type actions delegate to the
  personal avatar's tools (browser form tools — see step 5).
- `send_reply`: SMTP via the same stored credentials (`smtplib`, submission port derived
  from the IMAP host unless given). Nothing sends without an accepted interrupt.

### 3. Ingestion + "called as a tool"

- **Poll node / entry**: `POST /schedule_mailbox_poll` creates a LangGraph **cron** on a
  dedicated poller thread of the email graph at `MAILBOX_POLL_INTERVAL_SECONDS`
  (`client.crons.create_for_thread`); `DELETE` removes it. The poll run fetches unseen
  messages and **spawns one email-graph run per message on a fresh thread** (SDK
  `runs.create`) — that per-thread fan-out is what makes each email its own inbox item.
  Since nothing in the codebase has ever exercised the cron routes, verify early with a
  one-minute cron in dev before building on it.
- **Chat-side tool** `triage_mailbox_now` on the personal avatar (built alongside the
  data-analysis tools in `think`, same `is_personal_avatar` gate): thin SDK wrapper that
  triggers the same fetch-and-fan-out on demand.

### 4. Notifications in chat + preference learning

- **Chat-side tools** (personal avatar only):
  - `list_email_notifications` — `threads.search({status:"interrupted"})` on the email
    graph, summarize pending items (who/subject/proposed action).
  - `resolve_email_notification` — forward the owner's chat decision as the
    `Command(resume=[HumanResponse])` the graph expects (reuse
    `build_interrupt_resume_command` conventions from `graph_interrupts.py`).
- **Alerting**: in `load_consciousness`, for the personal avatar only, when interrupted
  email threads exist append a short `<EMAIL_NOTIFICATIONS>` block (count + subjects) so the
  avatar raises them naturally at the start of a conversation — same pattern as the
  `<MCP_CONNECTION_STATUS>` block added in Stage 4 (`nodes.py:707`).
- **Preference learning**: namespace `(user_id, "email_triage_preference")`. Every final
  outcome (`record_outcome`) writes/updates a compact preference record: sender, list-id,
  classification chosen by the model, decision the owner actually took (accept, override,
  ignore), and a short rationale extracted from any owner edit/response text.
  `recall_triage_preferences` retrieves top-K similar records (store vector index already
  embeds `page_content` — keep the record's text in that field, consistent with the
  640-dim index in `langgraph.json`) and injects them into the triage prompt as few-shot
  precedent. Owner decisions in chat and in Agent Inbox both flow through
  `apply_owner_decision`, so both surfaces teach the same store.

### 5. Browser form tools + personal-avatar gate (unchanged from prior plan; prerequisite
for "complete actions on my behalf")

- `src/anubis/utils/tools/browser/form_tools.py`: `fill_form_field`, `click_by_text`,
  `select_option`, `press_key`, `capture_page_screenshot` — built on the same
  per-conversation registry and `aget_current_page`, returned from
  `get_browser_toolkit_tools` under the same gate/lease.
- **Gate browser tools on `_user_personal_avatar`** in `think` (`graph.py:803`) — the flag
  is already computed at `graph.py:764`; one-line condition closing a live exposure path.

### 6. Agent Inbox fork patch

- `agent-inbox/src/lib/client.ts`: send the key as `API-KEY` (keep `x-api-key` too) so the
  inbox authenticates against our deployment with the owner's API key. Separate repo,
  separate commit.

### 7. Env vars (uppercase in `.env`, `.env.dev`, empty in `.env.example`; lowercase
`GlobalContext` fields)

`MAILBOX_CREDENTIAL_ENCRYPTION_KEY`, `MAILBOX_POLL_INTERVAL_SECONDS`,
`MAILBOX_FETCH_MAX_MESSAGES`.

### Files

Main repo: `src/anubis/utils/tools/email/imap_client.py` (new),
`src/anubis/utils/secret_store.py` (new), `src/subgraphs/email/utils/graph.py` (rewrite),
`langgraph.json`, `src/api/webapp.py` (mailbox endpoints + capability status + poll
schedule), `src/anubis/graph.py` (chat tools + browser gate),
`src/anubis/utils/nodes.py` (notification block),
`src/anubis/utils/prompts/system_prompts.py` (email capability prompt),
`src/anubis/utils/tools/browser/form_tools.py` (new), `context.py` + env files.
Other repo: `agent-inbox/src/lib/client.ts`.

Tests: `tests/unit_tests/test_mailbox_connection.py`, `test_email_triage_graph.py` (drive
the graph with a fake IMAP client + in-memory store; assert HumanInterrupt shape,
preference write/recall, nothing sends without accept), `test_browser_form_tools.py`,
plus the personal-avatar gate case in the browser-tools tests.

## Verification

Per repo rules: per-file `make test TEST_FILE=...` (never bare `make test` — known hang),
`make lint_diff` / `make format_diff` on changed files only (branch has ~1188 pre-existing
lint errors; hold the my-files-clean line as in Stage 4).

End-to-end (`docker compose up`):
1. `/connect_mailbox` with a real app password → `/list_mailboxes` shows it, never the
   secret; `GET /personal_avatar` shows `connected_mailboxes` populated.
2. One-minute cron in dev proves the never-exercised cron path, then send a test email →
   poll → one new email-graph thread; check `respond`/`notify` classification.
3. Point Agent Inbox (dev) at the deployment with the owner's API key → the interrupt
   appears; accept a draft → the reply actually sends.
4. In chat: avatar mentions the pending notification, `list_email_notifications` /
   `resolve_email_notification` round-trip works.
5. Override a triage decision twice; confirm the third similar email classifies per the
   learned preference.
6. Confirm a non-personal avatar has no email tools and no browser tools.
