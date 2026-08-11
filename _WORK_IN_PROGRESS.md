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
