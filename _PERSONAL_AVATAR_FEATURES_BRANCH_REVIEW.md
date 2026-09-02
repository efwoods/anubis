# Feature analysis: `f-personal-avatar-features` vs `main`

## Context

Branch `f-personal-avatar-features` is 9 commits / **+8,771 −1,120 across 43 files** ahead of `main`
(merge-base `319ac7e`, 2026-07-27 → 2026-08-12). Working tree clean. HEAD is 12 ahead of
`origin/main` and 0 behind — a clean fast-forward, nothing to rebase.

This document is a read-only assessment of what shipped, what is advertised but not built, and
what should be fixed before merge. No code changes were made.

## Churn by area

| Area | Added | Removed |
|---|---|---|
| tests | +4,034 | −381 |
| MCP / data-analysis | +1,187 | −307 |
| billing / Stripe | +1,067 | −98 |
| planning docs | +698 | −1 |
| src (other) | +618 | −96 |
| security | +580 | −97 |
| api | +507 | −136 |

---

## What shipped

### 1. Personal-avatar auto-provisioning (`src/anubis/utils/personal_avatar.py`, new, 395 lines)

Enforces the "every signed-up user has exactly one personal avatar" invariant.

- Triggered from **two auth paths only**: `auth.py:1139` (API-key, after the `_api_key_cache` write —
  ordering is load-bearing, since avatar creation re-enters `authenticate`) and `auth.py:1300`
  (refresh-token). Both non-fatal.
- Creates assistant + `creator_id` store item, flags `is_personal_avatar_of_creator`, then
  `demote_other_personal_avatars`.
- Idempotent via Auth0 `app_metadata["personal_avatar_provisioned"]` marker, plus a module-level
  `asyncio.Lock` re-entrancy guard. Marker written only on success, so failure retries.
- Self-healing: `resolve_personal_avatar` pops a stale marker and re-provisions.
- New route `GET /personal_avatar` (`webapp.py:2425`).

### 2. Dual-credential authentication (`src/security/auth.py`)

Endpoints now accept **API key OR Auth0 refresh token**. Both schemes are `auto_error=False`;
`_resolve_authenticated_user` (`auth.py:1313`) tries API key first, then Bearer.

The load-bearing trick is the **ephemeral API key bridge** (`auth.py:1152`): a refresh-token session
has no real API key, so one is minted and seeded directly into `_api_key_cache`, letting ~14
`current_user["API_KEY"]` LangGraph SDK call sites work unchanged. No cookies anywhere — credentials
come from the `API-KEY` and `Authorization: Bearer` headers only.

### 3. Multi-device MCP

Identity moved from **user-keyed to device-keyed** across all three store namespaces. New
`devices.py` derives a friendly label/platform per machine, dedupes collisions ("Ubuntu 2"), and
resolves a file path to its owning machine.

- **Auto-adopt replaced the consent interrupt** (`graph.py:1123`). Devices bind silently; the
  rationale is that the daemon registered with the user's own API key.
- **Fan-out**: `discover_data_files` gathers one leg per device with a 20 s per-leg deadline; a
  failed leg becomes `{"status": "offline"}` rather than raising.
- Directly addresses two recorded production incidents — unregister/heartbeat now scope to the
  calling device only, so a dev daemon's shutdown can no longer delete prod's record.

### 4. Report display

Deep-agent artifacts (reports, plots) are swept from the workspace and attached to
`final_message.response_metadata["created_artifacts"]` (`graph.py:1045`), which webapp already
forwards verbatim on the terminal SSE `done` frame.

### 5. Billing / Stripe

- **Provisioning became create-*and-mutate***: price drift is detected and replaced via
  `transfer_lookup_key=True` + archive, then `migrate_live_subscriptions` rewrites live subscriptions.
- Environment selection replaced `--allow-live` with `--live` (`.env.dev`/`sk_test_` vs `.env`/`sk_live_`).
- New `scripts/provision_stripe_webhook.py` (278 lines) automates prod webhook registration.
- **Hot-reload of billing config by file mtime** (`config.py:231`) — a reprovision takes effect
  without an API restart.
- New `usage_notification.py` pushes cumulative usage to the customer portal after each metered turn
  (HMAC-signed, fire-and-forget, fail-open) so portal meters do not lag Stripe aggregation.
- Anonymous billing now verifies the persisted Stripe customer is live and re-creates it if deleted.

### 6. Permissions hardening

- `share_avatar`: admin-only → **creator-or-admin**, and status changed **401 → 403** (client-visible break).
  Absent `metadata.user_id` fails closed.
- `get_thread_messages`: new thread-ownership check with a deliberate grandfather clause for
  pre-stamping threads.
- `/delete_user` no longer requires a verified email; the unverified path reads assistant ids
  straight from Postgres to avoid re-entering the verified-email gate via the SDK.

---

## Quality signals

- **127 new test functions** across 8 new files; roughly 1:1 test-to-source line ratio.
- **Zero new `TODO`/`FIXME`/`NotImplementedError`** — every marker in the changed files is pre-existing.
- All 5 new env vars are in `.env.example` (unset) *and* documented as `GlobalContext` fields.
- `test_mcp_discovery_flow.py` was deleted but **replaced, not dropped** — its two still-valid cases
  map 1:1 onto the new auto-adopt tests.
- Binary `.odt` assets were committed, but `.dockerignore` gained `assets/*` so they stay out of the
  build context.

---

## Findings, by severity

### P0 — confirmed correctness bug: duplicate device labels

Verified at every link:

1. `/mcp/register` dedupes labels (`webapp.py:3174`), but the **relay websocket path does not** —
   `webapp.py:2986` calls `derive_device_identity` with no `deduplicate_label`. (Only call sites are
   `webapp.py:3174` and `:3297`.)
2. `bound_connections_for` treats the live session as authoritative and **writes the session's
   non-deduped label back over the stored deduped one** (`discovery.py:608-616` compares
   `device_label` and re-saves on difference).
3. `connection_label_map` is a dict comprehension keyed on label (`devices.py:174`) — two machines
   sharing "Ubuntu" collapse to one.
4. `discover_data_files` returns `dict(outcomes)` (`analysis_tools.py:462`) — the duplicate key
   **silently drops one machine's entire result set**.

Net effect: two same-platform machines produce silently incomplete analysis results. No test covers
duplicate labels at the tool layer.

**Fix**: call `deduplicate_label` on the relay path, and stop the write-back from reverting labels.

### P1 — operational risk: free-tier change mutates live Stripe on next prod boot

`tiers.py:181` raises the free-tier `MESSAGING_TOKENS` allotment **200k → 2M (10×)**. Combined with
the new mutate behavior and the fact that `docker-compose-prod.yml` runs
`provision_stripe_billing.py --live` as a startup dependency, the first prod `up` after merge will:
archive the old free price, mint a replacement, and rewrite the item set of **every**
active/trialing/past_due/unpaid subscription holding it — including anonymous $0 metering subscriptions.

Aggravating details:
- `migrate_live_subscriptions` **deletes and re-adds every item**, dropping anything not in the tier's
  canonical price list.
- Multi-tier subscriptions are coerced to `held_superseded_tiers[0]`.
- Scheduled subscriptions are printed and skipped with no non-zero exit — an operator must notice.
- `langgraph-api-prod` hard-depends on provisioning succeeding, so a Stripe outage blocks API startup.

**Recommendation**: dry-run the migration against test mode and confirm the blast radius before merging.

### P2 — advertised-but-absent capabilities

`GET /personal_avatar` returns 5 capabilities, but `_resolve_personal_avatar_capability_statuses`
(`webapp.py:2514`) only ever fills `connected_data_servers` and **hardcodes
`adapter_training = "active"`** with no backing check. `mailbox`, `social_accounts`, and
`browser_analytics` can only ever report `not_configured`.

Verified genuinely absent from the branch: no IMAP/mailbox code at all (`src/subgraphs/email/utils/graph.py`
is still the broken stub importing a non-existent `State`), no social-account endpoints or namespace.

**Doc drift**: `_SOCIAL_MEDIA_ACCOUNT_CONNECTION.md:1` is literally headed `# Complete` for a feature
with zero code. `_WORK_IN_PROGRESS.md` marks stages 2–3 "Documentation only — no implementation yet."

### P3 — smaller items

- **Cache invalidation gap**: `_evict_api_key_cache_for_user` does not purge `_refresh_token_cache`, and
  every cached-session hit re-seeds the stale user dict back into `_api_key_cache`. A Stripe-webhook
  tier change does not take effect for a browser session for up to the 300 s TTL.
- **Device cap fails open**: `_search_device_records` swallows exceptions into `[]`, so an unreachable
  store makes `other_device_count == 0`. The relay websocket path has no cap at all.
- **`/disconnect_mcp` with no `device_id` deletes every device** — the exact behavior `/mcp/unregister`
  now refuses because of a production incident.
- **Artifacts lost on early returns**: `collect_turn_artifacts` runs at `graph.py:1045`, after two
  earlier `return {}` paths, while the `finally` still wipes the workspace.
- **Inline artifacts in checkpointed state**: up to 2 MiB *per artifact* with no per-turn count cap,
  and `response_metadata` is now also read for billing.
- **`check_data_server_connection` hardcodes `"connected": True`** — safe only because the tool is
  built solely when connections exist.
- Vestigial `"status": "pending_consent"` written after consent was deleted.
- `STRIPE_WEBHOOK_URL` (read by the new script) is missing from `.env.example`.
- `provision_stripe_webhook.py` has no test file and is not wired into compose/Makefile/CI.
- The two provisioning scripts now use **opposite safety flags** (`--live` vs `--allow-live`).
- `HANDLED_EVENT_NAMES` is triplicated by hand across three files with only a comment enforcing agreement.

### Explicitly NOT problems with this branch

I checked these and they are pre-existing on `main`, not introduced here:
- The 17 `logger.info("breakpoint")` debug lines (branch adds **zero**).
- The commented-out anonymous ban-enforcement block in `auth.py`.
- The `x-forwarded-for` hashing weakness in `resolve_request_hashed_ip`.
- Process-local cache concerns are **latent**: prod compose declares no `replicas`/`workers`.

---

## Verification

Baseline caveat: `make test` does **not** complete — `tests/unit_tests/test_think_interrupt_flow.py`
hangs, and 7 unrelated tests fail at baseline. Run the branch's own suites directly:

```bash
python -m pytest tests/unit_tests/test_mcp_multi_device.py \
  tests/unit_tests/test_mcp_auto_adopt_flow.py \
  tests/unit_tests/test_mcp_relay.py \
  tests/unit_tests/test_data_analysis_capability.py \
  tests/unit_tests/test_personal_avatar_provisioning.py \
  tests/unit_tests/test_refresh_token_authentication.py \
  tests/unit_tests/test_thread_ownership.py \
  tests/unit_tests/test_share_avatar_permissions.py \
  tests/unit_tests/test_stripe_catalog_provisioning.py \
  tests/unit_tests/test_anonymous_billing_recovery.py -q
```

Before merging, additionally:
1. **Stripe dry-run** — `python scripts/provision_stripe_billing.py` (test mode) and read the
   migration output to confirm the free-tier price replacement blast radius.
2. **Two same-platform daemons** connected simultaneously via relay, then ask the avatar to
   `discover_data_files` — this reproduces the P0 label collision.
3. **Refresh-token session** end to end: `/login` → call a graph-backed endpoint with
   `Authorization: Bearer <refresh_token>` → confirm the ephemeral-key bridge resolves.
