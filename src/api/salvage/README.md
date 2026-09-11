# Inert salvage from the `z-anubis` line

Two features were built and tested on the `z-anubis` branch, never merged into
`f-anubis`, and are preserved here as complete but **inert** code: per-avatar
**storage allotments** and **adapter routing**. Salvaged 2026-09-09.

Nothing in this directory runs. `src/api/webapp.py` does not import either
module and never calls `include_router` on `storage_route` or `adapter_route`,
so none of the six paths are served, no background task is started, and no
existing behaviour changes.

## What was salvaged, and from where

| File | Salvaged from | Origin |
|---|---|---|
| `src/anubis/utils/billing/storage.py` | `z-anubis` `c8a958e`, `6ed92cb` | same path |
| `src/api/salvage/storage_routes.py` | `z-anubis` `c8a958e` | inline in `webapp.py` |
| `src/anubis/utils/adapters/__init__.py`, `client.py` | `z-anubis` `d1b8473` | same path (verbatim) |
| `src/api/salvage/adapter_routes.py` | `z-anubis` `d1b8473` | inline in `webapp.py` |

The matching web-application code is salvaged in the frontend repository under
`src/components/salvage/` and `src/services/avatarStorageAndAdapter.js`
(from the orphaned frontend commits `c62c798` and `d8d2c93`).

The adapter **service** side of this feature — an OpenAI-compatible
`/v1/chat/completions` that attaches the caller's trained adapter — lives on
the `z-anubis-adapter` branch of the `anubis-adapter` repository
(`0381599`: `src/api/openai_compatible_routes.py`, `src/inference/tool_calls.py`,
`src/adapters/storage.py`, plus tests). It was **not** salvaged into
`f-anubis-adapter` and is still only on that branch.

## Deliberate divergences from the originals

These exist so that salvaging edits no live code. Undo each one on activation.

1. **`storage.py` carries its own tier table.** On `z-anubis` the per-tier byte
   allotment was a `storage_bytes_per_avatar` field on `TierDefinition` and the
   pack constants lived in `tiers.py`. Here they are module-local
   (`TIER_STORAGE_BYTES_PER_AVATAR`, `STORAGE_ADDON_*`). Move them into
   `tiers.py` as the `z-anubis` diff has them.
2. **`gating.resolve_use_adapter_inference` is untouched.** `z-anubis` changed
   its signature to a three-state `adapter_requested` plus `adapter_available`,
   so a Premium user with a trained adapter uses it automatically. That
   decision is carried here as `adapter_routes.resolve_adapter_preference`.
   Fold it into `gating.py` on activation.
3. **Route handlers are on `APIRouter`s, not `@app`.** They were inline in
   `webapp.py` upstream. Dependencies on `webapp` are imported inside each
   handler so these modules import standalone.
4. **`storage_addon_price_id` is read with `getattr`.** The field does not
   exist on f-anubis's `StripeBillingConfig` yet (step 3 below).
5. **`.gitignore` gained `!src/anubis/utils/adapters/`.** The first line of
   `.gitignore` is `adapters`, which would otherwise silently ignore the
   adapter-service client source. `z-anubis` carries the same negation.

## Activating storage allotments

1. Register the router: `app.include_router(storage_route)` in `webapp.py`.
2. Move `TIER_STORAGE_BYTES_PER_AVATAR` onto `TierDefinition.storage_bytes_per_avatar`
   and the `STORAGE_ADDON_*` constants into `tiers.py` (divergence 1).
3. Add `storage_addon_price_id` to `StripeBillingConfig`
   (`src/anubis/utils/billing/config.py`) and provision the one-time price in
   `scripts/provision_stripe_billing.py` under `STORAGE_ADDON_LOOKUP_KEY`, then
   re-run that script — the price does not exist in Stripe yet.
4. Call `grant_storage_packs_from_checkout` from the
   `checkout.session.completed` branch of the Stripe webhook.
5. Gate uploads: call `resolve_storage_allotment` plus
   `estimate_media_upload_bytes` in `_start_media_batch` and refuse with `402`
   and `storage_exhausted_detail(...)` when the estimate would exceed the
   allotment. Call `invalidate_storage_measurement(assistant_id)` after a
   successful upload or a document deletion.
6. Port `tests/unit_tests/test_storage_allotment.py` from `z-anubis` (125 lines).

## Activating adapter routing

1. Register the router: `app.include_router(adapter_route)` in `webapp.py`.
2. Add the three `GlobalContext` fields and their `.env` / `.env.dev` /
   `.env.example` entries (uppercase, blank in `.env.example`):
   `ADAPTER_INFERENCE_ENABLED`, `ADAPTER_SERVER_BASE_URL`,
   `ADAPTER_SERVER_TIMEOUT_SECONDS`, plus
   `ADAPTER_TRAINING_MIN_QUOTE_DOCUMENTS` and `ADAPTER_AUTO_TRAIN_ENABLED` if
   the automatic post-upload start is wanted. Until then every read here falls
   back through `getattr` and the feature reports "not configured".
3. Fold `resolve_adapter_preference` into `gating.resolve_use_adapter_inference`
   (divergence 2) and pass `avatar_has_trained_adapter(...)` as
   `adapter_available` at the `/message` call sites.
4. In the `think` node, when `config["configurable"]["use_adapter_inference"]`
   is set, build the reply model with `build_adapter_chat_model` and fall back
   to the standard model on any failure (404 `adapter_not_found`, timeout,
   unreachable service — `server_is_healthy` exists so a turn never waits on a
   down service). The billing side is already live: `think` stamps
   `is_adapter_inference` and the `ADAPTER_*` meters exist in `tiers.py`.
5. Call `register_turn_user_api_key(thread_id, api_key)` on the message
   endpoint and `pop_turn_user_api_key(thread_id)` in `think` — the caller's
   credential must not be checkpointed to PostgreSQL.
6. Deploy the adapter service from `z-anubis-adapter` (see above) and set
   `ADAPTER_SERVER_BASE_URL` to it.
7. Port `tests/unit_tests/test_adapter_routing.py` from `z-anubis` (147 lines).

## Why this was left inert

Adapter routing was tabled by the product owner on 2026-09-07 (plan:
`~/.claude/plans/this-plan-is-for-stateful-dove.md`) pending decisions on
hosting cost. Storage allotments were never scheduled onto the f line. Both
were at risk of being lost: the API code existed only on an unmerged branch and
the web-application code only in commits reachable from no branch at all.
