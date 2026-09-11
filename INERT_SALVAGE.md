# Inert salvaged code in this repository

**Read this before deleting anything that looks like dead code, and before
building storage allotments or adapter routing from scratch.**

Some complete, working features live in this repository in an **inert** state:
the code is present, importable and reviewed, but nothing imports it, no route
is registered, and no behaviour changes. It is here because it was built on a
branch that never merged, and leaving it only on that branch was losing it.

Inert code is **not** dead code. Do not delete it, and do not reimplement a
feature listed here — activate the existing code instead.

## How to activate

Each entry below names an activation document with an exact checklist. On an
explicit instruction to turn one of these on ("make the storage allotments
live", "wire up adapter routing"), follow that checklist rather than writing
new code. Activation is deliberately a checklist and not a single import,
because each feature has plumbing outside this repository — a Stripe price, a
deployed service — that must exist first.

Do not activate any of this incidentally, as a side effect of another task, or
because it looks unfinished. It is finished; it is switched off on purpose.

## What is currently inert

### 1. Per-avatar storage allotments and add-on packs

Byte allotments per avatar per tier, measured against the LangGraph store, with
one-time Stripe add-on packs and a `402 storage_exhausted` refusal on upload.

| File | Role |
|---|---|
| `src/anubis/utils/billing/storage.py` | Measurement, allotment resolution, refusal body |
| `src/api/salvage/storage_routes.py` | `GET`/`POST /avatar/{assistant_id}/storage[/purchase]`, webhook grant |

- **Activation document:** `src/api/salvage/README.md`
- **Origin:** branch `z-anubis`, commits `c8a958e`, `6ed92cb`
- **Web application:** the matching UI is inert in the frontend repository —
  see `INERT_SALVAGE.md` there.
- **Blocked on:** the one-time Stripe price does not exist yet; it must be
  provisioned before the purchase route can work.

### 2. Adapter routing (LoRA adapter inference and training)

Premium avatars answering through a LoRA adapter trained on their own words,
with automatic fallback to the standard model.

| File | Role |
|---|---|
| `src/anubis/utils/adapters/client.py` | `AdapterServerClient`, health probe, `build_adapter_chat_model` |
| `src/api/salvage/adapter_routes.py` | Train / status / progress / cancel routes, training watcher |

- **Activation document:** `src/api/salvage/README.md`
- **Origin:** branch `z-anubis`, commit `d1b8473`
- **Web application:** the adapter card is inert in the frontend repository.
- **Blocked on:** the adapter service's OpenAI-compatible endpoint exists only
  on branch `z-anubis-adapter` of the `anubis-adapter` repository (commit
  `0381599`) and is not deployed. Adapter routing was also tabled as a product
  decision on 2026-09-07 pending hosting cost, so do not activate it without an
  explicit instruction.
- **Already live, do not rebuild:** the billing half. `UsageMeter.ADAPTER_TRAINING_UNITS`,
  `UsageMeter.ADAPTER_INFERENCE_TOKENS` and `TierCapability.TRAIN_ADAPTER` exist
  in `src/anubis/utils/billing/tiers.py`, `resolve_use_adapter_inference` gates
  the `adapter` flag on `/message`, and `think` stamps `is_adapter_inference`.
  What is missing is only the call to the adapter service.

## How to tell inert code apart

Every inert file opens with a docstring beginning `INERT SALVAGE`. Route
modules live under `src/api/salvage/` and expose an `APIRouter` that
`src/api/webapp.py` never passes to `include_router`, so their paths are not
served. Confirm inertness at any time with:

```bash
grep -rn "api.salvage\|utils.adapters\|billing.storage" src/ --include=*.py \
  | grep -v "^src/api/salvage/" | grep -v "^src/anubis/utils/adapters/"
```

Empty output means nothing live references any of it.

## A trap to know about

The first line of `.gitignore` is `adapters`, which matches **any** path
segment of that name and would silently ignore `src/anubis/utils/adapters/`.
A negation (`!src/anubis/utils/adapters/`) keeps that source tracked. If you
add files under a directory named `adapters` anywhere else, check
`git check-ignore -v <path>` before assuming they are committed.
