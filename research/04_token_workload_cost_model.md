# 04 — Token Workload + Cost Model

> Goal: convert the user-stated workload (72 tokens/message, 10 messages/
> conversation/day, 1 conversation/day for 30 days) into defensible monthly
> spend numbers per avatar and per fleet, across **OpenAI / Together /
> NVIDIA / Llama** and across **scale tiers (1, 100, 1 000, 10 000 avatars)**.

All prices are May 2026 list pricing.

---

## 4.1 Reference prices (USD, per 1 M tokens unless noted)

| Model | Input \$/M | Output \$/M | Batch (-50 %) | Notes |
| --- | --- | --- | --- | --- |
| **OpenAI gpt-4o-mini** | **\$0.15** | **\$0.60** | \$0.075 / \$0.30 | Default thinking + responding |
| OpenAI gpt-4o | \$2.50 | \$10.00 | \$1.25 / \$5.00 | Higher-quality avatar |
| OpenAI text-embedding-3-small | \$0.02 | — | \$0.01 | 1 536 dim |
| OpenAI text-embedding-3-large | \$0.13 | — | \$0.065 | 3 072 dim |
| OpenAI gpt-4o-mini-transcribe | \$0.003 / minute | — | — | STT |
| OpenAI gpt-4o-transcribe | \$0.006 / minute | — | — | STT |
| Together Llama 3.3 70B Instruct Turbo | \$0.88 | \$0.88 | \$0.44 | |
| Together Llama 3 8B Instruct Lite | \$0.10 | \$0.10 | \$0.05 | |
| LangSmith trace overage | — | — | — | \$0.50 / 1 000 traces over 10 000 / mo included |
| LangSmith Fleet run overage | — | — | — | \$0.05 / run over 500 / mo included |

**Note about Output column for gpt-4o-mini:** OpenAI lists \$0.15 in / \$0.60
out. Some search excerpts swap these; the canonical listing is on
[openai.com/api/pricing](https://openai.com/api/pricing/) and is what we use
below.

---

## 4.2 Two interpretations of "72 tokens/message"

The user's workload is:

> "average of 72 tokens per message, an average of 10 messages per
> conversation per day, and one conversation per day for thirty days"

That is **72 × 10 × 30 = 21 600 tokens per avatar per month** of *raw user
text*. There are two reasonable ways to map raw user tokens into actual LLM
billable tokens:

### Interpretation A — "naive" (raw text only, no agent overhead)

The 21 600 tokens is roughly half the user's words and half the model's
reply. Cost at gpt-4o-mini:

```
input  = 10 800 tokens × $0.15 / 1 000 000 = $0.00162
output = 10 800 tokens × $0.60 / 1 000 000 = $0.00648
total  ≈ $0.0081 / avatar / month
```

This is the floor. **It is not realistic** because every message goes
through the Anubis pipeline (system prompt, identity context, retrieval,
agent thinking step). Use only as a sanity floor.

### Interpretation B — "agent-amplified" (recommended for budgeting)

Each `/message` invocation in `src/anubis/graph.py` runs:

1. `chat` (no LLM)
2. `resolve_human_message_images` (zero or one cheap image-to-text call)
3. `load_consciousness` — reads identity, builds system prompt
4. **`think` model pass** — full system prompt + identity + history + tools
5. **`process_thoughts`** — zero-or-more tool invocations (each writes/reads
   from store, often runs an embedding call)
6. **`respond` model pass** — full system prompt + identity + history,
   streamed
7. (optional) **`terms_and_services_content_moderation`** structured-output
   pass

Per message, we therefore burn approximately:

```
system prompt + identity + retrieved facts: ~1 800 input tokens (constant)
running history of conversation:            mean 5 prior messages × 72 = 360 input tokens
new user message:                            72 input tokens
think output:                                ~150 output tokens
respond output:                              ~200 output tokens
moderation pass (rare):                      ~600 input + 50 output, count occasionally
embedding for retrieval write:               ~300 tokens billable (if using OpenAI embeds)
```

So **per message** at gpt-4o-mini-equivalent:

| Item | Tokens | \$/message |
| --- | --- | --- |
| `think` input | 2 232 | 2232 × 0.15 / 1e6 = \$0.000335 |
| `think` output | 150 | 150 × 0.60 / 1e6 = \$0.000090 |
| `respond` input | 2 232 | \$0.000335 |
| `respond` output | 200 | \$0.000120 |
| Embeddings (avg) | 300 | 300 × 0.02 / 1e6 = \$0.000006 |
| **Per-message subtotal** | | **\$0.000886** |

Per **conversation** (10 messages): \$0.00886
Per **avatar / month** (30 conversations): **\$0.27**
Per **1 000 avatar-months**: **\$270**
Per **10 000 avatar-months**: **\$2 700**

---

## 4.3 Monthly LLM bill at four scale tiers

| Tier | Avatars active | Model choice | LLM \$/mo | Embedding \$/mo | LangSmith trace overage \$/mo | **Total LLM stack \$/mo** |
| --- | --- | --- | --- | --- | --- | --- |
| Demo / Kickstarter (Phase 1) | **1** | gpt-4o-mini | **\$0.27** | \$0.01 | \$0 (under 10 K) | **~\$0.28** |
| Early Backers | **100** | gpt-4o-mini | **\$27** | \$0.60 | 100 avatars × 30 conv × 2 = 6 000 traces; under 10 K free | **\$28** |
| Scale | **1 000** | gpt-4o-mini | **\$270** | \$6 | 60 000 traces – 10 000 free = 50 K × \$0.50/K = \$25 | **\$301** |
| Mid-tier | **10 000** | gpt-4o-mini | **\$2 700** | \$60 | 600 K traces – 10 K = 590 K × \$0.50/K = \$295 | **\$3 055** |

**If we move the `respond` step to gpt-4o** (better persona quality, ~17×
cost), the per-message LLM cost rises from \$0.000886 to ~\$0.0046; per
avatar / month from \$0.27 → ~\$1.40, and at 1 000 avatars from \$270 →
\$1 400.

---

## 4.4 STT / media-processing costs

Each "build my avatar" upload (`POST /update_avatar_identity_with_media`)
runs `process_media_graph`, which transcribes audio.

| Source | OpenAI gpt-4o-mini-transcribe | Per 1-hour video |
| --- | --- | --- |
| Long-form (e.g. podcast episode) | \$0.003 / minute | **\$0.18** |
| 10-hour Lex Fridman dump | | \$1.80 |

This is a one-time cost per avatar onboarding, not per conversation. Budget
**\$2 / avatar one-time** for typical Kickstarter-tier source material.

---

## 4.5 Putting it all together — Total Cost of an Avatar

| Cost line | One-time | Recurring |
| --- | --- | --- |
| Avatar onboarding STT | ~\$2 | — |
| Avatar onboarding embeddings (~30 K rows × 256 tokens = 7.7 M tokens) | ~\$0.15 | — |
| Avatar inference (Interpretation B, gpt-4o-mini) | — | \$0.27 / month |
| Storage on Postgres + S3/Blob | — | <\$0.05 / month |
| Pro-rated infra (Phase 2 fleet, 100 avatars on \$220/mo cluster) | — | \$2.20 / avatar / month |
| Pro-rated LangSmith Plus seat (\$39 / 1 000 backers) | — | \$0.04 / avatar / month |
| **Per-avatar total recurring (Phase 2 at 100 avatars)** | **\$2.15** | **\$2.51 / month** |
| **Per-avatar total recurring (Phase 2 at 1 000 avatars)** | **\$2.15** | **\$0.82 / month** |

The pro-rated infra dominates at low scale — that's the whole reason Phase 1
exists (Free Tier ≈ \$0 infra). The LLM bill dominates as we scale past
~1 000 avatars on shared cluster — that's when we either negotiate volume
pricing with OpenAI or switch the `respond` path to a Together-hosted Llama
3.3 70B (~\$0.88/M both ways → ~\$0.0048 per message → ~\$1.45/avatar/mo).

---

## 4.6 Sensitivity to higher real-world usage

The user's stated workload is light (1 conversation / day). For sensitivity,
here is the same per-avatar Interpretation-B table at higher loads:

| Conversations / day | Messages / day | Per-avatar / month (gpt-4o-mini) |
| --- | --- | --- |
| 1 (user's stated baseline) | 10 | **\$0.27** |
| 3 | 30 | \$0.80 |
| 10 (heavy fan / kiosk) | 100 | \$2.66 |
| 100 (interactive demo / conference booth) | 1 000 | \$26.60 |

This is why the Stripe subscription model (`/subscribe`) and rate-limiting
matter. Even a single runaway "demo at a conference" can move an avatar
from \$0.27/mo to \$26+/mo.

---

## 4.7 What to instrument before Phase 2 ships

We already capture `response_metadata.token_usage` in
`AsyncLlamaAPIClientWrapper` and the various `init_model` paths. To make
budgeting real:

1. Persist token counts per `(assistant_id, thread_id, run_id)` in Postgres
   (one row per LLM call). This is the source of truth for "did this
   backer's avatar exceed its monthly token quota?"
2. Wire those rows to `/metrics` Prometheus endpoint so Grafana shows
   live token burn.
3. Send a daily summary to LangSmith via `langsmith.Client.create_run`
   metadata.
4. Add a budget guard in `init_model` that short-circuits with an
   apology message if the avatar's monthly token limit is reached.

These four hooks turn the "budget" from a guess into a measured number
per backer.
