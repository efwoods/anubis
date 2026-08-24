# Feature Audit — Anubis / Neural Nexus (2026-08-18)

## Context

`features/milestones-roadmap.md` is the stated authoritative status tracker, but its last
content edit was **2026-07-03** — six weeks stale. In that window the project shipped
metering, the data-analysis deep agent, multi-device MCP, personal-avatar auto-provisioning,
the stylometric authenticity evaluator, HITL fact editing, browser tools, and dual-credential
auth. None of that is reflected. Meanwhile the roadmap asserts three things that are simply
untrue of the code.

This is a code-verified done/not-done audit of `features/` (84 docs) cross-checked against
`src/` and the 15 sibling repos under `/home/user/gh/anubis-project/`. **Deliverable is the
report itself — no repository files are modified.**

---

## 1. Where the tracker is wrong

Three claims in `features/milestones-roadmap.md` (echoed into `CLAUDE.md`) are contradicted
by the source:

| Claim | Reality |
|---|---|
| `response_only_workflow` "is already compiled" (roadmap:323, 326, 341) | The symbol **does not exist in any `.py` file**. Compiled graphs are `anubis_workflow` (`src/anubis/graph.py:1196`) and `message_workflow` (`graph.py:1226`). Every latency and Twitch plan that depends on it is planning against a phantom. |
| `/handle_email` "already stubbed and wired to the email config schema" (roadmap:332) | **No such route.** `src/subgraphs/email/utils/graph.py` is 35 lines of comments whose line 1 (`from langgraph.graph import StateGraph, State, END`) would raise `ImportError`. Nothing imports `subgraphs.email`. |
| Phase 8: VADER + five eval methods "are wired in" (roadmap:318) | **Zero `@traceable` decorators repo-wide.** No evaluators, no dataset upload, no `compute_test_metrics` (the symbol exists only in prose). VADER was *deliberately rejected* — `style_features.py:17` and `features/prompt_drafts/style/style.md:15`. |

Conversely the roadmap **understates** Phase 11 (metering, marked "planned") — it is the most
complete subsystem in the repo — and never mentions the stylometry evaluator at all, which is
the single most sophisticated thing built.

`_WORK_IN_PROGRESS.md` (2026-08-12) is the trustworthy status document. `CLAUDE.md` inherits
the roadmap's errors, including a `frontend/studio_chat_app.py` path that does not exist in
this checkout.

---

## 2. The schema/implementation divergence

The roadmap tracks Phases 3–5 by naming Pydantic classes. Those exact classes are **dead code
with zero call sites** — but the *functionality* was rebuilt under different names. Reading
the roadmap gives a systematically wrong picture in both directions.

| Roadmap artifact | Status | What actually runs |
|---|---|---|
| `ProprietaryContentClassification` | **Does not exist** | `ReferenceDocumentOrBiographicalConversationalInformation` (`schema.py:186`), live at `helper_functions.py:1038` |
| `TextualSituationalAwareness` | Dead (`schema.py:65`) | `ContentSituationClassification` (`schema.py:362`), live at `helper_functions.py:1102` |
| `MonologuePresentationOrSeriesOfQuotes` | Dead (`schema.py:141`) | Regex/line heuristic `_is_quotes_per_line_text` (`helper_functions.py:200`) |
| `NamedSpeakerMessageFormat` — "routing stubbed as TODO" | Dead (`schema.py:666`) | **Phase 3 is built**, in `process_media_graph/utils/text_dialogue_segmentation.py` (451 lines: windowed segmentation, narrator folding, target inference, relabelling), live at `helper_functions.py:1076,1265` |
| `TargetIdentificationInText`, `RoleConvertedMessageFormat` | Dead (`schema.py:748,828`) | **Phase 4 is built**, in `target_attribution.py:157` (diarizer-vote union-merge) + `_build_adapter_dialogue_document` (`helper_functions.py:609`) |
| `CHARACTERISTIC_EXTRACTORS` / `GeneralCharacteristicExtraction` (17 dims) | Dead (`schema.py:1698,1737`) | **Phase 5 is partly built**, via `ANALYSIS_SCAFFOLD_RUNNERS` (`analysis_methods.py:600`) — 15 analyzers fanned out by `analyze_documents` (`nodes.py:676`) |

Two caveats on Phase 5: `name`, `emotions`, `problems`, `strengths` have no runner at all, and
**10 of the 12 narrative analyzers use a generic stub prompt** (`build_stub_feature_prompt`,
`latent_feature_analysis_prompts.py:266-322`) — only `beliefs` and `relationships` are bespoke.
Myers-Briggs is absent (`meyers_briggs_personality_extraction.py` defines a bare class that is
not even a `BaseModel`, never imported).

---

## 3. Verified status map

### Genuinely built (several under-claimed by the docs)

- **Media ingestion + job orchestration** — master/child jobs, SSE progress, cancel with row
  rollback (`webapp.py:7198`), manifest + YouTube-playlist background expansion, batch
  semaphores, source-level dedup skip-sets. `src/api/media_jobs.py` (524 lines).
- **Preprocessing breadth** — PDF (per-page re-classification), URL/web + structured web
  extraction (498 lines), CSV with `csv.Sniffer` + LLM column identification, JSON/JSONL,
  audio transcription + chunked diarization with reference-speaker labelling, multi-layer
  dedup (`utility.py:159,179,355`).
- **Stripe metering** — 3 tiers, 4 meters, `Decimal`-exact pricing, 402 allotment gate, 429
  rolling-window token limiter, signature-verified webhooks, real billing portal, anonymous
  per-hashed-IP billing. Wired at ~14 real call sites, not just defined.
- **Stylometric authenticity evaluation** — 28-feature versioned Mahalanobis vector vs. a
  bundled ChatGPT baseline, SHAP explanation, per-avatar IsolationForest against the ground-
  truth quote cloud, signature key-phrase discovery injected into the prompt. Live in `think`
  at `graph.py:910`. Not on the roadmap at all.
- **Multi-device MCP** — HTTP-over-WebSocket relay, per-device registration/heartbeat,
  auto-adopt node, real unit tests. Closes the recorded prod incident where the dev daemon
  deleted prod's singleton record.
- **HITL fact editing** — per-document interrupt with 4-way action, resume endpoint, dedicated
  deep-agent checkpointer. `features/prompt_drafts/memory_edits/fp_correction/updates.md`
  documents 8/8 problems resolved.
- **Personal avatar invariant** — idempotent auto-provisioning with re-entrancy guard, other
  avatars demoted, share gate and MCP gate both enforce it.
- **Data-analysis deep agent** (9 tools), **browser tools** (Playwright, leased per turn),
  **auth** (Auth0 + API key + refresh token + anonymous), **observability** (Prometheus,
  Grafana, `api_metrics`).

### Built next door, not here — the roadmap misses this entirely

| Roadmap says | Sibling repo | Reality |
|---|---|---|
| Phase 15 Twitch bot "planned" | `anubis-twitch` (13 commits → 2026-08-05) | Real JS bot: OAuth, token store, rate limiter, channel store, web server |
| Discord "planned" | `anubis-discord` (8 commits → 2026-08-07) | Real bot: commands, SSE streaming from the API, voice `joinCall.js`, encrypted storage, migrations, 5 test files |
| Slack | `anubis-slack` (4 commits) | **Plan only** — 5 files, no code |
| Phase 7 adapter training | `anubis-adapter` (11 commits → 2026-07-16) | Real training: `grpo.py`, `rewards.py`, `dataset.py`, Burrows-Delta/style rewards, inference engine, training + cost routes |
| — | `adapters/one_B_avatar_singleturn/` | A **real trained LoRA** (Llama-3.2-1B, 45 MB safetensors, checkpoint-6). The 3B directory is empty. |
| Frontend | `Neural-Nexus-Frontend` (427 commits → 2026-08-12), `nn-streamlit-ui`, `anubis-customer-portal` | Active React app; no frontend in this repo |
| Phase 12 email | `agent-inbox` (LangChain fork) | Present and analysed; the Anubis side is unbuilt |

### Not built

- **Adapter loop closure** — datasets are written to `q_and_a_adapter`,
  `langsmith_factual_q_and_a`, `multi_turn_dataset_adapter` (`nodes.py:2748-2796`) and
  **nothing ever reads them back**. Zero `lora|peft|trl|SFTTrainer` in `src/`.
- **LangSmith instrumentation** — env-var tracing only.
- **Email ambient agent** — planned in full detail in `_WORK_IN_PROGRESS.md` (3 landings), no code.
- **Social account connection** — agent found **zero OAuth-social code**, despite branch
  `f-connect-social-media-accounts` being merged 2026-08-11. Worth confirming what that merge
  actually contained. `PersonalAvatarCapability("social_accounts")` can only ever report
  `not_configured`.
- **Deep research agent**, **Twilio/SMS**, **voice agent** (`voice_cloning.py` is 49 dead lines
  with 3 latent bugs), **`response_only_workflow`**.

---

## 4. Half-built and risky — things that look done and are not

1. **`webapp.py:2567` hardcodes `statuses["adapter_training"] = "active"`.** No implementation exists.
2. **`use_adapter_inference` (`graph.py:895`) performs no model swap** — it sets
   `is_adapter_inference` metadata whose only real effect is **selecting a billing meter**
   (`webapp.py:641`). Users can be billed on an adapter meter for base-model inference.
3. **`TierCapability.TRAIN_ADAPTER` is defined, mapped, and never enforced**; only `UPLOAD` is
   passed to `enforce_tier_capability`. `report_adapter_training_usage` has no production
   call site — a priced meter that never bills.
4. **`feedback` / `like` / `dislike` form fields on both `/message` routes are inert
   placeholders** (`webapp.py:3781`, `4016`) — while "continuous learning from user feedback"
   is a headline product claim.
5. **Single-process state** — the media job registry (`media_jobs.py:15`) and the MCP relay
   (`relay.py:29`) are per-process. Jobs are invisible across workers; multi-replica needs a
   broker. This caps real user load.
6. **Dead modules that will mislead the next planning pass** — `context_compression.py` (324
   lines), `quality.py` (LLM-judge + ROUGE + BERTScore), `knowledge_evaluator.py`,
   `build_profile.py`, `stylistic_profile.py`, `file_processing.py`, `tools/base.py`,
   `slack_tools.py`, `voice_cloning.py`: all zero importers, several with live bugs.
7. **`import debugpy` unconditionally at module scope** in production (`webapp.py:1219`).
8. **`GET /*` (`webapp.py:1366`) is a no-op** — FastAPI reads it as a literal path.
9. **26 bugs in `features/bugs/`, none marked resolved**, including *"API key is leaked when
   sharing an avatar/conversation"*, wrong-speaker selection under reference audio ("a user
   will not know this is the incorrect speaker ever"), FP facts from diarization, and
   `TARGET NOT VISIBLE` documents being injected into the system prompt.
10. **Credentials committed to the repo** — a live-shaped API key in
    `features/adapters/training_lora_adapters.ipynb`, and three more across
    `features/bugs/long_context.md`, `features/bugs/frontend_bugs.md`,
    `features/expandable_media_jobs_from_playlist.md`.
11. **`make test` does not complete** (hangs in `test_think_interrupt_flow.py`), with ~7
    unrelated baseline failures — so "tests pass" is not currently a usable gate.
12. **A committed git merge conflict** in `features/prompt_drafts/_6social_auth.md`.

---

## 5. Recommended build order

**Hard launch blockers**

1. **Security sweep** — rotate the four committed API keys; fix the share-time key leak
   (`bugs/frontend_bugs.md` #8); remove the `debugpy` import; confirm the browser toolkit's
   env-only gate (`graph.py:803`) is acceptable exposure.
2. **Stop billing for what does not exist** — either implement adapter inference or remove the
   `adapter_training: "active"` status, the adapter meter selection, and the `TRAIN_ADAPTER`
   tier promise. Charging an adapter meter for base-model inference is a real billing risk.
3. **Preprocessing accuracy** — the owner's own #1 baseline item and the root of *"the data is
   inaccurate; responses are incorrect and inauthentic"*. Concretely: VAD in `preprocess_audio`
   (`utility.py:1124`), wrong-speaker selection under reference audio, diarization
   misattribution, and the non-salient retrieval FP at `nodes.py:202`.
4. **Multi-replica readiness** — move the media-job registry and relay device map off
   per-process memory.

**Highest value per unit of effort**

5. **Close the adapter loop.** Both halves already exist — datasets land in the store here, and
   `anubis-adapter` has working GRPO/SFT training plus an inference engine, with a trained 1B
   LoRA already on disk. What is missing is only the read path, training trigger, and
   attach-at-selection. This is the tier-monetization story and it is nearly assembled.
6. **Wire the feedback fields.** They are already in the request schema and already promised
   to users; persisting them unlocks Phase 8 evaluation and "continuous learning".
7. **Add `@traceable`.** Cheap, and it is the prerequisite for every eval method in Phase 8.

**Owner's stated next stage**

8. **Email ambient agent**, per the three landings in `_WORK_IN_PROGRESS.md` — the clearest
   end-to-end use-case demo (email in → Agent Inbox item → approved reply out).
9. **Social account connection** — first confirm what `f-connect-social-media-accounts`
   actually merged, since no OAuth code was found.

**Cheap hygiene that prevents repeated wasted planning**

10. Delete or annotate the ~7 dead schema classes and 9 dead modules, and correct the three
    false roadmap claims. Every future plan written against `response_only_workflow`,
    `CHARACTERISTIC_EXTRACTORS`, or a stubbed `/handle_email` is planning against nothing.

---

## Verification

No code changes are proposed, so there is nothing to test. The findings above were produced by
three parallel read-only explorations and are anchored to `file:line`; the two claims worth
re-confirming before acting are (a) what `f-connect-social-media-accounts` merged on 2026-08-11,
and (b) whether the committed API keys are live and need rotation.
