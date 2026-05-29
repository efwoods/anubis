# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Anubis is an AI personality reconstruction system — a LangGraph agent that reconstructs a person's identity from uploaded media (text, audio, images, video, PDFs, URLs, CSV) and serves as that person's AI avatar. The product name is "Neural Nexus."

## Commands

### Development

```bash
# Start dev server (hot-reloads from local source via volume mount)
docker compose up

# LangGraph Studio dev server (alternative, no Docker)
langgraph dev

# Run Streamlit chat UI
streamlit run frontend/studio_chat_app.py
```

### Build & Deploy

```bash
# Two-stage Docker build (see dockerbuild.sh):
#   1. anubis-base:latest        from Dockerfile.anubis.base — FFmpeg + native audio libs + full dep install
#   2. evdev3/anubis-langgraph-api:latest  from Dockerfile — refreshes source on base, installs only changed deps
./dockerbuild.sh

# Run production stack (Prometheus + Grafana included)
./dcrp.sh                              # docker compose -f docker-compose-prod.yml down && up
```

Dependencies are managed with **uv** (`uv.lock` is the lockfile; Docker installs via `uv pip install --system`). The runtime base image is `langchain/langgraph-api:3.11-wolfi` — Python **3.11**. The audio stack (librosa, soundfile, moviepy, torch/torchaudio) dlopens native system libs that the base layer installs via `apk`: `ffmpeg-7`, `libsndfile`, `libgomp`. Any new native-dependent package must have a wheel compatible with Python 3.11/wolfi.

### Testing & Linting

```bash
make test                              # run all unit tests
make test TEST_FILE=tests/unit_tests/test_graph.py   # run single test file
make integration_tests                 # run integration tests
make lint                              # ruff + mypy --strict (runs mypy twice: uncached, then cached)
make lint_diff                         # same, but only files changed vs main
make format                            # ruff format + isort fix
make format_diff                       # format only files changed vs main
make spell_check                       # codespell
```

## Architecture

### Graph structure (`src/anubis/graph.py`)

The exported `graph` (named `"Anubis"` in `langgraph.json`) is a two-level composition:

**Outer `message_workflow`** (entry point):
```
START → chat → resolve_human_message_images → anubis → END
```
- `chat`: extracts user/assistant IDs from config into `GlobalState`
- `resolve_human_message_images`: converts base64 image blocks to text descriptions

**Inner `anubis` graph** (loops until no tool calls):
```
START → load_consciousness → think ⇄ process_thoughts → END
```
- `load_consciousness` (`src/anubis/utils/nodes.py`): builds the system prompt dynamically from identity documents, recalled memories, and assistant context
- `think`: streams tokens to the client; if a tool call is made, returns `internal_thoughts`; otherwise attaches Go Emotions sentiment and returns final `messages`
- `process_thoughts`: `ToolNode` that executes identity tools, then loops back to `load_consciousness`

### State and Context

- **`GlobalState`** (`src/anubis/utils/state.py`): LangGraph `TypedDict` with `messages`, `internal_thoughts`, `system_message`, identity documents, media upload queues, and model metrics
- **`GlobalContext`** (`src/anubis/utils/context.py`): LangGraph `context_schema` dataclass. `__post_init__` auto-reads env vars by uppercasing field names. **Every new env var must be added to both `.env` / `.env.dev` (uppercase) and as a field here (lowercase)**, then accessed only via `GlobalContext()` — never `os.environ` directly in business logic

### Store namespaces (LangGraph cross-thread store)

Three namespaces are used by identity tools (`src/anubis/utils/tools/identity/identity_tools.py`):
- **`memory`**: episodic events from conversation; retrieved by similarity query per conversation
- **`identity`**: primary-source facts (YouTube transcripts, tweets, uploaded documents); all loaded every turn without limit
- **`quote`**: verbatim historical quotes used for few-shot style grounding; top-K retrieved per turn

The store's vector index is configured in `langgraph.json` (not code): it auto-embeds `document.kwargs.page_content` with `huggingface:microsoft/harrier-oss-v1-270m` at **640 dims**. Anything written to the store must stay consistent with that model/dimension.

### Subgraphs (`src/subgraphs/`)

| Subgraph | Purpose |
|---|---|
| `vector_store_graph/index_graph.py` | Index documents into PostgreSQL pgvector |
| `vector_store_graph/retrieval_graph.py` | Retrieve documents by semantic similarity |
| `process_media_graph/` | Classify and convert uploaded media (audio → transcript/diarize, images → description, PDF/URL/CSV/JSON → Documents) |
| `email/` | Email response generation |

### Audio & diarization pipeline (`src/anubis/utils/utility.py`)

Speech APIs **always use the OpenAI client** (`_openai_client_for_speech`), independent of `MODEL_PROVIDER` — they need `OPENAI_API_KEY` (falls back to `llm_provider_api_key`).
- **Transcription** (`transcribe_audio`): OpenAI `whisper-1` (`audio_transcription_model`). Files larger than `whisper_max_bytes` (25 MiB) are split into segments and transcribed in chunks, then stitched.
- **Diarization** (`transcribe_audio_diarize`): OpenAI `gpt-4o-transcribe-diarize` with `response_format="diarized_json"`. A **reference audio clip** of the target speaker is passed via `known_speaker_names` (`audio_diarization_known_speaker_name`) so the target's turns are labeled. Reference audio is truncated to `reference_audio_diarize_max_seconds` to stay within a **single non-chunked diarizer call**, keeping speaker labels unified across the timeline; `isolate_dominant_speaker_audio_b64` extracts the dominant speaker for grounding.
- **Preprocessing**: `noisereduce` (spectral-gating) + `librosa`/`soundfile`; `demucs`/`deepfilternet` paths exist but are currently disabled. `extract_video_audio_b64` pulls audio out of uploaded video via moviepy.
- All audio env vars live under the "Audio Transcription & Diarization Model" block in `context.py` (pricing, model names, byte/second caps).

### API (`src/api/webapp.py`)

FastAPI app with SSE streaming on `POST /message/{assistant_id}`. Configured as the LangGraph HTTP app in `langgraph.json`. Authentication via Supabase JWT + Auth0 (`src/security/auth.py`); Stripe for subscription checks.

### Model initialization (`src/anubis/utils/model.py`)

`init_model()` dispatches to provider-specific SDK imports lazily based on the `MODEL_PROVIDER` env var — accepted values are `OPEN_AI`, `TOGETHER`, `NVIDIA`, `META` (Llama, via the OpenAI-compatible endpoint). All model providers must be configured via `GlobalContext` fields. Note speech (transcription/diarization) bypasses this and always uses OpenAI directly.

### Observability

Prometheus metrics are exposed on the FastAPI app. Grafana dashboard provisioned via `grafana/provisioning/`. Dev compose does not run Grafana/Prometheus; prod compose (`docker-compose-prod.yml`) does.

## Product roadmap & features

`features/` holds the full product vision (one file per idea; `features/milestones-roadmap.md` is the authoritative status tracker). These are planning docs, **not** spec — verify against code before relying on a status. The product targets a Y Combinator launch and is organized into 13 domains: infrastructure, data ingestion, response-quality analysis, LangGraph optimization, API, use cases (twitch/twitter/email/messaging/discord/slack), monetization, user management, database management, advertising, ML adapter training, project management, funding.

**Avatar quality has five dimensions** — relationships (who you know), knowledge (what you know), behavior (how you act), emotions (what you feel), sentence-structure (how you talk) — improved along four levers: **data quality, system prompt, trace tuning, adapters**. More/richer media (dialogue, two-party texts) + more funding ⇒ higher authenticity (fine-tuned adapters), at higher cost.

### Media → identity pipeline (the `process_media_graph` long-term design)

Numbered phases from `milestones-roadmap.md`, with current status:
- **Phase 1 — Data ingestion** ✅ — `/update_avatar_identity_with_media` accepts files + URLs, captures MIME type, packages `media_items`, feeds `process_media_graph`. Anonymous users via `get_current_user_or_anonymous_user`.
- **Phase 2 — Content classification** ✅ (partial) — `ProprietaryContentClassification` (single target vs generic → routes generic straight to vectorstore); `TextualSituationalAwareness` (`single_speaker` / `q_and_a_dialogue` / `multi_speaker` / `other`); `MonologuePresentationOrSeriesOfQuotes`; `ContentSituationClassification` adds `biographical_facts` / `presentation`.
- **Phase 3 — Named-speaker formatting** 🔲 schema done, routing stubbed — convert dialogue to `{speaker, content}` lists (narrator turns, `*actions*`). `q_and_a_dialogue` / `multi_speaker` branches in `process_text_to_document` are TODOs.
- **Phase 4 — Target identification & role conversion** 🔲 schema done, integration pending — relabel target turns `assistant`, others `user`. Adapter formatters (`llm_single_turn_dataset`, `llm_multiturn_dataset_one_conversation`, `langsmith_dataset`) in `formatting.py` consume this.
- **Phase 5 — Identity analysis (17 dimensions)** 🔲 schema done, integration pending — per-dimension Pydantic models + prompts: name, description, identity, history, emotions, beliefs, values, opinions, goals, wants, needs, fears, problems, flaws, strengths, secrets, relationships. `GeneralCharacteristicExtraction` runs all in one pass; `CHARACTERISTIC_EXTRACTORS` registry iterates them. OCEAN (`perform_ocean_analysis`) wired for quote series; rest pending. (Also planned: Myers-Briggs.)
- **Phase 6 — Synthetic question generation** ✅ — `create_question_list` makes prompts for monologues/presentations/tweet series → QA pairs for training.
- **Phase 7 — Adapter training-data storage** 🔲 partial — formatters ready; training loop, adapter storage, and attach-at-selection are future. See `features/adapters.md` (Llama-3.2-3B QLoRA, Together AI fine-tuning).
- **Phase 8 — Quality analysis & eval (LangSmith)** 🔲 planned — four tracks: **tracing** (`@traceable`), **eval** (prompt-regression, chatbot-simulation, single/multi-turn, extraction-chain, agent-trajectory; VADER post-hoc via `compute_test_metrics`), **feedback/monitoring** (capture `/message` `/chat` signals, hallucination flagging vs retrieved context, annotation queue), **optimization** (prompt bootstrapping, few-shot curation, fine-tuning export, Lilac dataset curation).
- **Phase 9 — Inference / avatar runtime** ✅ — `/message` and `/message/{assistant_id}`, file attachments, anonymous users, `AsyncPostgresSaver` thread persistence (`thread_id`, `most_recent_message`, `conversation_title`), `/conversations` + `/conversations/{thread_id}/messages`. A `response_only_workflow` graph is compiled for latency-sensitive paths.
- **Phase 10 — Latency reduction** 🔲 planned — target <1 s (currently 2–20 s). Use `response_only_workflow`, SSE streaming, async-gather parallel stages, cache identity profiles/embeddings, warm embedding model at lifespan startup, smaller models for cheap classification. Also cache `load_consciousness` until identity updates (`features/response_latency.md`).
- **Phase 11 — Stripe metering** 🔲 planned — usage records per message/upload/analysis, FastAPI rate-limit middleware by tier, webhook sync (`invoice.payment_failed`, `customer.subscription.deleted/updated`).
- **Phase 12 — Email ambient agent** 🔲 planned — triage inbox: ignore / respond / notify, draft in target's voice, human-in-the-loop interrupts. `/handle_email` stubbed (`src/subgraphs/email/`).
- **Phase 13 — Data-analysis agent** 🔲 planned — `create_deep_agent` over uploaded CSV/data, sandboxed exec (Daytona/Modal/Runloop), deliver to Slack/email; health/fitness/financial self-awareness reports.
- **Phase 14 — Deep research agent** 🔲 planned (prereq used now) — scoping + research + supervisor agents; auto-researches public facts about a target (with human source verification) before avatar creation.
- **Phase 15 — Twitch moderation bot** 🔲 planned — real-time chat moderation + engagement in the avatar's voice via `response_only_workflow`.

**Platform / avatar management** ✅ — full CRUD: `create_avatar`, `modify_avatar`, `share_avatar`, `delete_avatar`, `list_public_avatars`, `list_user_avatars`, `select_avatar`, plus `list_avatar_documents` / `delete_avatar_document`; Stripe + verification gating.

**Production-ready now:** custom git commit messages, text responses, factual self-awareness from uploaded media, restaurant menus/ordering, prayers (KJV-grounded Pastor avatar), persistent multi-turn threads.

### Other tracked feature ideas (`features/*.md`)

- **Ingestion convenience**: drag-and-drop ZIP of mixed media, bulk URL lists, linktree crawling, optional per-upload context to guide classification, YouTube playlist iteration (`_morning_ideas.md`, `todo.md`).
- **Social login & subscribe** (`login_and_pull_data_and_subscribe_from_social_media.md`): Auth0 social identity (YouTube/Instagram/Twitch/Twitter) → verify account → initial pull → subscribe to new posts for identity updates. One verified personal avatar per user; share only your own likeness; users own their data.
- **Use-case interfaces**: Slack bot (`src/anubis/utils/tools/slack/slack_tools.py`; do email/query/files via API), Discord voice bot, Twilio text ambient agent (`text_features.md`), realtime voice agent for passive listening (`build_a_realtime_voice_agent.md`).
- **Watch/listen/interact** (`watch_video_or_stream_and_interact_with_stream_of_text_in_public.md`): video→timestamped frame descriptions + transcribed audio → iteratively summarized conversation; podcast/music→text; live-stream→text; live-chat triage (ignore/respond/notify).
- **Emotion & media generation**: emotional-trigger analysis updating current emotions/events (`emotion.md`); generate reference-conditioned images per emotion as emoji-replacements (`create_images...md`); generative video / Genie-3 world models (far future).
- **Token metrics** (`token_usage_analysis.md`): per-request capture of tokens (total/prompt/completion), cost, latency — broken out by model and by inference type (image, structured-output, plain inference) and aggregated; log to `api_metrics` PG table, visualize in Grafana. Note `AsyncLlamaAPIClientWrapper` is free but cost is still computed; don't use it for real scaling.
- **Conversation summarization** when threads grow too long (`conversation_summarization.md`).
- **Far-future**: geo-located shareable avatars on a world map (memorials/markers/characters); neural data I/O (Neuralink thought-to-text); restaurant ordering to subsidize platform cost.

## Key conventions

- **Prompts**: follow the OpenAI GPT-5 prompting guide conventions
- **Env vars**: define in `.env` + `.env.dev` (uppercase), add a `field(default=None, metadata={"description": "..."})` in `GlobalContext` (lowercase), access via `context = GlobalContext()` at the top of functions and pass through nested calls
- **Imports for cold-start perf**: heavy SDK imports (`langchain_community`, provider SDKs) must be lazy (inside functions or conditional branches), not at module scope — several imports were already moved to avoid 7–13 s cold-start penalties
- **`internal_thoughts`**: parallel to `messages`; used for the agent's tool-call loop so tool turns don't emit `assistant_token` stream events meant for the user

## gstack (REQUIRED — global install)

**Before doing ANY work, verify gstack is installed:**

```bash
test -d ~/.claude/skills/gstack/bin && echo "GSTACK_OK" || echo "GSTACK_MISSING"
```

If GSTACK_MISSING: STOP. Do not proceed. Tell the user:

> gstack is required for all AI-assisted work in this repo.
> Install it:
> ```bash
> git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
> cd ~/.claude/skills/gstack && ./setup --team
> ```
> Then restart your AI coding tool.

Do not skip skills, ignore gstack errors, or work around missing gstack.

Using gstack skills: After install, skills like /qa, /ship, /review, /investigate,
and /browse are available. Use /browse for all web browsing.
Use ~/.claude/skills/gstack/... for gstack file paths (the global path).


when adding environment variables, include the environment variable in the .env.example without any set values.