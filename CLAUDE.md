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
make lint                              # ruff + mypy --strict
make format                            # ruff format + isort fix
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