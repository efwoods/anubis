# 01 — Current-State Audit

> Goal: understand exactly what is in the running container today so we can make defensible
> decisions about what fits in 1 GB, what must move out of process, and what costs we are
> actually paying for at scale.

Inputs reviewed:

- `Dockerfile`, `Dockerfile.anubis.base`, `langgraph.json`
- `docker-compose.yml`, `docker-compose-prod.yml`, `docker-compose-postgres.yml`
- `src/anubis/graph.py`, `src/anubis/utils/model.py`, `src/anubis/utils/context.py`
- `src/api/webapp.py` (route surface)
- `requirements.txt`, `pyproject.toml`
- `data/` directory on disk (training corpora used by `build_profile.py` etc.)

---

## 1.1 What is actually running in the avatar container today

The production image (`evdev3/anubis-langgraph-api:latest`) is built on top of
`langchain/langgraph-api:3.11-wolfi` and embeds:

| Layer | Component | Purpose | Approx RAM (resident) |
| --- | --- | --- | --- |
| Process supervisor | `langgraph-api` server | HTTP API + queue worker (single-host mode) | 80–120 MB |
| Web framework | FastAPI + Starlette + Uvicorn | `src/api/webapp.py` — 21 routes (avatar CRUD, message, conversations, media upload) | 40–60 MB |
| Graph runtime | LangGraph 1.x + LangChain 1.x | Compiled `Anubis` graph (`src/anubis/graph.py`) plus `process_media`, `vector_store`, `email`, `data-analysis` subgraphs | 60–100 MB |
| Embedding model (inline) | `microsoft/harrier-oss-v1-270m` via `sentence-transformers` (loaded by `LANGGRAPH_STORE` index) | 270 M parameters, dim=640, 32 K context, multilingual; FP32 weights | **~1080 MB FP32 / ~540 MB FP16** |
| Sentiment model (inline, on every `respond`) | `j-hartmann/emotion-english-distilroberta-base` via `transformers.pipeline` | DistilRoBERTa, 82 M params; lazily loaded but currently re-built per call in `respond()` | **~328 MB FP32 / ~164 MB FP16**, plus tokenizer ~50 MB |
| Tokenizer + tiktoken caches | `count_tokens`, BPE caches | Used per request | 30–50 MB |
| HF hub cache (mounted) | `~/.cache/huggingface` bind | Disk only — but model weights are mmap'd into RAM on first use | 0 MB explicit, but contributes to model RAM above |
| LangSmith tracer | `langsmith>=0.7.9` | Background trace shipping | 20–40 MB |
| Misc deps | `bert_score`, `rouge`, `moviepy`, `yt_dlp`, `streamlit`, `spacy`, `nltk`, `pandas`, `scipy`, `sentence-transformers`, `firebase-admin`, `supabase`, `stripe`, `slack-sdk`, `prometheus-client`, etc. | Imported at module load by various nodes | 150–250 MB resident pages |

**Floor estimate:** roughly **1.25–1.45 GB resident** in steady state. The user's
observed **1.3 GB** is consistent with this and confirms the embedding + sentiment
models are the dominant loaders.

The container also depends on two external services in compose:

- **Redis 7-alpine** (`redis-data` volume; pub/sub + checkpoint streaming)
- **Postgres 16 + pgvector** (`postgres_data` volume; LangGraph `BaseStore`,
  `langgraph-checkpoint-postgres`, plus `vectorstore_postgres_uri`).

Neither runs *inside* the avatar process. They are sibling containers.

## 1.2 What the graph actually does per message

Per call to `POST /message` (or `/message/{assistant_id}`), the request flows:

```
chat (extract_user_id_assistant_id)
  -> resolve_human_message_images (image -> text via image_model)
  -> anubis subgraph
       -> load_consciousness  (pull system prompt + identity context from store)
       -> think               (model + identity_tools, streamed)
       -> [conditional]
            process_thoughts  (ToolNode running identity tools, may write to store)
            -> back to load_consciousness  (loop)
          OR
            respond            (final model call, streams tokens to writer,
                               then loads emotion-english-distilroberta-base
                               and classifies the response)
```

Key implications for memory and cost:

1. The agent runs **two model passes per turn** (`think` then `respond`), plus a
   **third structured-output pass** for terms-of-service moderation when enabled.
   Token usage is therefore ~2–3× the naive raw-message count.
2. `respond()` re-instantiates the HF emotion pipeline every call
   (`pipeline("text-classification", ...)`). On a 1 GB box this is the single
   biggest avoidable RAM/CPU spike. **Move it to module-level singleton or a
   remote inference call.**
3. `LANGGRAPH_STORE` is configured to embed `document.kwargs.page_content` with
   harrier-oss-v1-270m at write time *and* search time. Every store write or
   semantic search re-embeds in-process.
4. `count_tokens` (tiktoken) is called from many code paths (`utility.py`,
   `nodes.py`, classes, model.py); it loads BPE files lazily per-encoding.

## 1.3 Persistent data on disk today

The repo's `data/` directory is **1.5 GB** and includes per-avatar training
corpora (Andrew Huberman, Lex Fridman, Elon Musk, Brian Johnson, Bradford
Smith, Julia Ann, Maria Brink, Kate Darling, Gracie Abrams, Curie, Bible,
neuralink urls, "mom", custom datasets, etc.). A `data.zip` weighing **520 MB**
is committed alongside.

Per-avatar persistent state (after upload + processing) lives in:

- **Postgres** (`pgvector/pgvector:pg16`)
  - `langgraph-checkpoint-postgres` tables (thread/checkpoint state)
  - LangGraph `BaseStore` (`async_postgres_store_uri`) — assistant identity,
    facts, episodic memories
  - Vector store (`vectorstore_postgres_uri`) — quote/identity Documents with
    640-dim embeddings
- **Redis** — only ephemeral pub/sub + queue signaling
- **Object store / disk** — uploaded media, generated reference audio,
  transcripts (currently bound through `volumes: - .:/deps/anubis`)

For one fully-built avatar with ~1 hour of source video (~500 MB raw),
~30 K embeddings, and identity/episodic memory:

| Datum | Estimate |
| --- | --- |
| Raw uploaded media (post-process kept) | 200–800 MB |
| Transcripts + diarization JSON | 5–30 MB |
| Vector embeddings @ 640 dims × float4 = 2.56 KB/row | 30 K rows ≈ 75 MB; with HNSW index ~1.5–2× = **120–150 MB** |
| Identity/episodic memory rows | 5–20 MB |
| Postgres WAL + bloat overhead | +30 % |
| **Per-avatar Postgres footprint** | **~250–400 MB** |
| **Per-avatar object-store footprint** | **300 MB – 1 GB** depending on whether raw media is retained |

## 1.4 Public API surface (must keep working in any new topology)

From `src/api/webapp.py`:

```
GET    /metrics
GET    /                           # health/landing
GET    /subscribe                  # Stripe
GET    /manage_subscription
GET    /cancel_subscription
GET    /verify_subscription_status

POST   /create_avatar
POST   /share_avatar
PATCH  /modify_avatar
DELETE /delete_avatar
GET    /list_public_avatars
GET    /list_user_avatars
POST   /select_avatar

POST   /message
POST   /message/{assistant_id}
GET    /conversations
GET    /conversations/{thread_id}/messages

GET    /avatar_reference_image
POST   /update_avatar_identity_with_media
GET    /list_avatar_documents
DELETE /delete_avatar_document
```

Any "single-avatar VM" must still serve `/message/{assistant_id}` and the
identity/document routes, plus Stripe subscription routes if billing is
co-located.

## 1.5 Conclusion of audit

The 1.3 GB number is dominated by **one 270 M-param embedding model** plus a
**second 82 M-param sentiment model** loaded **inside the Python process**.
The graph runtime, FastAPI, Redis client, and Postgres client are
collectively well under 300 MB.

This is the lever to pull for AWS Free Tier — see
[`02_phase1_aws_free_tier_single_avatar.md`](./02_phase1_aws_free_tier_single_avatar.md).
