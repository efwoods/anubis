# Anubis — Project Structure

## Root Layout
```
anubis/
├── src/                    # All application source code
│   ├── anubis/             # Core agent graph and utilities
│   ├── api/                # FastAPI web application
│   ├── security/           # Authentication middleware
│   ├── subgraphs/          # Specialized LangGraph subgraphs
│   └── url_loading_graph/  # URL content ingestion graph
├── data/                   # Training/reference data per persona
├── features/               # Feature planning markdown docs
├── grafana/                # Grafana dashboard provisioning
├── markdown/               # Dev notes, bugs, tasks, objectives
├── metrics/                # SQL schemas and Grafana setup
├── milestone/              # Roadmap and YC application materials
├── notebooks/              # Jupyter notebooks for analysis/dev
├── q_and_a_testing/        # Q&A evaluation scripts
├── static/                 # Static assets (images, base64)
├── tests/                  # Integration and unit tests
├── certificate/            # TLS certificates (Cloudflare)
├── langgraph.json          # LangGraph server config (entry point)
├── pyproject.toml          # Project metadata and dependencies
├── docker-compose.yml      # Local dev stack
├── docker-compose-prod.yml # Production stack
├── Dockerfile              # App container
├── nginx.conf              # Reverse proxy config
└── prometheus.yml          # Metrics scrape config
```

## Core Source: `src/anubis/`

### `graph.py` — Main LangGraph Entry Point
Defines two compiled graphs:
- `graph` (exported): `message_workflow` — outer graph wrapping `anubis` subgraph
  - Nodes: `chat` (message_interface) → `anubis`
- `anubis`: inner reasoning graph
  - Nodes: `load_consciousness` → `think` → `process_thoughts` (conditional) → `respond`
- `response_graph`: lightweight graph for API-only response (no tool loop)

### `utils/state.py` — State Definitions
- `GlobalState` (TypedDict): primary graph state with messages, internal_thoughts, system_message, identity docs, memory docs, media processing queues, routing
- `AssistantState` / `UserState`: identity metadata dicts
- `EmotionSummarization` (Pydantic BaseModel): Plutchik wheel emotion scores
- Index state dataclasses: `VectorstoreIndexState`, `AnalysisIndexState`, `AdapterIndexState`, `BaselineIndexState`

### `utils/nodes.py` — `load_consciousness` Node
Loads full assistant context into state before each reasoning cycle:
- Retrieves assistant identity, user identity, memories, direct quotes, knowledge documents from LangGraph store
- Builds dynamic system prompt via `DynamicPromptBuilder`
- Namespaces: `(creator_id, assistant_id, "identity")`, `(user_id, assistant_id, "memory")`, `(creator_id, assistant_id, "quote")`, `(creator_id, assistant_id, "document")`

### `utils/model.py` — Model Initialization
Central `init_model()` factory for LLM instances with optional tools and structured output.

### `utils/context.py` — Runtime Context
`GlobalContext`, `UserContext`, `AssistantContext` — passed via LangGraph `Runtime`.

### `utils/schema.py` — Pydantic Schemas
Shared structured output schemas (e.g., `RouteDecision`).

### `utils/utility.py` — Shared Utilities
`format_docs`, `reduce_docs`, `add_queries`, `extract_user_id_assistant_id`, `configure_assistant_context`.

### `utils/classes/`
- `DynamicPromptBuilder`: builds the system prompt from identity/memory/quote/knowledge documents
- `AsyncSpeakerDiarization`: async speaker diarization for audio processing

### `utils/tools/`
- `identity/identity_tools.py`: LangChain tools — `learn_information_about_the_user`, `update_self_identity_mem_from_user_txt`, `recall_memories`, `create_episodic_memory`
- `slack/slack_tools.py`: Slack messaging tools

### `utils/prompts/`
- `psycho_analysis/`: OCEAN, MBTI, Plutchik, 4-personality-types, Schwartz/attachment/moral foundations prompts
- `legal/`: `TERMS_OF_SERVICE.py`, `PRIVACY_POLICY.py`
- `system_prompts.py`, `email_prompts.py`, `Identify_general_characteristics.py`

### `utils/dataset/`
- `quality.py`: response quality evaluation (BERTScore, ROUGE)
- `formatting.py`: data formatting pipeline
- `file_processing.py`: file ingestion helpers
- `OceanStandardizedQuestionSet.jsonl`: OCEAN personality test questions

### `utils/analysis/`
- `analysis_methods.py`: general analysis utilities

## Subgraphs: `src/subgraphs/`

### `vector_store_graph/`
- `index_graph.py`: indexes documents into LangGraph store
- `retrieval_graph.py`: retrieves documents from vectorstore
- `utils/retrieval.py`, `helper_functions.py`, `utilities.py`

### `process_media_graph/`
- `process_media_graph_api_endpoint.py`: FastAPI endpoint for media upload
- `utils/nodes.py`: media processing nodes (audio→text, image→description, etc.)
- `utils/helper_functions.py`

### `email/`
- `utils/graph.py`: email sending/receiving subgraph

## API & Security

### `src/api/webapp.py`
FastAPI app (`app`) — custom HTTP endpoints mounted alongside LangGraph server.

### `src/security/auth.py`
Auth handler (`auth`) — Auth0 JWT validation integrated with LangGraph auth system.

## Data Directory: `data/`
Per-persona subdirectories containing:
- Biographies, transcripts, tweets, images, audio/video references
- Adapter training JSON files (text message conversations)
- LLM chat exports (ChatGPT, Claude, Grok)
- Identity JSON, URL lists, analysis results

## Architectural Patterns

1. **LangGraph Super-Graph + Subgraphs**: main graph delegates to specialized subgraphs as tools or nodes
2. **Runtime Context injection**: `Runtime[GlobalContext]` passes user/assistant context into every node
3. **Store-backed identity**: all identity, memory, quotes, and knowledge live in LangGraph's persistent store with structured namespaces
4. **Dynamic system prompt**: rebuilt on every `load_consciousness` call from live store data
5. **Agentic tool loop**: `think → process_thoughts → load_consciousness` cycles until no more tool calls
6. **Structured output via Pydantic**: all LLM structured responses use Pydantic BaseModel with `response_format`
