# Anubis — Technology Stack

## Language & Runtime
- Python >= 3.10
- Package manager: `uv` (lockfile: `uv.lock`) + `pip` (requirements.txt)
- Build system: `setuptools >= 73.0.0` with `wheel`

## Core Frameworks

### LangGraph / LangChain
- `langgraph >= 1.0.10` — graph orchestration, state management, store
- `langgraph-checkpoint-postgres >= 3.0.4` — persistent checkpointing via PostgreSQL
- `langgraph-cli[inmem] >= 0.4.19` — local dev server
- `langchain >= 1.2.10` — agent creation, prompt templates
- `langchain-core >= 1.2.16` — messages, runnables, documents
- `langchain-openai >= 1.1.10` — OpenAI LLM integration
- `langchain-together >= 0.4.0` — Together AI models
- `langchain-nvidia-ai-endpoints >= 1.2.1` — NVIDIA NIM endpoints
- `langchain-huggingface >= 1.2.0` — HuggingFace embeddings
- `langchain-community >= 0.4.1` — community integrations
- `langchain-unstructured >= 1.0.1` — document loading
- `langsmith >= 0.7.9` — tracing and evaluation

### Web Framework
- `fastapi >= 0.135.1` — REST API
- `python-multipart >= 0.0.22` — file upload support
- `httpx >= 0.28.1` — async HTTP client

### Authentication
- `auth0-fastapi-api >= 1.0.0b6` — Auth0 JWT validation
- `firebase-admin >= 7.3.0` — Firebase auth
- `supabase >= 2.28.3` — Supabase client (auth + DB)
- `python-jose[cryptography] >= 3.5.0` — JWT handling

### AI / ML
- `openai >= 2.24.0` — OpenAI API client
- `llama-api-client >= 0.6.0` — Llama API
- `huggingface-hub >= 0.36.2` — model hub access
- `sentence-transformers >= 5.2.3` — local embeddings (all-MiniLM-L6-v2, dims=384)
- `bert-score >= 0.3.13` — response quality evaluation
- `rouge >= 1.0.1` — text similarity scoring

### Data & Storage
- `pydantic >= 2.12.5` — data validation and structured output
- `pandas >= 2.3.3` — data analysis
- `psycopg-pool >= 3.3.0` — PostgreSQL async connection pool

### Media & Content
- `yt-dlp >= 2026.3.17` — YouTube video/audio download
- `beautifulsoup4 >= 4.14.3` — HTML parsing

### Integrations
- `slack-sdk >= 3.41.0` — Slack messaging
- `stripe >= 15.0.0` — subscription billing
- `prometheus-client >= 0.25.0` — metrics export

### Dev Tools
- `ruff >= 0.6.1` — linting and formatting (pycodestyle, pyflakes, isort, pydocstyle)
- `mypy >= 1.11.1` — strict type checking
- `pytest >= 9.0.2` — testing
- `pytest-asyncio >= 1.3.0` — async test support
- `debugpy >= 1.8.20` — remote debugging (port 5678)

## LangGraph Server Configuration (`langgraph.json`)
```json
{
  "graphs": { "Anubis": "./src/anubis/graph.py:graph" },
  "store": {
    "index": { "dims": 384, "embed": "huggingface:sentence-transformers/all-MiniLM-L6-v2", "fields": ["document.kwargs.page_content"] }
  },
  "http": { "app": "./src/api/webapp.py:app" },
  "auth": { "path": "./src/security/auth.py:auth" },
  "env": ".env",
  "image_distro": "wolfi"
}
```

## Infrastructure

### Docker
- `Dockerfile` + `Dockerfile.anubis.base` — app image (`evdev3/anubis-langgraph-api:latest`)
- `docker-compose.yml` — dev stack: LangGraph API (8123), Prometheus (9090), Grafana (3000)
- `docker-compose-prod.yml` — production stack
- `docker-compose-reverse-proxy.yml` — Nginx reverse proxy
- HuggingFace cache mounted: `~/.cache/huggingface:/root/.cache/huggingface`

### Observability
- Prometheus scraping LangGraph API metrics
- Grafana dashboards: `grafana/provisioning/dashboards/anubis-api-metrics.json`
- PostgreSQL datasource for Grafana
- Custom metrics schema: `metrics/sql/metrics_schema.sql`

### Networking
- Nginx reverse proxy (`nginx.conf`)
- Cloudflare TLS (`certificate/`)
- Custom DNS: `168.83.129.16`, `8.8.8.8`

## Development Commands

```bash
# Run LangGraph dev server
langgraph dev

# Run tests
make test                          # unit tests
make integration_tests             # integration tests
make test TEST_FILE=path/to/test   # specific test file

# Lint & format
make lint                          # ruff check + mypy strict
make format                        # ruff format + isort fix

# Spell check
make spell_check
make spell_fix

# Docker dev stack
docker compose up                  # starts API + Prometheus + Grafana

# Build Docker image
./dockerbuild.sh
```

## Code Style (ruff)
- Convention: Google docstring style
- Selected rules: E (pycodestyle), F (pyflakes), I (isort), D (pydocstyle), D401, T201, UP
- Ignored: UP006, UP007, UP035, D417, E501
- Tests directory: D and UP rules ignored
