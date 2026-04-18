# Anubis — Development Guidelines

## Code Quality Standards

### Linting & Formatting
- Formatter: `ruff format` (enforced via `make format`)
- Linter: `ruff check` with rules E, F, I, D, D401, T201, UP
- Type checker: `mypy --strict` (enforced via `make lint`)
- Docstring convention: **Google style** (`[tool.ruff.lint.pydocstyle] convention = "google"`)
- Spell checker: `codespell` (config in `pyproject.toml`)
- Tests directory is exempt from D (docstring) and UP (pyupgrade) rules

### Naming Conventions
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants and prompt strings: `UPPER_SNAKE_CASE` (e.g., `OCEAN_ANALYSIS_PROMPT`, `TERMS_OF_SERVICE`)
- LangGraph nodes: descriptive verb phrases (`load_consciousness`, `think`, `process_thoughts`, `respond`)
- State fields: verbose, self-documenting names (e.g., `documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant`)
- Pydantic models for structured LLM output: `PascalCase` with `Extraction` suffix (e.g., `NameExtraction`, `BeliefsExtraction`)

### File Headers
Every source file begins with a comment stating its module path:
```python
# src/anubis/utils/schema.py
```

### Imports
- `from __future__ import annotations` used in state/schema files
- Standard library → third-party → local imports (enforced by isort via ruff)
- Lazy imports inside functions to avoid circular imports (e.g., `from src.subgraphs... import workflow` inside endpoint handlers)

---

## Structural Conventions

### Pydantic Models (Structured LLM Output)
Every LLM structured output follows this pattern — model + system prompt as a paired constant:

```python
class NameExtraction(BaseModel):
    """Docstring describing what this extracts."""
    value: str = Field(description="...")
    evidence: str = Field(description="...")
    confidence: ConfidenceLevel = Field(description="...")
    reasoning: str = Field(description="...")

NAME_EXTRACTION_SYSTEM_PROMPT = """
<Role>...</Role>
<Instructions>...</Instructions>
<Rules>...</Rules>
"""
```

Key patterns:
- Every extraction model has a `reasoning` field for chain-of-thought
- `confidence: Literal["low", "medium", "high"]` used for epistemic uncertainty
- `Optional[...]` fields for data that may not be present in source text
- `Field(description=...)` on every field — descriptions are the LLM's instructions
- Master registry pattern: `CHARACTERISTIC_EXTRACTORS: dict[str, tuple[type[BaseModel], str]]` maps dimension name → (model, prompt)

### LangGraph State
- State classes use `@dataclass(kw_only=True)` for index states
- `GlobalState` uses `TypedDict` for the main graph state
- Annotated reducers on all list/sequence fields:
  ```python
  messages: Annotated[list[AnyMessage], add_messages]
  retrieved_docs: Annotated[list[Document], operator.add]
  user_identity_documents: Annotated[Sequence[Document], reduce_docs]
  ```
- State fields grouped by semantic category with `""" Section Header """` docstrings

### LangGraph Nodes
All nodes are `async def` and follow this signature:
```python
async def node_name(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]) -> dict:
```
- Return only the state keys being updated (partial state update)
- `logger.info("breakpoint")` used as a consistent debug marker throughout nodes
- Context accessed via `runtime.context`, store via `runtime.store`

### LangGraph Graph Construction
Graphs are built in three sections with clear comment separators:
```python
""" NODES """
workflow.add_node(...)

""" EDGES """
workflow.add_edge(...)
workflow.add_conditional_edges(...)

graph = workflow.compile()
graph.name = "Anubis"
```

---

## Semantic Patterns

### Prompt Engineering Pattern
All system prompts use XML-tagged sections for structure:
```
<Role>...</Role>
<Context>...</Context>
<Instructions>...</Instructions>
<Rules>...</Rules>
<Example>...</Example>
<Output_Format>...</Output_Format>
```
- Instructions are often repeated at the top and bottom of the prompt for emphasis
- `<Rules>` section contains hard constraints and edge cases
- `<Example>` sections provide few-shot examples inline

### Store Namespace Convention
LangGraph store namespaces follow a strict tuple pattern:
```python
(creator_id, assistant_id, "identity")    # assistant's identity facts
(user_id, assistant_id, "memory")         # episodic memories per user
(creator_id, assistant_id, "quote")       # direct quotes from the real person
(creator_id, assistant_id, "document")    # reference documents (bible, menu, etc.)
(creator_id, assistant_id, "reference_image")  # reference image description
(creator_id, assistant_id, "identity_memory")  # identity-level memories
(user_id, assistant_id, "preferences")   # user feedback preferences
```

### Model Initialization
Always use `init_model()` factory — never instantiate LLM clients directly:
```python
# With tools
model = init_model(context=runtime.context, tools=identity_tools)

# With structured output
model = init_model(model_without_tools=False, response_format=MyPydanticModel)

# Plain model
model = init_model(model_without_tools=False)
```

### Authentication Pattern
All protected endpoints use `Depends(get_current_user)`:
```python
@app.post("/endpoint")
async def endpoint(current_user: dict = Depends(get_current_user)):
    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"Authorization": f"Bearer {token}"})
```
- API key is hashed with SHA-256 before storage: `_hash_key(api_key)`
- TTLCache (maxsize=1000, ttl=300) used for API key → user lookups
- Anonymous users supported via `get_current_user_or_anonymous_user` dependency
- Auth0 Management API token cached with expiry: `_mgmt_token_cache`

### HTTP Retry Pattern
Retryable HTTP calls use `retry_async_httpx_request()`:
```python
response = await retry_async_httpx_request(
    method="PATCH",
    url=f"{BASE_AUTH_URL}/api/v2/users/{encoded_user_id}",
    headers=headers,
    json=payload,
    max_retries=5,
    base_delay=1.0,  # exponential backoff: delay * 2^attempt
)
```
Retryable status codes: 429, 500, 502, 503, 504.

### Metrics Pattern
Every API endpoint records Prometheus metrics + PostgreSQL metrics:
```python
# Prometheus counters/histograms (module-level)
REQUEST_COUNT = Counter("anubis_requests_total", ..., ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("anubis_request_duration_seconds", ..., ["method", "endpoint"])

# Per-request timing
start_time = time_ns()
# ... handler logic ...
latency_ms = (time_ns() - start_time) // 1_000_000

# Store to DB
await store_api_metrics(request_id=..., user_id=..., model=..., cost_usd=..., pool=pool)
```

### Error Handling
- All endpoint handlers wrap logic in `try/except` and raise `HTTPException`
- Database operations use `async with pool.connection() as conn: async with conn.cursor() as cur:`
- Errors logged with `logger.error(f"...")` before raising
- `response.raise_for_status()` called after every httpx request

### Quality Evaluation Pattern
Response quality uses a multi-metric approach:
```python
# Semantic similarity
bert_f_score = await get_bert_score(source_text, generated_response)

# Sentence structure
rouge_scores = await get_rouge_score(source_text, generated_response)

# LLM-as-judge (optional)
llm_scores = await get_llm_eval_scores(source_text, generated_response)
```
Metrics: Relevancy, Coherence, Consistency, Fluency, Tone, Style — all normalized to 0–1.

### Async Throughout
- All I/O operations are `async def` with `await`
- `asyncio.Lock()` for cache mutation: `async with _cache_lock:`
- `@asynccontextmanager` for FastAPI lifespan (startup/shutdown)
- `AsyncConnectionPool` for PostgreSQL with `min_size=1, max_size=5`

### Logging
Standard pattern across all modules:
```python
import logging
logger = logging.getLogger(__name__)

# Debug markers
logger.info("breakpoint")
logger.info(f"variable: {variable}")
logger.warning(f"IMPORTANT_DATA: {data}")
logger.error(f"Failed to do X: {e}")
```

### LangGraph SDK Client
Always instantiate with user's API key from `current_user`:
```python
token = current_user["API_KEY"]
client = get_client(headers={"Authorization": f"Bearer {token}"})
# or for LangGraph API-KEY header:
client = get_client(headers={"API-KEY": request.headers.get("api-key")})
```

### SQL Delete Pattern
Store deletions use a 4-pattern prefix match to catch all namespace variants:
```python
params = (
    user_id,
    f"{user_id}.%",
    f"%.{user_id}.%",
    f"%.{user_id}",
)
await cur.execute("DELETE FROM store WHERE prefix = %s OR prefix LIKE %s OR prefix LIKE %s OR prefix LIKE %s", params)
```

### Dataclass for Simple Value Objects
Simple value objects use `@dataclass` (not Pydantic):
```python
@dataclass
class SubscriptionStatus:
    status: str = None
    subscription_id: str = None
    def to_dict(self): ...
    def update(self, field: Literal[...], value): ...
```

---

## Frequently Used Idioms

- `getattr(obj, "field", default)` — safe attribute access on LangGraph messages
- `state["key"].get("subkey", default)` — safe dict access on state
- `reduce_docs([], items)` — coerce LangGraph store SearchItems into Document lists
- `datetime.now(tz=timezone.utc).isoformat()` — all timestamps in UTC ISO format
- `str(uuid4())` — generate new thread/assistant IDs
- `quote(user_id, safe="")` — URL-encode Auth0 user IDs before API calls
- `isinstance(x, SomeClass)` before `getattr` when runtime.context fields may be dict or dataclass
