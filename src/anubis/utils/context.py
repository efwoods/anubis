# src/anubis/utils/context.py

"""Define the runtime context information for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated


from src.anubis.utils.prompts import system_prompts
from src.anubis.utils.prompts.subgraphs import vector_store_graph_prompts

from langchain_core.messages import SystemMessage

from typing import Dict, Any
import typing
import types


def _unwrap_type_hint(tp):
    """Reduce Optional, PEP604 unions, and Annotated to the inner type for coercion."""
    if tp is None:
        return None
    while True:
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)
        if origin is typing.Union or origin is types.UnionType:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                tp = non_none[0]
                continue
            return tp
        if origin is Annotated:
            if not args:
                return tp
            tp = args[0]
            continue
        return tp


@dataclass
class IdentityContext:
    name: str = field(default=None)
    description: str = field(default=None)

    def update_metadata(self, key: str, value: Any):
        """Update a specific metadata field."""
        self.metadata[key] = value

    def merge_metadata(self, new_metadata: Dict[str, Any]):
        """Merge new metadata into existing."""
        self._deep_merge(self.metadata, new_metadata)

    def _deep_merge(self, base: Dict, update: Dict):
        """Recursively merge dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt injection."""
        return {"name": self.name, **self.metadata}  # Unpack all metadata at top level


@dataclass
class AssistantContext(IdentityContext):
    metadata: dict = field(
        default=None,
        metadata={
            "description": "This is metadata that includes the user_id of the creator."
        },
    )


@dataclass
class UserContext(IdentityContext):
    pass


@dataclass(kw_only=True)
class GlobalContext:
    """Main context class for the memory graph system."""

    assistant_ctx: AssistantContext = field(default_factory=AssistantContext)
    user_ctx: UserContext = field(default_factory=UserContext)

    # max_search_results: int = field(
    #     default=10,
    #     metadata={
    #         "description":"Maximum number of search results to return for each search query."
    #     },
    # )

    # response_system_prompt: str = field(
    #     default=vector_store_graph_prompts.RESPONSE_SYSTEM_PROMPT,
    #     metadata={"description": "The system prompt used for generating responses."},
    # )

    # query_system_prompt: str = field(
    #     default=vector_store_graph_prompts.QUERY_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for processing and refining queries."
    #     },
    # )

    """ Default Environment Variables """

    """ <Inference Model> """

    model_provider: str = field(
        default=None, metadata={"description": "Model inference provider."}
    )

    together_api_key: str = field(
        default=None,
        metadata={
            "description": "inference provider for production use and for adapter training."
        },
    )

    llm_provider_api_key: str = field(
        default=None,
        metadata={"description": "API key for llama models"},
    )

    llm_provider_base_url: str = field(
        default=None, metadata={"description": "base url for the llama model"}
    )

    model: str = field(
        default=None,
        metadata={
            "description": "Model Name Only; text response and tool use for thought processing."
        },
    )

    model_prompt_cost: float = 0.0
    # metadata={"description": "Cost of input tokens."},

    model_completion_cost: float = 0.0
    # metadata={"description": "Completion token cost."},

    """ </Inference Model> """

    """ <Image Model> """

    image_model: str = field(
        default=None,
        metadata={
            "description": "Model Name Only; used without tools for image to text descriptions."
        },
    )

    image_model_api_key: str = field(
        default=None,
        metadata={
            "description": "API Key; used without tools for image to text descriptions."
        },
    )

    image_model_base_url: str = field(
        default=None,
        metadata={
            "description": "Base Url; used without tools for image to text descriptions."
        },
    )

    image_model_prompt_cost: float = 0.0
    # metadata={"description": "Cost of input tokens."},

    image_model_completion_cost: float = 0.0
    # metadata={"description": "Completion token cost."},

    """ </Image Model> """

    """ <Llama Model> """

    llama_api_key: str = field(
        default=None, metadata={"description": "LLama developer api key."}
    )

    llama_model: str = field(
        default=None, metadata={"description": "LLama model name."}
    )

    llama_model_prompt_cost: float = 0.0
    # metadata={"description": "Cost of input tokens."},

    llama_model_completion_cost: float = 0.0
    # metadata={"description": "Completion token cost."},

    """ </Llama Model> """

    """ <Classification Model> """

    classification_model: str = field(
        default=None, metadata={"description": "Classification model name."}
    )

    classification_model_prompt_cost: float = 0.0
    # metadata={"description": "Cost of input tokens."},

    classification_model_completion_cost: float = 0.0
    # metadata={"description": "Completion token cost."},

    classification_model_base_url: str = field(
        default=None,
        metadata={
            "description": "Base Url; used with structured output for classification."
        },
    )

    classification_model_api_key: str = field(
        default=None,
        metadata={
            "description": "API Key; used with structured output for classification."
        },
    )

    """ </Classification Model> """


    """ <Audio Transcription & Diarization Model> """

    openai_api_key: str = field(
        default=None,
        metadata={
            "description": "OpenAI API key for speech-to-text; env OPENAI_API_KEY. Falls back to llm_provider_api_key in code if unset."
        },
    )

    whisper_max_bytes: int = field(
        default=26214400,
        metadata={
            "description": "Max audio bytes per single STT request (25 MiB). Env WHISPER_MAX_BYTES."
        },
    )

    chunk_source_bytes_target: int = field(
        default=20971520,
        metadata={
            "description": "Target source bytes per segment when chunking long files. Env CHUNK_SOURCE_BYTES_TARGET."
        },
    )

    reference_audio_clip_max_seconds: float = field(
        default=10.0,
        metadata={
            "description": "Max seconds kept when truncating reference audio. Env REFERENCE_AUDIO_CLIP_MAX_SECONDS."
        },
    )

    enable_target_speaker_attribution: str = field(
        default="TRUE",
        metadata={
            "description": "When TRUE, run the post-diarization LLM target-attribution pass that recovers the target's turns scattered across per-chunk speaker labels. Env ENABLE_TARGET_SPEAKER_ATTRIBUTION."
        },
    )

    target_speaker_attribution_transcript_character_limit: int = field(
        default=100000,
        metadata={
            "description": "Max rendered transcript characters passed to the target-attribution pass in one call; above this the transcript is adjudicated per diarization chunk. Env TARGET_SPEAKER_ATTRIBUTION_TRANSCRIPT_CHARACTER_LIMIT."
        },
    )

    text_dialogue_segmentation_window_characters: int = field(
        default=4000,
        metadata={
            "description": "Character size of each window when segmenting long-form text into speaker turns (the model echoes the window, so output length is the binding constraint). Env TEXT_DIALOGUE_SEGMENTATION_WINDOW_CHARACTERS."
        },
    )

    text_dialogue_segmentation_max_characters: int = field(
        default=250000,
        metadata={
            "description": "Cap on total text characters segmented into speaker turns; content beyond the cap is skipped with a warning. Env TEXT_DIALOGUE_SEGMENTATION_MAX_CHARACTERS."
        },
    )

    narrative_speech_extraction_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "When TRUE, reference documents (scripture / menus) additionally run text dialogue segmentation to extract inferred-target quote and adapter documents alongside the plain document-namespace chunks. Env NARRATIVE_SPEECH_EXTRACTION_ENABLED."
        },
    )

    structured_web_extraction_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "When TRUE, structured web pages (character wikis, personal homepages) are parsed with BeautifulSoup to extract the inferred subject's biographical prose and verbatim direct quotes. Env STRUCTURED_WEB_EXTRACTION_ENABLED."
        },
    )

    browser_tools_enabled: str = field(
        default="FALSE",
        metadata={
            "description": "Set to TRUE to expose the Playwright browser tool suite (navigate to URL, click element, extract text, extract hyperlinks, get elements, current page, navigate back) to the avatar deep agent for live web browsing. Env BROWSER_TOOLS_ENABLED."
        },
    )

    browser_chromium_executable_path: str = field(
        default=None,
        metadata={
            "description": "Filesystem path of a system-installed Chromium binary for the Playwright browser tools. The production wolfi image installs the apk chromium package and sets this variable to /usr/bin/chromium. When empty, Playwright launches the Playwright-managed Chromium download instead (requires `playwright install chromium` on the host). Env BROWSER_CHROMIUM_EXECUTABLE_PATH."
        },
    )

    browser_conversation_idle_timeout_seconds: int = field(
        default=900,
        metadata={
            "description": "Seconds a conversation's dedicated headless Chromium may sit unused before the browser tools close that browser (browsing state for the conversation is then lost; the next browsing turn starts a fresh browser). Env BROWSER_CONVERSATION_IDLE_TIMEOUT_SECONDS."
        },
    )

    browser_max_concurrent_conversations: int = field(
        default=4,
        metadata={
            "description": "Maximum number of conversations that may each hold a dedicated headless Chromium process at once (each idle Chromium is roughly 100-200 MiB resident). The least-recently-used conversation's browser is closed when a new conversation needs one beyond this cap. Env BROWSER_MAX_CONCURRENT_CONVERSATIONS."
        },
    )

    media_processing_concurrency: int = field(
        default=5,
        metadata={
            "description": "Max media items converted in parallel inside process_media_graph (bounds OpenAI diarization / yt_dlp fan-out so a large playlist or batch upload does not exhaust rate limits or memory). Env MEDIA_PROCESSING_CONCURRENCY."
        },
    )

    standardized_question_analysis_concurrency: int = field(
        default=8,
        metadata={
            "description": "Max standardized identity questions asked in parallel per document by the standardized-question analyzer (each question is a separate structured-output call; bounds the per-document fan-out so the full question bank does not exhaust LLM rate limits). Env STANDARDIZED_QUESTION_ANALYSIS_CONCURRENCY."
        },
    )

    enable_document_analysis: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to run the analyze_documents branch (OCEAN, emotional triggers, standardized questions, narrative analyzers) in process_media_graph. FALSE skips it entirely; documents are still indexed via the direct convert->index_docs path. Env ENABLE_DOCUMENT_ANALYSIS."
        },
    )

    audio_transcription_model: str = field(
        default=None, metadata={"description": "Audio transcription model name."}
    )

    audio_transcription_price_per_minute: float = 0.0

    audio_diarization_estimated_price_per_minute: float = 0.0
    audio_diarization_model: str = field(
        default=None, metadata={"description": "Audio diarization model name."}
    )

    audio_diarization_price_per_million_tokens_input: float = 0.0
    audio_diarization_price_per_million_tokens_output: float = 0.0
    audio_diarization_context_window: int = field(
        default=0,
        metadata={"description": "Context window hint for diarization pricing or prompts."},
    )

    audio_diarization_known_speaker_name: str = field(
        default="avatar",
        metadata={
            "description": "Speaker id passed as known_speaker_names[0] with reference audio. Env AUDIO_DIARIZATION_KNOWN_SPEAKER_NAME."
        },
    )

    openai_speech_max_retries: int = field(
        default=4,
        metadata={
            "description": "Max retries for transient OpenAI speech (transcription/diarization) failures — 429 rate_limit_exceeded, timeouts, connection errors, 5xx — retried with exponential backoff. Permanent errors (insufficient_quota, auth) are NOT retried and surface immediately as item errors. Env OPENAI_SPEECH_MAX_RETRIES."
        },
    )

    openai_speech_retry_base_seconds: float = field(
        default=1.0,
        metadata={
            "description": "Base delay (seconds) for exponential backoff between transient OpenAI speech retries; delay = base * 2**attempt + jitter. Env OPENAI_SPEECH_RETRY_BASE_SECONDS."
        },
    )

    """ </Audio Transcription & Diarization Model> """

    """ <Stylistic + Knowledge Profile thresholds> """

    min_quotes_for_profile: int = field(
        default=20,
        metadata={
            "description": "Minimum number of quote-namespace Documents required to build the stylistic profile. Env MIN_QUOTES_FOR_PROFILE."
        },
    )
    profile_refresh_threshold: int = field(
        default=20,
        metadata={
            "description": "Minimum number of new quote Documents added since the last build to trigger a profile refresh. Env PROFILE_REFRESH_THRESHOLD."
        },
    )
    min_identity_docs_for_knowledge_profile: int = field(
        default=10,
        metadata={
            "description": "Minimum number of identity-namespace Documents required to build the knowledge profile. Env MIN_IDENTITY_DOCS_FOR_KNOWLEDGE_PROFILE."
        },
    )
    knowledge_profile_top_k: int = field(
        default=8,
        metadata={
            "description": "Top-K bounded retrieval for the knowledge evaluator's atomic-fact index. Env KNOWLEDGE_PROFILE_TOP_K."
        },
    )

    """ </Stylistic + Knowledge Profile thresholds> """

    """ <Deep Agent (think node) tuning> """

    deep_agent_summarization_max_tokens: int = field(
        default=120000,
        metadata={
            "description": "Token threshold above which SummarizationMiddleware compacts the deep agent's message history. Env DEEP_AGENT_SUMMARIZATION_MAX_TOKENS."
        },
    )

    deep_agent_summarization_keep_last_n_messages: int = field(
        default=20,
        metadata={
            "description": "Number of most-recent messages preserved verbatim when SummarizationMiddleware compacts the deep agent's history. Env DEEP_AGENT_SUMMARIZATION_KEEP_LAST_N_MESSAGES."
        },
    )

    deep_agent_recursion_limit: int = field(
        default=50,
        metadata={
            "description": "LangGraph recursion limit for the deep agent's inner tool-call loop invoked by the think node. Env DEEP_AGENT_RECURSION_LIMIT."
        },
    )

    """ </Deep Agent (think node) tuning> """

    """ <Data Analysis (MCP filesystem -> deep agent) tuning> """

    data_analysis_enabled: str = field(
        default="FALSE",
        metadata={
            "description": "Set to TRUE to enable the data preprocessing pipeline. Env DATA_ANALYSIS_ENABLED. NOTE: this gates data PREPROCESSING only — it does NOT gate the avatar's MCP data-analysis capability, which is gated solely by a discovered+consented per-user MCP connection (see data_analysis_mcp_discovery_url)."
        },
    )

    data_analysis_mcp_url: str = field(
        default="http://localhost:8000/mcp",
        metadata={
            "description": "Fallback URL of the Model Context Protocol filesystem server's tool endpoint. Normally the avatar saves the URL supplied by the server's discovery announcement; this default is used only when an announcement omits one. Env DATA_ANALYSIS_MCP_URL."
        },
    )

    data_analysis_mcp_discovery_url: str = field(
        default="http://localhost:8000/discovery",
        metadata={
            "description": "Server-Sent-Events discovery endpoint the avatar subscribes to in order to discover an available Model Context Protocol filesystem server and its connection details. Env DATA_ANALYSIS_MCP_DISCOVERY_URL."
        },
    )

    data_analysis_discovery_timeout_seconds: float = field(
        default=2.0,
        metadata={
            "description": "Maximum seconds the avatar waits for a discovery announcement before proceeding without offering a connection this turn. Kept small so a missing server never stalls a conversation turn. Env DATA_ANALYSIS_DISCOVERY_TIMEOUT_SECONDS."
        },
    )

    data_analysis_mcp_transport: str = field(
        default="streamable_http",
        metadata={
            "description": "Transport for the Model Context Protocol filesystem server connection. streamable_http is the supported value; the Server-Sent-Events transport is deprecated by the Model Context Protocol specification. Env DATA_ANALYSIS_MCP_TRANSPORT."
        },
    )

    data_analysis_mcp_server_name: str = field(
        default="Ubuntu-OS-Filesystem",
        metadata={
            "description": "Registered name of the Model Context Protocol filesystem server inside the MultiServerMCPClient configuration. Env DATA_ANALYSIS_MCP_SERVER_NAME."
        },
    )

    data_analysis_execution_backend: str = field(
        default="local_shell",
        metadata={
            "description": "Execution backend for deep-agent data analysis. local_shell runs shell commands inside this container's per-turn temporary workspace; hosted sandbox provider names are reserved for the future. Env DATA_ANALYSIS_EXECUTION_BACKEND."
        },
    )

    data_analysis_workspace_root: str = field(
        default="/tmp/anubis-analysis",
        metadata={
            "description": "Root directory under which each analysis turn creates an ephemeral workspace; the workspace is deleted when the turn ends. Env DATA_ANALYSIS_WORKSPACE_ROOT."
        },
    )

    data_analysis_store_max_bytes: int = field(
        default=52428800,
        metadata={
            "description": "Per-user-per-avatar byte quota for the ingested-data store buffer; least-recently-updated items are evicted beyond this size. Default 50 MiB. Env DATA_ANALYSIS_STORE_MAX_BYTES."
        },
    )

    data_analysis_store_max_age_days: int = field(
        default=90,
        metadata={
            "description": "Maximum age in days for items in the ingested-data store buffer; older items are evicted as a backstop. Env DATA_ANALYSIS_STORE_MAX_AGE_DAYS."
        },
    )

    data_analysis_registration_stale_seconds: float = field(
        default=120.0,
        metadata={
            "description": "Maximum age in seconds of a local MCP daemon's last heartbeat (POST /mcp/heartbeat) for its pushed registration to still count as online, for tunnel/local connection modes that have no live socket. Relay mode ignores this and uses live-socket presence instead. Env DATA_ANALYSIS_REGISTRATION_STALE_SECONDS."
        },
    )

    data_analysis_relay_request_timeout_seconds: float = field(
        default=120.0,
        metadata={
            "description": "Maximum seconds the /mcp/relay bridge waits for the local MCP daemon to return a proxy_response for one tunneled HTTP call before failing the request. Matches the daemon's own 120s local-proxy timeout. Env DATA_ANALYSIS_RELAY_REQUEST_TIMEOUT_SECONDS."
        },
    )

    """ </Data Analysis (MCP filesystem -> deep agent) tuning> """

    dev: str = field(
        default=None,
        metadata={
            "description": "development mode; single user model; 10 requests/minute; no adapters/training"
        },
    )

    huggingface_token: str = field(
        default=None, metadata={"description": "Token to use huggingface models"}
    )

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="microsoft/harrier-oss-v1-270m",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    vectorstore_postgres_uri: str = field(
        default=None,
        metadata={
            "description": "Connection string to postgres db for persistent document storage via vector store"
        },
    )

    async_postgres_store_uri: str = field(
        default=None,
        metadata={
            "description": "Connection string to async postgres store for persistent storage of avatar metadata for contextual prompt injection"
        },
    )

    model_token_limit: int = field(
        default=400000,
        metadata={
            "description": "Maximum context window for the primary inference model, in tokens (absolute count, not thousands)."
        },
    )

    context_completion_reserve_tokens: int = field(
        default=65536,
        metadata={
            "description": "Tokens reserved for model completion, tool outputs, and overhead when budgeting prompt size."
        },
    )

    conversation_verbatim_tail_messages: int = field(
        default=24,
        metadata={
            "description": "Number of most recent chat messages to keep verbatim before rolling summarization."
        },
    )

    context_summarization_max_chunks: int = field(
        default=32,
        metadata={
            "description": "Maximum number of text chunks processed per map-reduce summarization pass."
        },
    )

    map_reduce_chunk_max_tokens: int = field(
        default=120000,
        metadata={
            "description": "Maximum tokens per chunk when map-reducing a single oversized user message."
        },
    )

    system_prompt_max_tokens: int = field(
        default=120000,
        metadata={
            "description": "Upper bound on token count for the assembled identity/system prompt before truncation."
        },
    )

    memory_retrieval_max_items: int = field(
        default=200,
        metadata={
            "description": "Maximum episodic memory items retrieved from the store per turn (caps store search)."
        },
    )

    langsmith_api_key: str = field(default=None, metadata={"description": "api key"})

    deployment: str = field(
        default=None,
        metadata={
            "description": "True for langsmith deployments to use autoconfiguration of store; disables functionality of api yet allows the graph to run for deployments."
        },
    )

    supabase_url: str = field(
        default=None, metadata={"description": "url for user authentication"}
    )

    supabase_key: str = field(
        default=None, metadata={"description": "api key for user authentication"}
    )

    admin_user_id: str = field(
        default=None,
        metadata={
            "description": "user_id to allow the creation of public avatars. Reserved for CEO."
        },
    )

    anonymous_user_id: str = field(
        default=None,
        metadata={
            "description": "user_id to allow the creation of public avatars. Reserved for anonymous users to store the creation of avatars in a cookie."
        },
    )

    anonymous_api_key: str = field(
        default=None,
        metadata={
            "description": "api key for anonymous user data analytics to monitor content."
        },
    )

    stripe_secret_key: str = field(
        default=None,
        metadata={"description": "API key for interacting with the stripe API."},
    )

    stripe_product_id: str = field(
        default=None,
        metadata={"description": "Neural Nexus API monthly subscription product id."},
    )

    stripe_payment_url: str = field(
        default=None, metadata={"description": "Payment URL for subscriptions."}
    )

    stripe_publishable_key: str = field(
        default=None,
        metadata={
            "description": "Stripe publishable (client-side) key used to render checkout."
        },
    )

    stripe_manage_subscription_url: str = field(
        default=None,
        metadata={
            "description": "Stripe customer-portal login URL for managing/cancelling a subscription."
        },
    )

    stripe_webhook_secret: str = field(
        default=None,
        metadata={
            "description": (
                "Signing secret used to verify inbound Stripe webhook events "
                "(Dashboard 'Your account' endpoint, or a fixed whsec_). "
                "When empty, the API falls back to stripe_webhook_secret_file "
                "(written by the docker-compose stripe-cli service)."
            )
        },
    )

    stripe_webhook_secret_file: str = field(
        default=None,
        metadata={
            "description": (
                "Path to a file containing a whsec_ signing secret. Used when "
                "STRIPE_WEBHOOK_SECRET is unset — typically "
                "/run/stripe/webhook_secret from the compose stripe-cli service. "
                "Env STRIPE_WEBHOOK_SECRET_FILE."
            )
        },
    )

    stripe_billing_config_json: str = field(
        default=None,
        metadata={
            "description": (
                "JSON emitted by scripts/provision_stripe_billing.py mapping the four "
                "meter event names to meter ids and each tier to its flat base price id "
                "and per-meter graduated price ids. Parsed via "
                "src.anubis.utils.billing.config.load_stripe_billing_config. When "
                "empty, the API falls back to stripe_billing_config_file (written by "
                "the docker-compose stripe-provision service)."
            )
        },
    )

    stripe_billing_config_file: str = field(
        default=None,
        metadata={
            "description": (
                "Path to a file containing the billing-config JSON. Used when "
                "STRIPE_BILLING_CONFIG_JSON is unset — typically "
                "/run/stripe/billing_config.json, written by the compose "
                "stripe-provision service so no JSON is pasted into the env. "
                "Env STRIPE_BILLING_CONFIG_FILE."
            )
        },
    )

    message_rate_limit_window_seconds: int = field(
        default=60,
        metadata={
            "description": (
                "Length, in seconds, of the rolling window used by the per-user "
                "token rate limit on the message endpoints. Combined with "
                "MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW: a message request is "
                "refused with HTTP 429 and a Retry-After header when the user's "
                "summed messaging plus adapter-inference token usage inside this "
                "window already meets the cap."
            )
        },
    )

    message_rate_limit_tokens_per_window: int = field(
        default=0,
        metadata={
            "description": (
                "Maximum messaging plus adapter-inference tokens one user may "
                "consume inside each MESSAGE_RATE_LIMIT_WINDOW_SECONDS rolling "
                "window (a tokens-per-minute style limit, in the spirit of the "
                "OpenAI rate-limit guide). This is an abuse guard independent of "
                "the monthly allotment and of pay-per-use, so a runaway client "
                "cannot burn a month's budget or an unbounded overage bill in "
                "minutes. Zero disables the limit."
            )
        },
    )

    media_upload_rate_limit_window_seconds: int = field(
        default=60,
        metadata={
            "description": (
                "Length, in seconds, of the rolling window used by the per-user "
                "token rate limit on the update_avatar_identity_with_media "
                "endpoint. Combined with MEDIA_UPLOAD_RATE_LIMIT_TOKENS_PER_WINDOW."
            )
        },
    )

    media_upload_rate_limit_tokens_per_window: int = field(
        default=0,
        metadata={
            "description": (
                "Maximum document-upload token-equivalents one user may consume "
                "inside each MEDIA_UPLOAD_RATE_LIMIT_WINDOW_SECONDS rolling "
                "window on the update_avatar_identity_with_media endpoint. Zero "
                "disables the limit."
            )
        },
    )

    usage_period_days: int = field(
        default=0,
        metadata={
            "description": (
                "Length, in days, of the local usage-allotment period read by "
                "allotment gating and the subscription-status endpoint. Zero "
                "(the default) means calendar-month periods, matching Stripe's "
                "monthly billing cycle; a positive value means fixed-length "
                "windows counted from the user's usage_period_anchor (or the "
                "deterministic global anchor when the user has none)."
            )
        },
    )

    estimated_analysis_passes_per_document: int = field(
        default=2,
        metadata={
            "description": (
                "Number of identity-analysis passes that re-read one uploaded "
                "item's extracted content (transcript or text) — used by the "
                "pre-request token estimate as extracted-content tokens times "
                "this pass count. Two models the current pipeline "
                "(classification plus identity-dimension analysis); set to "
                "zero if the analysis stage is dropped so estimates reflect "
                "the change in advance of any model call."
            )
        },
    )

    system_prompt_token_estimate_cache_ttl_seconds: int = field(
        default=300,
        metadata={
            "description": (
                "Maximum age, in seconds, of a cached system-prompt token "
                "measurement used by the pre-request message estimate. Every "
                "load_consciousness build refreshes the measurement, so the "
                "time-to-live only bounds staleness between a large identity "
                "upload and the next message turn."
            )
        },
    )

    anonymous_billing_enabled: str = field(
        default="FALSE",
        metadata={
            "description": (
                "TRUE enables per-hashed-ip Stripe metering for anonymous "
                "users: each anonymous visitor lazily receives a Stripe "
                "customer with a $0 free-tier subscription so anonymous "
                "usage is visible in Stripe cost analysis. FALSE (the "
                "default) keeps anonymous metering local-only (api_metrics), "
                "avoiding Stripe customer fan-out in development."
            )
        },
    )

    message_expected_output_tokens_estimate: int = field(
        default=512,
        metadata={
            "description": (
                "Expected completion-token budget for one message reply, used "
                "by the manual pre-request message estimate (billed usage "
                "covers prompt AND completion tokens). Calibrate from "
                "observed api_metrics completion_tokens."
            )
        },
    )

    baseline_response_threshold: float = field(
        default=47.66322963655769,
        metadata={
            "description": "Pre-calculated IQR threshold for the empirical representation of the squared mahalanobis distances of the features presented from the unmodified chatgpt responses using a leave-one-out method. Recalibrated by data/build_baseline_features_arr.py whenever the feature vector changes (current: 28-wide v3 vector)."
        }
    )

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args; coerce int/float hints from str."""
        hints = typing.get_type_hints(self.__class__)

        for f in fields(self):
            if not f.init:
                continue

            field_type = hints.get(f.name)
            scalar_type = _unwrap_type_hint(field_type)

            if getattr(self, f.name) == f.default:
                env_val = os.environ.get(f.name.upper(), f.default)

                # An env var that is declared but left empty (e.g. `MODEL_TOKEN_LIMIT=`
                # in .env) reads back as "" rather than being absent. Treat an
                # empty/whitespace-only string as "unset" and keep the field default,
                # so int("")/float("") coercion below cannot crash startup.
                if isinstance(env_val, str) and env_val.strip() == "":
                    env_val = f.default

                if env_val is not None:
                    if scalar_type is float:
                        env_val = float(env_val)
                    elif scalar_type is int:
                        env_val = int(env_val)

                setattr(self, f.name, env_val)

            val = getattr(self, f.name)
            if scalar_type is float and isinstance(val, str) and val.strip() != "":
                try:
                    setattr(self, f.name, float(val))
                except ValueError:
                    pass
            elif scalar_type is int and isinstance(val, str) and val.strip() != "":
                try:
                    setattr(self, f.name, int(val, 10))
                except ValueError:
                    pass
