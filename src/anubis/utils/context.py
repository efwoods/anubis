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

    reference_audio_diarize_max_seconds: int = field(
        default=180,
        metadata={
            "description": "Cap on audio length fed to dominant-speaker diarization for reference uploads, in seconds. Keeps the input to a single non-chunked diarizer call so speaker labels stay unified. Env REFERENCE_AUDIO_DIARIZE_MAX_SECONDS."
        },
    )

    media_processing_concurrency: int = field(
        default=5,
        metadata={
            "description": "Max media items converted in parallel inside process_media_graph (bounds OpenAI diarization / yt_dlp fan-out so a large playlist or batch upload does not exhaust rate limits or memory). Env MEDIA_PROCESSING_CONCURRENCY."
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
