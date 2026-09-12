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

    reference_audio_minimum_seconds: float = field(
        default=1.3,
        metadata={
            "description": "Least seconds of the avatar speaking a reference clip must hold before the clip may anchor the diarizer. Env REFERENCE_AUDIO_MINIMUM_SECONDS."
        },
    )

    media_preprocessing_seconds_per_media_second: float = field(
        default=1.0,
        metadata={
            "description": (
                "Wall-clock seconds the media pipeline is expected to spend per "
                "second of uploaded audio or video (download, speaker isolation, "
                "diarization, transcription, indexing). Multiplied by the probed "
                "media duration to produce the processing-time estimate shown on "
                "the upload progress card and stamped on every progress frame. "
                "Env MEDIA_PREPROCESSING_SECONDS_PER_MEDIA_SECOND."
            )
        },
    )

    remote_duration_probe_timeout_seconds: float = field(
        default=30.0,
        metadata={
            "description": (
                "Seconds allowed for the yt_dlp metadata probe that reads a remote "
                "video's duration at upload time. A YouTube probe takes roughly ten "
                "seconds; when the probe exceeds this limit the item is estimated "
                "with ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS instead, which makes "
                "both the billing estimate and the processing-time estimate wrong. "
                "Env REMOTE_DURATION_PROBE_TIMEOUT_SECONDS."
            )
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

    ground_truth_calibration_timeout_seconds: float = field(
        default=1800.0,
        metadata={
            "description": "Ceiling on the once-per-upload refit of the avatar's direct-quote cloud (empirical Mahalanobis threshold + IsolationForest) that runs after a media batch finishes indexing. The fit is quadratic in corpus size before the MAX_CALIBRATION_ROWS subsample caps it, and it is awaited before the batch reports finished, so this bound stops a pathological corpus from wedging an upload's terminal progress event. Exceeding the ceiling is not an error: the upload completes and the direct-quote comparison keeps its previous fit until the next upload or an explicit recalibration. Env GROUND_TRUTH_CALIBRATION_TIMEOUT_SECONDS."
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

    standardized_question_analysis_enabled: str = field(
        default="FALSE",
        metadata={
            "description": "TRUE to let the standardized-question analyzer run. It asks the whole standardized identity question bank about a document, which is roughly two hundred structured-output calls for that one document, so it is off by default and is narrowed to biographical documents by the analysis_scaffolds metadata key. Env STANDARDIZED_QUESTION_ANALYSIS_ENABLED."
        },
    )

    """ <Psychological analysis (passive latent-feature analysis of the target)> """

    enable_psychological_analysis: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to run the psycho_analysis_graph branch of process_media_graph, which reads the target's latent psychology (love languages, emotional triggers in dialogue, attachment style, values, moral foundations, typology, defenses, drives, conversational habits) from uploaded media and writes the consolidated psychological profile injected into the avatar's system prompt. FALSE skips the branch; every other upload stage is unaffected. Env ENABLE_PSYCHOLOGICAL_ANALYSIS."
        },
    )

    psychological_analysis_concurrency: int = field(
        default=6,
        metadata={
            "description": "Maximum psychological dimensions analyzed in parallel per upload. Every dimension is one structured-output call per selected document, so this bounds the fan-out of a single upload against the model provider's rate limit. Env PSYCHOLOGICAL_ANALYSIS_CONCURRENCY."
        },
    )

    psychological_analysis_dimensions: str = field(
        default=None,
        metadata={
            "description": "Comma-separated list of psychological dimensions to analyze, naming keys of the dimension registry in src/subgraphs/psycho_analysis_graph/utils/dimensions.py (for example love_languages,attachment_style). Empty runs every registered dimension except the ones disabled by their own flag. Env PSYCHOLOGICAL_ANALYSIS_DIMENSIONS."
        },
    )

    psychological_analysis_max_documents: int = field(
        default=6,
        metadata={
            "description": "Maximum documents from one upload that the psychological analysis reads. Every document costs one structured-output call per dimension, so this is the single biggest lever on what an upload costs: at six documents and twelve dimensions an upload spends seventy-two calls. The dimensions describe stable traits, so reading every chunk of a long transcript pays many times over to reach the same conclusion and produces near-duplicate findings; the selection keeps the documents richest in the target's own words. Env PSYCHOLOGICAL_ANALYSIS_MAX_DOCUMENTS."
        },
    )

    enable_dark_trait_analysis: str = field(
        default="FALSE",
        metadata={
            "description": "TRUE to let the dark-trait dimension (Machiavellianism, narcissism, psychopathy, sadism) run. Off by default: the reading is easy to draw from thin evidence and it is not needed to reproduce how a person actually speaks. Env ENABLE_DARK_TRAIT_ANALYSIS."
        },
    )

    psychological_profile_max_characters: int = field(
        default=6000,
        metadata={
            "description": "Ceiling on the rendered psychological profile injected into the avatar's system prompt. The profile is loaded on every turn, so an unbounded profile would grow the billed input tokens of every message. Env PSYCHOLOGICAL_PROFILE_MAX_CHARACTERS."
        },
    )

    current_emotion_decay_hours: float = field(
        default=6.0,
        metadata={
            "description": "Half-life in hours over which the avatar's current emotional state decays back toward the baseline emotional profile read from the uploaded media. Zero disables decay, so the state holds until the next conversation moves it. Env CURRENT_EMOTION_DECAY_HOURS."
        },
    )

    current_emotion_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to track the avatar's current emotional state across a conversation and render it into the YOUR EMOTIONS section of the system prompt. The state is refreshed inside the existing observe_user branch and adds no model call. Env CURRENT_EMOTION_ENABLED."
        },
    )

    """ <AI monitoring (content moderation and account bans)> """

    content_moderation_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to screen every message and every uploaded document against the terms of service and the privacy policy; a confirmed violation bans the account. FALSE disables both the inline screen and the background judge. Env CONTENT_MODERATION_ENABLED."
        },
    )

    content_moderation_fast_screen_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to run the OpenAI moderation endpoint as the inline first pass on every message. It is free and answers in roughly a tenth of a second, so it is the only moderation stage allowed on the critical path of a reply. Env CONTENT_MODERATION_FAST_SCREEN_ENABLED."
        },
    )

    content_moderation_fast_screen_model: str = field(
        default="omni-moderation-latest",
        metadata={
            "description": "The OpenAI moderation model used for the inline first pass. Moderation is an OpenAI-only API, so this is used regardless of MODEL_PROVIDER, reading OPENAI_API_KEY and falling back to the general provider key. Env CONTENT_MODERATION_FAST_SCREEN_MODEL."
        },
    )

    content_moderation_fast_screen_threshold: float = field(
        default=0.9,
        metadata={
            "description": "Category score, from zero to one, at or above which the inline screen refuses the turn outright instead of deferring to the background judge. Lower catches more and refuses more false positives; the value is clamped into the endpoint's own zero-to-one range. Env CONTENT_MODERATION_FAST_SCREEN_THRESHOLD."
        },
    )

    content_moderation_deep_judge_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to run the structured-output terms-of-service and privacy-policy judge. On the chat path it runs after the reply has streamed, so it never costs the reader latency; on the upload path it gates indexing. Env CONTENT_MODERATION_DEEP_JUDGE_ENABLED."
        },
    )

    content_moderation_max_characters: int = field(
        default=6000,
        metadata={
            "description": "Characters per judge window. Long documents are judged in windows of this size, up to a bounded number of windows, so one very large upload cannot cost an unbounded classification call. Env CONTENT_MODERATION_MAX_CHARACTERS."
        },
    )

    content_moderation_concurrency: int = field(
        default=4,
        metadata={
            "description": "Maximum documents judged in parallel when an upload is moderated. Env CONTENT_MODERATION_CONCURRENCY."
        },
    )

    ban_refund_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to cancel the banned account's Stripe subscription and refund the latest paid invoice when a ban is recorded. Env BAN_REFUND_ENABLED."
        },
    )

    ban_appeal_contact_email: str = field(
        default="contact@neuralnexus.site",
        metadata={
            "description": "The address a banned person is told to contact to appeal the ban. Env BAN_APPEAL_CONTACT_EMAIL."
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

    """ <Signature key-phrase discovery> """

    key_phrase_judgement_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE runs the language-model judge that separates a speaker's style markers from their subject matter after the statistical shortlist is built. FALSE keeps the shortlist unjudged, which is the rollback. Env KEY_PHRASE_JUDGEMENT_ENABLED."
        },
    )
    key_phrase_judgement_batch_size: int = field(
        default=20,
        metadata={
            "description": "How many candidate phrases each structured-output judging call classifies. Kept modest because a structured-output call does not reliably return one record per input and stops early on long answers; candidates a round leaves out are re-asked with the batch halved. Env KEY_PHRASE_JUDGEMENT_BATCH_SIZE."
        },
    )
    key_phrase_judgement_concurrency: int = field(
        default=4,
        metadata={
            "description": "How many judging calls may run at once. Env KEY_PHRASE_JUDGEMENT_CONCURRENCY."
        },
    )
    key_phrase_judgement_timeout_seconds: float = field(
        default=120.0,
        metadata={
            "description": "Ceiling on the whole judging fan-out. Exceeding it falls back to the statistical shortlist rather than consuming the much larger ground-truth calibration budget. Env KEY_PHRASE_JUDGEMENT_TIMEOUT_SECONDS."
        },
    )
    key_phrase_candidate_shortlist_size: int = field(
        default=200,
        metadata={
            "description": "How many statistically-selected candidates are handed to the judge. This is the direct cost dial for judging. Env KEY_PHRASE_CANDIDATE_SHORTLIST_SIZE."
        },
    )
    key_phrase_profile_maximum_phrases: int = field(
        default=25,
        metadata={
            "description": "Hard ceiling on how many signature phrases are stored for one avatar. The set is re-ranked and capped on every calibration rather than accumulated, so this bounds growth across uploads. Env KEY_PHRASE_PROFILE_MAXIMUM_PHRASES."
        },
    )
    key_phrase_prompt_maximum_phrases: int = field(
        default=25,
        metadata={
            "description": "Ceiling on how many signature phrases are rendered into the SIGNATURE PHRASES system-prompt section. Also bounds phrase lists stored before the ceiling existed. Env KEY_PHRASE_PROMPT_MAXIMUM_PHRASES."
        },
    )
    key_phrase_incumbency_score_bonus: float = field(
        default=0.25,
        metadata={
            "description": "Score bonus, in log-ratio units, given to a phrase already stored for this avatar so the capped list does not oscillate when two phrases score almost identically. Zero disables it. Env KEY_PHRASE_INCUMBENCY_SCORE_BONUS."
        },
    )

    """ </Signature key-phrase discovery> """

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
            "description": "Set to TRUE to enable the data preprocessing pipeline. Env DATA_ANALYSIS_ENABLED. NOTE: this gates data PREPROCESSING only — it does NOT gate the avatar's MCP data-analysis capability, which is gated solely by the per-device MCP connections adopted for the personal avatar (see data_analysis_mcp_discovery_url)."
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

    data_analysis_inline_artifact_max_bytes: int = field(
        default=2097152,
        metadata={
            "description": "Maximum size in bytes of one created artifact (report or plot) whose content is inlined on the assistant reply for display in the client. Larger artifacts stay in durable storage but are reported as metadata only, so an oversized file cannot bloat the checkpointed message. Default 2 MiB. Env DATA_ANALYSIS_INLINE_ARTIFACT_MAX_BYTES."
        },
    )

    data_analysis_device_fanout_timeout_seconds: float = field(
        default=20.0,
        metadata={
            "description": "Maximum seconds one connected machine is given to answer its leg of a fan-out data-analysis call (for example discover_data_files across every connected machine) before that machine is reported as offline. Fan-out legs run concurrently, so this is the ceiling the whole call adds to the turn no matter how many machines are connected. Kept well below data_analysis_relay_request_timeout_seconds so a sleeping laptop cannot stall a conversation turn. Env DATA_ANALYSIS_DEVICE_FANOUT_TIMEOUT_SECONDS."
        },
    )

    data_analysis_max_devices_per_user: int = field(
        default=10,
        metadata={
            "description": "Maximum number of local MCP daemon devices one user may register simultaneously (Ubuntu desktop, macOS, mobile, Windows, and so on). POST /mcp/register rejects a new device beyond this count. Guards the fan-out cost of a data-analysis call and the store against an unbounded set of stale device records. Env DATA_ANALYSIS_MAX_DEVICES_PER_USER."
        },
    )

    """ </Data Analysis (MCP filesystem -> deep agent) tuning> """

    """ <Connected accounts (mailbox and social) for the personal avatar> """

    connected_account_encryption_key: str = field(
        default=None,
        metadata={
            "description": "Fernet key encrypting the third-party credentials the owner connects to their personal avatar (Google OAuth refresh tokens for Gmail, tokens for custom connectors). This is the only secret in the platform that must be recoverable rather than merely comparable, because the avatar has to present the original credential to a mail server on a later turn. Generate one with src.anubis.utils.secret_store.generate_encryption_key(). Rotating this key invalidates every stored credential, which surfaces to the owner as a request to reconnect the account rather than as silent corruption. Env CONNECTED_ACCOUNT_ENCRYPTION_KEY."
        },
    )

    max_connected_accounts_per_user: int = field(
        default=10,
        metadata={
            "description": "Maximum number of external accounts (mailboxes and, later, social accounts) one user may connect simultaneously. POST /connect_mailbox rejects a new account beyond this count. Guards the store against an unbounded set of stale credential records and bounds the cost of listing accounts on every capability check. Mirrors data_analysis_max_devices_per_user. Env MAX_CONNECTED_ACCOUNTS_PER_USER."
        },
    )

    mailbox_fetch_max_messages: int = field(
        default=25,
        metadata={
            "description": "Ceiling on how many messages one mailbox search may return, regardless of the limit the model asks for. Keeps a request for 'all my email' from spending the whole context window on message summaries. Env MAILBOX_FETCH_MAX_MESSAGES."
        },
    )

    mailbox_request_timeout_seconds: float = field(
        default=30.0,
        metadata={
            "description": "Maximum seconds one IMAP socket operation is given before the mailbox is reported as unreachable for that turn. Mail servers are reached over the public internet from inside a conversation turn, so this is the ceiling a sleeping or throttled server can add to a reply. Env MAILBOX_REQUEST_TIMEOUT_SECONDS."
        },
    )

    mailbox_send_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the personal avatar may transmit email through a connected mailbox's submission server when the owner explicitly asks in conversation. Set to false to keep the avatar draft-only while still reading mail; the send tool is then withheld and the avatar says the draft is waiting. Env MAILBOX_SEND_ENABLED."
        },
    )

    max_custom_mcp_connectors_per_user: int = field(
        default=10,
        metadata={
            "description": "Maximum number of custom Model Context Protocol servers (custom connectors) one user may connect simultaneously. POST /connect_account refuses a new custom connector beyond this count. Each connector's tool list is fetched and attached to every turn, so this bounds the per-turn tool count and the prompt describing it. Env MAX_CUSTOM_MCP_CONNECTORS_PER_USER."
        },
    )

    mcp_connector_probe_timeout_seconds: float = field(
        default=20.0,
        metadata={
            "description": "Maximum seconds a custom Model Context Protocol server is given to list its tools, both when the owner connects the server (the address is proved before it is stored) and when the avatar loads the server's tools for a turn. A server that does not answer in time contributes no tools for that turn. Env MCP_CONNECTOR_PROBE_TIMEOUT_SECONDS."
        },
    )

    google_oauth_client_id: str = field(
        default=None,
        metadata={
            "description": "Client ID of the Google Cloud OAuth 2.0 web client the connect card opens Google sign-in with (Gmail). Created in the Google Cloud console under Google Auth Platform > Clients; the client's authorized redirect URIs must include {CONNECT_OAUTH_REDIRECT_BASE_URL}/connect_account/oauth/callback. While the consent screen is in Testing publishing status only listed test users can sign in and refresh tokens expire after seven days; the mailbox is then reported as needing reconnection and the card is raised again. Env GOOGLE_OAUTH_CLIENT_ID."
        },
    )

    google_oauth_client_secret: str = field(
        default=None,
        metadata={
            "description": "Client secret paired with GOOGLE_OAUTH_CLIENT_ID. Sent only to Google's token endpoint; never logged or returned. Env GOOGLE_OAUTH_CLIENT_SECRET."
        },
    )

    connect_oauth_redirect_base_url: str = field(
        default=None,
        metadata={
            "description": "Public base URL of this API as a browser reaches it (for example http://localhost:9600 in development, https://api.neuralnexus.site in production). The OAuth callback the popup returns to is {base}/connect_account/oauth/callback; the same value is registered as the redirect URI with Google and with Model Context Protocol authorization servers during dynamic client registration. Env CONNECT_OAUTH_REDIRECT_BASE_URL."
        },
    )

    connect_oauth_popup_target_origins: str = field(
        default=None,
        metadata={
            "description": "Comma-separated browser origins of the Neural Nexus web UI (for example http://localhost:5173,https://neuralnexus.site). The OAuth callback page posts its non-secret result to the window that opened the popup, and only to these origins — never to '*'. Env CONNECT_OAUTH_POPUP_TARGET_ORIGINS."
        },
    )

    connect_oauth_state_secret: str = field(
        default=None,
        metadata={
            "description": "Secret that signs the OAuth 'state' parameter carried through the popup, so the callback can trust which user, provider, and avatar a returning login belongs to. Leave empty to derive the signing key from CONNECTED_ACCOUNT_ENCRYPTION_KEY. Env CONNECT_OAUTH_STATE_SECRET."
        },
    )

    connect_oauth_state_max_age_seconds: int = field(
        default=600,
        metadata={
            "description": "How long a started OAuth login stays valid before the owner must press Sign in again. Bounds both the signed state and the pending-login record holding the PKCE verifier. Env CONNECT_OAUTH_STATE_MAX_AGE_SECONDS."
        },
    )

    connect_oauth_http_timeout_seconds: float = field(
        default=15.0,
        metadata={
            "description": "Timeout for each HTTP call to an OAuth provider during a connection: token exchange, token refresh, userinfo, and Model Context Protocol authorization-server discovery and client registration. Env CONNECT_OAUTH_HTTP_TIMEOUT_SECONDS."
        },
    )

    mcp_oauth_client_name: str = field(
        default="Neural Nexus",
        metadata={
            "description": "The client_name this API registers under when a Model Context Protocol server's authorization server supports dynamic client registration; shown on that server's consent screen. Env MCP_OAUTH_CLIENT_NAME."
        },
    )

    github_oauth_client_id: str = field(
        default=None,
        metadata={
            "description": "Client ID of the GitHub OAuth App the connect card opens GitHub sign-in with. Created at github.com/settings/developers with the callback {CONNECT_OAUTH_REDIRECT_BASE_URL}/connect_account/oauth/callback. Env GITHUB_OAUTH_CLIENT_ID."
        },
    )

    github_oauth_client_secret: str = field(
        default=None,
        metadata={
            "description": "Client secret paired with GITHUB_OAUTH_CLIENT_ID. Sent only to GitHub's token endpoint; never logged or returned. Env GITHUB_OAUTH_CLIENT_SECRET."
        },
    )

    x_oauth_client_id: str = field(
        default=None,
        metadata={
            "description": "Client ID of the X (Twitter) OAuth 2.0 app (developer.x.com, user authentication settings, type Web App) the connect card opens X sign-in with, using PKCE. Env X_OAUTH_CLIENT_ID."
        },
    )

    x_oauth_client_secret: str = field(
        default=None,
        metadata={
            "description": "Client secret paired with X_OAUTH_CLIENT_ID, sent as HTTP Basic to X's token endpoint. Env X_OAUTH_CLIENT_SECRET."
        },
    )

    coinbase_oauth_client_id: str = field(
        default=None,
        metadata={
            "description": "Client ID of the Coinbase OAuth application the connect card opens Coinbase sign-in with, requesting read-only wallet scopes. Callback {CONNECT_OAUTH_REDIRECT_BASE_URL}/connect_account/oauth/callback. Env COINBASE_OAUTH_CLIENT_ID."
        },
    )

    coinbase_oauth_client_secret: str = field(
        default=None,
        metadata={
            "description": "Client secret paired with COINBASE_OAUTH_CLIENT_ID. Sent only to Coinbase's token endpoint; never logged or returned. Env COINBASE_OAUTH_CLIENT_SECRET."
        },
    )

    vercel_oauth_client_id: str = field(
        default=None,
        metadata={
            "description": "Client ID of the Vercel integration the connect card opens Vercel sign-in with. Env VERCEL_OAUTH_CLIENT_ID."
        },
    )

    vercel_oauth_client_secret: str = field(
        default=None,
        metadata={
            "description": "Client secret paired with VERCEL_OAUTH_CLIENT_ID. Env VERCEL_OAUTH_CLIENT_SECRET."
        },
    )

    plaid_client_id: str = field(
        default=None,
        metadata={
            "description": "Plaid client id (dashboard.plaid.com, Team settings, Keys) used by the Finance connector's Plaid Link popup and transaction sync. Env PLAID_CLIENT_ID."
        },
    )

    plaid_secret: str = field(
        default=None,
        metadata={
            "description": "Plaid secret for the environment named by PLAID_ENVIRONMENT (the Sandbox secret works with Plaid's test bank credentials; production needs Plaid's approval). Never logged or returned. Env PLAID_SECRET."
        },
    )

    plaid_environment: str = field(
        default="sandbox",
        metadata={
            "description": "Which Plaid environment the Finance connector talks to: sandbox or production. Env PLAID_ENVIRONMENT."
        },
    )

    plaid_products: str = field(
        default="transactions",
        metadata={
            "description": "Comma-separated Plaid products requested when a bank is linked. Env PLAID_PRODUCTS."
        },
    )

    plaid_country_codes: str = field(
        default="US",
        metadata={
            "description": "Comma-separated country codes Plaid Link offers institutions for. Env PLAID_COUNTRY_CODES."
        },
    )

    browser_session_keepalive_hours: float = field(
        default=12.0,
        metadata={
            "description": "How often the API revisits every site the owner signed in to through the live browser, re-saving refreshed cookies so the session stays alive for days without a new sign-in. A visit that lands on a login page marks the account as needing sign-in again and notifies the owner through the inbox. Env BROWSER_SESSION_KEEPALIVE_HOURS."
        },
    )

    browser_session_keepalive_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the browser-session keepalive task runs in this process. Set to false in a process that must not open browsers. Env BROWSER_SESSION_KEEPALIVE_ENABLED."
        },
    )

    browser_session_login_ttl_seconds: int = field(
        default=900,
        metadata={
            "description": "How long a live browser sign-in window stays open waiting for the owner before the login is discarded. Env BROWSER_SESSION_LOGIN_TTL_SECONDS."
        },
    )

    browser_session_max_concurrent_logins: int = field(
        default=3,
        metadata={
            "description": "How many live browser sign-in windows this process serves at once; further requests answer 409 until one finishes. Env BROWSER_SESSION_MAX_CONCURRENT_LOGINS."
        },
    )

    browser_session_max_open: int = field(
        default=6,
        metadata={
            "description": "How many signed-in browser sessions stay open in this process at once for the connected-site tools; the least recently used is closed beyond this count and reopened from the stored session on demand. Env BROWSER_SESSION_MAX_OPEN."
        },
    )

    browser_session_idle_seconds: int = field(
        default=900,
        metadata={
            "description": "Seconds an open browser session may sit unused before the process closes the window (the stored session survives; the next tool call reopens the window). Env BROWSER_SESSION_IDLE_SECONDS."
        },
    )

    browser_session_frame_interval_ms: int = field(
        default=250,
        metadata={
            "description": "Milliseconds between frames of a live browser sign-in window when the browser cannot stream frames on the browser's own schedule. Env BROWSER_SESSION_FRAME_INTERVAL_MS."
        },
    )

    website_crawl_max_pages: int = field(
        default=50,
        metadata={
            "description": "Ceiling on how many pages one website crawl or audit fetches, regardless of what the model asks for. Env WEBSITE_CRAWL_MAX_PAGES."
        },
    )

    report_scheduler_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether scheduled reports (weekly sprint digest, monthly spend digest, and any report the owner schedules in conversation) run in this process and are delivered to the inbox. Env REPORT_SCHEDULER_ENABLED."
        },
    )

    report_scheduler_poll_seconds: float = field(
        default=60.0,
        metadata={
            "description": "How often the report scheduler looks for due schedules. Env REPORT_SCHEDULER_POLL_SECONDS."
        },
    )

    report_schedule_run_timeout_seconds: float = field(
        default=600.0,
        metadata={
            "description": "Ceiling on one scheduled report run through the graph. Env REPORT_SCHEDULE_RUN_TIMEOUT_SECONDS."
        },
    )

    tool_call_log_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether every tool the avatar calls is recorded in the tool_calls table (tool name, avatar, thread, duration) so feature usage per avatar can be reported. Env TOOL_CALL_LOG_ENABLED."
        },
    )

    finance_sync_min_interval_minutes: int = field(
        default=360,
        metadata={
            "description": "Minimum minutes between two Plaid transaction syncs of the same institution; a finance question inside that window answers from the stored transactions. Env FINANCE_SYNC_MIN_INTERVAL_MINUTES."
        },
    )

    inbox_account_poll_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the agent inbox also polls every non-mailbox connected account (signed-in sites, vendor APIs) for new items addressed to the owner, through the account's own tools. Env INBOX_ACCOUNT_POLL_ENABLED."
        },
    )

    inbox_account_poll_interval_seconds: float = field(
        default=1800.0,
        metadata={
            "description": "Minimum seconds between two discovery passes on the same non-mailbox account; each pass costs a model call and, for a signed-in site, a browser visit. Env INBOX_ACCOUNT_POLL_INTERVAL_SECONDS."
        },
    )

    inbox_discovery_max_items: int = field(
        default=10,
        metadata={
            "description": "Ceiling on new items one discovery pass may report for one account. Env INBOX_DISCOVERY_MAX_ITEMS."
        },
    )

    inbox_discovery_max_steps: int = field(
        default=8,
        metadata={
            "description": "Ceiling on tool calls one discovery or delivery pass may make on one account. Env INBOX_DISCOVERY_MAX_STEPS."
        },
    )

    """ <Content subscriptions: the personal avatar's own accounts, crawled and subscribed> """

    social_webhook_callback_base_url: str = field(
        default=None,
        metadata={
            "description": "Public https base address platforms call when the owner publishes, e.g. https://api.example.com. The callback becomes <base>/social_webhook/<provider>. Must be reachable from the public internet: in development this is a tunnel address, and leaving it empty disables every push transport (notification transports still work). Env SOCIAL_WEBHOOK_CALLBACK_BASE_URL."
        },
    )

    social_webhook_lease_seconds: int = field(
        default=432000,
        metadata={
            "description": "Lease length requested from a WebSub hub, in seconds. A lease that lapses stops delivering silently, so the renewal task refreshes each one before this elapses. Env SOCIAL_WEBHOOK_LEASE_SECONDS."
        },
    )

    social_subscription_renewal_interval_seconds: float = field(
        default=3600.0,
        metadata={
            "description": "How often the renewal task looks for leases about to expire. This is a timer over stored expiry times, not a poll of any platform. Env SOCIAL_SUBSCRIPTION_RENEWAL_INTERVAL_SECONDS."
        },
    )

    social_subscription_renewal_margin_seconds: float = field(
        default=86400.0,
        metadata={
            "description": "How long before a lease expires it is renewed. Generous on purpose: a missed renewal is invisible until the owner notices content has stopped arriving. Env SOCIAL_SUBSCRIPTION_RENEWAL_MARGIN_SECONDS."
        },
    )

    social_crawl_max_items_free: int = field(
        default=5,
        metadata={
            "description": "Most items the initial crawl may ingest for a free-tier owner. Every item costs a transcription or description call, so this is a spending limit rather than a quality setting. Env SOCIAL_CRAWL_MAX_ITEMS_FREE."
        },
    )

    social_crawl_max_items_pro: int = field(
        default=50,
        metadata={
            "description": "Most items the initial crawl may ingest for a pro-tier owner. Env SOCIAL_CRAWL_MAX_ITEMS_PRO."
        },
    )

    social_crawl_max_items_premium: int = field(
        default=300,
        metadata={
            "description": "Most items the initial crawl may ingest for a premium-tier owner. Env SOCIAL_CRAWL_MAX_ITEMS_PREMIUM."
        },
    )

    social_crawl_max_depth: int = field(
        default=3,
        metadata={
            "description": "How many levels out from the account's own profile the breadth-first crawl may walk. Each level multiplies, and the material actually belonging to the person is concentrated in the first two. Env SOCIAL_CRAWL_MAX_DEPTH."
        },
    )

    social_crawl_max_nodes: int = field(
        default=200,
        metadata={
            "description": "Ceiling on addresses one crawl may visit, counting pages that are judged and then pruned. Bounds the relevance calls, which are paid for whether or not the page is kept. Env SOCIAL_CRAWL_MAX_NODES."
        },
    )

    social_crawl_relevance_minimum_score: float = field(
        default=0.5,
        metadata={
            "description": "How strongly a page must belong to the avatar's person before it is ingested and its links followed, from 0 to 1. Raising it makes the crawl stricter about what counts as the person's own content. Env SOCIAL_CRAWL_RELEVANCE_MINIMUM_SCORE."
        },
    )

    require_social_proof_for_sharing: str = field(
        default="false",
        metadata={
            "description": "Whether sharing a personal avatar requires at least one social account proven to belong to the owner. Sharing a likeness is a claim about a real person, and a proven account is what backs that claim. Turning this on refuses sharing for an already-shared avatar that has no connected social account, so enable it deliberately. Env REQUIRE_SOCIAL_PROOF_FOR_SHARING."
        },
    )

    imap_idle_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether connected mailboxes are watched with IMAP IDLE, so a new message wakes triage in seconds instead of at the next poll. Servers that refuse IDLE fall back to the interval poll on their own. Env IMAP_IDLE_ENABLED."
        },
    )

    imap_idle_refresh_seconds: float = field(
        default=1500.0,
        metadata={
            "description": "How often an IDLE connection is renewed. The IMAP specification requires a client to re-issue IDLE at least every 29 minutes, so this stays below that. Env IMAP_IDLE_REFRESH_SECONDS."
        },
    )

    twitch_client_id: str = field(
        default=None,
        metadata={
            "description": "Twitch application client id, used to register EventSub subscriptions for the owner's own channel. Empty means Twitch notices are read from the owner's mailbox instead. Env TWITCH_CLIENT_ID."
        },
    )

    twitch_app_access_token: str = field(
        default=None,
        metadata={
            "description": "Twitch application access token for EventSub registration. Env TWITCH_APP_ACCESS_TOKEN."
        },
    )

    twitch_eventsub_secret: str = field(
        default=None,
        metadata={
            "description": "Fallback secret for verifying Twitch EventSub deliveries when a subscription carries none of its own. Env TWITCH_EVENTSUB_SECRET."
        },
    )

    meta_webhook_verify_token: str = field(
        default=None,
        metadata={
            "description": "Token Meta echoes when verifying the Instagram and Facebook webhook callback. Env META_WEBHOOK_VERIFY_TOKEN."
        },
    )

    meta_app_secret: str = field(
        default=None,
        metadata={
            "description": "Meta application secret, used to verify the signature on Instagram and Facebook deliveries. Env META_APP_SECRET."
        },
    )

    """ </Content subscriptions: the personal avatar's own accounts, crawled and subscribed> """

    """ </Connected accounts (mailbox and social) for the personal avatar> """

    """ <Emotion media generation (xAI images and idle-loop videos)> """

    xai_api_key: str = field(
        default=None,
        metadata={
            "description": "Bearer key for the xAI API, used to derive an avatar's six emotion stills from its reference image and to animate each still into a six-second idle loop. Leaving this unset disables emotion media generation without affecting anything else. Env XAI_API_KEY."
        },
    )

    emotion_media_generation_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether emotion stills and idle-loop videos may be generated from the reference image. Idle-loop videos (and the stills they need) wait for POST /avatar_emotion_media/regenerate. Uploading a reference image does not generate stills. Set to false to skip generation (for example in a test environment) while keeping the rest of the media pipeline. Env EMOTION_MEDIA_GENERATION_ENABLED."
        },
    )

    emotion_media_minimum_tier: str = field(
        default="premium",
        metadata={
            "description": "Lowest subscription tier whose owner may generate an avatar's emotion stills and idle loops from the explicit regenerate control — free, pro, or premium (premium is the enterprise-grade tier today). A tier below this value is refused by POST /avatar_emotion_media/regenerate. Env EMOTION_MEDIA_MINIMUM_TIER."
        },
    )

    xai_image_edit_model: str = field(
        default="grok-imagine-image-2.0",
        metadata={
            "description": "xAI image-editing model that turns the reference image into an emotion still. Env XAI_IMAGE_EDIT_MODEL."
        },
    )

    xai_image_cost_per_image_usd: float = field(
        default=0.04,
        metadata={
            "description": "Vendor price of one generated image, recorded per call in api_metrics as inference_type image_generation. Env XAI_IMAGE_COST_PER_IMAGE_USD."
        },
    )

    xai_video_model: str = field(
        default="grok-imagine-video-1.5",
        metadata={
            "description": "xAI image-to-video model that animates an emotion still into an idle loop. Env XAI_VIDEO_MODEL."
        },
    )

    xai_video_cost_per_second_usd: float = field(
        default=0.08,
        metadata={
            "description": "Vendor price per second of generated video, recorded per call in api_metrics as inference_type video_generation. Env XAI_VIDEO_COST_PER_SECOND_USD."
        },
    )

    xai_idle_loop_duration_seconds: int = field(
        default=6,
        metadata={
            "description": "Length in seconds of each generated idle loop. The prompt requires the first and final frames to match the still so the loop plays seamlessly. Env XAI_IDLE_LOOP_DURATION_SECONDS."
        },
    )

    xai_video_resolution: str = field(
        default="720p",
        metadata={
            "description": "Resolution requested for idle loops: 480p, 720p, or 1080p. Env XAI_VIDEO_RESOLUTION."
        },
    )

    xai_video_aspect_ratio: str = field(
        default="9:16",
        metadata={
            "description": "Aspect ratio requested for idle loops, matching the portrait framing the voice-mode stage displays. Env XAI_VIDEO_ASPECT_RATIO."
        },
    )

    xai_video_poll_interval_seconds: float = field(
        default=5.0,
        metadata={
            "description": "Seconds between status checks while an idle loop renders. Env XAI_VIDEO_POLL_INTERVAL_SECONDS."
        },
    )

    xai_video_poll_timeout_seconds: float = field(
        default=600.0,
        metadata={
            "description": "Maximum seconds to wait for one idle loop before recording the generation as failed and moving on. Env XAI_VIDEO_POLL_TIMEOUT_SECONDS."
        },
    )

    """ </Emotion media generation (xAI images and idle-loop videos)> """

    """ <Remote URL fetching> """

    url_fetch_allow_private_hosts: str = field(
        default="false",
        metadata={
            "description": "Whether the API may download a URL whose host resolves to an address inside this deployment's network. Off in every deployment that faces the internet: a URL chosen by a search engine or a language model must never be able to reach the container's own services or the cloud metadata endpoint. Turn it on only in development, and only when a test fixture is served from localhost. Env URL_FETCH_ALLOW_PRIVATE_HOSTS."
        },
    )

    """ </Remote URL fetching> """

    """ <Deep research with web-based fact verification> """

    deep_research_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether an avatar's creator may run deep research on the avatar's subject (POST /avatar/{assistant_id}/deep_research). Set to false to disable the research button and the endpoint without affecting the rest of the media pipeline. Env DEEP_RESEARCH_ENABLED."
        },
    )

    tavily_api_key: str = field(
        default=None,
        metadata={
            "description": "Tavily search API key. When the key is set, Tavily results (which carry the page content) are merged with the DuckDuckGo results the browser tools read; without the key the research runs on DuckDuckGo alone. Env TAVILY_API_KEY."
        },
    )

    deep_research_max_topics: int = field(
        default=4,
        metadata={
            "description": "Maximum research topics one research job delegates to concurrent researchers. Env DEEP_RESEARCH_MAX_TOPICS."
        },
    )

    deep_research_max_queries: int = field(
        default=4,
        metadata={
            "description": "Maximum web search queries one researcher runs per round for one topic. Env DEEP_RESEARCH_MAX_QUERIES."
        },
    )

    deep_research_max_sources: int = field(
        default=12,
        metadata={
            "description": "Maximum source pages one researcher reads and extracts facts from per round. Env DEEP_RESEARCH_MAX_SOURCES."
        },
    )

    media_fact_verification_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the facts in freshly indexed media are extracted, cross-checked against what the avatar already holds, and verified — applying what agrees and holding contradictions for the owner to settle. Env MEDIA_FACT_VERIFICATION_ENABLED."
        },
    )

    media_fact_verification_max_documents: int = field(
        default=40,
        metadata={
            "description": "Ceiling on how many freshly indexed documents one media batch fact-checks. Extraction and judging both cost model calls, so a larger upload indexes everything and verifies the first this-many documents. Zero switches the verification off. Env MEDIA_FACT_VERIFICATION_MAX_DOCUMENTS."
        },
    )

    deep_research_max_media_items: int = field(
        default=24,
        metadata={
            "description": "Ceiling on how many verified sources one deep-research run sends through the media pipeline, where each one may be transcribed. Sources are ranked by how many verified facts each supported, so the cap keeps the ones the research leaned on most. Zero switches the media hand-off off entirely. Env DEEP_RESEARCH_MAX_MEDIA_ITEMS."
        },
    )

    deep_research_follow_up_rounds: int = field(
        default=1,
        metadata={
            "description": "How many further search rounds a researcher may run after reflecting that the topic is still unanswered. Zero searches once per topic. Env DEEP_RESEARCH_FOLLOW_UP_ROUNDS."
        },
    )

    deep_research_concurrency: int = field(
        default=4,
        metadata={
            "description": "Concurrent page reads and fact-extraction calls per researcher. Env DEEP_RESEARCH_CONCURRENCY."
        },
    )

    research_on_create_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether creating an avatar starts a deep-research job for that avatar on its own. Switching this off leaves the manual 'Research & verify facts' button working exactly as before; only the automatic trigger stops. Env RESEARCH_ON_CREATE_ENABLED."
        },
    )

    personal_avatar_research_on_naming_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether giving the personal avatar a real person's name starts one deep-research job for the account holder. Separate from RESEARCH_ON_CREATE_ENABLED because this trigger researches the account holder rather than a character the account holder invented. The run happens at most once per personal avatar, and a name that is only the email local part never starts one. Env PERSONAL_AVATAR_RESEARCH_ON_NAMING_ENABLED."
        },
    )

    conversation_starters_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the standard set of three conversation starters is generated once per avatar and stored on the assistant record (metadata.conversation_starters): on avatar creation, again when deep research finishes, and on the owner's request through POST /avatar/{assistant_id}/conversation_starters. Switching this off stops the generation and the storage; the browser then paints its local identity-leaned starters. Env CONVERSATION_STARTERS_ENABLED."
        },
    )

    research_account_discovery_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether a finished research run asks the owner, through the agent inbox, about the accounts that run turned up. Switching this off leaves the research itself untouched and stops only the questions. Env RESEARCH_ACCOUNT_DISCOVERY_ENABLED."
        },
    )

    research_account_discovery_maximum: int = field(
        default=3,
        metadata={
            "description": "Ceiling on how many account questions one research run may put in the owner's inbox. Each question is one item waiting on the owner, so the ceiling is about the owner's attention rather than about cost. Env RESEARCH_ACCOUNT_DISCOVERY_MAXIMUM."
        },
    )

    conversation_training_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether what a person types or says to Neural Nexus is used, like anything else that person gives the product, to build that person's own avatar. The words are read from the conversations already stored and never copied anywhere; switching this off stops the reading. Env CONVERSATION_TRAINING_ENABLED."
        },
    )

    speaker_quote_minimum_characters: int = field(
        default=40,
        metadata={
            "description": "Shortest message that counts as a quote of the person who wrote it. A turn below this length is read past when the person's own words are gathered, so 'ok' and 'yes' never reach the style profile or the training data. Env SPEAKER_QUOTE_MINIMUM_CHARACTERS."
        },
    )

    deep_research_bootstrap_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether a deep-research run may acquire the reference image and reference audio an avatar is missing. Switching this off leaves fact research untouched and stops every image download, video probe, and transcription the acquisition would have started. Env DEEP_RESEARCH_BOOTSTRAP_ENABLED."
        },
    )

    deep_research_bootstrap_max_image_candidates: int = field(
        default=8,
        metadata={
            "description": "Ceiling on how many candidate photographs one avatar's acquisition may download and put in front of the vision model before giving up on finding a portrait. Each candidate costs one download and one vision call. Env DEEP_RESEARCH_BOOTSTRAP_MAX_IMAGE_CANDIDATES."
        },
    )

    deep_research_bootstrap_image_max_bytes: int = field(
        default=10485760,
        metadata={
            "description": "Largest candidate photograph the acquisition will download, in bytes. Deliberately tighter than the 25 MiB ceiling on a person's own upload, because this URL came from a search engine rather than from the account holder. Env DEEP_RESEARCH_BOOTSTRAP_IMAGE_MAX_BYTES."
        },
    )

    deep_research_bootstrap_min_portrait_confidence: float = field(
        default=0.7,
        metadata={
            "description": "How sure the vision model must be that a candidate photograph shows the named subject before that photograph becomes the avatar's reference image. Below this the candidate is rejected and the next one is tried. Env DEEP_RESEARCH_BOOTSTRAP_MIN_PORTRAIT_CONFIDENCE."
        },
    )

    deep_research_bootstrap_max_video_candidates: int = field(
        default=6,
        metadata={
            "description": "Ceiling on how many candidate videos one avatar's acquisition may probe for duration and title before choosing the recording to learn the voice from. Each probe is a yt_dlp metadata call and downloads nothing. Env DEEP_RESEARCH_BOOTSTRAP_MAX_VIDEO_CANDIDATES."
        },
    )

    deep_research_bootstrap_min_video_seconds: float = field(
        default=120.0,
        metadata={
            "description": "Shortest candidate video the acquisition will accept as the voice source. A clip below this rarely holds enough of the subject speaking to seed a voice model. Env DEEP_RESEARCH_BOOTSTRAP_MIN_VIDEO_SECONDS."
        },
    )

    deep_research_bootstrap_max_video_seconds: float = field(
        default=2400.0,
        metadata={
            "description": "Longest candidate video the acquisition will accept as the voice source. This is the ceiling on what one automatically created avatar may cost in transcription and diarization, so it is the main cost bound on acquisition running for every tier. Env DEEP_RESEARCH_BOOTSTRAP_MAX_VIDEO_SECONDS."
        },
    )

    deep_research_bootstrap_media_wait_seconds: float = field(
        default=900.0,
        metadata={
            "description": "How long the acquisition waits for the media job it started to finish before reporting the asset as still pending. The media job carries on regardless; this only decides when the research run stops watching it. Env DEEP_RESEARCH_BOOTSTRAP_MEDIA_WAIT_SECONDS."
        },
    )

    """ </Deep research with web-based fact verification> """

    """ <Voice cloning and speech (ElevenLabs)> """

    elevenlabs_api_key: str = field(
        default=None,
        metadata={
            "description": "ElevenLabs key used for instant and professional voice cloning, speech synthesis, and lip-sync video. Pro tier or higher is required for API access and one professional-clone slot. Leaving this unset disables every voice feature without affecting anything else. Env ELEVENLABS_API_KEY."
        },
    )

    elevenlabs_instant_voice_clone_minimum_seconds: float = field(
        default=60.0,
        metadata={
            "description": "Seconds of the avatar's own speech required before an instant voice clone is created. Env ELEVENLABS_INSTANT_VOICE_CLONE_MINIMUM_SECONDS."
        },
    )

    elevenlabs_instant_voice_clone_target_seconds: float = field(
        default=120.0,
        metadata={
            "description": "Seconds of speech the instant clone is rebuilt from once available; a clone built from less is replaced when the corpus reaches this. Non-personal avatars stop collecting here. Env ELEVENLABS_INSTANT_VOICE_CLONE_TARGET_SECONDS."
        },
    )

    elevenlabs_professional_voice_clone_minimum_seconds: float = field(
        default=1800.0,
        metadata={
            "description": "Seconds of the personal avatar's own speech required before a professional voice clone is prepared for verification and training. Env ELEVENLABS_PROFESSIONAL_VOICE_CLONE_MINIMUM_SECONDS."
        },
    )

    elevenlabs_professional_voice_clone_maximum_seconds: float = field(
        default=10800.0,
        metadata={
            "description": "Ceiling on the personal avatar's voice corpus; clips beyond this are not stored. Env ELEVENLABS_PROFESSIONAL_VOICE_CLONE_MAXIMUM_SECONDS."
        },
    )

    elevenlabs_professional_voice_clone_training_model: str = field(
        default="eleven_multilingual_v2",
        metadata={
            "description": "Model the professional clone is trained against. Env ELEVENLABS_PROFESSIONAL_VOICE_CLONE_TRAINING_MODEL."
        },
    )

    professional_voice_clone_poll_interval_seconds: float = field(
        default=300.0,
        metadata={
            "description": "Seconds between checks on a professional clone that is training (training takes three to six hours). Env PROFESSIONAL_VOICE_CLONE_POLL_INTERVAL_SECONDS."
        },
    )

    elevenlabs_text_to_speech_model: str = field(
        default="eleven_flash_v2_5",
        metadata={
            "description": "Speech model the cloned voice is rendered with for the speak button and voice mode. Flash is the low-latency choice; eleven_v3 is more expressive and slower. Env ELEVENLABS_TEXT_TO_SPEECH_MODEL."
        },
    )

    elevenlabs_text_to_speech_cost_per_1000_characters_usd: float = field(
        default=0.05,
        metadata={
            "description": "Vendor price per thousand characters of speech, recorded per call in api_metrics as inference_type speech_synthesis. Env ELEVENLABS_TEXT_TO_SPEECH_COST_PER_1000_CHARACTERS_USD."
        },
    )

    elevenlabs_lip_sync_model: str = field(
        default="creatify-aurora",
        metadata={
            "description": "ElevenLabs lip-sync model (image + audio to video) used for voice-mode replies when video is enabled. Env ELEVENLABS_LIP_SYNC_MODEL."
        },
    )

    elevenlabs_lip_sync_resolution: str = field(
        default="720p",
        metadata={
            "description": "Resolution requested for lip-sync clips: 480p or 720p. Env ELEVENLABS_LIP_SYNC_RESOLUTION."
        },
    )

    elevenlabs_lip_sync_cost_per_second_usd: float = field(
        default=0.14,
        metadata={
            "description": "Estimated vendor price per second of lip-sync video, recorded per clip in api_metrics as inference_type lip_sync. Env ELEVENLABS_LIP_SYNC_COST_PER_SECOND_USD."
        },
    )

    elevenlabs_lip_sync_poll_interval_seconds: float = field(
        default=5.0,
        metadata={
            "description": "Seconds between status checks while a lip-sync clip renders. Env ELEVENLABS_LIP_SYNC_POLL_INTERVAL_SECONDS."
        },
    )

    lip_sync_enabled: str = field(
        default="true",
        metadata={
            "description": "Process-wide switch for lip-sync video generation; tier capability still applies per user. Env LIP_SYNC_ENABLED."
        },
    )

    voice_mode_capture_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "Whether speaking to a personal avatar in voice mode accrues the owner's speech to that avatar's voice corpus. Accrual additionally requires a stored reference audio clip and the owner's consent. Env VOICE_MODE_CAPTURE_ENABLED."
        },
    )

    voice_mode_capture_min_segment_seconds: float = field(
        default=1.0,
        metadata={
            "description": "Shortest owner segment in a spoken turn that is worth keeping for the voice corpus. Shorter windows are mostly onset and release and make a clone worse. Env VOICE_MODE_CAPTURE_MIN_SEGMENT_SECONDS."
        },
    )

    voice_mode_capture_max_seconds_per_turn: float = field(
        default=30.0,
        metadata={
            "description": "Most seconds one spoken turn may contribute to the voice corpus, so a single long monologue cannot dominate the clone. Env VOICE_MODE_CAPTURE_MAX_SECONDS_PER_TURN."
        },
    )

    instant_voice_refresh_at_target_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "Whether an instant voice clone built from less than ELEVENLABS_INSTANT_VOICE_CLONE_TARGET_SECONDS is rebuilt once, the first time the corpus reaches that target. Env INSTANT_VOICE_REFRESH_AT_TARGET_ENABLED."
        },
    )

    """ </Voice cloning and speech (ElevenLabs)> """

    """ <Agent inbox (triage of incoming messages for the personal avatar)> """

    inbox_poll_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the API polls every connected mailbox on a schedule and triages new mail through the inbox graph. Set to false to triage only on demand (POST /inbox/poll or the avatar's triage_inbox_now tool). Env INBOX_POLL_ENABLED."
        },
    )

    inbox_poll_interval_seconds: float = field(
        default=300.0,
        metadata={
            "description": "Seconds between scheduled inbox polls of every connected mailbox. Env INBOX_POLL_INTERVAL_SECONDS."
        },
    )

    inbox_fetch_max_messages: int = field(
        default=20,
        metadata={
            "description": "Ceiling on unseen messages fetched from one mailbox per poll, so a flooded inbox is triaged in batches rather than all at once. Env INBOX_FETCH_MAX_MESSAGES."
        },
    )

    inbox_auto_send_confidence: float = field(
        default=0.9,
        metadata={
            "description": "Confidence (alignment with the owner's recorded preferences times the decision prior) at or above which the inbox graph sends a drafted reply without asking the owner. Below it, the draft waits for the owner's approval. Env INBOX_AUTO_SEND_CONFIDENCE."
        },
    )

    """ </Agent inbox (triage of incoming messages for the personal avatar)> """

    """ <Group conversations (Slack, Discord, Twitch)> """

    group_conversation_enabled: str = field(
        default="true",
        metadata={
            "description": "Process-wide switch for the avatar taking part in group conversations. When this is not true, the /groups routes answer 404 and no bot can reach the triage graph. Env GROUP_CONVERSATION_ENABLED."
        },
    )

    group_conversation_concurrency: int = field(
        default=4,
        metadata={
            "description": "How many messages from one batch are decided in parallel. Each message is one graph run, so this bounds the model calls a single busy room can start at once. Env GROUP_CONVERSATION_CONCURRENCY."
        },
    )

    group_max_events_per_request: int = field(
        default=100,
        metadata={
            "description": "Ceiling on the messages one bot may send in a single batch; a larger batch is refused rather than truncated, so a bot is never left believing messages were decided when the messages were dropped. Env GROUP_MAX_EVENTS_PER_REQUEST."
        },
    )

    group_auto_respond_confidence: float = field(
        default=0.9,
        metadata={
            "description": "Confidence at or above which the avatar posts a reply in a room without being asked and without the owner seeing the reply first. Below it, the reply waits for the owner. Env GROUP_AUTO_RESPOND_CONFIDENCE."
        },
    )

    group_auto_moderate_confidence: float = field(
        default=0.97,
        metadata={
            "description": "Confidence at or above which the avatar carries out a moderation action without the owner. Higher than the reply threshold because moderating somebody is harder to undo than saying something. A timeout or a ban additionally requires the owner to have allowed that same action in that same room before, whatever this value is. Env GROUP_AUTO_MODERATE_CONFIDENCE."
        },
    )

    group_recent_events_for_triage: int = field(
        default=12,
        metadata={
            "description": "How many preceding messages from the room are handed to the classifier as context, so the avatar reads the room rather than one line out of context. Env GROUP_RECENT_EVENTS_FOR_TRIAGE."
        },
    )

    group_precedent_recall_limit: int = field(
        default=8,
        metadata={
            "description": "How many of the owner's rules and past decisions are retrieved by similarity for each message being decided. Env GROUP_PRECEDENT_RECALL_LIMIT."
        },
    )

    group_auto_react_confidence: float = field(
        default=0.75,
        metadata={
            "description": "Confidence at or above which the avatar adds an emoji reaction without being asked. Deliberately lower than the reply threshold: a reaction is cheap to be wrong about and is most of what makes somebody feel present in a room, whereas an avatar that only reacts when it is nearly certain reacts almost never. Env GROUP_AUTO_REACT_CONFIDENCE."
        },
    )

    group_follow_up_max_delay_seconds: int = field(
        default=86400,
        metadata={
            "description": "The longest the avatar may defer something it said it would come back to. A follow-up further out than this is clamped, because an avatar that resurfaces a week-old message reads as broken rather than conscientious. Env GROUP_FOLLOW_UP_MAX_DELAY_SECONDS."
        },
    )

    group_history_catch_up_messages: int = field(
        default=50,
        metadata={
            "description": "How many messages a bot reads back when the avatar joins a room or returns after being offline, so the avatar comes back knowing what it missed rather than starting blank. Env GROUP_HISTORY_CATCH_UP_MESSAGES."
        },
    )

    """ </Group conversations (Slack, Discord, Twitch)> """

    """ <Ambient vision (webcam / screen snapshots as hidden conversation context)> """

    ambient_capture_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether POST /message/{assistant_id} accepts ambient observations (ambient=true: webcam and screen snapshots described into hidden conversation context and triaged as ignore / respond / notify). Set to false to refuse them with 404. Env AMBIENT_CAPTURE_ENABLED."
        },
    )

    ambient_capture_min_interval_seconds: float = field(
        default=10.0,
        metadata={
            "description": "Server-side floor, in seconds, between two ambient observations on the same conversation thread; a faster client receives 429 with Retry-After. The browser paces itself with VITE_AMBIENT_CAPTURE_INTERVAL_SECONDS (default 30). Env AMBIENT_CAPTURE_MIN_INTERVAL_SECONDS."
        },
    )

    ambient_capture_max_image_bytes: int = field(
        default=2_000_000,
        metadata={
            "description": "Largest snapshot accepted as an ambient observation, in bytes per image; larger uploads receive 413. Env AMBIENT_CAPTURE_MAX_IMAGE_BYTES."
        },
    )

    ambient_preference_recall_limit: int = field(
        default=8,
        metadata={
            "description": "How many of the conversation partner's recorded ambient-observation decisions (dismissals, replies, notes on notification cards) are recalled by similarity from the store and handed to the triage classifier as precedent. Env AMBIENT_PREFERENCE_RECALL_LIMIT."
        },
    )

    ambient_respond_salience_floor: float = field(
        default=0.55,
        metadata={
            "description": "Least salience, from zero to one, at which a triaged ambient observation classified 'respond' is allowed to reach the avatar and become a spoken reply. A 'respond' below this floor is demoted to 'ignore': the observation is still kept as hidden conversation context, but the avatar says nothing. Raise this floor to make the avatar speak up less often about what the avatar notices. Env AMBIENT_RESPOND_SALIENCE_FLOOR."
        },
    )

    ambient_notify_salience_floor: float = field(
        default=0.70,
        metadata={
            "description": "Least salience, from zero to one, at which a triaged ambient observation classified 'notify' is allowed to become a notification card. This floor is higher than AMBIENT_RESPOND_SALIENCE_FLOOR because a card interrupts the conversation partner more than a spoken reply does. A 'notify' below this floor is demoted to 'ignore'. Env AMBIENT_NOTIFY_SALIENCE_FLOOR."
        },
    )

    ambient_respond_cooldown_seconds: float = field(
        default=300.0,
        metadata={
            "description": "Seconds of quiet required on one conversation thread after the avatar speaks about something the avatar noticed, before another ambient observation on that same thread may become a reply or a notification card. This cooldown bounds how often ambient vision can interrupt regardless of what the classifier decides; an observation refused by the cooldown is demoted to 'ignore' and kept as hidden context. Env AMBIENT_RESPOND_COOLDOWN_SECONDS."
        },
    )

    ambient_respond_cooldown_override_salience: float = field(
        default=0.90,
        metadata={
            "description": "Salience, from zero to one, at or above which an ambient observation ignores AMBIENT_RESPOND_COOLDOWN_SECONDS and reaches the conversation partner anyway. This exists so that something genuinely urgent the avatar notices is not silenced by a cooldown started by an ordinary remark. Env AMBIENT_RESPOND_COOLDOWN_OVERRIDE_SALIENCE."
        },
    )

    """ </Ambient vision (webcam / screen snapshots as hidden conversation context)> """

    """ <Scene narration (the accessibility mode: what is in view, read aloud)> """

    scene_narration_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the avatar may switch on scene narration, the accessibility mode in which the conversation partner's outward-facing camera is described continuously and every description is read aloud to them. Set to false to withhold the set_scene_narration tool from every turn; observations already flowing are unaffected. Env SCENE_NARRATION_ENABLED."
        },
    )

    scene_narration_min_interval_seconds: float = field(
        default=3.0,
        metadata={
            "description": "Server-side floor, in seconds, between two observations on one conversation thread while scene narration is on. Lower than AMBIENT_CAPTURE_MIN_INTERVAL_SECONDS on purpose: an ordinary ambient look is context the avatar may never use, while a narration look is an answer somebody is waiting for, often while walking. The browser paces itself with VITE_SCENE_NARRATION_INTERVAL_SECONDS. Env SCENE_NARRATION_MIN_INTERVAL_SECONDS."
        },
    )

    """ </Scene narration (the accessibility mode: what is in view, read aloud)> """

    """ <Motion wireframe (how the person moves)> """

    motion_learning_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether motion tracks (named body joints and the face mesh through time) are recorded and folded into the avatar's motion profile, from the live camera, from uploaded video, or from a decoder. Set to false to refuse every motion window while leaving the rest of the platform untouched. Env MOTION_LEARNING_ENABLED."
        },
    )

    motion_landmark_set_version: str = field(
        default="mediapipe_body33_face478_v1",
        metadata={
            "description": "The landmark set new tracks are written against (see src/anubis/utils/motion/landmarks.py). Old rows keep the version they were written with and stay decodable; bump this when a denser set (hands, iris, a decoder's limb set) is registered. Env MOTION_LANDMARK_SET_VERSION."
        },
    )

    motion_face_sample_rate_hz: float = field(
        default=30.0,
        metadata={
            "description": "Face-mesh frames per second recorded from uploaded video. Micro-expressions last 40 to 200 milliseconds, so a rate under about 25 Hz aliases them away. The browser paces itself with VITE_MOTION_FACE_FPS. Env MOTION_FACE_SAMPLE_RATE_HZ."
        },
    )

    motion_body_sample_rate_hz: float = field(
        default=15.0,
        metadata={
            "description": "Body-joint frames per second recorded from uploaded video; gestures are slower than expressions, so half the face rate is enough. The browser paces itself with VITE_MOTION_BODY_FPS. Env MOTION_BODY_SAMPLE_RATE_HZ."
        },
    )

    motion_track_window_seconds: float = field(
        default=10.0,
        metadata={
            "description": "How many seconds of motion one window holds. Windows are the unit every source sends and the unit primitives are cut from; the browser flushes one per VITE_MOTION_TRACK_WINDOW_SECONDS onto the ambient observation it already sends. Env MOTION_TRACK_WINDOW_SECONDS."
        },
    )

    motion_track_max_bytes: int = field(
        default=4_000_000,
        metadata={
            "description": "Largest motion window accepted from a client, in bytes of decoded buffers; a dense bootstrap window of ten seconds with the full face mesh at 30 Hz is about 900 KB, a basis-encoded window about 40 KB. Larger windows receive 413. Env MOTION_TRACK_MAX_BYTES."
        },
    )

    motion_track_retention_seconds_per_avatar: float = field(
        default=600.0,
        metadata={
            "description": "How many seconds of compact motion track to keep per avatar per emotion; older tracks are pruned first. At roughly 376 bytes a frame this is a few megabytes per avatar, smaller than one idle-loop clip. Env MOTION_TRACK_RETENTION_SECONDS_PER_AVATAR."
        },
    )

    motion_golden_seconds_per_avatar: float = field(
        default=120.0,
        metadata={
            "description": "How many seconds of raw, uncompressed face mesh to keep per avatar as the golden set: the ground truth the expression basis is fitted from, and what a denser landmark set or a future motion-transfer model would train on. Curated for identity confidence and emotion coverage, not recency. Env MOTION_GOLDEN_SECONDS_PER_AVATAR."
        },
    )

    motion_basis_components: int = field(
        default=48,
        metadata={
            "description": "How many principal components the per-avatar expression basis keeps. Forty-eight coefficients stand in for the 1,434 values of a 478-point mesh frame; the reconstruction error is recorded on the basis row. Env MOTION_BASIS_COMPONENTS."
        },
    )

    motion_basis_min_seconds: float = field(
        default=60.0,
        metadata={
            "description": "Seconds of golden face mesh needed before the first expression basis is fitted for an avatar. Until then dense windows are kept whole; the browser sends dense windows only during this bootstrap and coefficient windows afterwards. Env MOTION_BASIS_MIN_SECONDS."
        },
    )

    motion_primitive_max_count: int = field(
        default=8,
        metadata={
            "description": "How many recurring movements (primitives) to keep per avatar per emotion per channel; the least frequent are dropped when the dictionary is full. Env MOTION_PRIMITIVE_MAX_COUNT."
        },
    )

    motion_primitive_min_occurrences: int = field(
        default=3,
        metadata={
            "description": "How many times a recurring movement must have been seen before the prompt may describe it. Env MOTION_PRIMITIVE_MIN_OCCURRENCES."
        },
    )

    motion_signature_min_seconds: float = field(
        default=30.0,
        metadata={
            "description": "Seconds of motion a scalar measurement (blink rate, resting head tilt, gesture rate) must rest on before the prompt may state it; below this the reading stays in the signature but never reaches a prompt. Env MOTION_SIGNATURE_MIN_SECONDS."
        },
    )

    motion_identity_min_confidence: float = field(
        default=0.8,
        metadata={
            "description": "Least confidence, from zero to one, the vision comparison against the reference image must report before a camera or video window is attributed to the avatar's person. Anything below is discarded, never attributed. Env MOTION_IDENTITY_MIN_CONFIDENCE."
        },
    )

    motion_identity_reverify_seconds: float = field(
        default=600.0,
        metadata={
            "description": "How long a live-camera identity verification stays valid before the next window costs another vision comparison. Env MOTION_IDENTITY_REVERIFY_SECONDS."
        },
    )

    motion_analysis_max_video_seconds: float = field(
        default=600.0,
        metadata={
            "description": "Longest span of an uploaded video that is wireframed, in seconds; the rest of a longer video is skipped for motion (its transcript and other analyses are unaffected). Landmarking is local CPU, so this caps time rather than money. Env MOTION_ANALYSIS_MAX_VIDEO_SECONDS."
        },
    )

    motion_prompt_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the rendered motion block reaches the prompts that generate video and stills, and the avatar's HOW YOU MOVE section. Set to false to keep recording motion without letting the block drive anything. Env MOTION_PROMPT_ENABLED."
        },
    )

    lip_sync_prompt_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the lip-sync generation sends a prompt field: the cinematic foundation line plus the person's measured motion block, which is the vendor's behavioural channel (gestures, head, gaze, demeanour on creatify-aurora). A vendor 4xx naming the field falls back to the image-and-audio-only request once and records that on the job. Env LIP_SYNC_PROMPT_ENABLED."
        },
    )

    lip_sync_cinematic_prompt: str = field(
        default="Medium close-up of the person, same framing, lighting, clothing and background as the supplied image. No camera movement, no zoom, no new objects.",
        metadata={
            "description": "The stable first line of every lip-sync prompt: framing, lighting and what must not change. The person's measured motion block is appended under it as the behavioural layer. Env LIP_SYNC_CINEMATIC_PROMPT."
        },
    )

    motion_fidelity_check_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether a finished lip-sync or idle-loop clip is wireframed with the same extractor and scored against the person's motion profile, measurement by measurement and movement by movement, as motion_fidelity on the profile. Costs local CPU only. Env MOTION_FIDELITY_CHECK_ENABLED."
        },
    )

    motion_model_cache_dir: str = field(
        default="/tmp/anubis-motion-models",
        metadata={
            "description": "Directory the MediaPipe landmarker model files are downloaded to on first use for server-side wireframing of uploaded video. Env MOTION_MODEL_CACHE_DIR."
        },
    )

    motion_pose_model_url: str = field(
        default="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        metadata={
            "description": "Where the PoseLandmarker model file is fetched from on first use. Env MOTION_POSE_MODEL_URL."
        },
    )

    motion_face_model_url: str = field(
        default="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        metadata={
            "description": "Where the FaceLandmarker model file is fetched from on first use. Env MOTION_FACE_MODEL_URL."
        },
    )

    """ </Motion wireframe (how the person moves)> """

    """ Browsing Insights (what the owner's own web browsing says about the owner) """

    browsing_insights_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to read the owner's browsing history from their connected machines and learn facts and traits from it. Env BROWSING_INSIGHTS_ENABLED."
        },
    )

    browsing_insights_poll_seconds: int = field(
        default=300,
        metadata={
            "description": "How often, in seconds, the background browsing sweep asks each connected machine whether any new browsing has happened. The question costs one indexed count per browser profile and no model call. Env BROWSING_INSIGHTS_POLL_SECONDS."
        },
    )

    browsing_insights_minimum_new_visits: int = field(
        default=25,
        metadata={
            "description": "How many new page visits a machine must have recorded before a background pass spends a model call on them. A first pass over a machine ignores this. Env BROWSING_INSIGHTS_MINIMUM_NEW_VISITS."
        },
    )

    browsing_insights_minimum_seconds_between_analyses: int = field(
        default=900,
        metadata={
            "description": "The shortest gap, in seconds, between two background analyses of one machine's browsing, whatever the visit count. Env BROWSING_INSIGHTS_MINIMUM_SECONDS_BETWEEN_ANALYSES."
        },
    )

    browsing_insights_backfill_days: int = field(
        default=30,
        metadata={
            "description": "How many days of existing browsing history are read the first time a machine is analysed, so a newly connected machine teaches the avatar immediately. Env BROWSING_INSIGHTS_BACKFILL_DAYS."
        },
    )

    browsing_insights_max_visits_per_pass: int = field(
        default=2000,
        metadata={
            "description": "The most page visits one pass reads from one machine; the rest are left for the next pass rather than skipped. Env BROWSING_INSIGHTS_MAX_VISITS_PER_PASS."
        },
    )

    browsing_insights_max_digest_characters: int = field(
        default=24000,
        metadata={
            "description": "The size cap, in characters, of the browsing digest one analysis reads. The digest is trimmed from the end, so the counts and the searches survive a trim. Env BROWSING_INSIGHTS_MAX_DIGEST_CHARACTERS."
        },
    )

    browsing_insights_report_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to save a readable report to the reports list for every browsing pass that found something. Env BROWSING_INSIGHTS_REPORT_ENABLED."
        },
    )

    """ </Browsing Insights> """

    """ Continuous Learning & Personalization """

    learning_sweep_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to run the background learning sweep (conversation sentiment history, rating aggregation, preference inference) from the API lifespan. Env LEARNING_SWEEP_ENABLED."
        },
    )

    learning_sweep_interval_seconds: int = field(
        default=300,
        metadata={
            "description": "How often, in seconds, the background learning sweep wakes to look for idle accounts with unprocessed conversations. Env LEARNING_SWEEP_INTERVAL_SECONDS."
        },
    )

    learning_idle_seconds: int = field(
        default=600,
        metadata={
            "description": "An account is swept only once the user's newest message on that account is at least this many seconds old. Env LEARNING_IDLE_SECONDS."
        },
    )

    learning_prompt_retrieval_limit: int = field(
        default=10,
        metadata={
            "description": "Maximum records retrieved per learning section (feedback messages, rated messages, sentiment history, what feels real, preferences) for the system prompt. Env LEARNING_PROMPT_RETRIEVAL_LIMIT."
        },
    )

    ask_what_feels_real_after_messages: int = field(
        default=5,
        metadata={
            "description": "Once the user has sent this many messages to an avatar and nothing is recorded about what feels real to the user, the avatar naturally asks. Zero disables the question. Env ASK_WHAT_FEELS_REAL_AFTER_MESSAGES."
        },
    )

    conversation_sentiment_per_turn_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to refresh the structured sentiment summary of the current conversation on every user turn (one classification-model call per turn). Env CONVERSATION_SENTIMENT_PER_TURN_ENABLED."
        },
    )

    """ <Automatic conversation naming (sidebar titles written by the messaging service)> """

    conversation_title_enabled: str = field(
        default="TRUE",
        metadata={
            "description": "TRUE to let the messaging service name a conversation from its transcript: once when the conversation is started, and again when the reader leaves the conversation. FALSE leaves every thread unnamed unless the reader types a name. Env CONVERSATION_TITLE_ENABLED."
        },
    )

    conversation_title_max_characters: int = field(
        default=60,
        metadata={
            "description": "Longest automatic conversation name, in characters; a longer name is cut on a word boundary and ends with an ellipsis so the sidebar row cannot be overrun. Env CONVERSATION_TITLE_MAX_CHARACTERS."
        },
    )

    conversation_title_transcript_tail_messages: int = field(
        default=20,
        metadata={
            "description": "How many of the most recent visible messages the conversation namer reads. A conversation is named after what it is about, which the recent turns carry; reading the whole transcript would spend classification tokens on turns that cannot change the name. Env CONVERSATION_TITLE_TRANSCRIPT_TAIL_MESSAGES."
        },
    )

    conversation_title_timeout_seconds: float = field(
        default=20.0,
        metadata={
            "description": "How long the conversation namer waits on the classification model before giving up and leaving the conversation unnamed. Env CONVERSATION_TITLE_TIMEOUT_SECONDS."
        },
    )

    """ <Usage analytics (opt-in action log and described page captures)> """

    usage_analytics_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether the /usage_analytics/* routes accept action events and page captures from a browser whose account opted in. Set to false to refuse them with 404 while keeping consent records. Env USAGE_ANALYTICS_ENABLED."
        },
    )

    usage_analytics_capture_min_interval_seconds: float = field(
        default=10.0,
        metadata={
            "description": "Server-side floor, in seconds, between two page captures from the same browser session; a faster client receives 429 with Retry-After. The browser paces itself with VITE_USAGE_ANALYTICS_CAPTURE_INTERVAL_SECONDS (default 30). Env USAGE_ANALYTICS_CAPTURE_MIN_INTERVAL_SECONDS."
        },
    )

    usage_analytics_max_image_bytes: int = field(
        default=3_000_000,
        metadata={
            "description": "Largest page capture accepted, in bytes; larger uploads receive 413. Env USAGE_ANALYTICS_MAX_IMAGE_BYTES."
        },
    )

    usage_analytics_thumbnail_width: int = field(
        default=640,
        metadata={
            "description": "Width, in pixels, of the JPEG thumbnail kept beside each capture's description; the full-size capture is discarded after the description is made. Env USAGE_ANALYTICS_THUMBNAIL_WIDTH."
        },
    )

    usage_analytics_max_events_per_request: int = field(
        default=200,
        metadata={
            "description": "Largest batch of action events one POST /usage_analytics/events accepts. Env USAGE_ANALYTICS_MAX_EVENTS_PER_REQUEST."
        },
    )

    usage_analytics_retention_days: int = field(
        default=90,
        metadata={
            "description": "How many days of action events and page captures are kept per consenting user before the purge task deletes them; zero keeps everything. Env USAGE_ANALYTICS_RETENTION_DAYS."
        },
    )

    """ </Usage analytics (opt-in action log and described page captures)> """

    """ <Geo-located avatars (avatars pinned to a real-world place)> """

    geo_checkin_min_interval_seconds: int = field(
        default=300,
        metadata={
            "description": "Shortest interval, in seconds, between two recorded visits by the same person to the same geo-located avatar; POST /geo/checkin drops a repeat check-in inside this window so a moving phone cannot flood the visit records. Env GEO_CHECKIN_MIN_INTERVAL_SECONDS."
        },
    )

    geo_nearby_default_radius_meters: int = field(
        default=500,
        metadata={
            "description": "Search radius, in meters, used by GET /avatars/nearby and POST /geo/checkin when the caller does not ask for one. Env GEO_NEARBY_DEFAULT_RADIUS_METERS."
        },
    )

    geo_nearby_max_radius_meters: int = field(
        default=50_000,
        metadata={
            "description": "Largest search radius, in meters, that GET /avatars/nearby will honor; a larger request is capped at this value. Env GEO_NEARBY_MAX_RADIUS_METERS."
        },
    )

    geo_notify_cooldown_seconds: int = field(
        default=3600,
        metadata={
            "description": "Shortest interval, in seconds, between two proximity notifications naming the same geo-located avatar to the same person, so walking in and out of a geofence does not notify repeatedly. Env GEO_NOTIFY_COOLDOWN_SECONDS."
        },
    )

    """ </Geo-located avatars (avatars pinned to a real-world place)> """

    """ <Who is speaking (live voice: label utterances by speaker)> """

    voice_speaker_labels_enabled: str = field(
        default="true",
        metadata={
            "description": "Whether POST /message/{assistant_id} accepts diarize=true on an attached live-voice utterance: the utterance is transcribed with speaker labels (the owner by the voice-clone recordings, other people as Speaker N remembered per thread) instead of plain transcription. Env VOICE_SPEAKER_LABELS_ENABLED."
        },
    )
    voice_speaker_memory_max_speakers: int = field(
        default=3,
        metadata={
            "description": "How many other people (besides the owner) are remembered per conversation thread as reference clips so their Speaker N label stays stable across utterances. The diarizer accepts four known speakers per call, one of which is the owner. Env VOICE_SPEAKER_MEMORY_MAX_SPEAKERS."
        },
    )
    voice_speaker_min_segment_seconds: float = field(
        default=2.0,
        metadata={
            "description": "Shortest stretch of speech (seconds) a new voice must produce in one utterance before a clip of that voice is remembered for later labelling. Env VOICE_SPEAKER_MIN_SEGMENT_SECONDS."
        },
    )
    voice_speaker_reference_max_seconds: float = field(
        default=9.0,
        metadata={
            "description": "Length (seconds, at most ten) of the owner's reference clip cut from the voice-clone recordings and handed to the diarizer as the known owner voice. Env VOICE_SPEAKER_REFERENCE_MAX_SECONDS."
        },
    )

    voice_transcription_language: str = field(
        default="en",
        metadata={
            "description": "ISO-639-1 language hint (for example en) passed to the speech model for live-voice utterances (POST /transcribe and diarize=true turns). A language hint stops the model from inventing captions in other languages for silent or noisy clips. The value none (or auto) lets the model guess the language. Uploaded media is never affected. Env VOICE_TRANSCRIPTION_LANGUAGE."
        },
    )
    voice_transcription_prompt: str = field(
        default="",
        metadata={
            "description": "Optional text prompt handed to whisper-1 for live-voice utterances (POST /transcribe) to steer spelling and vocabulary; the diarization model does not accept a prompt. Keep the prompt to a comma-separated list of names and terms (Neural Nexus, webcam, screen share). Never put example sentences or questions in the prompt: whisper-1 treats the prompt as the previous transcript and, on a clip with no clear speech, returns those sentences as if the person had said them (\"What is on my screen right now?\"). A transcript made only of the prompt's words is dropped as an echo. Empty sends none. Env VOICE_TRANSCRIPTION_PROMPT."
        },
    )
    voice_no_speech_probability_max: float = field(
        default=0.6,
        metadata={
            "description": "Live-voice transcripts are requested from whisper-1 as verbose_json; a segment whose no_speech_prob is at or above this value is dropped as noise the model captioned anyway. 0 disables the check. Env VOICE_NO_SPEECH_PROBABILITY_MAX."
        },
    )
    voice_average_logprob_min: float = field(
        default=-1.0,
        metadata={
            "description": "A live-voice whisper segment whose avg_logprob is at or below this value (the model was guessing the words) is dropped. 0 disables the check. Env VOICE_AVERAGE_LOGPROB_MIN."
        },
    )
    voice_compression_ratio_max: float = field(
        default=2.4,
        metadata={
            "description": "A live-voice whisper segment whose text compression_ratio is at or above this value (a caption repeated over and over) is dropped. 0 disables the check. Env VOICE_COMPRESSION_RATIO_MAX."
        },
    )
    voice_silence_max_volume_db: float = field(
        default=-45.0,
        metadata={
            "description": "Live-voice clips whose loudest sample (dBFS, measured with ffmpeg volumedetect) is below this value hold no speech and are never sent to the speech model, which would otherwise invent a caption for the silence. Spoken words with browser gain control peak well above -30 dBFS. A value of 0 or higher disables the gate. Env VOICE_SILENCE_MAX_VOLUME_DB."
        },
    )

    """ </Who is speaking (live voice: label utterances by speaker)> """


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

    admin_metering_bypass_identifiers: str = field(
        default=None,
        metadata={
            "description": (
                "Comma-separated metering identifiers that skip usage enforcement "
                "and metering writes exactly like admin_user_id, for testing flows "
                "that admin_user_id cannot cover. Anonymous requesters have no "
                "account, so an entry is the hashed IP that appears in "
                "identities[0].user_id (sha256 of the x-forwarded-for value); "
                "authenticated user ids are accepted too. Leave EMPTY in "
                "production: every listed identifier is unmetered and unenforced."
            )
        },
    )

    dev_metered_enforcement_bypass_identifiers: str = field(
        default=None,
        metadata={
            "description": (
                "Comma-separated metering identifiers that skip usage ENFORCEMENT "
                "(the 402 exhausted-allotment refusal and the 429 token rate limit) "
                "while still being metered to Stripe and to api_metrics, unlike "
                "admin_metering_bypass_identifiers which also suppresses those "
                "writes. Intended for driving the anonymous free-tier flows past "
                "the allotment during development while the customer portal, "
                "/verify_subscription_status and the SSE usage frames keep "
                "advancing in step. An entry is the hashed IP that appears in "
                "identities[0].user_id (sha256 of the x-forwarded-for value); "
                "authenticated user ids are accepted too. Honored ONLY when "
                "DEV=TRUE, so a leftover entry is inert in production."
            )
        },
    )

    unrestricted_anonymous_messaging_avatar_identifiers: str = field(
        default=None,
        metadata={
            "description": (
                "Comma-separated avatar (assistant) identifiers that anonymous "
                "visitors may message without usage ENFORCEMENT — neither the 402 "
                "exhausted-allotment refusal nor the 429 token rate limit applies "
                "to an anonymous request aimed at one of these avatars. Every such "
                "turn is STILL metered to Stripe and to api_metrics, so the cost of "
                "the demonstration stays visible; only the refusals are lifted. The "
                "exemption is keyed on the avatar rather than on the requester "
                "because a public demonstration avatar is messaged by visitors whose "
                "hashed IP is not known in advance, and the exemption is limited to "
                "anonymous requesters so an authenticated account can never obtain "
                "unlimited free messaging by aiming at a listed avatar. Unlike "
                "dev_metered_enforcement_bypass_identifiers this list is honored in "
                "production, which is the point: leave EMPTY unless a listed avatar "
                "is genuinely intended to answer unlimited anonymous traffic."
            )
        },
    )

    unrestricted_metered_account_identifiers: str = field(
        default=None,
        metadata={
            "description": (
                "Comma-separated identifiers of accounts that are UNCAPPED "
                "WITHIN THEIR TIER: the HTTP 402 exhausted-allotment refusal and "
                "the HTTP 429 token rate limit stop applying, so a listed account "
                "may run past the allotment of whatever tier the account holds. "
                "The tier itself is NOT changed and no capability is granted — "
                "the HTTP 403 tier-capability gate still applies in full, so a "
                "listed account on the free tier is refused uploads exactly like "
                "any other free-tier account and reaches uploads by changing "
                "tier, which a listed account is free to do at any time. "
                "Every token is STILL metered to Stripe and to api_metrics, so "
                "the cost of demonstrating and testing the product stays visible "
                "wherever real usage appears. An entry is preferably the "
                "account's email address, because Auth0 mints a NEW user id "
                "whenever an account is deleted and signs up again while the "
                "email address does not change; the prefixed "
                "'auth0|<subject>' user id and the bare subject are both "
                "accepted as well, because those two spellings are already used "
                "side by side (resolve_metering_user_id returns the prefixed "
                "form, while admin_user_id and every avatar-ownership check use "
                "the bare form) and an entry written in either spelling has to "
                "work. An email entry matches only when the account's email "
                "address is verified, so an unverified account claiming a listed "
                "address cannot inherit the exemption. Anonymous requesters "
                "never match: the exemptions written for anonymous traffic are "
                "admin_metering_bypass_identifiers, "
                "dev_metered_enforcement_bypass_identifiers and "
                "unrestricted_anonymous_messaging_avatar_identifiers. Unlike "
                "dev_metered_enforcement_bypass_identifiers this list is honored "
                "in production, which is the whole purpose: a demonstration "
                "account has to work against the deployed API, which runs "
                "DEV=FALSE. Leave EMPTY unless an account is genuinely intended "
                "to be free of every limit, and expect to pay for the usage that "
                "account meters."
            )
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

    stripe_usage_source_of_truth_enabled: str = field(
        default="TRUE",
        metadata={
            "description": (
                "TRUE (the default) reads period usage for allotment gating and "
                "every usage display from Stripe's Billing Meter aggregation — "
                "the same number the customer portal shows — using the local "
                "api_metrics sum only as a floor for usage Stripe has not "
                "finished aggregating, and as the fallback when Stripe cannot "
                "be read. FALSE returns to local-only accounting, which drifts "
                "from the portal whenever an api_metrics insert fails while the "
                "Stripe meter event succeeds."
            )
        },
    )

    stripe_usage_cache_ttl_seconds: int = field(
        default=60,
        metadata={
            "description": (
                "Maximum age, in seconds, of a cached Stripe usage reading per "
                "(customer, meter, usage period). Allotment enforcement runs on "
                "the message hot path, so this bounds how often a message turn "
                "pays for a Stripe usage call; zero disables the cache and "
                "reads Stripe on every metered request."
            )
        },
    )

    portal_usage_event_url: str = field(
        default=None,
        metadata={
            "description": (
                "Customer portal endpoint that receives a usage event after each "
                "metered turn, so the portal can show usage immediately instead "
                "of waiting for Stripe's meter aggregation (for example "
                "http://host.docker.internal:8200/internal/usage-event). Delivery "
                "is fire-and-forget and fail-open; leaving this empty disables "
                "the push entirely and the portal falls back to reading Stripe "
                "on its own schedule. Env PORTAL_USAGE_EVENT_URL."
            )
        },
    )

    portal_usage_event_secret: str = field(
        default=None,
        metadata={
            "description": (
                "Shared secret signing usage events sent to "
                "portal_usage_event_url, as an HMAC-SHA256 over "
                "'<timestamp>.<body>' — the same construction Stripe uses for "
                "webhook signatures. Must match the portal's "
                "USAGE_EVENT_SHARED_SECRET exactly or every event is rejected. "
                "Empty disables the push. Env PORTAL_USAGE_EVENT_SECRET."
            )
        },
    )

    billing_portal_exchange_secret: str = field(
        default=None,
        metadata={
            "description": (
                "Shared secret for customer-portal single sign-on. It signs the "
                "short-lived exchange codes issued by "
                "/create_billing_portal_exchange_code and authenticates the "
                "portal's call to /redeem_billing_portal_exchange_code, as an "
                "HMAC-SHA256 over '<timestamp>.<body>'. Must match the portal's "
                "NN_EXCHANGE_SHARED_SECRET exactly. Empty disables single "
                "sign-on: both endpoints refuse and the portal shows its own "
                "sign-in card. Env BILLING_PORTAL_EXCHANGE_SECRET."
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
        default=50.6303957447559,
        metadata={
            "description": "Pre-calculated IQR threshold for the empirical representation of the squared mahalanobis distances of the features presented from the unmodified chatgpt responses using a leave-one-out method. Recalibrated and written back by scripts/retrain_chatgpt_baseline.py whenever the inference model is upgraded, and by data/build_baseline_features_arr.py whenever the feature vector changes (current: 28-wide v4 vector)."
        }
    )

    baseline_auto_retrain_on_model_change: bool = field(
        default=True,
        metadata={
            "description": "When TRUE (the default), an API boot whose MODEL differs from the model recorded in data/unmodified_inference_model_baseline_corpus.meta.json triggers ONE retrain of the unmodified-inference-model style baseline (scripts/retrain_chatgpt_baseline.py, run detached in this container) coordinated through a lock row in the shared LangGraph store, so every other container or checkout booting with that MODEL adopts the published result instead of regenerating. Set FALSE to only log the mismatch. Env BASELINE_AUTO_RETRAIN_ON_MODEL_CHANGE."
        },
    )

    baseline_auto_retrain_lock_stale_after_seconds: int = field(
        default=7200,
        metadata={
            "description": "Age in seconds after which a baseline_retrain_lock row in the store is considered abandoned (the container that took it died mid-retrain) and may be taken over by the next boot. A full retrain takes minutes, so two hours is generous. Env BASELINE_AUTO_RETRAIN_LOCK_STALE_AFTER_SECONDS."
        },
    )

    baseline_auto_retrain_poll_seconds: int = field(
        default=60,
        metadata={
            "description": "How often, in seconds, a boot that found another container already retraining the baseline for its MODEL re-reads the store provenance row, so it can adopt the published artifacts as soon as that retrain finishes. Env BASELINE_AUTO_RETRAIN_POLL_SECONDS."
        },
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
                # Prefer the field's own name (ELEVENLABS_API_KEY). Fall back to
                # the NN_ prefix some deployments use (NN_ELEVENLABS_API_KEY) so
                # a key that is only set under NN_ still configures the field —
                # otherwise voice speaks refuse with "Voice features are not
                # configured" even when the avatar already has a clone.
                env_val = os.environ.get(f.name.upper())
                if env_val is None or (
                    isinstance(env_val, str) and env_val.strip() == ""
                ):
                    env_val = os.environ.get(f"NN_{f.name.upper()}")

                # An env var that is declared but left empty (e.g. `MODEL_TOKEN_LIMIT=`
                # in .env) reads back as "" rather than being absent. Treat an
                # empty/whitespace-only string as "unset" and keep the field default,
                # so int("")/float("") coercion below cannot crash startup.
                if env_val is None or (
                    isinstance(env_val, str) and env_val.strip() == ""
                ):
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
