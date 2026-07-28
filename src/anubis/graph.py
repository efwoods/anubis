# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import asyncio
import logging
import math

logger = logging.getLogger(__name__)

# NOTE: ``retrieval_graph`` was imported here but never referenced in this module.
# Runtime logs showed ``from ... retrieval_graph import retrieval_graph`` alone took
# ~13 s per cold webapp worker because ``retrieval_graph`` pulls heavy retrieval deps.
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from src.anubis.utils.schema import RouteDecision
from src.anubis.utils.utility import (
    configure_assistant_context,
    extract_user_id_assistant_id,
)

logger = logging.getLogger(__name__)

# NOTE: ``langchain.agents.create_agent`` and ``SummarizationMiddleware`` were
# imported here and below but never referenced in this file (the only
# ``create_agent(...)`` site is commented out at ~line 313).  Removed to skip the
# eager ``langchain.agents`` package load on every cold start.
import json
import pickle
from typing import Any

import numpy as np
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.deep_agent import build_avatar_deep_agent
from src.anubis.utils.graph_interrupts import (
    build_interrupt_resume_command,
    collect_pending_interrupts,
)
from src.anubis.utils.emotion_mapping import EMOTION_MAPPING
from src.anubis.utils.huggingface_prefetch import (
    ensure_huggingface_models_cached,
)
from src.anubis.utils.model import STRUCTURED_OUTPUT_STREAM_TAG, init_model
from src.anubis.utils.nltk_prefetch import ensure_nltk_corpora_cached
from src.anubis.utils.nodes import load_consciousness, resolve_human_message_images
from src.anubis.utils.prompts.legal import PRIVACY_POLICY, TERMS_OF_SERVICE
from src.anubis.utils.runtime_handles import get_deep_agent_checkpointer
from src.anubis.utils.state import GlobalState
from src.anubis.utils.tools.browser import (
    get_browser_toolkit_tools,
    release_conversation_browser,
)
from src.anubis.utils.tools.data_analysis import (
    bound_connection_for,
    build_analysis_backend,
    build_connect_tool,
    build_data_analysis_tools,
    cleanup_analysis_workspace,
    collect_turn_artifacts,
    is_declined,
    mark_declined,
    read_user_connection,
    resolve_available_connection,
    save_user_connection,
)
from src.anubis.utils.utility import format_docs

""" NODES """

def _coalesce_ai_message(full: AIMessage | AIMessageChunk) -> AIMessage:
    """Merge streamed chunks into a single AIMessage for graph state."""
    if isinstance(full, AIMessage):
        return full
    return AIMessage(
        content=full.content,
        additional_kwargs=dict(full.additional_kwargs or {}),
        response_metadata=dict(full.response_metadata or {}),
        id=full.id,
        tool_calls=list(full.tool_calls or []),
        invalid_tool_calls=list(full.invalid_tool_calls or []),
    )


def _attach_token_usage_metadata(
    avatar_response: AIMessage, turn_messages: list
) -> None:
    """Fold the turn's aggregate token usage into ``response_metadata["token_usage"]``.

    Sums ``usage_metadata`` across every AI message the deep agent produced this
    turn (tool-planning calls consume tokens too, not just the final reply) and
    writes the total onto the final message's ``response_metadata`` in the
    ``token_usage`` shape the API metering layer reads (``prompt_tokens`` /
    ``completion_tokens`` / ``total_tokens``). ``usage_metadata`` itself is a
    message attribute that never survives the API layer's ``response_metadata``
    serialization, which is why the fold is necessary.
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for message in turn_messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        prompt_tokens += usage.get("input_tokens") or 0
        completion_tokens += usage.get("output_tokens") or 0
        total_tokens += usage.get("total_tokens") or 0
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    if total_tokens <= 0:
        return
    avatar_response.response_metadata = dict(avatar_response.response_metadata or {})
    avatar_response.response_metadata["token_usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _attach_go_emotions_metadata(avatar_response: AIMessage) -> None:
    """Mutate ``avatar_response.response_metadata`` with Go Emotions classifier output."""
    from src.anubis.utils.emotion_classifier import classify_go_emotions

    sentiment = classify_go_emotions(avatar_response.content)
    if sentiment is None:
        return
    avatar_response.response_metadata = dict(avatar_response.response_metadata or {})
    avatar_response.response_metadata.update({"sentiment": sentiment})


async def _attach_analyzed_features(avatar_response: AIMessage, runtime: Runtime[GlobalContext], assistant_id: str, user_id: str) -> None:
    """ analyze the avatar_response features, compare against unmodified chatgpt responses and any existing direct quotes if possible.
    Update the metadata with the feature analysis and the results of comparison.
    """
    import json

    from src.anubis.utils.dataset.style_features import (
        GROUND_TRUTH_FEATURES_DICT_KEY,
        baseline_feature_array_is_current,
        compute_mahalanobis_distance,
        deserialize_features_by_doc_id,
        extract_style_features,
        features_by_doc_id_to_arr,
        load_bundled_baseline_features_arr,
        sanitize_ground_truth_feature_matrix,
    )

    avatar_response.response_metadata = dict(avatar_response.response_metadata or {})

    # BASELINE KEY PHRASES: the ChatGPT baseline's self-discovered signature
    # phrases (the reference set the baseline matrix's key_phrase_rate column was
    # measured against). Cached under the namespace-root pattern the other
    # baseline artifacts use — ("baseline_key_phrase_profile",) with the same
    # key — as {"value": json.dumps(list)}; bundled JSON on disk is the fallback.
    baseline_key_phrase_profile_ITEM = await runtime.store.aget(
        ("baseline_key_phrase_profile",), key="baseline_key_phrase_profile"
    )
    baseline_key_phrases_str = (getattr(baseline_key_phrase_profile_ITEM, "value", None) or {}).get("value", None)

    if baseline_key_phrases_str:
        baseline_key_phrases = json.loads(baseline_key_phrases_str)
    else:
        # Load the bundled baseline key phrases from disk and cache in the store.
        from src.anubis.utils.dataset.style_features import BASELINE_KEY_PHRASES_PATH

        with open(BASELINE_KEY_PHRASES_PATH, encoding="utf-8") as fp:
            baseline_key_phrases = json.load(fp)
        await runtime.store.aput(
            ("baseline_key_phrase_profile",),
            key="baseline_key_phrase_profile",
            value={"value": json.dumps(baseline_key_phrases)},
        )

    features_dict = await asyncio.to_thread(
        extract_style_features,
        text=avatar_response.content,
        key_phrases=baseline_key_phrases,
    )
    # The metadata copy must be STRICT JSON: a degenerate reply (emoji-only,
    # all-punctuation) legitimately carries NaN cells, but json.dumps would
    # emit the bare NaN token, producing an invalid SSE "done" event. The raw
    # dict (NaN intact) still feeds the comparison math below.
    # key_phrase_rate is the one reference-set-relative feature, and the
    # reference set differs per comparison, so the features block carries a
    # description pair spelling out which phrase set applies where — the same
    # convention the two *_shap_values_description keys follow.
    avatar_response.response_metadata.update(
        {
            "features": {
                **{
                    name: (value if math.isfinite(value) else None)
                    for name, value in features_dict.items()
                },
                "key_phrase_rate_description": (
                    "The key_phrase_rate is the rate of detected avatar "
                    "signature key phrases per total word when compared "
                    "against the ground truth dataset (direct quotes of the "
                    "avatar), and is the rate of baseline ChatGPT signature "
                    "key phrases per total word when compared against the "
                    "baseline ChatGPT dataset."
                ),
            }
        }
    )

    features_arr = np.array(list(features_dict.values()))

    baseline_response_threshold = runtime.context.baseline_response_threshold

    try:
        # Baseline artifacts live at their namespace ROOT (namespace = key); the
        # per-avatar ground-truth artifacts are owner-scoped under
        # (user_id, assistant_id, <artifact_name>) with the artifact name as key.
        baseline_features_namespace = ("baseline_features_arr_list_str",)
        ground_truth_text_features_by_doc_id_namespace = (user_id, assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY)
        ground_truth_text_empirical_threshold_namespace = (user_id, assistant_id, "ground_truth_text_empirical_threshold_list_str")

        baseline_features_arr_list_str_ITEM = await runtime.store.aget(
            baseline_features_namespace, key="baseline_features_arr_list_str"
        )

        baseline_features_arr_list_str = (
            getattr(baseline_features_arr_list_str_ITEM, "value", None) or {}
        ).get("value", None)

        # If the baseline_features_arr has not yet been stored, store the array:
        if not baseline_features_arr_list_str:
            _BASELINE_ANSWERS_RESPONSES_ARR_DIR = (
                "src/anubis/utils/dataset/baseline_features_arr.npy"
            )
            baseline_features_arr = np.load(
                _BASELINE_ANSWERS_RESPONSES_ARR_DIR, allow_pickle=False
            )

            baseline_features_arr_list_str = json.dumps(baseline_features_arr.tolist())

            await runtime.store.aput(
                baseline_features_namespace,
                key="baseline_features_arr_list_str",
                value={"value": baseline_features_arr_list_str},
            )

        # Convert from str to np.array
        if isinstance(baseline_features_arr_list_str, str):
            baseline_features_arr = np.array(json.loads(baseline_features_arr_list_str))

        # Feature-version self-heal: an existing deployment may have cached a
        # previous-width baseline matrix in the store. Comparing a current-width
        # candidate row against it would raise on the shape mismatch, so reload
        # the freshly-bundled .npy and overwrite the stale cache.
        if not baseline_feature_array_is_current(baseline_features_arr):
            baseline_features_arr = load_bundled_baseline_features_arr()
            await runtime.store.aput(
                baseline_features_namespace,
                key="baseline_features_arr_list_str",
                value={"value": json.dumps(baseline_features_arr.tolist())},
            )

        # Compare the difference between the synthetic text and the unaltered chatgpt responses
        M_d_square_synth_from_baseline_chatgpt = await asyncio.to_thread(
            compute_mahalanobis_distance, features_arr, baseline_features_arr
        )

        # Explain the result
        from src.anubis.utils.utility import compute_shap_values_against_baseline

        shap_values_dict = await compute_shap_values_against_baseline(
            features_arr, runtime.store
        )

        # Nest the distance verdict together with the SHAP explanation under a single key
        # (verdict first to match the documented output order).
        comparison_to_unmodified_llm_response_analysis = {
            "no_statistically_significantly_difference_from_unmodified_llm_response_using_squared_mahalanobis_distance": bool(
                M_d_square_synth_from_baseline_chatgpt[0] <= baseline_response_threshold
            ),
            **shap_values_dict,
        }
        avatar_response.response_metadata.update(
            {
                "comparison_to_unmodified_llm_response_analysis": comparison_to_unmodified_llm_response_analysis
            }
        )

        # AVATAR SIGNATURE KEY PHRASES — the ``features`` block's
        # key_phrase_rate was measured against the ChatGPT BASELINE's phrases
        # (matching the baseline cloud it is compared to); here the same reply
        # is re-measured against the AVATAR's own discovered phrases to build
        # the ground-truth candidate row (matching the direct-quote cloud).
        # Loaded OUTSIDE the ground-truth-artifacts gate because the phrase
        # profile is written on every calibration, before the corpus reaches
        # the calibration floor. Stored phrases pass through
        # phrase_is_well_formed so sets polluted before discovery cleaned its
        # corpus never score here.
        from src.anubis.utils.dataset.key_phrases import phrase_is_well_formed

        key_phrase_profile_ITEM = await runtime.store.aget(
            (user_id, assistant_id, "key_phrase_profile"), key="key_phrase_profile"
        )
        avatar_key_phrases_str = (getattr(key_phrase_profile_ITEM, "value", None) or {}).get("value", None)
        avatar_key_phrases = json.loads(avatar_key_phrases_str) if avatar_key_phrases_str else None
        if avatar_key_phrases:
            avatar_key_phrases = [
                phrase for phrase in avatar_key_phrases if phrase_is_well_formed(phrase)
            ]

        # Swap ONLY key_phrase_rate to the avatar-referenced value; every other
        # feature is text-only and carries over from the baseline-scored dict.
        ground_truth_features_dict = await asyncio.to_thread(
            extract_style_features,
            avatar_response.content,
            key_phrases=avatar_key_phrases,
            update_key_phrases_only=True,
            features_dict=features_dict,
        )
        # Compare against ground truth quotes if available:
        ground_truth_text_features_model_namespace = (user_id, assistant_id, "ground_truth_text_features_model_b64_pkl")

        ground_truth_text_features_model_b64_pkl_ITEM = await runtime.store.aget(
            ground_truth_text_features_model_namespace,
            key="ground_truth_text_features_model_b64_pkl",
        )

        ground_truth_text_features_by_doc_id_ITEM = await runtime.store.aget(
            ground_truth_text_features_by_doc_id_namespace,
            key=GROUND_TRUTH_FEATURES_DICT_KEY,
        )

        ground_truth_text_empirical_threshold_list_str_ITEM = await runtime.store.aget(
            ground_truth_text_empirical_threshold_namespace,
            key="ground_truth_text_empirical_threshold_list_str",
        )

        ground_truth_text_features_model_b64_pkl = (
            getattr(ground_truth_text_features_model_b64_pkl_ITEM, "value", None) or {}
        ).get("value", None)

        ground_truth_text_features_by_doc_id_str = (
            getattr(ground_truth_text_features_by_doc_id_ITEM, "value", None) or {}
        ).get("value", None)

        ground_truth_text_empirical_threshold_list_str = (
            getattr(ground_truth_text_empirical_threshold_list_str_ITEM, "value", None)
            or {}
        ).get("value", None)

        # Reconstruct the (n_docs, len(FEATURE_NAMES)) corpus array from the
        # per-document dict, then sanitize: a corpus persisted before the
        # write-time all-NaN filter existed can still hold NaN cells, and the
        # StandardScaler inside compute_mahalanobis_distance (plus the
        # IsolationForest predict below) raises on NaN input.
        ground_truth_text_features_arr = sanitize_ground_truth_feature_matrix(
            features_by_doc_id_to_arr(
                deserialize_features_by_doc_id(ground_truth_text_features_by_doc_id_str)
            )
        )

        if ground_truth_text_features_arr.shape[0] > 0 and ground_truth_text_empirical_threshold_list_str and ground_truth_text_features_model_b64_pkl:
            # The ground-truth cloud's key_phrase_rate column was measured against
            # the AVATAR's discovered signature phrases, so the candidate row uses
            # the avatar-referenced dict computed above (key_phrase_rate swapped,
            # every other feature carried over from the baseline-scored dict).
            ground_truth_candidate_arr = np.array(list(ground_truth_features_dict.values()))

            import base64, shap, pandas as pd
            from src.anubis.utils.dataset.style_features import FEATURE_NAMES

            if isinstance(ground_truth_text_empirical_threshold_list_str, str):
                ground_truth_text_empirical_threshold = np.array(
                    json.loads(ground_truth_text_empirical_threshold_list_str)
                ).flatten()
            else:
                ground_truth_text_empirical_threshold = np.array(
                    ground_truth_text_empirical_threshold_list_str
                ).flatten()

            ground_truth_text_features_model = pickle.loads(
                base64.b64decode(ground_truth_text_features_model_b64_pkl)
            )


            # Feature-version self-heal: this per-avatar IsolationForest may have
            # been fit under a previous vector width and cached in the store. Unlike
            # the bundled ChatGPT baseline there is nothing to reload it from — it is
            # rebuilt only on the next media upload (calibrate_ground_truth) once the
            # corpus reaches MIN_ROWS_FOR_CALIBRATION current-width rows. Until then,
            # scoring a current-width candidate against it would raise, so skip the
            # ground-truth comparison cleanly rather than tripping the best-effort
            # handler with a noisy error every turn.
            ground_truth_model_feature_width = getattr(
                ground_truth_text_features_model, "n_features_in_", len(FEATURE_NAMES)
            )
            if ground_truth_model_feature_width != len(FEATURE_NAMES):
                logger.info(
                    "skipping ground-truth comparison: cached model width %s != "
                    "current %s; will rebuild on next media upload.",
                    ground_truth_model_feature_width,
                    len(FEATURE_NAMES),
                )
                return

            # Compute the difference between the synthetic text and the direct quotes.
            M_d_square_synth_from_ground_truth_corpus = await asyncio.to_thread(
                compute_mahalanobis_distance,
                ground_truth_candidate_arr,
                ground_truth_text_features_arr,
            )

            # Predict and explain the classification
            ground_truth_prediction = bool(
                await asyncio.to_thread(
                    ground_truth_text_features_model.predict,
                    ground_truth_candidate_arr.reshape(1, -1),
                )
                == 1
            )

            # KernelExplainer weights model.predict over EVERY background row per
            # explanation, so passing the full corpus (which can be thousands of
            # quote rows after calibration) makes this step effectively hang.
            # Summarize to a bounded background sample using kmeans clustering — standard SHAP practice.
            if ground_truth_text_features_arr.shape[0] > 100:
                shap_background = await asyncio.to_thread(
                    shap.kmeans, ground_truth_text_features_arr, 100
                )
            else:
                shap_background = ground_truth_text_features_arr
            explainer = shap.KernelExplainer(ground_truth_text_features_model.predict, shap_background)
            # KernelExplainer.shap_values re-runs model.predict across the background
            # sample — the dominant CPU cost of this block — so keep it off the loop.
            ground_truth_shap_values = await asyncio.to_thread(
                explainer.shap_values, ground_truth_candidate_arr.reshape(1, -1)
            )
            # shap_values for a single sample comes back shaped (1, n_features);
            # flatten to (n_features,) so it aligns with the FEATURE_NAMES index
            # (the DataFrame expects (n_features, 1), not (1, n_features)).
            ground_truth_shap_values = np.asarray(ground_truth_shap_values).ravel()
            ground_truth_shap_values_df = pd.DataFrame(
                data=ground_truth_shap_values,
                index=FEATURE_NAMES,
                columns=["ground_truth_shap_values"],
            )

            ground_truth_shap_values_dict = ground_truth_shap_values_df[
                ground_truth_shap_values_df["ground_truth_shap_values"] != 0
            ].to_dict()

            # Nest the distance verdict, SHAP explanation, and isolation-forest verdict under a single key.
            comparison_to_direct_quote_response_analysis = {
                "no_statistically_significant_difference_from_direct_quotes_using_squared_mahalanobis_distance": bool(
                    M_d_square_synth_from_ground_truth_corpus[0]
                    < ground_truth_text_empirical_threshold
                ),
                "no_statisically_significant_difference_between_sample_and_direct_quotes_dataset_according_to_isolation_forest": ground_truth_prediction,
                "ground_truth_comparison_isolation_forest_shap_values_description": "Negative values indicate dissimilarity from direct quotes dataset. Positive values indicate similarity to direct quotes. Scale is -1 to 1.",
                **ground_truth_shap_values_dict,
            }
            avatar_response.response_metadata.update(
                {
                    "comparison_to_direct_quote_response_analysis": comparison_to_direct_quote_response_analysis
                }
            )

    except Exception as e:
        # Post-stream analysis is best-effort: the user has already received the
        # reply, so a failure here must never fail the response. Log and keep
        # whatever metadata was attached before the error (features/sentiment,
        # and the baseline comparison if it got that far).
        logger.error(f"error analyzing features: {e}")


async def message_interface(
    state: MessagesState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> GlobalState:
    assistant_state = {}
    user_state = {}

    # Assert the user is loggedin and the assistant has an id from the config:
    # Otherwise use an anonymouse user id

    logger.info("breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        config
    )

    user_state.update(updated_user_state)
    assistant_state.update(updated_assistant_state)
    
    assistant_id = assistant_state.get("assistant_id", None)
    user_id = user_state.get("user_id", None)
    user_is_creator = state.get("user_is_creator", None)

    # verify the user is creator
    if (
        assistant_id is not None 
        and user_id is not None 
        and user_is_creator is None
    ):
        creator_id_dict = await runtime.store.aget((assistant_id, 'creator_id'), key='creator_id')
        creator_id = getattr(creator_id_dict,"value", {}).get("value", "")
        user_is_creator = user_id == creator_id

    return {
        "messages": state["messages"],
        "assistant_state": assistant_state,
        "user_state": user_state,
        "user_is_creator": user_is_creator
    }


# TODO: COERCE OUTPUT OF MESSAGE ONTO FINAL MESSAGE
async def terms_and_services_content_moderation(
    config: RunnableConfig, runtime: Runtime[GlobalContext]
):

    message = runtime.state["messages"][-1]

    class TermsAndServicesContentModeration(BaseModel):
        violation: bool = Field(
            description="If the user has violated the provided terms and services or the privacy policy then this value is TRUE. If the user has not violated the privacy policy or the terms and services then this value is FALSE"
        )
        reasoning: str = Field(
            description="This is the reason the user has violated the terms of service or the privacy policy. This is a clear reason. The exact terms of service violation or the exact violation of the privacy policy or both MUST be included and unaltered in any way."
        )

    TERMS_AND_SERVICES_CONTENT_MODERATION_SYSTEM_PROMPT = """
    <ROLE>
    You are an expert judge of identifying violations of the terms of service and privacy policy from human messages.
    </ROLE>
    
    <INSTRUCTIONS>
    Determine if the user MESSAGE is a violation of the included TERMS_OF_SERVICE or the PRIVACY_POLICY. 
    Include a reason the user MESSAGE has violated the terms of service or the privacy policy or both. 
    The reason the MESSAGE is a violation must be clear.
    The MESSAGE may not be in violation of the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a TRUE violation if the MESSAGE has violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a FALSE violation if the MESSAGE has NOT violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    MUST include EVERY exact line in the TERMS_OF_SERVICE or the PRIVACY_POLICY that give reason that the MESSAGE is a violation of either the TERMS_OF_SERVICE or the PRIVACY_POLICY whenever there is a violation in the MESSAGE of either or the TERMS_OF_SERVICE or the PRIVACY_POLICY or both.
    </INSTRUCTIONS>

    <TERMS_OF_SERVICE>
    {terms_of_service}
    </TERMS_OF_SERVICE>
    
    <PRIVACY_POLICY>
    {privacy_policy}
    </PRIVACY_POLICY>
    
    <INSTRUCTIONS>
    Determine if the user MESSAGE is a violation of the included TERMS_OF_SERVICE or the PRIVACY_POLICY. 
    Include a reason the user MESSAGE has violated the terms of service or the privacy policy or both. 
    The reason the MESSAGE is a violation must be clear.
    The MESSAGE may not be in violation of the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a TRUE violation if the MESSAGE has violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a FALSE violation if the MESSAGE has NOT violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    MUST include EVERY exact line in the TERMS_OF_SERVICE or the PRIVACY_POLICY that give reason that the MESSAGE is a violation of either the TERMS_OF_SERVICE or the PRIVACY_POLICY whenever there is a violation in the MESSAGE of either or the TERMS_OF_SERVICE or the PRIVACY_POLICY or both.
    </INSTRUCTIONS>
    
    <ROLE>
    You are an expert judge of identifying violations of the terms of service and privacy policy from human messages.
    </ROLE>
"""

    # TODO: CALCULATE TOKEN USAGE response['response_metadata']
    system_message = SystemMessage(
        content=TERMS_AND_SERVICES_CONTENT_MODERATION_SYSTEM_PROMPT.format(
            terms_of_service=TERMS_OF_SERVICE, privacy_policy=PRIVACY_POLICY
        )
    )
    # TODO: response_metrics_aggregation
    model_with_structured_output = init_model(
        model_without_tools=False, response_format=TermsAndServicesContentModeration
    )

    chat_prompt_template = [system_message] + [message]

    response = await model_with_structured_output.ainvoke(
        input=chat_prompt_template.messages
    )
    moderation_response = {
        "violation": response.violation,
        "reasoning": response.reasoning,
    }
    return {"moderation_response": moderation_response}


def _deep_agent_config(
    config: RunnableConfig, turn_key: int
) -> tuple[RunnableConfig, str | None]:
    """Derive the deep agent's own config + thread id from the outer config.

    The deep agent is checkpointed on a deterministic per-turn thread derived from
    ``"<outer-thread>::deepagent::<turn_key>"``. The ``turn_key`` (the outer
    conversation length) makes the thread **unique per turn** — so checkpointed
    deep-agent state doesn't accumulate across turns — while staying **stable across
    the interrupt → resume of the same turn** (no message is added while paused).
    The outer ``checkpoint_id``/``checkpoint_ns`` are dropped so the inner run isn't
    pinned to an outer checkpoint.

    The derived value is hashed through ``uuid5`` because the shared
    ``AsyncPostgresSaver`` (langgraph-api managed schema) types its ``thread_id``
    column as ``uuid`` — a raw ``"<uuid>::deepagent::<n>"`` string fails on the first
    ``aget_state`` with ``InvalidTextRepresentation``. ``uuid5`` is deterministic, so
    it preserves the "stable while paused, unique per turn" property while yielding a
    valid UUID.
    """
    outer_configurable = dict((config or {}).get("configurable", {}) or {})
    outer_thread = outer_configurable.get("thread_id")
    deep_agent_configurable = dict(outer_configurable)
    deep_agent_configurable.pop("checkpoint_id", None)
    deep_agent_configurable.pop("checkpoint_ns", None)
    if outer_thread:
        deep_agent_configurable["thread_id"] = str(
            uuid.uuid5(uuid.NAMESPACE_OID, f"{outer_thread}::deepagent::{turn_key}")
        )
    deep_agent_config: RunnableConfig = {"configurable": deep_agent_configurable}
    return deep_agent_config, outer_thread


async def _stream_deep_agent(
    deep_agent, agent_input, deep_agent_config, context, writer
):
    """Run the deep agent (fresh input or ``Command(resume=...)``), streaming only
    the final user-visible reply's tokens. Returns the deep agent's terminal state
    dict (carrying ``messages`` + identity-doc snapshots), or ``None`` if it produced
    no terminal output (e.g. it paused on an interrupt).

    Same streaming heuristic as the legacy ``think`` body: tokens are emitted per
    LLM call only while the running merged chunk shows no ``tool_calls``; once a tool
    call appears, that call is a tool-planning turn and its tokens are dropped.
    """
    stream_buffers: dict[str, dict[str, Any]] = {}
    final_output: dict[str, Any] | None = None

    async for event in deep_agent.astream_events(
        agent_input,
        config=deep_agent_config,
        context=context,
        version="v2",
    ):
        ev_name = event.get("event")
        if ev_name == "on_chat_model_stream":
            # Skip internal structured-output calls (e.g. the per-document fact-correction
            # ``ProposedFactEdit`` analyses, run concurrently inside tools). Their tokens are
            # raw JSON, not a reply — streaming them leaks interleaved JSON into the chat.
            if STRUCTURED_OUTPUT_STREAM_TAG in (event.get("tags") or []):
                continue
            run_id = event.get("run_id")
            chunk = event["data"].get("chunk")
            if chunk is None or run_id is None:
                continue
            buf = stream_buffers.setdefault(
                run_id, {"merged": None, "streamed_text": False}
            )
            merged_prev = buf["merged"]
            buf["merged"] = chunk if merged_prev is None else merged_prev + chunk
            if not (getattr(buf["merged"], "tool_calls", None) or []):
                delta = chunk.content
                if isinstance(delta, str) and delta:
                    writer({"type": "assistant_token", "text": delta})
                    buf["streamed_text"] = True
        elif ev_name == "on_chat_model_end":
            stream_buffers.pop(event.get("run_id"), None)
        elif ev_name == "on_chain_end":
            data = event.get("data") or {}
            output = data.get("output")
            if isinstance(output, dict) and "messages" in output:
                final_output = output

    return final_output


async def think(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Drive the avatar's deep agent and stream only the final user-visible reply.

    Replaces the old single-LLM-call ``think`` node + ``process_thoughts``
    tool loop. The deep agent (see ``build_avatar_deep_agent``) owns its
    own internal loop: think → tool calls → tools execute → (optional
    synthetic ``load_consciousness`` refresh) → think → ... until the
    model emits an ``AIMessage`` with no tool calls.

    Streaming contract: same as the legacy node. ``assistant_token``
    events are emitted only for token chunks belonging to an LLM call
    that ultimately produces zero tool calls — i.e. the final reply.
    Token chunks for tool-planning LLM calls are silently dropped.

    When the data-analysis capability is enabled (``DATA_ANALYSIS_ENABLED``)
    and the Model Context Protocol filesystem server is reachable, the deep
    agent additionally receives the analysis tool set and a
    ``CompositeBackend`` (local-shell execution workspace + persistent
    per-user-per-avatar store routes). The ephemeral workspace is cleaned in
    the ``finally`` below — including when a human-in-the-loop ``interrupt``
    pauses the turn: on resume this node re-runs and builds a fresh
    workspace, and previously ingested data is recoverable from the
    persistent store via the ``hydrate_ingested_data`` tool.

    Returns:
        Outer-graph state delta with:

        - ``messages``: single final ``AIMessage`` (Go Emotions sentiment
          metadata attached).
        - ``internal_thoughts``: every intermediate ``AIMessage`` /
          ``ToolMessage`` the deep agent produced, for auditing.
        - ``system_message`` / identity-doc snapshots forwarded from the
          deep agent's final state so the outer state stays in sync.
    """
    # The deep agent is checkpointed on its own deterministic thread so a
    # human-in-the-loop ``interrupt`` raised mid-tool (e.g. ``edit_identity_fact``)
    # is durable. ``store`` is passed explicitly because under its own checkpointer the
    # agent no longer inherits it implicitly from the parent run.
    checkpointer = get_deep_agent_checkpointer()
    # Key the deep-agent thread on the conversation length so each turn is isolated
    # but the interrupt/resume of one turn shares a thread (stable while paused).
    deep_agent_config, outer_thread = _deep_agent_config(config, len(state["messages"]))

    # The factory's ``with_config({"recursion_limit": ...})`` binding is honored
    # by ``ainvoke`` but silently DROPPED by ``astream_events`` (verified against
    # langgraph 1.x) — and ``_stream_deep_agent`` streams via ``astream_events``.
    # Without this line the deep agent runs at langgraph's default limit of 25
    # regardless of DEEP_AGENT_RECURSION_LIMIT, which multi-step analysis turns
    # (discover → ingest → execute → persist) routinely exceed.
    deep_agent_run_context = runtime.context or GlobalContext()
    deep_agent_config["recursion_limit"] = (
        deep_agent_run_context.deep_agent_recursion_limit
    )

    # Data-analysis capability gate: the SOLE condition is a saved MCP
    # connection that (a) the user established through the discovery/consent
    # flow (the ``mcp_discovery`` node) and (b) is bound to THIS avatar. No
    # environment switch, no per-avatar enable flag — the connection is the
    # gate. Unbound avatars (e.g. a test avatar) get nothing; an unreachable
    # server degrades to a normal turn (empty tool list) inside the tools.
    analysis_bundle = None
    analysis_extra_tools: list[Any] | None = None
    # Exclusive to the user's own personal avatar: a bound connection on a
    # demoted (no-longer-personal) avatar must NOT re-enable live MCP access.
    is_personal_avatar = _user_personal_avatar(config, state)
    connection = await bound_connection_for(
        runtime.store,
        state["user_state"]["user_id"],
        state["assistant_state"]["assistant_id"],
    )
    if connection is not None and is_personal_avatar:
        analysis_bundle = build_analysis_backend(
            deep_agent_run_context,
            state["user_state"]["user_id"],
            state["assistant_state"]["assistant_id"],
            store=runtime.store,
        )
        analysis_extra_tools = build_data_analysis_tools(
            deep_agent_run_context, analysis_bundle, connection
        )
    elif runtime.store is not None and is_personal_avatar:
        # Unconnected owned avatar: carry only the explicit-connect tool so a
        # natural-language "connect to the Neural Nexus MCP server" always
        # works, even after a decline or disconnect suppressed the automatic
        # discovery offer.
        analysis_extra_tools = [
            build_connect_tool(
                deep_agent_run_context,
                state["user_state"]["user_id"],
                state["assistant_state"]["assistant_id"],
            )
        ]

    # Browser capability gate: a process-wide environment switch
    # (BROWSER_TOOLS_ENABLED) rather than a per-user connection — the browser
    # tools read the public web and carry no per-user credentials. Keyed on
    # the outer workflow thread so each conversation browses in a dedicated
    # Chromium process (isolated pages, cookies, history). Returns an empty
    # list when the gate is off or Chromium is unavailable, so this line
    # never degrades the turn.
    browser_toolkit_tools = await get_browser_toolkit_tools(
        deep_agent_run_context, conversation_key=outer_thread
    )

    extra_tools = [*(analysis_extra_tools or []), *browser_toolkit_tools]
    deep_agent = build_avatar_deep_agent(
        runtime.context,
        checkpointer=checkpointer,
        store=runtime.store,
        extra_tools=extra_tools or None,
        backend=analysis_bundle.backend if analysis_bundle is not None else None,
    )
    try:
        return await _run_avatar_deep_agent_turn(
            state,
            config,
            runtime,
            deep_agent,
            deep_agent_config,
            outer_thread,
            checkpointer,
            analysis_bundle=analysis_bundle,
        )
    finally:
        if analysis_bundle is not None:
            cleanup_analysis_workspace(analysis_bundle)
        if browser_toolkit_tools:
            # Drop the turn lease on the conversation's browser so idle /
            # least-recently-used eviction may consider the browser again.
            await release_conversation_browser(outer_thread)


# Cadence of the keepalive frames emitted while post-reply analysis (Go Emotions
# + SHAP style comparison) runs. That analysis produces no user-visible tokens
# yet gates the terminal ``done`` SSE frame, so without these the client's
# idle-read can time out ("Error in input stream") before ``done`` arrives. Kept
# well under common client read timeouts; promote to GlobalContext if it ever
# needs per-deployment tuning.
_POST_REPLY_ANALYSIS_HEARTBEAT_SECONDS = 5.0

# Ceiling on the whole post-reply analysis window. Everything inside that window
# is metadata enrichment, so a dependency that stalls there must cost the client
# metadata, never the reply: without a ceiling the keepalive loop above streams
# forever and the terminal ``done`` frame never arrives, which is exactly what a
# ~20 MB NLTK corpus download over a slow link once did to a first request.
# Sized well above the observed steady-state cost (tens of seconds, dominated by
# the SHAP KernelExplainer pass over a large ground-truth corpus) so a healthy
# turn never trips it; promote to GlobalContext if it ever needs per-deployment
# tuning.
_POST_REPLY_ANALYSIS_TIMEOUT_SECONDS = 180.0


async def _emit_analysis_keepalives(writer, interval_seconds: float) -> None:
    """Emit a ``keepalive`` custom stream event every ``interval_seconds``.

    Run as a background task wrapping the post-reply analysis so the SSE
    generator keeps writing bytes to the client during the token-less analysis
    window. The caller cancels this task once analysis completes. ``writer`` is
    the LangGraph stream writer (a no-op when the run is not being streamed, so
    this is harmless off the streaming path).
    """
    while True:
        await asyncio.sleep(interval_seconds)
        writer({"type": "keepalive"})


async def _attach_post_reply_analysis(
    final_message: Any,
    *,
    new_messages: list,
    state: GlobalState,
    config: RunnableConfig,
    runtime: Runtime[GlobalContext],
) -> None:
    """Attach every token-less enrichment the terminal ``done`` frame reports.

    Go Emotions sentiment, token-usage accounting, and — when the caller asked
    for ``include_quality_metrics`` — the authenticity comparison against the
    target author and the ChatGPT baseline. All of it mutates
    ``final_message.response_metadata`` in place.

    Split out of the turn body so the caller can place the entire window under a
    single ``asyncio.wait_for`` ceiling. The whole window gates ``done``, so from
    the client's side an unbounded step in here is indistinguishable from a hung
    request; a partial attach is strictly better than no reply.
    """
    if isinstance(final_message, AIMessage) and not final_message.tool_calls:
        # Go Emotions is a RoBERTa forward pass — offload to a worker thread so it
        # does not block the event loop between the streamed reply and the terminal
        # ``done`` SSE frame (which is gated behind this whole block).
        await asyncio.to_thread(_attach_go_emotions_metadata, final_message)
        _attach_token_usage_metadata(final_message, new_messages)
        if config.get("configurable", {}).get("use_adapter_inference"):
            final_message.response_metadata = dict(final_message.response_metadata or {})
            final_message.response_metadata["is_adapter_inference"] = True
    # Authenticity metrics: score the (already-streamed) reply against the
    # target author + ChatGPT baseline and attach to response_metadata.
    if config.get("configurable", {}).get("include_quality_metrics", False):
        # The per-avatar artifacts (ground-truth cloud, key_phrase_profile) are
        # owner-scoped, so pass the avatar OWNER's id — the same first namespace
        # element calibrate_ground_truth wrote under — not the conversing user.
        retrieved_user_id = (
            config.get("configurable", {})
            .get("assistant_ctx", {})
            .get("metadata", {})
            .get("user_id")
        ) or state["user_state"]["user_id"]
        await _attach_analyzed_features(
            final_message,
            runtime=runtime,
            assistant_id=state["assistant_state"]["assistant_id"],
            user_id=retrieved_user_id,
        )


async def _run_avatar_deep_agent_turn(
    state: GlobalState,
    config: RunnableConfig,
    runtime: Runtime[GlobalContext],
    deep_agent: Any,
    deep_agent_config: RunnableConfig,
    outer_thread: str | None,
    checkpointer: Any,
    analysis_bundle: Any = None,
):
    """Body of one ``think`` turn: run/resume the deep agent, slice output.

    Split out of ``think`` so the data-analysis workspace cleanup can wrap
    the whole run in a ``try``/``finally`` without re-indenting the flow.

    ``analysis_bundle`` is the turn's data-analysis bundle when the avatar has
    a bound MCP connection. It is read here — not in ``think`` — because the
    created artifacts must be attached to the final message before the caller's
    ``finally`` wipes the workspace they were produced in.
    """
    deep_agent_input = {
        "messages": list(state["messages"]),
        "system_message": list(state.get("system_message") or []),
        "user_identity_documents": list(state.get("user_identity_documents") or []),
        "assistant_identity_documents": list(
            state.get("assistant_identity_documents") or []
        ),
        "recalled_memory_documents": list(state.get("recalled_memory_documents") or []),
        "user_state": state["user_state"],
        "assistant_state": state["assistant_state"],
        "internal_thoughts": [],
    }
    # Slice new messages from the deep agent's persisted conversation against the
    # outer conversation length — stable across the interrupt/resume re-run.
    input_messages_count = len(state["messages"])

    writer = get_stream_writer()

    # Idempotency guard: this whole node re-runs when the OUTER graph resumes. If the
    # deep-agent thread is already paused mid-interrupt, skip the fresh run (which
    # would re-stream the same tokens) and go straight to resuming it below.
    can_persist = checkpointer is not None and bool(outer_thread)
    already_paused = False
    if can_persist:
        snapshot = await deep_agent.aget_state(deep_agent_config)
        already_paused = bool(snapshot.next)

    final_output: dict[str, Any] | None = None
    if not already_paused:
        final_output = await _stream_deep_agent(
            deep_agent, deep_agent_input, deep_agent_config, runtime.context, writer
        )

    # If the deep agent paused on an interrupt, surface it through the OUTER graph so
    # its ``AsyncPostgresSaver`` persists the pause and the API can present the
    # approve/edit/reject preview. On resume, the outer ``interrupt`` returns the
    # owner's decision, which we forward into the deep agent on its own thread.
    if can_persist:
        snapshot = await deep_agent.aget_state(deep_agent_config)
        pending_interrupts = collect_pending_interrupts(snapshot)
        if pending_interrupts:
            decision = interrupt(pending_interrupts[0].value)
            resume_cmd = build_interrupt_resume_command(
                pending_interrupts, decision
            )
            final_output = await _stream_deep_agent(
                deep_agent,
                resume_cmd,
                deep_agent_config,
                runtime.context,
                writer,
            )

    if final_output is None:
        logger.warning(
            "Deep agent produced no final output; returning empty state delta."
        )
        return {}

    all_messages = list(final_output.get("messages") or [])
    new_messages = all_messages[input_messages_count:]
    if not new_messages:
        logger.warning(
            "Deep agent produced no new messages; returning empty state delta."
        )
        return {}

    final_message = new_messages[-1]
    intermediate = new_messages[:-1]

    # The reply is already fully streamed to the client by this point; what
    # follows (Go Emotions sentiment + SHAP style comparison) only enriches the
    # terminal ``done`` frame's metadata. It emits no tokens and can run for tens
    # of seconds, so a background task streams keepalive frames throughout to keep
    # the client's connection from idling out before ``done``.
    heartbeat_task = asyncio.create_task(
        _emit_analysis_keepalives(writer, _POST_REPLY_ANALYSIS_HEARTBEAT_SECONDS)
    )
    try:
        await asyncio.wait_for(
            _attach_post_reply_analysis(
                final_message,
                new_messages=new_messages,
                state=state,
                config=config,
                runtime=runtime,
            ),
            timeout=_POST_REPLY_ANALYSIS_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        # Degrade rather than hang: whatever was attached before the deadline
        # stays on the message and the rest is simply absent from ``done``.
        # The cancellation reaches the awaiting coroutine, NOT any
        # ``asyncio.to_thread`` worker already running inside it — a wedged
        # thread runs to its own completion in the background, but the client is
        # released now instead of waiting on it.
        logger.warning(
            "Post-reply analysis exceeded %.0fs; emitting the reply with whatever "
            "metadata was attached before the deadline.",
            _POST_REPLY_ANALYSIS_TIMEOUT_SECONDS,
        )
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    # Reports and plots this turn produced ride out on ``response_metadata``:
    # the terminal ``done`` SSE frame already forwards that field verbatim, it
    # is checkpointed on the message (so a client reloading the thread still
    # gets the artifacts), and — unlike ``additional_kwargs`` — it is never
    # converted back into a provider request, so the inlined base64 is not
    # re-sent to the model on later turns.
    if analysis_bundle is not None:
        try:
            created_artifacts = await collect_turn_artifacts(
                runtime.context, analysis_bundle
            )
            if created_artifacts:
                final_message.response_metadata = dict(
                    final_message.response_metadata or {}
                )
                final_message.response_metadata["created_artifacts"] = created_artifacts
        except Exception:
            # Never lose an already-streamed reply over a display concern.
            logger.exception("Could not collect this turn's analysis artifacts.")

    update: dict[str, Any] = {
        "messages": [final_message],
        "internal_thoughts": [*intermediate, final_message],
    }

    # ``system_message`` replaces via its pinned UUID (add_messages). The document
    # channels are forwarded as replace-snapshots: the deep agent's final lists are
    # already merged/deduped/pruned by ``load_consciousness``, so the outer state must
    # adopt them verbatim — the default append reducer would resurrect stale copies
    # (e.g. a document the edit/delete tools just removed).
    if final_output.get("system_message") is not None:
        update["system_message"] = final_output["system_message"]
    for key in (
        "user_identity_documents",
        "assistant_identity_documents",
        "recalled_memory_documents",
    ):
        if key in final_output and final_output[key] is not None:
            update[key] = {"op": "replace", "docs": list(final_output[key])}

    return update


def _user_owns_avatar(config: RunnableConfig, state: GlobalState) -> bool:
    """Whether the conversing user is the avatar's owner (creator).

    A personal Neural Nexus MCP data server is only ever offered to — or
    connectable by — the avatar's own creator, never a visitor conversing
    with someone else's shared avatar.
    """
    owner_id = (
        config.get("configurable", {})
        .get("assistant_ctx", {})
        .get("metadata", {})
        .get("user_id")
    )
    return owner_id is not None and owner_id == state["user_state"]["user_id"]


def _user_personal_avatar(config: RunnableConfig, state: GlobalState) -> bool:
    """Whether the conversing user is the creator AND this is their personal avatar.

    The desktop MCP data server (and future personal analytics) are exclusive to
    the one avatar a user has flagged ``PERSONAL_AVATAR_OF_THE_CREATOR`` — never a
    visitor on someone else's avatar, and never the user's other, non-personal
    avatars. Combines the owner check with the ``is_personal_avatar_of_creator``
    metadata flag set by ``/create_avatar`` / ``/modify_avatar``.
    """
    metadata = (
        config.get("configurable", {}).get("assistant_ctx", {}).get("metadata", {})
    )
    return (
        _user_owns_avatar(config, state)
        and metadata.get("is_personal_avatar_of_creator") is True
    )


async def mcp_discovery(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Offer to connect a discovered MCP data server, and persist the choice.

    Runs before ``load_consciousness`` so a just-approved connection is visible
    to both the capability prompt and the ``think`` gate in the same turn. It
    offers a connection only when ALL of the following hold, and otherwise
    returns an empty delta (a normal turn):

    - this is the user's own PERSONAL avatar (owner match AND the
      ``is_personal_avatar_of_creator`` flag) — never a visitor on someone
      else's avatar, and never the user's other, non-personal avatars;
    - the user has NO connection yet (a user has at most one MCP connection —
      once established, bound to one avatar, no further offers are made);
    - this avatar has not previously DECLINED (a decline on one avatar never
      suppresses the offer on the user's other avatars);
    - a server actually ANNOUNCES itself over the discovery SSE channel.

    When those hold, it raises a durable ``interrupt`` — surfaced to the client
    as the SSE ``interrupt`` event and resolved via ``POST
    /message/{assistant_id}/resume`` — deferring the connect decision to the
    user. ``{"type": "apply"}`` saves the connection bound to this avatar;
    anything else records a per-avatar decline.
    """
    store = runtime.store
    if store is None:
        return {}

    user_id = state["user_state"]["user_id"]
    assistant_id = state["assistant_state"]["assistant_id"]

    if not _user_personal_avatar(config, state):
        return {}

    if await read_user_connection(store, user_id) is not None:
        return {}
    if await is_declined(store, user_id, assistant_id):
        return {}

    context = runtime.context or GlobalContext()
    connection = await resolve_available_connection(store, user_id, context)
    if connection is None:
        return {}

    decision = interrupt(
        {
            "kind": "mcp_connect_consent",
            "prompt": (
                "An MCP data server is available and not yet connected — "
                "connect it to this avatar for data analysis?"
            ),
            "server": {
                "server_name": connection.server_name,
                "url": connection.url,
                "transport": connection.transport,
                "allowed_roots": list(connection.allowed_roots),
            },
        }
    )

    approved = isinstance(decision, dict) and decision.get("type") == "apply"
    if approved:
        await save_user_connection(
            store, user_id, connection=connection, assistant_id=assistant_id
        )
    else:
        await mark_declined(store, user_id, assistant_id, connection)
    return {}


""" GRAPH """

# Build minimal graph: START -> load_consciousness -> think -> END
anubis_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=GlobalState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

""" ANUBIS WORKFLOW NODES """

anubis_workflow.add_node("mcp_discovery", mcp_discovery)
anubis_workflow.add_node("load_consciousness", load_consciousness)
anubis_workflow.add_node("think", think)

""" ANUBIS WORKFLOW EDGES """

anubis_workflow.add_edge(START, "mcp_discovery")
anubis_workflow.add_edge("mcp_discovery", "load_consciousness")
anubis_workflow.add_edge("load_consciousness", "think")
anubis_workflow.add_edge("think", END)

# workflow.add_edge("chat", "terms_and_services_content_moderation")

# COERCION
# workflow.add_conditional_edges("respond", avatar_tools_condition, {'avatar_tools':'avatar_tools', END:"evaluate_response_quality"})
# workflow.add_edge("evaluate_response_quality", "update_response_metadata")
# workflow.add_edge("terms_and_services_content_moderation", "update_response_metadata")
# workflow.add_edge("update_response_metadata", END)

anubis = anubis_workflow.compile()

message_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=MessagesState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

# workflow.add_edge("terms_and_services_content_moderation", END)
message_workflow.add_node("chat", message_interface)
message_workflow.add_node("resolve_human_message_images", resolve_human_message_images)
message_workflow.add_node("anubis", anubis)

message_workflow.add_edge(START, "chat")
message_workflow.add_edge("chat", "resolve_human_message_images")
message_workflow.add_edge("resolve_human_message_images", "anubis")
message_workflow.add_edge("anubis", END)

graph = message_workflow.compile()

graph.name = "Anubis"

ensure_huggingface_models_cached(GlobalContext())
ensure_nltk_corpora_cached()

__all__ = ["graph"]
