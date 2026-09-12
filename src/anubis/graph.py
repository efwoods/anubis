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

# The fan-in edge below compiles to a ``NamedBarrierValue`` channel whose
# checkpoint the LangGraph platform reads back as a list, which breaks
# reading any thread that was cancelled between the fan-out and the fan-in.
# Applied here because this module is imported before any graph is compiled
# or any thread state is read.
from src.anubis.utils.barrier_channel_checkpoint_compatibility import (
    apply_barrier_channel_checkpoint_compatibility_patch,
)

apply_barrier_channel_checkpoint_compatibility_patch()

load_dotenv()

import logging
import uuid
from datetime import datetime, timezone
from collections import OrderedDict
from typing import Any, Literal

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

from src.anubis.utils.ambient.observations import ambient_details
from src.anubis.utils.ambient.observations import mark_view_currency
from src.anubis.utils.client_harvest_turns import without_stale_client_harvest_turns
from src.anubis.utils.tools.vision.look_tools import normalize_live_shares
from src.anubis.utils.ambient.triage_node import (
    AMBIENT_TRIAGE_NODE,
    ambient_triage,
    route_after_ambient_triage,
    route_after_image_resolution,
)
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.deep_agent import build_avatar_deep_agent
from src.anubis.utils.middleware.avatar_summarization import (
    CONVERSATION_SUMMARY_EVENT_KEY,
    CONVERSATION_SUMMARY_SESSION_ID_KEY,
    clamp_summary_event,
)
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
from src.anubis.utils.nodes import (
    join_user_observation,
    load_consciousness,
    observe_user,
    resolve_human_message_images,
)
from src.anubis.utils.runtime_handles import get_deep_agent_checkpointer
from src.anubis.utils.state import GlobalState
from src.anubis.utils.tools.browser import (
    get_browser_toolkit_tools,
    release_conversation_browser,
)
from src.anubis.utils.tools.data_analysis import (
    bound_connections_for,
    build_analysis_backend,
    build_connect_tool,
    build_data_analysis_tools,
    cleanup_analysis_workspace,
    collect_turn_artifacts,
    read_user_connections,
    resolve_available_connections,
    save_user_connection,
    suppressed_device_ids,
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
    avatar_response: AIMessage,
    turn_messages: list,
    context: GlobalContext | None = None,
) -> None:
    """Fold the turn's aggregate token usage into ``response_metadata["token_usage"]``.

    Sums ``usage_metadata`` across every AI message the deep agent produced this
    turn (tool-planning calls consume tokens too, not just the final reply) and
    writes the total onto the final message's ``response_metadata`` in the
    ``token_usage`` shape the API metering layer reads (``prompt_tokens`` /
    ``completion_tokens`` / ``total_tokens``). ``usage_metadata`` itself is a
    message attribute that never survives the API layer's ``response_metadata``
    serialization, which is why the fold is necessary.

    When ``context`` is given the turn's dollar cost is derived here too, as
    ``response_metadata["total_cost"]`` = prompt tokens × ``MODEL_PROMPT_COST`` +
    completion tokens × ``MODEL_COMPLETION_COST`` (both dollars per single token).
    The API layer reads that key into the Prometheus ``MODEL_COST_TOTAL`` counter
    and the ``api_metrics.cost_usd`` column; before this derivation nothing on the
    message path set the key, so both ledgers recorded 0 for every reply. Every
    turn is priced at the inference model's rates, including an adapter-inference
    turn, which runs through the Llama wrapper at a different tariff — a known
    approximation until adapter pricing is configured separately.
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
    if context is not None:
        avatar_response.response_metadata["total_cost"] = (
            prompt_tokens * float(context.model_prompt_cost or 0.0)
            + completion_tokens * float(context.model_completion_cost or 0.0)
        )


def _attach_go_emotions_metadata(avatar_response: AIMessage) -> None:
    """Mutate ``avatar_response.response_metadata`` with Go Emotions classifier output."""
    from src.anubis.utils.emotion_classifier import classify_go_emotions

    sentiment = classify_go_emotions(avatar_response.content)
    if sentiment is None:
        return
    avatar_response.response_metadata = dict(avatar_response.response_metadata or {})
    avatar_response.response_metadata.update({"sentiment": sentiment})


def _publishable_avatar_key_phrase_rate(
    avatar_key_phrases: Any, ground_truth_features_dict: dict
) -> float | None:
    """Return the avatar-referenced key_phrase_rate, or None when it is unknown.

    ``extract_style_features`` yields 0.0 for an empty phrase set, which a client
    cannot tell apart from a genuine "this reply reuses none of the avatar's
    signature phrases". An avatar with no calibrated phrase profile has an
    UNKNOWN rate, not a zero one, so publish null there. NaN is unpublishable for
    the same reason the features block above sanitizes it: the metadata copy must
    be strict JSON or the terminal SSE frame is invalid.
    """
    if not avatar_key_phrases:
        return None
    avatar_referenced_rate = ground_truth_features_dict.get("key_phrase_rate")
    if not isinstance(avatar_referenced_rate, float) or not math.isfinite(
        avatar_referenced_rate
    ):
        return None
    return avatar_referenced_rate


# Bounded, process-local cache of per-avatar ground-truth SHAP explainers.
#
# Rebuilding the explainer on every reply costs a kmeans summarization of the
# quote corpus plus the KernelExplainer constructor. Measured against a 156-row
# corpus that is ~50 ms of a ~435 ms ground-truth SHAP step. The remaining
# ~385 ms is ``shap_values`` itself, which re-runs model.predict across the
# background sample for each new candidate and therefore cannot be cached at all
# — the ChatGPT baseline pole pays exactly the same per-reply cost.
#
# Deliberately NOT persisted to the store the way the baseline explainer is:
# pickling one of these for a 156-row corpus produces a ~66 MB base64 blob (it
# drags the whole IsolationForest and SHAP's internal state along), so reading it
# back would cost far more per message than the 50 ms it saves, and a restored
# copy does not even reproduce the same values. The kmeans half of the saving
# grows with corpus size, so this cache earns more as an avatar ingests more
# media, which is the direction every avatar moves.
#
# Keyed by avatar and validated against the exact serialized model the store just
# returned, so a recalibration — which rewrites that blob — can never be served a
# stale explainer and no separate invalidation hook is needed.
_GROUND_TRUTH_EXPLAINER_CACHE_MAX_ENTRIES = 32
_ground_truth_explainer_cache: "OrderedDict[tuple[str, str], tuple[str, int, Any]]" = (
    OrderedDict()
)


def _cached_ground_truth_explainer(
    user_id: str, assistant_id: str, model_b64_pkl: str, feature_width: int
) -> Any | None:
    """Return this avatar's cached SHAP explainer, or None if one must be built.

    The serialized model is compared in full rather than fingerprinted: it is the
    same string the store returned this turn, and a direct string comparison is a
    memcmp measured in microseconds against the ~50 ms rebuild it guards.
    """
    entry = _ground_truth_explainer_cache.get((user_id, assistant_id))
    if entry is None:
        return None
    cached_model_b64_pkl, cached_feature_width, explainer = entry
    if cached_model_b64_pkl != model_b64_pkl or cached_feature_width != feature_width:
        return None
    _ground_truth_explainer_cache.move_to_end((user_id, assistant_id))
    return explainer


def _remember_ground_truth_explainer(
    user_id: str,
    assistant_id: str,
    model_b64_pkl: str,
    feature_width: int,
    explainer: Any,
) -> None:
    """Cache this avatar's explainer, evicting the least recently used entry."""
    cache_key = (user_id, assistant_id)
    _ground_truth_explainer_cache[cache_key] = (
        model_b64_pkl,
        feature_width,
        explainer,
    )
    _ground_truth_explainer_cache.move_to_end(cache_key)
    while len(_ground_truth_explainer_cache) > _GROUND_TRUTH_EXPLAINER_CACHE_MAX_ENTRIES:
        _ground_truth_explainer_cache.popitem(last=False)


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
                    "key_phrase_rate is the rate of baseline ChatGPT signature "
                    "key phrases per total word, the reference set the baseline "
                    "ChatGPT comparison is measured against. "
                    "key_phrase_rate_against_avatar_signature_phrases is the "
                    "same reply measured against the avatar's own discovered "
                    "signature phrases, the reference set the direct-quote "
                    "comparison is measured against, and is null until the "
                    "avatar has a calibrated signature phrase profile."
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

        # Publish the avatar-referenced rate alongside the baseline-referenced
        # one. Both were already computed; only the baseline number reached the
        # client, so a reply dense with the target's own recurring phrasing still
        # reported key_phrase_rate 0.0 — the single most misleading number in the
        # features block, because it reads as "sounds nothing like the avatar"
        # when it actually means "was scored against ChatGPT's phrases".
        # Added as a NEW key rather than replacing key_phrase_rate: that value is
        # the input to the baseline comparison and clients already read it.
        # Null, not 0.0, when the avatar has no calibrated phrase set — an
        # uncalibrated avatar has an UNKNOWN rate, and reporting zero there would
        # be indistinguishable from a genuine zero.
        published_features = avatar_response.response_metadata.get("features")
        if isinstance(published_features, dict):
            published_features["key_phrase_rate_against_avatar_signature_phrases"] = (
                _publishable_avatar_key_phrase_rate(
                    avatar_key_phrases, ground_truth_features_dict
                )
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

            # Reuse this avatar's explainer across replies when the fitted model
            # has not changed; see _cached_ground_truth_explainer for why this is
            # a process-local cache rather than a persisted artifact.
            explainer = _cached_ground_truth_explainer(
                user_id,
                assistant_id,
                ground_truth_text_features_model_b64_pkl,
                len(FEATURE_NAMES),
            )
            if explainer is None:
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
                explainer = await asyncio.to_thread(
                    shap.KernelExplainer,
                    ground_truth_text_features_model.predict,
                    shap_background,
                )
                _remember_ground_truth_explainer(
                    user_id,
                    assistant_id,
                    ground_truth_text_features_model_b64_pkl,
                    len(FEATURE_NAMES),
                    explainer,
                )
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


MODERATION_REFUSAL_TEXT = (
    "This message violates the terms of service that every person on this platform "
    "agreed to, so this conversation cannot continue. The account has been suspended "
    "from Neural Nexus and every Afterlife Systems product; to appeal, contact "
    "{appeal_contact}."
)

# The node names of the inline moderation branch and the refusal it can route to.
MODERATE_CONTENT_NODE = "moderate_content_fast"
REFUSE_FOR_VIOLATION_NODE = "refuse_for_violation"


def moderation_is_skipped(config: RunnableConfig | None) -> bool:
    """Whether this run opted out of moderating its own latest message.

    Set ``configurable["skip_content_moderation"]`` when the graph is re-entered
    with words that are NOT the caller's own — the live-stream responder answering
    a viewer, a harvested client turn — so that another person's words can never
    ban the account whose credential carries the request. Mirrors the
    ``skip_observation`` flag that guards the learning observation the same way.
    """
    return bool((config or {}).get("configurable", {}).get("skip_content_moderation"))


async def moderate_content_fast(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """AI monitoring, stage one: screen the latest human message inline.

    Runs in the outer workflow in parallel with image resolution and the learning
    observation, and only the CHEAP screen runs here — one OpenAI moderation call
    that answers in roughly a tenth of a second, underneath two branches that
    already take longer. The turn's critical path is therefore unchanged.

    The deep terms-of-service judge deliberately does NOT run here. It reads the
    same message after the reply has streamed (``schedule_background`` in
    ``src.api.webapp``), so a person never waits on it; a violation it finds bans
    the account, and the ban refuses the caller's next request.

    Fail-open: a screening outage leaves the verdict clean and logs. See
    ``src.anubis.utils.moderation.fast_screen``.
    """
    from src.anubis.utils.learning.sentiment import message_text
    from src.anubis.utils.moderation.content_moderation import (
        clean_verdict,
        moderation_flag_enabled,
    )
    from src.anubis.utils.moderation.fast_screen import (
        FAST_SCREEN_BLOCK,
        clean_screen,
        screen_to_verdict,
    )
    from src.subgraphs.moderation_graph.graph import (
        MODERATION_MODE_MESSAGE,
        moderate_text_with_graph,
    )

    clean = {"screen": clean_screen(), "verdict": clean_verdict()}
    context = runtime.context or GlobalContext()
    if not moderation_flag_enabled(
        getattr(context, "content_moderation_enabled", "TRUE")
    ):
        return {"moderation_response": clean}
    if moderation_is_skipped(config):
        return {"moderation_response": clean}
    messages = state.get("messages") or []
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {"moderation_response": clean}
    latest_text = message_text(messages[-1].content).strip()
    if not latest_text:
        return {"moderation_response": clean}

    try:
        result = await moderate_text_with_graph(
            latest_text, mode=MODERATION_MODE_MESSAGE, context=context
        )
    except Exception as moderation_error:  # noqa: BLE001 - fail open, never cost a reply
        logger.error(
            "Inline content moderation failed (treating as clean): %s", moderation_error
        )
        return {"moderation_response": clean}

    # Only a hard block refuses inline. A "suspect" screen lets the reply through
    # and is settled by the background judge.
    if (result.get("screen") or {}).get("outcome") != FAST_SCREEN_BLOCK:
        return {"moderation_response": {**result, "verdict": clean_verdict()}}
    return {
        "moderation_response": {
            "screen": result.get("screen"),
            "verdict": screen_to_verdict(result.get("screen") or {}),
        }
    }


def route_after_moderation(
    state: GlobalState,
) -> Literal["refuse_for_violation", "ambient_triage", "anubis"]:
    """Refuse a blocked turn; otherwise fall through to the existing ambient routing."""
    verdict = (state.get("moderation_response") or {}).get("verdict") or {}
    if verdict.get("violation"):
        return REFUSE_FOR_VIOLATION_NODE
    return route_after_image_resolution(state)


async def refuse_for_violation(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Refuse the turn after a confirmed violation and tell the API layer to ban.

    The refusal text is streamed as an ordinary ``assistant_token`` so every
    existing client renders the reason with no change, and a
    ``moderation_violation`` custom event carries the verdict to the API layer,
    which owns the ban side effects (the database row, the Stripe refund and
    cancellation, the cache eviction) because those need the connection pool and
    the Stripe client that the graph does not hold. The verdict is also stamped on
    the reply's ``response_metadata`` so the non-streaming path and the
    checkpointed transcript both carry the reason.
    """
    context = runtime.context or GlobalContext()
    verdict = dict((state.get("moderation_response") or {}).get("verdict") or {})
    appeal_contact = (
        getattr(context, "ban_appeal_contact_email", None) or "contact@neuralnexus.site"
    )
    refusal_text = MODERATION_REFUSAL_TEXT.format(appeal_contact=appeal_contact)
    try:
        writer = get_stream_writer()
        writer({"type": "moderation_violation", **verdict})
        writer({"type": "assistant_token", "text": refusal_text})
    except Exception:  # noqa: BLE001 - outside a graph run there is no stream writer
        pass
    refusal = AIMessage(
        content=refusal_text,
        id=str(uuid.uuid4()),
        response_metadata={"moderation_violation": verdict},
    )
    return {"messages": [refusal], "internal_thoughts": [refusal]}


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
        # The conversation's own thread id, kept for the tool-call log so
        # feature usage is counted per conversation, not per deep-agent turn.
        deep_agent_configurable["outer_thread_id"] = str(outer_thread)
    deep_agent_config: RunnableConfig = {"configurable": deep_agent_configurable}
    return deep_agent_config, outer_thread


# Human-readable activity for each tool the deep agent can call, keyed by the
# tool's registered name. Streamed to the client as a ``status`` frame the moment
# the tool starts, so a person waiting through a long data-analysis turn (which
# emits no reply tokens until the analysis is finished) can see what the avatar
# is doing instead of a silent "thinking" indicator. Tool names absent from this
# table fall back to a generic phrase built from the name. ``{device}`` is
# replaced by the ``device_label`` argument when the tool call names a machine.
_TOOL_ACTIVITY_DESCRIPTIONS: dict[str, str] = {
    "list_git_repositories": "Finding repositories on {device}",
    "git_log": "Reading commit history on {device}",
    "git_diff_stat": "Measuring the change on {device}",
    "git_status": "Checking uncommitted work on {device}",
    "list_claude_code_sessions": "Listing coding sessions on {device}",
    "read_claude_code_session": "Reading a coding session on {device}",
    "connect_account": "Offering an account connection",
    "open_connected_site": "Opening a connected site",
    "read_connected_page": "Reading a page of a connected site",
    "fetch_connected_json": "Reading figures from a connected site",
    "run_provider_recipe": "Reading vendor usage through the signed-in session",
    "crawl_website": "Reading the website's pages",
    "website_audit": "Auditing the website",
    "website_traffic": "Reading website traffic",
    "finance_transactions": "Reading bank transactions",
    "finance_spend_summary": "Summarising spending",
    "query_platform_metrics": "Querying platform usage",
    "query_finances": "Querying finances",
    "query_vendor_usage": "Querying vendor usage",
    "make_chart": "Drawing a chart",
    "save_report": "Saving the report",
    "schedule_report": "Scheduling the report",
    "forecast_metric": "Forecasting",
    "github_activity": "Reading GitHub activity",
    "github_issues": "Reading GitHub issues",
    "github_pull_requests": "Reading pull requests",
    "check_data_server_connection": "Checking which machines are reachable",
    "discover_data_files": "Listing files on {device}",
    "preview_data_file": "Previewing a data file on {device}",
    "ingest_data_files": "Reading data files from {device}",
    "hydrate_ingested_data": "Loading saved data into the workspace",
    "list_persisted_data": "Checking previously saved data",
    "execute": "Running analysis code",
    "persist_created_artifact": "Saving the report or plot",
    "connect_data_server": "Connecting to a machine",
    "disconnect_data_server": "Disconnecting a machine",
    "write_todos": "Planning the steps",
    "ls": "Looking through the workspace",
    "read_file": "Reading a workspace file",
    "write_file": "Writing a workspace file",
    "edit_file": "Editing a workspace file",
    "glob": "Searching the workspace",
    "grep": "Searching the workspace",
    "update_avatar_identity_with_media": "Learning from the attached media",
    # Tools of the machine's own Model Context Protocol server, called from
    # inside the data-analysis tools above; they stream as nested tool events.
    "list_all_files": "Listing files on the machine",
    "get_file_info": "Checking a file on the machine",
    "preview_data": "Previewing data on the machine",
    "read_file_bytes": "Reading a file from the machine",
    "read_files_for_sandbox": "Copying files from the machine into the workspace",
}
_DEFAULT_DEVICE_PHRASE = "the connected machines"
def _record_tool_call_event(
    event: dict[str, Any], config: dict[str, Any], *, status: str, duration_ms: float
) -> None:
    """Log one tool call to the ``tool_calls`` table (fire-and-forget)."""
    try:
        from src.anubis.utils.analytics.tool_calls import record_tool_call
    except ImportError:
        return
    configurable = (config or {}).get("configurable", {}) or {}
    try:
        record_tool_call(
            user_id=configurable.get("user_id"),
            assistant_id=configurable.get("assistant_id"),
            thread_id=configurable.get("outer_thread_id") or configurable.get("thread_id"),
            tool_name=str(event.get("name") or ""),
            status=status,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.debug("Could not record a tool call", exc_info=True)


class _ToolCallTimerFallback:
    """Timer used when the analytics package is absent."""

    def start(self, run_id: Any) -> None:
        """Ignore the start of a run."""

    def finish(self, run_id: Any) -> float:
        """Report no duration."""
        return 0.0


try:
    from src.anubis.utils.analytics.tool_calls import ToolCallTimer as _ToolCallTimer

    _tool_call_timer: Any = _ToolCallTimer()
except ImportError:
    _tool_call_timer = _ToolCallTimerFallback()

_TOOL_FINISHED_ACTIVITY = "Thinking about the results"


def _describe_tool_activity(tool_name: str, tool_input: Any) -> str:
    """One short present-tense phrase saying what a tool call is doing.

    Only the phrase leaves the server: tool arguments (host paths, addresses)
    never ride on the frame, so the status can be shown to anyone who can see
    the conversation.
    """
    device_phrase = _DEFAULT_DEVICE_PHRASE
    if isinstance(tool_input, dict):
        device_label = tool_input.get("device_label")
        if isinstance(device_label, str) and device_label.strip():
            device_phrase = device_label.strip()
    template = _TOOL_ACTIVITY_DESCRIPTIONS.get(tool_name)
    if template is None:
        readable_name = tool_name.replace("_", " ").strip() or "a tool"
        return f"Using {readable_name}"
    return template.format(device=device_phrase)


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
        elif ev_name == "on_tool_start":
            # Say what the avatar is doing. A data-analysis turn spends most of
            # its wall-clock time inside tools and streams no reply token until
            # the analysis is done; without these frames the client can only
            # show a generic "thinking" indicator for the whole stretch.
            if STRUCTURED_OUTPUT_STREAM_TAG in (event.get("tags") or []):
                continue
            tool_name = event.get("name") or ""
            _tool_call_timer.start(event.get("run_id"))
            writer(
                {
                    "type": "status",
                    "text": _describe_tool_activity(
                        tool_name, (event.get("data") or {}).get("input")
                    ),
                    "tool": tool_name,
                }
            )
        elif ev_name == "on_tool_end":
            if STRUCTURED_OUTPUT_STREAM_TAG in (event.get("tags") or []):
                continue
            _record_tool_call_event(
                event, deep_agent_config, status="success",
                duration_ms=_tool_call_timer.finish(event.get("run_id")),
            )
            writer(
                {
                    "type": "status",
                    "text": _TOOL_FINISHED_ACTIVITY,
                    "tool": event.get("name") or "",
                }
            )
        elif ev_name == "on_tool_error":
            _record_tool_call_event(
                event, deep_agent_config, status="error",
                duration_ms=_tool_call_timer.finish(event.get("run_id")),
            )
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

    # Data-analysis capability gate: the SOLE condition is one or more saved MCP
    # connections that (a) the ``mcp_auto_adopt`` node bound for this user and
    # (b) are bound to THIS avatar. No environment switch, no per-avatar enable
    # flag — the connections are the gate. Unbound avatars (e.g. a test avatar)
    # get nothing; an unreachable machine degrades to an offline entry inside the
    # tools rather than costing the turn.
    analysis_bundle = None
    analysis_extra_tools: list[Any] | None = None
    # Exclusive to the user's own personal avatar: bound connections on a
    # demoted (no-longer-personal) avatar must NOT re-enable live MCP access.
    is_personal_avatar = _user_personal_avatar(config, state)
    connections = await bound_connections_for(
        runtime.store,
        state["user_state"]["user_id"],
        state["assistant_state"]["assistant_id"],
    )
    if is_personal_avatar and runtime.store is not None:
        # The explicit-connect tool is attached whether or not machines are
        # already connected: with several machines a user can have one connected
        # and another suppressed by an earlier explicit disconnect, and
        # "reconnect my laptop" has to work in that state.
        analysis_extra_tools = [
            build_connect_tool(
                deep_agent_run_context,
                state["user_state"]["user_id"],
                state["assistant_state"]["assistant_id"],
            )
        ]
    # Only machines reachable from THIS process receive tools. A relay-mode
    # record whose device holds no relay socket here is offline for this turn:
    # the prompt (``load_consciousness``) still names the machine as offline,
    # but the fan-out must never dial a bridge address that the OTHER API
    # process sharing the Postgres store wrote (development and production
    # share one store, with different bridge ports).
    live_connections = [
        connection for connection in connections if connection.online
    ]
    if live_connections and is_personal_avatar:
        analysis_bundle = build_analysis_backend(
            deep_agent_run_context,
            state["user_state"]["user_id"],
            state["assistant_state"]["assistant_id"],
            store=runtime.store,
        )
        from src.anubis.utils.tools.data_analysis.development_tools import (
            build_development_tools,
        )

        analysis_extra_tools = [
            *(analysis_extra_tools or []),
            *build_data_analysis_tools(
                deep_agent_run_context, analysis_bundle, live_connections
            ),
            # Git history and Claude Code sessions on the owner's machines, for
            # "what happened since", "what is in progress", "how long did it take".
            *build_development_tools(deep_agent_run_context, live_connections),
        ]
        # What the owner's own web browsing says about the owner. The
        # background sweep keeps this current on its own; the tool is for the
        # owner asking directly, and it waives the sweep's thresholds because
        # a person who asked has already decided the pass is worth running.
        if (
            str(getattr(deep_agent_run_context, "browsing_insights_enabled", "TRUE") or "")
            .strip()
            .upper()
            == "TRUE"
        ):
            from src.anubis.utils.browsing.tools import build_browsing_insight_tools

            analysis_extra_tools = [
                *analysis_extra_tools,
                *build_browsing_insight_tools(
                    deep_agent_run_context,
                    store=runtime.store,
                    user_id=state["user_state"]["user_id"],
                    assistant_id=state["assistant_state"]["assistant_id"],
                    target_name=str(
                        (state.get("assistant_state") or {}).get("assistant_name") or ""
                    ),
                ),
            ]

    # Browser capability gate: the process-wide environment switch
    # (BROWSER_TOOLS_ENABLED) AND the personal avatar. The browser follows
    # links out of the owner's own mail and will carry the owner's signed-in
    # sessions, so a shared avatar or a demoted one must never drive it. Keyed
    # on the outer workflow thread so each conversation browses in a dedicated
    # Chromium process (isolated pages, cookies, history). Returns an empty
    # list when the gate is off or Chromium is unavailable, so this line
    # never degrades the turn.
    browser_toolkit_tools: list[Any] = []
    if is_personal_avatar:
        browser_toolkit_tools = await get_browser_toolkit_tools(
            deep_agent_run_context, conversation_key=outer_thread
        )

    # Connected-account capability gate: the accounts bound to THIS avatar, and
    # only when this avatar is the owner's personal avatar. Same two-part gate as
    # the data-analysis tools, and for a stronger reason — these tools carry the
    # owner's own credentials, so a shared avatar or a demoted one must reach
    # nothing. No environment switch: a connected account is the gate. Tools are
    # built per account KIND through the factory table, so a mailbox, a custom
    # Model Context Protocol server, and whatever kind lands next all attach
    # through this one block.
    # Which avatar's connected accounts this turn may act on.
    #
    # The owner's accounts belong to ONE avatar — their personal one. When the
    # owner is talking to a different avatar of their own, that avatar does not
    # get its own copy of the credentials; the personal avatar acts on its
    # behalf, and any account connected during the conversation binds to the
    # personal avatar rather than scattering credentials across every avatar the
    # owner has made. The owner is always themselves, whichever of their avatars
    # they happen to be speaking to.
    #
    # A visitor is unaffected: _user_owns_avatar is false for them, so a shared
    # avatar still reaches nothing. That is the property the gate protects, and
    # brokering does not weaken it.
    accounts_avatar_id: str | None = None
    if is_personal_avatar:
        accounts_avatar_id = state["assistant_state"]["assistant_id"]
    elif _user_owns_avatar(config, state) and runtime.store is not None:
        from src.anubis.utils.personal_avatar import (
            personal_avatar_id_for_owner,
            read_personal_avatar_id,
        )
        from src.anubis.utils.runtime_handles import get_postgres_pool as _pool

        owner_id_for_broker = state["user_state"]["user_id"]
        # The pointer first because it is a single store read; the assistant
        # table second because the pointer is only written once the personal
        # avatar has taken a turn, and an owner who has never used theirs still
        # has one.
        accounts_avatar_id = await read_personal_avatar_id(
            runtime.store, owner_id_for_broker
        ) or await personal_avatar_id_for_owner(_pool(), owner_id_for_broker)

    mailbox_tools: list[Any] = []
    connection_tools: list[Any] = []
    if accounts_avatar_id is not None and runtime.store is not None:
        from src.anubis.utils.connected_accounts import bound_accounts_for
        from src.anubis.utils.connected_accounts.connection_tools import (
            build_connection_tools,
        )
        from src.anubis.utils.connected_accounts.tool_factories import (
            build_tools_for_accounts,
        )

        from src.anubis.utils.connected_accounts.store import stale_accounts_for
        from src.anubis.utils.runtime_handles import get_postgres_pool

        owner_user_id = state["user_state"]["user_id"]
        # The accounts (and any card raised this turn) belong to the personal
        # avatar, which may not be the avatar answering.
        answering_assistant_id = accounts_avatar_id
        connected_accounts = await bound_accounts_for(
            runtime.store,
            owner_user_id,
            answering_assistant_id,
        )
        stale_accounts = await stale_accounts_for(
            runtime.store, owner_user_id, answering_assistant_id
        )
        # A scheduled (unattended) run must never pause on a sign-in card.
        scheduled_run = bool(
            (config.get("configurable", {}) or {}).get("scheduled", False)
        )
        mailbox_tools = await build_tools_for_accounts(
            runtime.context,
            connected_accounts,
            store=runtime.store,
            pool=get_postgres_pool(),
            bundle=analysis_bundle,
        )
        # Business analytics: charts, reports, schedules, and — for the
        # platform administrator — platform metrics. Personal avatar only;
        # the pool is published by the lifespan.
        try:
            from src.anubis.utils.analytics.analytics_tools import (
                build_analytics_tools,
            )

            mailbox_tools = [
                *mailbox_tools,
                *build_analytics_tools(
                    runtime.context,
                    store=runtime.store,
                    pool=get_postgres_pool(),
                    user_id=owner_user_id,
                    assistant_id=answering_assistant_id,
                    connected_accounts=connected_accounts,
                    analysis_bundle=analysis_bundle,
                    thread_id=outer_thread,
                    timezone_name=(config.get("configurable", {}) or {}).get(
                        "user_timezone"
                    ),
                ),
            ]
        except ImportError:
            logger.debug("Analytics tools are not installed; skipping")
        except Exception:
            logger.exception("Could not build analytics tools; skipping")
        try:
            from src.anubis.utils.analytics.development_report import (
                build_development_report_tools,
            )

            mailbox_tools = [
                *mailbox_tools,
                *build_development_report_tools(
                    deep_agent_run_context,
                    live_connections=live_connections,
                    connected_accounts=connected_accounts,
                    store=runtime.store,
                ),
            ]
        except ImportError:
            pass
        except Exception:
            logger.exception("Could not build the development report tool; skipping")
        # The agent inbox is the personal avatar's: report and resolve pending
        # items in conversation, or trigger a poll now.
        from src.anubis.utils.inbox.inbox_tools import build_inbox_tools
        from src.anubis.utils.inbox.repository import get_inbox_repository

        if get_inbox_repository() is not None:
            mailbox_tools = [
                *mailbox_tools,
                *build_inbox_tools(
                    runtime.context,
                    user_id=owner_user_id,
                    assistant_id=answering_assistant_id,
                ),
            ]
        # Offered whether or not anything is connected. The owner with no mailbox
        # is precisely the owner who needs to connect one, so gating the connect
        # tool on having a connection would leave no way in.
        connection_tools = build_connection_tools(
            runtime.context,
            store=runtime.store,
            user_id=owner_user_id,
            assistant_id=answering_assistant_id,
            connected_accounts=connected_accounts,
            stale_accounts=stale_accounts,
            allow_interrupt=not scheduled_run,
        )

    # Making a plan: finding a real, named place to go. Gated on ownership, not
    # on the personal-avatar flag and not on any connection — the owner asking an
    # avatar they own to plan something is exactly the case, and the search runs
    # whether or not a calendar has been connected yet, so the avatar can offer
    # a real place and THEN offer to connect the calendar it needs to book into.
    # A visitor on a shared avatar never plans on the owner's behalf.
    place_tools: list[Any] = []
    if _user_owns_avatar(config, state):
        try:
            from src.anubis.utils.scheduling import build_place_tools

            place_tools = build_place_tools(runtime.context)
        except Exception:  # noqa: BLE001 - planning is never worth a failed turn
            logger.exception("Could not build the place-finding tool; skipping")

    # Learning from media in conversation: the creator of THIS avatar, never a
    # visitor. Personal and non-personal avatars alike — every avatar's identity
    # is taught by its creator. The subscription tier (UPLOAD capability) and
    # the upload allotment are enforced when the tool runs, by the same code
    # path the settings upload uses, so no per-request flag is needed here.
    identity_media_tools: list[Any] = []
    if _user_owns_avatar(config, state):
        from src.anubis.utils.tools.identity.identity_media_tools import (
            build_identity_media_tools,
        )

        identity_media_tools = build_identity_media_tools(
            deep_agent_run_context,
            user_id=state["user_state"]["user_id"],
            assistant_id=state["assistant_state"]["assistant_id"],
            assistant_ctx=dict(config.get("configurable", {}).get("assistant_ctx") or {}),
            thread_id=outer_thread,
        )

    # Fresh-look gate: the browser reported a live webcam or screen share on
    # THIS turn. No environment switch and no avatar gate — a share is the gate,
    # and it is the whole gate, because the tool pauses the run to ask the
    # browser for a frame and only a browser with something live can answer.
    # A turn with nothing shared never sees the tool, so ordinary messaging
    # neither pays for the tool's description nor risks the pause.
    from src.anubis.utils.ambient.observations import (
        ambient_details,
        is_ambient_observation,
        is_speech_observation,
    )
    from src.anubis.utils.tools.vision.look_tools import build_look_tools

    # An ambient observation IS a fresh look — the frame that started this turn
    # was captured seconds ago — so an observation turn is not offered another
    # one. Only a turn the conversation partner started can ask to look.
    answering_an_observation = bool(
        (state.get("messages") or [])
        and is_ambient_observation((state.get("messages") or [])[-1])
    )
    # A look is a pause, and a pause needs a checkpointer to be persisted,
    # surfaced to the browser, and resumed. Without one (``langgraph dev``, a
    # run outside the lifespan) the pause would strand the turn, so the tool is
    # not offered at all and the avatar answers from the observations it has.
    look_tools = (
        []
        if answering_an_observation or checkpointer is None
        else build_look_tools(
            runtime.context,
            live_shares=(config.get("configurable", {}) or {}).get("live_shares"),
            # With nothing live the tool still attaches on a conversation that
            # holds observations, so the avatar can CHECK whether a source is
            # in view rather than assert it from a description that is history.
            conversation_has_scene_observations=any(
                is_ambient_observation(message)
                and not is_speech_observation(ambient_details(message) or {})
                for message in (state.get("messages") or [])
            ),
            may_control_shares=bool(
                (config.get("configurable", {}) or {}).get("may_control_shares")
            ),
            # What the browser can open for ONE look right now: the camera when
            # its peek permission is granted in this browser, the desktop when
            # the person granted a desktop peek and the browser still holds it.
            peekable_shares=(config.get("configurable", {}) or {}).get(
                "peekable_shares"
            ),
        )
    )

    # Scene narration: the accessibility mode a conversation partner who cannot
    # see the scene switches on by asking. Gated on one thing only — the
    # browser reported the ``scene_narration`` field, which is how it says it
    # can point a camera and read descriptions aloud. Every avatar on such a
    # browser gets the tool, the personal avatar and the help avatar included:
    # the switch belongs to the person listening, not to any one avatar. An
    # observation turn is not offered it, for the same reason it is not offered
    # a look — the person did not speak, so there is nothing to have asked.
    from src.anubis.utils.tools.vision.accessibility_tools import (
        build_scene_narration_tools,
    )

    scene_narration_tools = (
        []
        if answering_an_observation
        else build_scene_narration_tools(
            runtime.context,
            scene_narration=(config.get("configurable", {}) or {}).get(
                "scene_narration"
            ),
            # How often the browser is describing the scene right now, so
            # "describe more often" becomes a number instead of a guess.
            scene_narration_seconds=(config.get("configurable", {}) or {}).get(
                "scene_narration_seconds"
            ),
        )
    )

    extra_tools = [
        *(analysis_extra_tools or []),
        *browser_toolkit_tools,
        *mailbox_tools,
        *connection_tools,
        *identity_media_tools,
        *place_tools,
        *look_tools,
        *scene_narration_tools,
    ]
    deep_agent = build_avatar_deep_agent(
        runtime.context,
        checkpointer=checkpointer,
        store=runtime.store,
        extra_tools=extra_tools or None,
        backend=analysis_bundle.backend if analysis_bundle is not None else None,
    )
    # Charts made with ``make_chart`` during this turn are collected on a
    # context variable and attached to the reply after the run.
    try:
        from src.anubis.utils.analytics.charts import TurnChartCollector

        TurnChartCollector.begin_turn()
    except ImportError:
        pass
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


def _latest_ambient_observation(messages: list) -> dict[str, Any] | None:
    """The triage record of the latest human turn when that turn is ambient."""
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            return ambient_details(message)
    return None


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
        _attach_token_usage_metadata(
            final_message, new_messages, context=runtime.context or GlobalContext()
        )
        if config.get("configurable", {}).get("use_adapter_inference"):
            final_message.response_metadata = dict(final_message.response_metadata or {})
            final_message.response_metadata["is_adapter_inference"] = True
    # A reply to an ambient observation (a hidden webcam/screen turn the
    # conversation partner never typed) carries the observation's triage record
    # so the client can render a ``notify`` reply as a notification card — on
    # the live stream (``done`` forwards ``response_metadata``) and after a
    # reload (the field is checkpointed on the message).
    ambient_record = _latest_ambient_observation(state.get("messages") or [])
    if isinstance(final_message, AIMessage) and ambient_record is not None:
        final_message.response_metadata = dict(final_message.response_metadata or {})
        final_message.response_metadata["ambient"] = ambient_record
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
    # The model reads the conversation without past browser harvest turns (the
    # ``[neural-nexus:...]`` requests for follow-up chips or a description) and
    # without the JSON lists those produced: a thread that kept one from an
    # earlier browser build would otherwise teach the avatar to answer every
    # message as a JSON list, which the browser then hides as leaked JSON.
    model_facing_messages = without_stale_client_harvest_turns(list(state["messages"]))
    # Every webcam / screen observation is marked as the current view or an
    # earlier one, in this copy only. The observations themselves read as flat
    # present-tense descriptions — "screen: a terminal showing three
    # repositories" says nothing about whether that is the screen now or the
    # screen twenty minutes ago, before the share ended — and telling the two
    # apart is the difference between answering and inventing. Whether an
    # observation is current is true of the moment it is read, not of the
    # observation, so the marks are never written back to the thread.
    model_facing_messages = mark_view_currency(
        model_facing_messages,
        normalize_live_shares(
            (config.get("configurable", {}) or {}).get("live_shares")
        ),
    )
    deep_agent_input = {
        "messages": model_facing_messages,
        "system_message": list(state.get("system_message") or []),
        "user_identity_documents": list(state.get("user_identity_documents") or []),
        "assistant_identity_documents": list(
            state.get("assistant_identity_documents") or []
        ),
        "recalled_memory_documents": list(state.get("recalled_memory_documents") or []),
        "user_state": state["user_state"],
        "assistant_state": state["assistant_state"],
        "internal_thoughts": [],
        # Continuous learning: the immediate emotion reading and the running
        # sentiment summary of this turn, so a consciousness rebuild inside
        # the agent (after an identity or learning tool ran) keeps both sections.
        "current_user_emotions": state.get("current_user_emotions") or "",
        "current_conversation_sentiment": state.get("current_conversation_sentiment") or "",
        # The conversation's summarization event from earlier turns, so the
        # summarizer reuses the compaction instead of summarizing again.
        CONVERSATION_SUMMARY_EVENT_KEY: state.get(CONVERSATION_SUMMARY_EVENT_KEY),
        CONVERSATION_SUMMARY_SESSION_ID_KEY: state.get(
            CONVERSATION_SUMMARY_SESSION_ID_KEY
        ),
    }
    # Slice new messages from the deep agent's persisted conversation against the
    # length of what the deep agent was given — stable across the
    # interrupt/resume re-run.
    input_messages_count = len(model_facing_messages)

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

    # Connect cards ride the same channel: every ``connect_account`` result this
    # turn produced (or the card a "+"-menu acknowledgement turn carried) is
    # kept on the reply so the transcript shows "Gmail · Added · 6 tools" after
    # a reload instead of nothing.
    try:
        from src.anubis.utils.connected_accounts.connection_cards import (
            connection_acknowledgement_card,
            connection_records_from_messages,
        )

        connection_cards = connection_records_from_messages(
            new_messages
        ) or connection_acknowledgement_card(state.get("messages") or [])
        if connection_cards:
            final_message.response_metadata = dict(final_message.response_metadata or {})
            final_message.response_metadata["connections"] = connection_cards
    except Exception:
        logger.exception("Could not attach this turn's connection cards.")

    # Charts made with ``make_chart`` this turn, as specs the client renders.
    try:
        from src.anubis.utils.analytics.charts import TurnChartCollector

        turn_charts = TurnChartCollector.collect()
        if turn_charts:
            final_message.response_metadata = dict(final_message.response_metadata or {})
            final_message.response_metadata["charts"] = turn_charts
    except ImportError:
        pass
    except Exception:
        logger.exception("Could not attach this turn's charts.")

    update: dict[str, Any] = {
        "messages": [final_message],
        "internal_thoughts": [*intermediate, final_message],
    }

    # Carry the summarizer's event across turns. The deep agent saw the outer
    # conversation followed by this turn's intermediate tool messages, and only
    # the final reply is written back, so the cutoff is clamped to the outer
    # count before the event is stored on the outer thread.
    summary_event = clamp_summary_event(
        final_output.get(CONVERSATION_SUMMARY_EVENT_KEY),
        outer_message_count=input_messages_count,
    )
    if summary_event is not None:
        update[CONVERSATION_SUMMARY_EVENT_KEY] = summary_event
        update[CONVERSATION_SUMMARY_SESSION_ID_KEY] = final_output.get(
            CONVERSATION_SUMMARY_SESSION_ID_KEY
        )

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


async def mcp_auto_adopt(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Bind every reachable Neural Nexus machine to the user's personal avatar.

    Runs before ``load_consciousness`` so a machine that just came online is
    visible to both the capability prompt and the ``think`` gate in the same
    turn.

    Adoption is automatic and raises no interrupt. A daemon registers by calling
    this API with the user's OWN API key, so a registration is already proof that
    the machine belongs to the account — asking the user to approve each of four
    machines would add a consent step that the credential has already satisfied.

    A machine is adopted when ALL of the following hold, and the node otherwise
    returns an empty delta (a normal turn):

    - this is the user's own PERSONAL avatar (owner match AND the
      ``is_personal_avatar_of_creator`` flag) — never a visitor on someone
      else's avatar, and never the user's other, non-personal avatars;
    - the machine is reachable right now (a live relay socket, or a
      tunnel/local registration with a fresh heartbeat);
    - the machine is not already bound to this avatar;
    - the user has not explicitly disconnected this machine from this avatar.
      That suppression marker is what keeps an explicit disconnect from being
      silently undone on the very next conversation turn.
    """
    store = runtime.store
    if store is None:
        return {}

    user_id = state["user_state"]["user_id"]
    assistant_id = state["assistant_state"]["assistant_id"]

    if not _user_personal_avatar(config, state):
        return {}

    context = runtime.context or GlobalContext()
    available = await resolve_available_connections(store, user_id, context)
    if not available:
        return {}

    already_bound = {
        record.get("device_id")
        for record in await read_user_connections(store, user_id)
        if record.get("assistant_id") == assistant_id
        and record.get("status") == "connected"
    }
    suppressed = await suppressed_device_ids(store, user_id, assistant_id)

    for connection in available:
        if not connection.device_id:
            continue
        if connection.device_id in already_bound:
            continue
        if connection.device_id in suppressed:
            continue
        await save_user_connection(
            store, user_id, connection=connection, assistant_id=assistant_id
        )
        logger.info(
            "Adopted Model Context Protocol device %r (%s) for user %s on avatar %s",
            connection.device_label,
            connection.device_id,
            user_id,
            assistant_id,
        )
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

anubis_workflow.add_node("mcp_auto_adopt", mcp_auto_adopt)
anubis_workflow.add_node("load_consciousness", load_consciousness)
anubis_workflow.add_node("think", think)

""" ANUBIS WORKFLOW EDGES """

anubis_workflow.add_edge(START, "mcp_auto_adopt")
anubis_workflow.add_edge("mcp_auto_adopt", "load_consciousness")
anubis_workflow.add_edge("load_consciousness", "think")
anubis_workflow.add_edge("think", END)


# COERCION
# workflow.add_conditional_edges("respond", avatar_tools_condition, {'avatar_tools':'avatar_tools', END:"evaluate_response_quality"})
# workflow.add_edge("evaluate_response_quality", "update_response_metadata")
# workflow.add_edge("update_response_metadata", END)

anubis = anubis_workflow.compile()

message_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=MessagesState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

message_workflow.add_node("chat", message_interface)
message_workflow.add_node("resolve_human_message_images", resolve_human_message_images)
# Continuous learning: the user's latest message is observed (immediate
# sentiment, running conversation sentiment, engagement counters, pending
# learning marker) in parallel with image resolution, so the observation costs
# the turn only the slower of the two branches; the join waits for both.
message_workflow.add_node("observe_user", observe_user)
# AI monitoring: the latest human message is screened by the cheap OpenAI
# moderation endpoint in a third parallel branch, so the check hides underneath
# the two branches that already take longer and the turn costs only its slowest
# branch. The deep terms-of-service judge is NOT here — it reads the same message
# after the reply has streamed, from the API layer.
message_workflow.add_node(MODERATE_CONTENT_NODE, moderate_content_fast)
message_workflow.add_node("join_user_observation", join_user_observation)
message_workflow.add_node(REFUSE_FOR_VIOLATION_NODE, refuse_for_violation)
message_workflow.add_node("anubis", anubis)
message_workflow.add_node(AMBIENT_TRIAGE_NODE, ambient_triage)

message_workflow.add_edge(START, "chat")
message_workflow.add_edge("chat", "resolve_human_message_images")
message_workflow.add_edge("chat", "observe_user")
message_workflow.add_edge("chat", MODERATE_CONTENT_NODE)
message_workflow.add_edge(
    ["resolve_human_message_images", "observe_user", MODERATE_CONTENT_NODE],
    "join_user_observation",
)
# An ambient observation (a hidden webcam/screen turn sent through /message
# with ambient=true) is triaged before the avatar runs: ``ignore`` ends the run
# with the observation persisted as context, ``respond`` / ``notify`` reach the
# avatar. Every other turn goes straight to the avatar as before.
message_workflow.add_conditional_edges(
    "join_user_observation",
    route_after_moderation,
    {
        REFUSE_FOR_VIOLATION_NODE: REFUSE_FOR_VIOLATION_NODE,
        AMBIENT_TRIAGE_NODE: AMBIENT_TRIAGE_NODE,
        "anubis": "anubis",
    },
)
message_workflow.add_edge(REFUSE_FOR_VIOLATION_NODE, END)
message_workflow.add_conditional_edges(
    AMBIENT_TRIAGE_NODE,
    route_after_ambient_triage,
    {END: END, "anubis": "anubis"},
)
message_workflow.add_edge("anubis", END)

graph = message_workflow.compile()

graph.name = "Anubis"

ensure_huggingface_models_cached(GlobalContext())
ensure_nltk_corpora_cached()

__all__ = ["graph"]
