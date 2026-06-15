# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END

# NOTE: ``retrieval_graph`` was imported here but never referenced in this module.
# Runtime logs showed ``from ... retrieval_graph import retrieval_graph`` alone took
# ~13 s per cold webapp worker because ``retrieval_graph`` pulls heavy retrieval deps.

from dotenv import load_dotenv

load_dotenv()


from langchain_core.runnables import RunnableConfig, Runnable
from src.anubis.utils.utility import (
    extract_user_id_assistant_id,
    configure_assistant_context,
)

from src.anubis.utils.schema import RouteDecision

from pydantic import BaseModel, Field
from typing import Literal
from langgraph.types import Command

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# NOTE: ``langchain.agents.create_agent`` and ``SummarizationMiddleware`` were
# imported here and below but never referenced in this file (the only
# ``create_agent(...)`` site is commented out at ~line 313).  Removed to skip the
# eager ``langchain.agents`` package load on every cold start.
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from src.anubis.utils.model import init_model

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.huggingface_prefetch import (
    ensure_huggingface_models_cached,
)
from src.anubis.utils.utility import format_docs

from langgraph.graph import MessagesState

from src.anubis.utils.nodes import load_consciousness, resolve_human_message_images
from src.anubis.utils.deep_agent import build_avatar_deep_agent
from src.anubis.utils.emotion_mapping import EMOTION_MAPPING
from src.anubis.utils.prompts.legal import TERMS_OF_SERVICE, PRIVACY_POLICY
import numpy as np
import pickle

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


def _attach_go_emotions_metadata(avatar_response: AIMessage) -> None:
    """Mutate ``avatar_response.response_metadata`` with Go Emotions classifier output."""
    from src.anubis.utils.emotion_classifier import classify_go_emotions

    sentiment = classify_go_emotions(avatar_response.content)
    if sentiment is None:
        return
    avatar_response.response_metadata = dict(avatar_response.response_metadata or {})
    avatar_response.response_metadata.update({"sentiment": sentiment})


async def _attach_analyzed_features(avatar_response: AIMessage, runtime: Runtime[GlobalContext], assistant_id: str) -> None:
    """ analyze the avatar_response features, compare against unmodified chatgpt responses and any existing direct quotes if possible.
    Update the metadata with the feature analysis and the results of comparison.
    """
    from src.anubis.utils.dataset.style_features import extract_style_features, compute_mahalanobis_distance
    import json

    avatar_response.response_metadata = dict(avatar_response.response_metadata or {})

    features_dict = extract_style_features(avatar_response.content)
    avatar_response.response_metadata.update({"features": features_dict})

    features_arr = np.array(list(features_dict.values()))
    baseline_response_threshold = runtime.context.baseline_response_threshold

    try:
        baseline_features_namespace = ("baseline_features_arr_list_str",)
        ground_truth_text_features_arr_namespace = (assistant_id, "ground_truth_text_features_arr_list_str")
        ground_truth_text_empirical_threshold_namespace = (assistant_id, "ground_truth_text_empirical_threshold_list_str")

        baseline_features_arr_item = await runtime.store.aget(baseline_features_namespace, key="baseline_features_arr_list_str")

        baseline_features_arr_list_str = getattr(baseline_features_arr_item, "value", None)

        # If the baseline_features_arr has not yet been stored, store the array:
        if not baseline_features_arr_list_str:
            _BASELINE_ANSWERS_RESPONSES_ARR_DIR = "src/anubis/utils/dataset/baseline_features_arr.npy"
            baseline_features_arr = np.load(_BASELINE_ANSWERS_RESPONSES_ARR_DIR, allow_pickle=False)
            
            baseline_features_arr_list_str = json.dumps(baseline_features_arr.tolist())

            await runtime.store.aput(baseline_features_namespace, key="baseline_features_arr_list_str", value=baseline_features_arr_list_str)

        # Convert from str to np.array
        if isinstance(baseline_features_arr_list_str, str):
            baseline_features_arr = np.array(json.loads(baseline_features_arr_list_str))
        else:
            baseline_features_arr = np.array(baseline_features_arr_list_str)

        # Compare the difference between the synthetic text and the unaltered chatgpt responses
        M_d_square_synth_from_baseline_chatgpt = compute_mahalanobis_distance(features_arr, baseline_features_arr)
        
        avatar_response.response_metadata.update({"significantly_different_from_baseline_chatgpt_response":bool(M_d_square_synth_from_baseline_chatgpt[0] > baseline_response_threshold)})

        # Explain the result
        from src.anubis.utils.utility import compute_shap_values_against_baseline
        
        shap_values_dict = await compute_shap_values_against_baseline(features_arr)
        avatar_response.response_metadata.update(shap_values_dict)

        # Compare against ground truth quotes if available:
        ground_truth_text_features_arr_item = await runtime.store.aget(
            ground_truth_text_features_arr_namespace, 
            key="ground_truth_text_features_arr_list_str")

        ground_truth_text_features_arr_list_str = getattr(ground_truth_text_features_arr_item,"value", None)

        ground_truth_text_empirical_threshold_item = await runtime.store.aget(
            ground_truth_text_empirical_threshold_namespace, 
            key="ground_truth_text_empirical_threshold_list_str")
        
        ground_truth_text_empirical_threshold_list_str = getattr(ground_truth_text_empirical_threshold_item,"value", None)

        if ground_truth_text_features_arr_list_str and ground_truth_text_empirical_threshold_list_str:
            
            # Convert from str to np format
            if isinstance(ground_truth_text_features_arr_list_str, str):
                ground_truth_text_features_arr = np.array(json.loads(ground_truth_text_features_arr_list_str))
            else:
                ground_truth_text_features_arr = np.array(ground_truth_text_features_arr_list_str)

            if isinstance(ground_truth_text_empirical_threshold_list_str, str):
                ground_truth_text_empirical_threshold = np.array(json.loads(ground_truth_text_empirical_threshold_list_str)).flatten()
            else:
                ground_truth_text_empirical_threshold = np.array(ground_truth_text_empirical_threshold_list_str).flatten()
            
            # Compute the difference between the synthetic text and the direct quotes.
            M_d_square_synth_from_ground_truth_corpus = compute_mahalanobis_distance(features_arr, ground_truth_text_features_arr)

            avatar_response.response_metadata.update({"no_significant_difference_from_direct_quotes":bool(M_d_square_synth_from_ground_truth_corpus[0] < ground_truth_text_empirical_threshold)})

    except Exception as e:
        logger.error(f"error analyzing features: {e}")
        raise Exception(f"error analyzing features: {e}")


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

    return {
        "messages": state["messages"],
        "assistant_state": assistant_state,
        "user_state": user_state,
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

    Returns:
        Outer-graph state delta with:

        - ``messages``: single final ``AIMessage`` (Go Emotions sentiment
          metadata attached).
        - ``internal_thoughts``: every intermediate ``AIMessage`` /
          ``ToolMessage`` the deep agent produced, for auditing.
        - ``system_message`` / identity-doc snapshots forwarded from the
          deep agent's final state so the outer state stays in sync.
    """
    deep_agent = build_avatar_deep_agent(runtime.context)

    deep_agent_input = {
        "messages": list(state["messages"]),
        "system_message": list(state.get("system_message") or []),
        "user_identity_documents": list(state.get("user_identity_documents") or []),
        "assistant_identity_documents": list(
            state.get("assistant_identity_documents") or []
        ),
        "recalled_memory_documents": list(
            state.get("recalled_memory_documents") or []
        ),
        "user_state": state["user_state"],
        "assistant_state": state["assistant_state"],
        "internal_thoughts": [],
    }
    input_messages_count = len(deep_agent_input["messages"])

    writer = get_stream_writer()
    # Per-LLM-call streaming buffers keyed by `run_id`. We emit tokens
    # incrementally as long as the running merged chunk shows no
    # tool_calls — once one appears, we know this call is a tool-planning
    # turn and silently drop the rest. Same heuristic the legacy node used,
    # applied independently to each LLM call in the deep agent's loop.
    stream_buffers: dict[str, dict[str, Any]] = {}
    final_output: dict[str, Any] | None = None

    async for event in deep_agent.astream_events(
        deep_agent_input,
        config=config,
        context=runtime.context,
        version="v2",
    ):
        ev_name = event.get("event")
        if ev_name == "on_chat_model_stream":
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
            # Capture the outermost graph's terminal output. The deep
            # agent's compiled graph is the chain we're streaming; its
            # final on_chain_end event carries the resulting state.
            data = event.get("data") or {}
            output = data.get("output")
            if isinstance(output, dict) and "messages" in output:
                final_output = output

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

    if isinstance(final_message, AIMessage) and not final_message.tool_calls:
        _attach_go_emotions_metadata(final_message)
    # TODO: Authenticity metrics: score the (already-streamed) reply against the
    # target author + ChatGPT baseline and attach to response_metadata. The
    # user has seen the reply by now, so this adds no perceived latency.
        await _attach_analyzed_features(final_message, runtime=runtime, assistant_id=state['assistant_state']['assistant_id'])


    update: dict[str, Any] = {
        "messages": [final_message],
        "internal_thoughts": [*intermediate, final_message],
    }

    for key in (
        "system_message",
        "user_identity_documents",
        "assistant_identity_documents",
        "recalled_memory_documents",
    ):
        if key in final_output and final_output[key] is not None:
            update[key] = final_output[key]

    return update

""" GRAPH """

# Build minimal graph: START -> load_consciousness -> think -> END
anubis_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=GlobalState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

""" ANUBIS WORKFLOW NODES """

anubis_workflow.add_node("load_consciousness", load_consciousness)
anubis_workflow.add_node("think", think)

""" ANUBIS WORKFLOW EDGES """

anubis_workflow.add_edge(START, "load_consciousness")
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
message_workflow.add_node(
    "resolve_human_message_images", resolve_human_message_images
)
message_workflow.add_node("anubis", anubis)

message_workflow.add_edge(START, "chat")
message_workflow.add_edge("chat", "resolve_human_message_images")
message_workflow.add_edge("resolve_human_message_images", "anubis")
message_workflow.add_edge("anubis", END)

graph = message_workflow.compile()

graph.name = "Anubis"

ensure_huggingface_models_cached(GlobalContext())

__all__ = ["graph"]
