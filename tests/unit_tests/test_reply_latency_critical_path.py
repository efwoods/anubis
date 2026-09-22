"""Nothing but the reply sits in front of the avatar's first token.

Covers the four changes that took slow work off the reply's critical path:

* ``observe_user`` returns within the inline Go Emotions budget however long
  the conversation sentiment summary takes, and the summary still lands in the
  store on a detached task;
* the summary runs on the text inference model, not the classification model;
* ``message_graph_sse`` screens the message beside the reply, and a hard block
  replaces the streamed reply with the refusal;
* the pre-request token estimate uses a measurement of any age instead of
  rebuilding the system prompt.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

import src.anubis.graph as graph_module
import src.anubis.utils.nodes as nodes_module
from src.anubis.utils.background_tasks import drain_detached_tasks
from src.api import webapp as webapp_module

USER = "user-1"
AVATAR = "avatar-1"
THREAD = "thread-1"


def _observation_arguments(store):
    state = {
        "messages": [HumanMessage(content="How was the regatta?")],
        "user_state": {"user_id": USER},
        "assistant_state": {"assistant_id": AVATAR},
    }
    config = {"configurable": {"thread_id": THREAD}}
    runtime = SimpleNamespace(
        store=store,
        context=SimpleNamespace(
            observe_user_inline_emotion_budget_milliseconds=100,
            conversation_sentiment_per_turn_enabled="TRUE",
            current_emotion_enabled="FALSE",
        ),
    )
    return state, config, runtime


# ── observe_user ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_slow_conversation_summary_never_delays_observe_user(monkeypatch):
    summary_calls: list[str] = []

    async def fast_classify(text):
        return {"emotion": "joy", "base_emotion": "joy", "score": 0.9}

    async def slow_summary(store, user_id, assistant_id, thread_id, messages):
        await asyncio.sleep(0.5)
        summary_calls.append(thread_id)
        return {"sentiment_summary": "Upbeat."}

    monkeypatch.setattr(nodes_module, "classify_user_message_sentiment", fast_classify)
    monkeypatch.setattr(nodes_module, "update_current_conversation_sentiment", slow_summary)

    started_at = time.monotonic()
    update = await nodes_module.observe_user(*_observation_arguments(InMemoryStore()))
    elapsed_seconds = time.monotonic() - started_at

    assert elapsed_seconds < 0.2
    assert "joy" in update["current_user_emotions"]
    assert summary_calls == []
    await drain_detached_tasks()
    assert summary_calls == [THREAD]


@pytest.mark.asyncio
async def test_a_reading_that_misses_the_budget_is_left_out_of_the_prompt(monkeypatch):
    async def slow_classify(text):
        await asyncio.sleep(0.3)
        return {"emotion": "joy", "base_emotion": "joy", "score": 0.9}

    async def no_summary(store, user_id, assistant_id, thread_id, messages):
        return None

    monkeypatch.setattr(nodes_module, "classify_user_message_sentiment", slow_classify)
    monkeypatch.setattr(nodes_module, "update_current_conversation_sentiment", no_summary)

    update = await nodes_module.observe_user(*_observation_arguments(InMemoryStore()))

    # An empty string, not a missing key: the previous turn's checkpointed
    # reading must not be shown as this message's.
    assert update == {"current_user_emotions": ""}
    await drain_detached_tasks()


@pytest.mark.asyncio
async def test_summary_updates_for_one_thread_run_one_at_a_time(monkeypatch):
    running_updates = 0
    most_concurrent_updates = 0

    async def counting_summary(store, user_id, assistant_id, thread_id, messages):
        nonlocal running_updates, most_concurrent_updates
        running_updates += 1
        most_concurrent_updates = max(most_concurrent_updates, running_updates)
        await asyncio.sleep(0.05)
        running_updates -= 1

    monkeypatch.setattr(nodes_module, "update_current_conversation_sentiment", counting_summary)

    await asyncio.gather(
        *(
            nodes_module._update_conversation_sentiment_one_at_a_time(
                None, USER, AVATAR, THREAD, []
            )
            for _ in range(3)
        )
    )

    assert most_concurrent_updates == 1
    assert THREAD not in nodes_module._conversation_sentiment_summary_locks


# ── the summary's model ─────────────────────────────────────────────────────


def _captured_chat_openai_arguments(monkeypatch, **init_model_arguments):
    import langchain_openai

    from src.anubis.utils import model as model_module

    captured_arguments: dict = {}

    class RecordingChatOpenAI:
        def __init__(self, **keyword_arguments):
            captured_arguments.update(keyword_arguments)

        def with_structured_output(self, schema):
            return self

        def with_config(self, **keyword_arguments):
            return self

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", RecordingChatOpenAI)
    monkeypatch.setenv("MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("CLASSIFICATION_MODEL", "gpt-5-nano")
    model_module.init_model(response_format=dict, **init_model_arguments)
    return captured_arguments


def test_summaries_run_on_the_inference_model_without_reasoning(monkeypatch):
    captured_arguments = _captured_chat_openai_arguments(
        monkeypatch, structured_output_on_inference_model=True
    )
    assert captured_arguments["model"] == "gpt-5.6-luna"
    assert captured_arguments["reasoning_effort"] == "none"
    assert "top_p" not in captured_arguments


def test_other_structured_calls_keep_the_classification_model(monkeypatch):
    captured_arguments = _captured_chat_openai_arguments(monkeypatch)
    assert captured_arguments["model"] == "gpt-5-nano"


# ── concurrent moderation on the streaming endpoint ─────────────────────────


class _RecordingGraph:
    """A graph that streams a reply and records what the endpoint writes back."""

    def __init__(self, events, token_delay_seconds=0.0):
        self._events = events
        self._token_delay_seconds = token_delay_seconds
        self.state_updates: list[tuple[dict, str]] = []

    async def astream(self, input, config, context, stream_mode, subgraphs):
        for event in self._events:
            await asyncio.sleep(self._token_delay_seconds)
            yield event

    async def aget_state(self, config):
        return SimpleNamespace(next=(), tasks=(), interrupts=())

    async def aupdate_state(self, config, values, as_node=None):
        self.state_updates.append((values, as_node))


async def _stream_frames(graph, monkeypatch):
    async def no_thread_metadata(*arguments, **keyword_arguments):
        return None

    monkeypatch.setattr(webapp_module, "_write_thread_metadata", no_thread_metadata)
    monkeypatch.setattr(webapp_module, "get_client", lambda headers=None: None)
    generator = webapp_module.message_graph_sse(
        graph,
        HumanMessage(content="a message"),
        {"configurable": {"thread_id": THREAD}},
        SimpleNamespace(),
        thread_id=THREAD,
        user_id=USER,
        assistant_id=AVATAR,
        conversation_title_value=THREAD,
        start_time_ns=time.time_ns(),
        request_id="request-1",
        langgraph_client_headers={},
        app_state=None,
        current_user=None,
        include_usage_metrics=False,
    )
    frames = []
    async for chunk in generator:
        if chunk.startswith("data: "):
            frames.append(json.loads(chunk[len("data: ") :].strip()))
    return frames


@pytest.mark.asyncio
async def test_a_hard_block_mid_stream_replaces_the_reply_with_the_refusal(monkeypatch):
    async def blocking_screen(message_text, context):
        await asyncio.sleep(0.05)
        return {"violation": True, "reasoning": "blocked"}

    monkeypatch.setattr(graph_module, "screen_message_for_hard_block", blocking_screen)
    graph = _RecordingGraph(
        [((), "custom", {"type": "assistant_token", "text": "word "})] * 20,
        token_delay_seconds=0.02,
    )

    frames = await _stream_frames(graph, monkeypatch)

    frame_types = [frame["type"] for frame in frames]
    assert frame_types[0] == "turn_started"
    assert frame_types[-2:] == ["moderation_violation", "done"]
    done = frames[-1]
    assert done["moderation"]["banned"] is True
    assert "terms of service" in done["content"]
    # The partial reply is discarded: only the refusal is written, as the
    # avatar node's output.
    (written_values, written_as_node), = graph.state_updates
    assert written_as_node == "anubis"
    assert [message.content for message in written_values["messages"]] == [done["content"]]


@pytest.mark.asyncio
async def test_a_hard_block_after_the_reply_finished_removes_the_finished_reply(monkeypatch):
    screen_may_answer = asyncio.Event()

    async def late_blocking_screen(message_text, context):
        await screen_may_answer.wait()
        return {"violation": True, "reasoning": "blocked"}

    monkeypatch.setattr(graph_module, "screen_message_for_hard_block", late_blocking_screen)
    finished_reply = AIMessage(content="whole reply", id="reply-1")
    monkeypatch.setattr(
        webapp_module, "_latest_ai_from_stream_update", lambda payload: finished_reply
    )
    graph = _RecordingGraph([((), "updates", {"anubis": {"messages": [finished_reply]}})])

    stream_task = asyncio.create_task(_stream_frames(graph, monkeypatch))
    await asyncio.sleep(0.05)
    screen_may_answer.set()
    frames = await stream_task

    assert frames[-1]["type"] == "done"
    assert frames[-1]["moderation"]["banned"] is True
    (written_values, _written_as_node), = graph.state_updates
    removed_message, refusal_message = written_values["messages"]
    assert removed_message.id == "reply-1"
    assert refusal_message.content == frames[-1]["content"]


@pytest.mark.asyncio
async def test_a_clean_screen_leaves_the_stream_untouched_and_reports_latency(monkeypatch):
    graph = _RecordingGraph([((), "custom", {"type": "assistant_token", "text": "hello"})])

    frames = await _stream_frames(graph, monkeypatch)

    assert [frame["type"] for frame in frames] == ["turn_started", "assistant_token", "done"]
    assert graph.state_updates == []
    latency_breakdown = frames[-1]["latency_breakdown_ms"]
    assert {"graph_started", "first_token", "done"} <= set(latency_breakdown)


def test_the_inline_screen_stands_down_when_the_caller_screens():
    assert graph_module.moderation_is_screened_by_caller(
        {"configurable": {graph_module.CONTENT_MODERATION_SCREENED_BY_CALLER_KEY: True}}
    )
    assert not graph_module.moderation_is_screened_by_caller({"configurable": {}})


# ── the pre-request token estimate ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stale_system_prompt_measurement_is_used_without_rebuilding(monkeypatch):
    from src.anubis.utils.billing import system_prompt_estimate_cache as cache_module

    cache_module.record_system_prompt_token_estimate("user-stale", "avatar-stale", "word " * 30)
    recorded_at, estimated_tokens = cache_module._system_prompt_estimate_cache[
        ("user-stale", "avatar-stale")
    ]
    cache_module._system_prompt_estimate_cache[("user-stale", "avatar-stale")] = (
        recorded_at - 10_000,
        estimated_tokens,
    )

    async def forbidden_rebuild(*arguments, **keyword_arguments):
        raise AssertionError("the system prompt must not be rebuilt on the request path")

    monkeypatch.setattr(nodes_module, "build_system_prompt_text_for_estimation", forbidden_rebuild)

    measured_tokens = await webapp_module._measure_system_prompt_tokens_for_request(
        SimpleNamespace(),
        {"configurable": {"user_id": "user-stale", "assistant_id": "avatar-stale"}},
        "hello",
    )

    assert measured_tokens == estimated_tokens
    cache_module.invalidate_system_prompt_token_estimate("user-stale", "avatar-stale")
