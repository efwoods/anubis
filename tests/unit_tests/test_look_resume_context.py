"""A look must survive its own pause.

``look_now`` pauses the run and is answered by a SECOND request — ``POST
/message/{assistant_id}/resume`` — and the resumed run rebuilds every tool from
THAT request's configuration. The rebuilt ``look_now`` decides between taking
the pause (and so collecting the frame this very request is carrying) and
reporting that there is nothing to look at, purely from three fields:
``live_shares``, ``peekable_shares`` and ``may_control_shares``.

When the resume did not carry them, the rebuilt tool believed the browser was
sharing nothing, returned ``not_shared`` before ever reaching the ``interrupt``
holding the browser's answer, and the described frame was discarded unread. To
the person that looked like an avatar opening their camera and then saying it
could not see anything — which is the exact failure ``look_now`` exists to
prevent, arriving through the back door.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict

from src.anubis.utils.tools.vision.look_tools import build_look_tools
from src.api.look_context import LookContext, LookContextRegistry


async def _resume_a_look_with(configurable: dict) -> dict:
    """Pause a look on a live webcam, then resume it with this configuration.

    The pause is taken with the share report the turn really had; the resume
    rebuilds the tool from ``configurable``, exactly as ``think`` does.
    """
    turn_report = {"live_shares": "webcam", "may_control_shares": True}
    built: dict[str, object] = {"configurable": turn_report}

    class LookState(TypedDict):
        answer: Annotated[list, lambda left, right: (left or []) + (right or [])]

    async def take_a_look(state: LookState):
        # Rebuilt on every entry, which is the whole point: the resumed run
        # re-enters this node and builds the tool again from what it is given.
        report = built["configurable"]
        tool = build_look_tools(
            None,
            live_shares=report.get("live_shares"),
            may_control_shares=bool(report.get("may_control_shares")),
            peekable_shares=report.get("peekable_shares"),
            conversation_has_scene_observations=bool(
                report.get("conversation_has_scene_observations")
            ),
        )[0]
        return {"answer": [await tool.ainvoke({"sources": ["webcam"]})]}

    builder = StateGraph(LookState)
    builder.add_node("take_a_look", take_a_look)
    builder.add_edge(START, "take_a_look")
    builder.add_edge("take_a_look", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "resume-thread"}}

    paused = await graph.ainvoke({"answer": []}, config)
    assert paused["__interrupt__"], "the look should have paused the run"
    built["configurable"] = configurable
    resumed = await graph.ainvoke(
        Command(
            resume={
                "type": "looked",
                "observations": [
                    {"source": "webcam", "description": "a person at a desk"}
                ],
            }
        ),
        config,
    )
    (answer,) = resumed["answer"]
    return answer


def test_a_resume_that_forgets_everything_has_no_tool_left_to_answer_with():
    # The bluntest shape of the regression: with no share report and nothing in
    # the thread, the resumed run does not rebuild the tool at all, and the
    # paused tool call has nothing left to execute.
    assert build_look_tools(None, live_shares="", may_control_shares=False) == []


@pytest.mark.asyncio
async def test_a_resume_that_forgets_what_was_shared_throws_the_look_away():
    # The quieter shape, and the one the person actually saw: the thread holds
    # earlier observations, so the tool IS rebuilt — believing the browser
    # shares nothing. It answers "not shared" and returns before reaching the
    # interrupt holding the frame. Keep this test: it is the only thing that
    # fails if the resume stops carrying the report.
    answer = await _resume_a_look_with({"conversation_has_scene_observations": True})
    assert answer["status"] == "not_shared", (
        "a rebuilt tool that believes nothing is shared never reads the frame"
    )
    assert "observations" not in answer


@pytest.mark.asyncio
async def test_a_resume_carrying_the_report_reads_the_frame_the_browser_sent():
    answer = await _resume_a_look_with(
        {"live_shares": "webcam", "may_control_shares": True}
    )
    assert answer["status"] == "looked"
    assert answer["observations"][0]["description"] == "a person at a desk"


@pytest.mark.asyncio
async def test_a_peeked_camera_also_needs_its_report_back():
    # Nothing is shared; the camera is opened for the one look. The resume has
    # to report the same permission or the peeked frame is discarded too.
    answer = await _resume_a_look_with(
        {"live_shares": "", "peekable_shares": '["webcam"]'}
    )
    assert answer["status"] == "looked"
    assert answer["opened_for_this_look"] == ["webcam"]


# --- The registry that answers a client which forgot ------------------------


def test_what_a_turn_reported_is_remembered_for_its_own_pause():
    registry = LookContextRegistry()
    registry.remember("t1", LookContext(live_shares="webcam", may_control_shares=True))
    recalled = registry.recall("t1")
    assert recalled is not None
    assert recalled.live_shares == "webcam" and recalled.may_control_shares is True


def test_a_turn_reporting_nothing_clears_what_the_thread_had():
    # A browser that has stopped sharing and stopped allowing looks must not be
    # answered out of what it allowed earlier.
    registry = LookContextRegistry()
    registry.remember("t1", LookContext(live_shares="webcam"))
    registry.remember("t1", LookContext())
    assert registry.recall("t1") is None


def test_a_context_nobody_came_back_for_expires():
    registry = LookContextRegistry(memory_seconds=60.0)
    registry.remember("t1", LookContext(live_shares="screen"), now=1_000.0)
    assert registry.recall("t1", now=1_030.0) is not None
    assert registry.recall("t1", now=1_100.0) is None


def test_the_table_cannot_grow_without_bound():
    registry = LookContextRegistry(max_threads=3)
    for index in range(10):
        registry.remember(
            f"t{index}", LookContext(live_shares="webcam"), now=1_000.0 + index
        )
    surviving = [
        f"t{index}"
        for index in range(10)
        if registry.recall(f"t{index}", now=1_010.0)
    ]
    assert surviving == ["t7", "t8", "t9"], "the oldest go first"


def test_a_thread_nobody_named_is_not_remembered():
    registry = LookContextRegistry()
    registry.remember(None, LookContext(live_shares="webcam"))
    registry.remember("  ", LookContext(live_shares="webcam"))
    assert registry.recall(None) is None and registry.recall("  ") is None
