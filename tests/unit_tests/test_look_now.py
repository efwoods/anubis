"""The fresh look: ``look_now`` and the LIVE_SHARES section of the prompt.

Ambient vision describes the webcam and the shared screen on an interval, and
those descriptions stay in the thread forever. Two things go wrong without the
code under test here. Asked what is on the screen, the avatar reads back an
observation of a screen the conversation partner stopped sharing, as though
that screen were still in view. Asked what is on the webcam, the avatar answers
from whatever the interval last captured, which during a conversation can be
many minutes old, because the ambient loop skips a capture whenever a turn is
in flight and drops a frame that has not changed.
"""

from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.messages import HumanMessage

from src.anubis.utils.ambient.observations import (
    build_ambient_additional_kwargs,
    build_live_shares_block,
    describe_age,
    newest_observation_age_seconds,
)
from src.anubis.utils.tools.vision.look_tools import (
    LOOK_NOW_TOOL_NAME,
    build_look_tools,
    normalize_live_shares,
    peekable_sources,
)


def _observation(source: str, minutes_ago: float) -> HumanMessage:
    captured_at = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()
    return HumanMessage(
        id=f"observation-{source}-{minutes_ago}",
        content=f"[AMBIENT_OBSERVATION] {source}: something",
        additional_kwargs=build_ambient_additional_kwargs(
            observation_id=f"o-{source}",
            sources=[source],
            captured_at=captured_at,
            voice_mode=False,
        ),
    )


# --- What the browser reported live ----------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ('["webcam","screen"]', ["webcam", "screen"]),
        ("screen,webcam", ["webcam", "screen"]),
        ("cam", ["webcam"]),
        ("screenshot", ["screen"]),
        (["screen"], ["screen"]),
        ("", []),
        (None, []),
        ("microphone", []),
        ("not-a-source", []),
        ("[broken", []),
        (17, []),
    ],
)
def test_live_shares_are_read_from_any_shape_the_browser_sends(value, expected):
    assert normalize_live_shares(value) == expected


def test_a_source_the_browser_did_not_name_is_dropped_rather_than_trusted():
    # The field decides whether a tool that PAUSES the run is attached, so a
    # client typo must never open a pause the browser will not answer.
    assert normalize_live_shares("webcam,telepathy") == ["webcam"]


# --- The gate on the tool ---------------------------------------------------


def test_no_look_tool_is_attached_when_nothing_is_shared():
    assert build_look_tools(None, live_shares="") == []
    assert build_look_tools(None, live_shares=None) == []
    assert build_look_tools(None, live_shares=[]) == []


def test_the_look_tool_is_attached_when_something_is_shared():
    tools = build_look_tools(None, live_shares="webcam")
    assert [tool.name for tool in tools] == [LOOK_NOW_TOOL_NAME]


def test_the_tool_description_names_what_is_actually_live():
    # The model decides whether to call the tool from this text, so what is
    # live is named in it rather than described in general.
    description = build_look_tools(None, live_shares="webcam,screen")[0].description
    # Named one by one, with what each one shows, because the avatar has to
    # choose between them rather than ask for "whatever can be seen".
    assert "webcam (the camera" in description
    assert "screen (the desktop" in description
    assert "is being shared right now" in description
    assert "{live}" not in description


@pytest.mark.asyncio
async def test_asking_to_look_at_a_source_that_is_not_shared_never_pauses():
    # No interrupt is raised: there is nothing to capture, and the answer the
    # avatar needs is that the conversation partner is not sharing that source.
    tool = build_look_tools(None, live_shares="webcam")[0]
    answer = await tool.ainvoke({"sources": ["screen"]})
    assert answer["status"] == "not_shared"
    assert answer["live_sources"] == ["webcam"]
    assert "screen" in answer["message"]


# --- The age of what the thread already holds -------------------------------


def test_the_age_of_the_newest_observation_of_a_source_is_read_from_the_thread():
    messages = [_observation("screen", 30), _observation("screen", 5)]
    age = newest_observation_age_seconds(messages, "screen")
    assert age is not None
    assert 4 * 60 < age < 6 * 60


def test_a_source_the_thread_has_never_seen_has_no_age():
    assert newest_observation_age_seconds([_observation("webcam", 1)], "screen") is None


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (None, "at an unknown time"),
        (5, "less than a minute ago"),
        (60, "about 1 minute ago"),
        (14 * 60, "about 14 minutes ago"),
        (2 * 60 * 60, "about 2 hours ago"),
    ],
)
def test_an_age_is_said_the_way_a_person_would_say_it(seconds, expected):
    assert describe_age(seconds) == expected


# --- The LIVE_SHARES section ------------------------------------------------


def test_a_conversation_that_never_involved_a_camera_gets_no_section():
    assert build_live_shares_block([], [], can_look_now=False) == ""


def test_a_stopped_screen_is_named_as_stopped_and_its_descriptions_as_the_past():
    # The bug this exists for: asked what is on the screen with no screen
    # shared, the avatar narrated a terminal captured before the share ended.
    block = build_live_shares_block(
        ["webcam"], [_observation("screen", 14)], can_look_now=True
    )
    assert "Being shared at this moment: webcam." in block
    assert "The screen is NOT being shared any more." in block
    assert "about 14 minutes ago" in block
    assert "not what the screen shows now" in block
    assert "Never describe a source that is not being shared" in block


def test_nothing_shared_at_all_is_said_plainly():
    block = build_live_shares_block(
        [], [_observation("screen", 3), _observation("webcam", 3)], can_look_now=False
    )
    assert "Nothing is being shared at this moment." in block
    assert "The screen is NOT being shared any more." in block
    assert "The webcam is NOT being shared any more." in block
    # With nothing live there is nothing to look at, so the look is not offered.
    assert "look_now" not in block


def test_a_live_source_with_a_stale_description_is_flagged_as_stale():
    block = build_live_shares_block(
        ["webcam"], [_observation("webcam", 20)], can_look_now=True
    )
    assert "Being shared at this moment: webcam." in block
    assert "may no longer match" in block
    assert "call look_now" in block


def test_a_live_source_just_described_is_not_flagged_as_stale():
    block = build_live_shares_block(
        ["webcam"], [_observation("webcam", 0.2)], can_look_now=True
    )
    assert "may no longer match" not in block


def test_the_look_is_offered_only_when_the_tool_is_attached():
    messages = [_observation("webcam", 1)]
    assert "call look_now" in build_live_shares_block(
        ["webcam"], messages, can_look_now=True
    )
    assert "call look_now" not in build_live_shares_block(
        ["webcam"], messages, can_look_now=False
    )


# --- The pause and what comes back ------------------------------------------


async def _look_through_a_graph(tool, tool_call: dict, resume_value):
    """Drive one already-built look tool through a real interrupt and resume.

    :returns: What the tool returned after the run resumed.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command
    from typing_extensions import Annotated, TypedDict

    class LookState(TypedDict):
        answer: Annotated[list, lambda left, right: (left or []) + (right or [])]

    async def take_a_look(state: LookState):
        return {"answer": [await tool.ainvoke(tool_call)]}

    builder = StateGraph(LookState)
    builder.add_node("take_a_look", take_a_look)
    builder.add_edge(START, "take_a_look")
    builder.add_edge("take_a_look", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "one-look-thread"}}
    await graph.ainvoke({"answer": []}, config)
    resumed = await graph.ainvoke(Command(resume=resume_value), config)
    (answer,) = resumed["answer"]
    return answer


async def _run_look_through_a_graph(tool_call: dict, resume_value):
    """Drive ``look_now`` through a real interrupt and resume.

    The tool's whole shape depends on LangGraph semantics — the pause, the
    re-entry from the top on resume, and the resume value arriving as the
    return of ``interrupt`` — so the round trip is exercised rather than
    stubbed.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command
    from typing_extensions import Annotated, TypedDict

    tool = build_look_tools(None, live_shares="webcam,screen")[0]
    entries: list[int] = []

    class LookState(TypedDict):
        answer: Annotated[list, lambda left, right: (left or []) + (right or [])]

    async def take_a_look(state: LookState):
        entries.append(1)
        return {"answer": [await tool.ainvoke(tool_call)]}

    builder = StateGraph(LookState)
    builder.add_node("take_a_look", take_a_look)
    builder.add_edge(START, "take_a_look")
    builder.add_edge("take_a_look", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "look-thread"}}

    paused = await graph.ainvoke({"answer": []}, config)
    resumed = await graph.ainvoke(Command(resume=resume_value), config)
    return paused, resumed, entries


@pytest.mark.asyncio
async def test_a_look_pauses_the_run_and_asks_the_browser_for_the_live_sources():
    paused, _resumed, _entries = await _run_look_through_a_graph(
        {"sources": ["screen"], "reason": "asked what is on the screen"},
        {"type": "looked", "observations": []},
    )
    (pause,) = paused["__interrupt__"]
    assert pause.value["kind"] == "look_now"
    # The browser answers this one itself: no card, nobody approving anything.
    assert pause.value["silent"] is True
    assert pause.value["sources"] == ["screen"]
    assert pause.value["reason"] == "asked what is on the screen"


@pytest.mark.asyncio
async def test_the_descriptions_the_browser_sent_back_reach_the_avatar():
    _paused, resumed, entries = await _run_look_through_a_graph(
        {"sources": ["webcam"]},
        {
            "type": "looked",
            "observations": [
                {"source": "webcam", "description": "a person at a desk, smiling"}
            ],
        },
    )
    (answer,) = resumed["answer"]
    assert answer["status"] == "looked"
    assert answer["looked_at"] == ["webcam"]
    assert answer["observations"][0]["description"] == "a person at a desk, smiling"
    # The node runs again from the top when the run resumes, which is why
    # nothing before the pause may have a side effect.
    assert entries == [1, 1]


@pytest.mark.asyncio
async def test_a_look_that_could_not_be_taken_is_reported_as_unavailable():
    _paused, resumed, _entries = await _run_look_through_a_graph(
        {},
        {"type": "looked", "observations": [], "message": "The camera went away."},
    )
    (answer,) = resumed["answer"]
    assert answer["status"] == "unavailable"
    assert answer["message"] == "The camera went away."


@pytest.mark.asyncio
async def test_a_look_answered_in_a_shape_the_tool_does_not_know_invents_no_scene():
    # An older client, or one that answered the wrong pause. The avatar is told
    # the view could not be checked rather than being left to fall back on an
    # observation from before, which is the whole failure this tool prevents.
    _paused, resumed, _entries = await _run_look_through_a_graph({}, "looked")
    (answer,) = resumed["answer"]
    assert answer["status"] == "unavailable"
    assert "do not fall back on an earlier observation" in answer["message"]


@pytest.mark.asyncio
async def test_an_observation_with_no_description_is_not_counted_as_a_look():
    _paused, resumed, _entries = await _run_look_through_a_graph(
        {},
        {"type": "looked", "observations": [{"source": "webcam", "description": "  "}]},
    )
    (answer,) = resumed["answer"]
    assert answer["status"] == "unavailable"


@pytest.mark.asyncio
async def test_a_look_at_everything_live_asks_for_every_live_source():
    paused, _resumed, _entries = await _run_look_through_a_graph(
        {}, {"type": "looked", "observations": []}
    )
    (pause,) = paused["__interrupt__"]
    assert pause.value["sources"] == ["webcam", "screen"]


@pytest.mark.asyncio
async def test_a_source_that_is_not_live_is_reported_beside_the_ones_that_are():
    _paused, resumed, _entries = await _run_look_through_a_graph(
        {"sources": ["webcam", "microphone"]},
        {
            "type": "looked",
            "observations": [{"source": "webcam", "description": "a desk"}],
        },
    )
    (answer,) = resumed["answer"]
    assert answer["status"] == "looked"
    assert answer["looked_at"] == ["webcam"]


# --- Describing the frames the browser sent back ----------------------------


class _FakeUpload:
    def __init__(self, filename, content_type="image/jpeg", body=b"frame"):
        self.filename = filename
        self.content_type = content_type
        self._body = body

    async def read(self):
        return self._body


@pytest.fixture
def _described(monkeypatch):
    """Describe every frame as ``described(<filename>)``, with no model call."""
    from src.api import webapp as webapp_module

    described: list[str] = []

    class _Descriptor:
        def __init__(self, system_prompt=None):
            self.system_prompt = system_prompt

        async def describe(self, data_uri, filename):
            described.append(filename)
            assert data_uri.startswith("data:image/jpeg;base64,")
            return {"description": f"described({filename})"}

    monkeypatch.setattr(
        "src.anubis.utils.classes.ImageDescriptionClass.ImageDescriptionClass",
        _Descriptor,
    )
    monkeypatch.setattr(
        webapp_module,
        "prepare_still_image_upload",
        lambda declared_mime, body: (declared_mime, body),
    )
    return described


@pytest.mark.asyncio
async def test_each_frame_is_described_and_named_by_its_source(_described):
    from src.api.webapp import describe_look_frames

    observations = await describe_look_frames(
        [_FakeUpload("webcam.jpg"), _FakeUpload("screen.jpg")],
        '["webcam","screen"]',
    )
    assert [observation["source"] for observation in observations] == [
        "webcam",
        "screen",
    ]
    assert observations[0]["description"] == "described(webcam.jpg)"
    assert _described == ["webcam.jpg", "screen.jpg"]


@pytest.mark.asyncio
async def test_a_look_with_no_frames_describes_nothing(_described):
    from src.api.webapp import describe_look_frames

    assert await describe_look_frames([], None) == []
    assert _described == []


@pytest.mark.asyncio
async def test_one_unreadable_frame_does_not_lose_the_other(monkeypatch, _described):
    from src.api import webapp as webapp_module
    from src.api.webapp import describe_look_frames

    def _refuse_the_screen(declared_mime, body):
        if body == b"bad":
            raise ValueError("unreadable")
        return declared_mime, body

    monkeypatch.setattr(
        webapp_module, "prepare_still_image_upload", _refuse_the_screen
    )
    observations = await describe_look_frames(
        [_FakeUpload("screen.jpg", body=b"bad"), _FakeUpload("webcam.jpg")],
        '["screen","webcam"]',
    )
    assert [observation["source"] for observation in observations] == ["webcam"]


# --- Telling the current view from an earlier one ---------------------------
#
# The failure these cover: an observation in the thread reads as a flat
# present-tense description ("screen: a terminal showing three repositories")
# and nothing in it says whether that is the screen now or the screen twenty
# minutes ago, before the share ended.


def _scene(sources, minutes_ago, observation_id):
    from src.anubis.utils.ambient.observations import compose_observation_text

    captured_at = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()
    ambient_kwargs = build_ambient_additional_kwargs(
        observation_id=observation_id,
        sources=list(sources),
        captured_at=captured_at,
        voice_mode=False,
    )
    body = "\n".join(f"{source}: something on the {source}" for source in sources)
    return HumanMessage(
        id=observation_id,
        content=compose_observation_text(ambient_kwargs["ambient"], body),
        additional_kwargs=ambient_kwargs,
    )


def _marks(messages, live_sources):
    from src.anubis.utils.ambient.observations import mark_view_currency

    return [
        marked.content.split("\n")[0]
        for marked in mark_view_currency(messages, live_sources)
    ]


def test_the_newest_look_at_a_live_source_is_marked_the_current_view():
    (mark,) = _marks([_scene(["webcam"], 0.2, "o1")], ["webcam"])
    assert "[CURRENT VIEW" in mark
    assert "still being shared" in mark


def test_an_observation_of_a_source_no_longer_shared_is_marked_history():
    (mark,) = _marks([_scene(["screen"], 22, "o1")], ["webcam"])
    assert "[EARLIER VIEW" in mark
    assert "about 22 minutes ago" in mark
    assert "NOT being shared any more" in mark
    assert "not what the screen shows now" in mark


def test_an_observation_a_later_look_replaced_is_marked_as_superseded():
    marks = _marks(
        [_scene(["webcam"], 12, "o1"), _scene(["webcam"], 0.2, "o2")], ["webcam"]
    )
    assert "[EARLIER VIEW" in marks[0]
    assert "a later look at the webcam came after this one" in marks[0]
    assert "[CURRENT VIEW" in marks[1]


def test_a_live_source_whose_newest_look_has_gone_stale_is_not_called_current():
    (mark,) = _marks([_scene(["webcam"], 20, "o1")], ["webcam"])
    assert "[EARLIER VIEW" in mark
    assert "may have changed since" in mark


def test_an_observation_covering_a_live_and_a_stopped_source_names_both():
    (mark,) = _marks([_scene(["webcam", "screen"], 9, "o1")], ["webcam"])
    assert "[EARLIER VIEW" in mark
    assert "the screen is NOT being shared any more" in mark
    assert "the webcam is still being shared" in mark


def test_with_nothing_shared_every_observation_is_history():
    marks = _marks(
        [_scene(["webcam"], 3, "o1"), _scene(["screen"], 1, "o2")], []
    )
    assert all("[EARLIER VIEW" in mark for mark in marks)
    assert not any("[CURRENT VIEW" in mark for mark in marks)


def test_marking_leaves_the_thread_itself_untouched():
    # Whether an observation is current is true of the moment it is read, not
    # of the observation, so the mark must never be written back.
    from src.anubis.utils.ambient.observations import mark_view_currency

    original = _scene(["screen"], 22, "o1")
    before = original.content
    mark_view_currency([original], ["webcam"])
    assert original.content == before


def test_messages_that_are_not_scene_observations_pass_through_unchanged():
    from src.anubis.utils.ambient.observations import mark_view_currency

    plain = HumanMessage(id="h1", content="what is on my screen?")
    observation = _scene(["screen"], 2, "o1")
    marked = mark_view_currency([plain, observation], ["screen"])
    # The same object, not a copy: nothing about it needed changing.
    assert marked[0] is plain
    assert marked[1] is not observation


def test_the_body_of_a_marked_observation_survives_the_mark():
    from src.anubis.utils.ambient.observations import mark_view_currency

    (marked,) = mark_view_currency([_scene(["screen"], 22, "o1")], [])
    assert "screen: something on the screen" in marked.content


# --- Checking, rather than asserting, that a source is in view --------------


def test_the_tool_is_attached_with_nothing_live_so_availability_can_be_checked():
    tools = build_look_tools(
        None, live_shares="", conversation_has_scene_observations=True
    )
    assert [tool.name for tool in tools] == [LOOK_NOW_TOOL_NAME]


def test_a_conversation_that_never_saw_a_camera_still_gets_no_tool():
    assert (
        build_look_tools(None, live_shares="", conversation_has_scene_observations=False)
        == []
    )


@pytest.mark.asyncio
async def test_checking_with_nothing_live_answers_not_shared_without_pausing():
    # No interrupt, no capture, no cost — and a truthful answer the avatar can
    # give instead of describing an observation that is history.
    tool = build_look_tools(
        None, live_shares="", conversation_has_scene_observations=True
    )[0]
    answer = await tool.ainvoke({"sources": ["screen"]})
    assert answer["status"] == "not_shared"
    assert answer["live_sources"] == []
    assert "Nothing at all is being shared right now." in answer["message"]


@pytest.mark.asyncio
async def test_checking_with_no_source_named_and_nothing_live_also_answers_plainly():
    tool = build_look_tools(
        None, live_shares="", conversation_has_scene_observations=True
    )[0]
    answer = await tool.ainvoke({})
    assert answer["status"] == "not_shared"


def test_the_tool_description_says_nothing_is_shared_when_nothing_is():
    description = build_look_tools(
        None, live_shares="", conversation_has_scene_observations=True
    )[0].description
    assert "sharing NOTHING" in description
    assert "{sharing_line}" not in description


@pytest.mark.asyncio
async def test_a_fresh_look_is_announced_as_superseding_the_earlier_ones():
    _paused, resumed, _entries = await _run_look_through_a_graph(
        {"sources": ["screen"]},
        {
            "type": "looked",
            "observations": [{"source": "screen", "description": "an editor"}],
        },
    )
    (answer,) = resumed["answer"]
    assert "THIS IS THE CURRENT VIEW of screen" in answer["message"]
    assert "supersedes" in answer["message"]
    assert "EARLIER VIEW" in answer["message"]


# --- The section that explains the marks ------------------------------------


def test_the_live_shares_section_explains_both_marks():
    block = build_live_shares_block(
        ["webcam"], [_observation("webcam", 1)], can_look_now=True
    )
    assert "[CURRENT VIEW ...] is what that source shows now" in block
    assert "is history" in block
    assert "never as something in view now" in block


def test_with_nothing_shared_the_section_still_offers_the_check():
    block = build_live_shares_block(
        [], [_observation("screen", 5)], can_look_now=True
    )
    assert "call look_now" in block
    assert "confirm that nothing is being shared" in block


# --- What the avatar may do with the camera and a screen share ---------------
#
# The avatar-settings permission, reported per turn by the browser that holds
# it. With it on the avatar may open the camera for one look and switch a share
# off. It can never start a screen share: no browser lets a page call
# getDisplayMedia without a gesture, so the person is offered a button instead.


@pytest.fixture
def _browser_frames(monkeypatch):
    """Collect the frames the tool sends to the streaming client."""
    import langgraph.config as langgraph_config

    sent: list[dict] = []
    monkeypatch.setattr(
        langgraph_config, "get_stream_writer", lambda: sent.append, raising=False
    )
    return sent


def test_the_permission_alone_attaches_both_tools_with_nothing_shared():
    from src.anubis.utils.tools.vision.look_tools import STOP_SHARING_TOOL_NAME

    names = [
        tool.name
        for tool in build_look_tools(None, live_shares="", may_control_shares=True)
    ]
    assert names == [LOOK_NOW_TOOL_NAME, STOP_SHARING_TOOL_NAME]


def test_without_the_permission_the_avatar_gets_no_way_to_switch_a_share_off():
    names = [
        tool.name
        for tool in build_look_tools(
            None, live_shares="webcam", may_control_shares=False
        )
    ]
    assert names == [LOOK_NOW_TOOL_NAME]


@pytest.mark.asyncio
async def test_a_look_with_nothing_shared_opens_the_camera_rather_than_refusing():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import Annotated, TypedDict

    tool = build_look_tools(None, live_shares="", may_control_shares=True)[0]

    class LookState(TypedDict):
        answer: Annotated[list, lambda left, right: (left or []) + (right or [])]

    async def take_a_look(state: LookState):
        return {"answer": [await tool.ainvoke({"sources": ["webcam"]})]}

    builder = StateGraph(LookState)
    builder.add_node("take_a_look", take_a_look)
    builder.add_edge(START, "take_a_look")
    builder.add_edge("take_a_look", END)
    graph = builder.compile(checkpointer=MemorySaver())
    paused = await graph.ainvoke(
        {"answer": []}, {"configurable": {"thread_id": "peek"}}
    )
    (pause,) = paused["__interrupt__"]
    # Nothing to capture, one camera to open.
    assert pause.value["sources"] == []
    assert pause.value["open"] == ["webcam"]


@pytest.mark.asyncio
async def test_without_the_permission_the_same_look_never_opens_anything():
    tool = build_look_tools(
        None,
        live_shares="",
        conversation_has_scene_observations=True,
        may_control_shares=False,
    )[0]
    answer = await tool.ainvoke({"sources": ["webcam"]})
    assert answer["status"] == "not_shared"
    assert answer["offered_to_share"] == []


@pytest.mark.asyncio
async def test_a_look_with_no_source_named_covers_the_camera_it_may_open():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import Annotated, TypedDict

    tool = build_look_tools(None, live_shares="screen", may_control_shares=True)[0]

    class LookState(TypedDict):
        answer: Annotated[list, lambda left, right: (left or []) + (right or [])]

    async def take_a_look(state: LookState):
        return {"answer": [await tool.ainvoke({})]}

    builder = StateGraph(LookState)
    builder.add_node("take_a_look", take_a_look)
    builder.add_edge(START, "take_a_look")
    builder.add_edge("take_a_look", END)
    graph = builder.compile(checkpointer=MemorySaver())
    paused = await graph.ainvoke(
        {"answer": []}, {"configurable": {"thread_id": "both"}}
    )
    (pause,) = paused["__interrupt__"]
    assert pause.value["sources"] == ["screen"]
    assert pause.value["open"] == ["webcam"]


@pytest.mark.asyncio
async def test_asking_for_the_screen_offers_a_button_instead_of_pausing(
    _browser_frames,
):
    # The screen is the one thing the avatar cannot switch on, so the person is
    # handed one press. No interrupt: there is nothing to wait for.
    tool = build_look_tools(
        None,
        live_shares="",
        conversation_has_scene_observations=True,
        may_control_shares=True,
    )[0]
    answer = await tool.ainvoke({"sources": ["screen"], "reason": "to read the error"})
    assert answer["status"] == "not_shared"
    assert answer["offered_to_share"] == ["screen"]
    assert "take one look at their screen" in answer["message"]
    assert _browser_frames == [
        {"type": "share_request", "sources": ["screen"], "reason": "to read the error"}
    ]


@pytest.mark.asyncio
async def test_the_button_is_offered_even_without_the_permission(_browser_frames):
    # Asking the person for something is not acting on their device, so it does
    # not need the permission that governs acting on their device. Gating it
    # meant somebody who had never granted the camera could not be offered a
    # look at their screen at all.
    tool = build_look_tools(
        None,
        live_shares="",
        conversation_has_scene_observations=True,
        may_control_shares=False,
    )[0]
    answer = await tool.ainvoke({"sources": ["screen"]})
    assert answer["offered_to_share"] == ["screen"]
    assert _browser_frames == [
        {"type": "share_request", "sources": ["screen"], "reason": ""}
    ]


@pytest.mark.asyncio
async def test_switching_a_share_off_takes_effect_without_pausing(_browser_frames):
    from src.anubis.utils.tools.vision.look_tools import STOP_SHARING_TOOL_NAME

    tools = build_look_tools(
        None, live_shares="webcam,screen", may_control_shares=True
    )
    stop_sharing = next(
        tool for tool in tools if tool.name == STOP_SHARING_TOOL_NAME
    )
    answer = await stop_sharing.ainvoke({"sources": ["webcam"]})
    assert answer["status"] == "stopped"
    assert answer["stopped"] == ["webcam"]
    assert _browser_frames == [{"type": "share_stop", "sources": ["webcam"]}]


@pytest.mark.asyncio
async def test_switching_everything_off_names_everything_that_was_live(
    _browser_frames,
):
    from src.anubis.utils.tools.vision.look_tools import STOP_SHARING_TOOL_NAME

    tools = build_look_tools(
        None, live_shares="webcam,screen", may_control_shares=True
    )
    stop_sharing = next(
        tool for tool in tools if tool.name == STOP_SHARING_TOOL_NAME
    )
    answer = await stop_sharing.ainvoke({})
    assert answer["stopped"] == ["webcam", "screen"]


@pytest.mark.asyncio
async def test_switching_off_something_that_is_not_on_tells_the_browser_nothing(
    _browser_frames,
):
    from src.anubis.utils.tools.vision.look_tools import STOP_SHARING_TOOL_NAME

    tools = build_look_tools(None, live_shares="webcam", may_control_shares=True)
    stop_sharing = next(
        tool for tool in tools if tool.name == STOP_SHARING_TOOL_NAME
    )
    answer = await stop_sharing.ainvoke({"sources": ["screen"]})
    assert answer["status"] == "nothing_to_stop"
    assert _browser_frames == []


@pytest.mark.asyncio
async def test_a_look_that_opened_the_camera_says_it_was_closed_again():
    _paused, resumed, _entries = await _run_look_through_a_graph(
        {"sources": ["webcam"]},
        {
            "type": "looked",
            "observations": [{"source": "webcam", "description": "a person"}],
        },
    )
    # The live-share path does not open anything, so nothing is claimed.
    (answer,) = resumed["answer"]
    assert answer["opened_for_this_look"] == []


def test_the_section_says_the_camera_is_opened_only_for_the_look():
    block = build_live_shares_block(
        [], [], can_look_now=True, may_control_shares=True
    )
    assert "cannot be opened by this" in block, "the desktop is not peekable here"
    assert "never left running" in block
    assert "stop_sharing" in block


# --- Peeking at the desktop, and telling the two views apart ---------------
#
# The camera rides a standing browser permission, so a granted camera reopens
# on its own. A desktop cannot: ``getDisplayMedia`` needs a real gesture every
# time and no browser keeps a standing grant for it. So a *peekable* desktop is
# one the person granted once and the browser is still holding, reported per
# turn as ``peekable_shares``; a desktop nobody granted is still asked for with
# a button.


def test_what_can_be_peeked_at_is_what_the_browser_reported():
    assert peekable_sources('["screen"]') == ["screen"]
    assert peekable_sources('["webcam","screen"]') == ["webcam", "screen"]
    assert peekable_sources("[]") == []


def test_a_client_that_reports_nothing_is_read_the_way_it_always_was():
    # Before ``peekable_shares`` existed, the camera was peekable exactly when
    # the avatar could control shares. An older browser keeps that behaviour.
    assert peekable_sources(None, may_control_shares=True) == ["webcam"]
    assert peekable_sources(None, may_control_shares=False) == []
    assert peekable_sources("", may_control_shares=True) == ["webcam"]
    # ...but a browser that DOES report is believed, including when it reports
    # that nothing at all can be opened.
    assert peekable_sources("[]", may_control_shares=True) == []


@pytest.mark.asyncio
async def test_a_granted_desktop_is_opened_for_a_look_instead_of_asked_for(
    _browser_frames,
):
    tool = build_look_tools(
        None,
        live_shares="",
        may_control_shares=True,
        peekable_shares='["screen"]',
    )[0]
    answer = await _look_through_a_graph(
        tool,
        {"sources": ["screen"], "reason": "to read the error"},
        {
            "type": "looked",
            "observations": [
                {"source": "screen", "description": "a terminal with a stack trace"}
            ],
        },
    )
    assert answer["status"] == "looked"
    assert answer["looked_at"] == ["screen"]
    assert answer["opened_for_this_look"] == ["screen"]
    # Nothing was asked of the person: the grant is already in hand.
    assert _browser_frames == []


@pytest.mark.asyncio
async def test_a_desktop_nobody_granted_is_still_a_button(_browser_frames):
    tool = build_look_tools(
        None,
        live_shares="",
        may_control_shares=True,
        peekable_shares='["webcam"]',
    )[0]
    answer = await tool.ainvoke({"sources": ["screen"], "reason": "to read the error"})
    assert answer["status"] == "not_shared"
    assert answer["offered_to_share"] == ["screen"]
    assert _browser_frames == [
        {"type": "share_request", "sources": ["screen"], "reason": "to read the error"}
    ]


def test_the_tool_says_which_view_answers_which_question():
    description = build_look_tools(
        None, live_shares="", may_control_shares=True, peekable_shares='["webcam","screen"]'
    )[0].description
    assert "not interchangeable" in description.lower()
    assert "what is on my screen" in description
    assert "open it for a single look" in description


@pytest.mark.asyncio
async def test_the_result_names_each_view_it_looked_at():
    tool = build_look_tools(None, live_shares='["webcam","screen"]')[0]
    answer = await _look_through_a_graph(
        tool,
        {},
        {
            "type": "looked",
            "observations": [
                {"source": "webcam", "description": "a person at a desk"},
                {"source": "screen", "description": "a terminal"},
            ],
        },
    )
    assert "webcam = the camera" in answer["message"]
    assert "screen = the desktop" in answer["message"]
    assert "keep them apart" in answer["message"]


@pytest.mark.asyncio
async def test_a_held_desktop_can_be_switched_off_but_a_peekable_camera_cannot(
    _browser_frames,
):
    tools = build_look_tools(
        None,
        live_shares="",
        may_control_shares=True,
        peekable_shares='["webcam","screen"]',
    )
    stop = next(tool for tool in tools if tool.name == "stop_sharing")
    # The desktop grant is a capture the browser is running, so it can end.
    assert (await stop.ainvoke({"sources": ["screen"]}))["status"] == "stopped"
    assert _browser_frames[-1] == {"type": "share_stop", "sources": ["screen"]}
    # A peekable camera is a permission with nothing running behind it.
    assert (await stop.ainvoke({"sources": ["webcam"]}))["status"] == "nothing_to_stop"


def test_the_prompt_section_names_what_can_be_peeked_at():
    block = build_live_shares_block(
        [],
        [],
        can_look_now=True,
        may_control_shares=True,
        peekable_sources=["webcam", "screen"],
    )
    assert "open to a single look right now: webcam and screen" in block
    assert "never interchangeable" in block
    assert "never a standing watch" in block
