"""Scene narration: the accessibility mode a person who cannot see switches on by asking.

Two things have to hold. First, the mode is reachable by voice alone — the
avatar is given ``set_scene_narration`` whenever the browser says it can
narrate, and calling it tells the browser through a stream frame that pauses
nothing. Second, once the mode is on, an observation is never silenced: it
skips the classifier, the salience floors and the quiet period, is described
for a listener rather than for a triage, and reaches the avatar with the
narration instruction. A person standing somewhere they cannot see, waiting,
is the case every assertion here is about.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import src.anubis.utils.ambient.triage_node as triage_node_module
import src.anubis.utils.nodes as nodes_module
from src.anubis.utils.ambient.observations import (
    NARRATE_INSTRUCTION,
    REASON_LINE_PREFIX,
    build_ambient_additional_kwargs,
    build_live_shares_block,
    compose_observation_text,
    is_narration_observation,
    observation_header,
)
from src.anubis.utils.ambient.triage_node import (
    AMBIENT_TRIAGE_NODE,
    NARRATION_OBSERVATION_KIND,
    ambient_triage,
    route_after_ambient_triage,
    route_after_image_resolution,
)
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState
from src.anubis.utils.tools.vision.accessibility_tools import (
    SLOWEST_NARRATION_SECONDS,
    NARRATION_OFF,
    NARRATION_ON,
    NARRATION_UNSUPPORTED,
    SCENE_NARRATION_EVENT,
    SET_SCENE_NARRATION_TOOL_NAME,
    build_scene_narration_tools,
    normalize_scene_narration_state,
)

IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}


# --- What the browser reports -------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("on", NARRATION_ON),
        ("ON", NARRATION_ON),
        ("true", NARRATION_ON),
        ("1", NARRATION_ON),
        ("off", NARRATION_OFF),
        ("false", NARRATION_OFF),
        ("0", NARRATION_OFF),
        ("", NARRATION_UNSUPPORTED),
        (None, NARRATION_UNSUPPORTED),
        ("maybe", NARRATION_UNSUPPORTED),
        (17, NARRATION_UNSUPPORTED),
    ],
)
def test_the_browser_report_is_read_from_any_shape_it_sends(value, expected):
    assert normalize_scene_narration_state(value) == expected


# --- The gate on the tool -----------------------------------------------------


def test_a_client_that_never_said_it_could_narrate_gets_no_tool():
    # The Discord bot, the Slack bot, an API caller: none can point a camera
    # or read a description aloud, so none may be given a way to promise it.
    assert build_scene_narration_tools(None, scene_narration=None) == []
    assert build_scene_narration_tools(None, scene_narration="") == []
    assert build_scene_narration_tools(None, scene_narration="banana") == []


@pytest.mark.parametrize("state", ["on", "off"])
def test_a_browser_that_can_narrate_gets_the_tool_either_way(state):
    tools = build_scene_narration_tools(None, scene_narration=state)
    assert [tool.name for tool in tools] == [SET_SCENE_NARRATION_TOOL_NAME]


def test_the_tool_tells_the_avatar_which_way_the_switch_is_set():
    on = build_scene_narration_tools(None, scene_narration="on")[0].description
    off = build_scene_narration_tools(None, scene_narration="off")[0].description
    assert "Scene narration is ON right now" in on
    assert "Scene narration is OFF right now" in off
    assert "{state_line}" not in on and "{state_line}" not in off


def test_the_tool_names_the_words_a_blind_person_might_use():
    description = build_scene_narration_tools(None, scene_narration="off")[0].description
    for phrase in ("surroundings", "cannot see", "accessibility", "look_now"):
        assert phrase in description


# --- Calling the tool ---------------------------------------------------------


def _browser_frames(monkeypatch):
    """Collect the frames the tool sends to the streaming client."""
    import langgraph.config as langgraph_config

    sent: list[dict] = []
    monkeypatch.setattr(
        langgraph_config, "get_stream_writer", lambda: sent.append, raising=False
    )
    return sent


@pytest.mark.asyncio
async def test_switching_on_sends_the_browser_one_frame_and_pauses_nothing(monkeypatch):
    sent = _browser_frames(monkeypatch)
    tool = build_scene_narration_tools(None, scene_narration="off")[0]
    result = await tool.ainvoke({"enabled": True, "reason": "asked to be guided"})

    assert sent == [
        {
            "type": SCENE_NARRATION_EVENT,
            "enabled": True,
            # Unset: the request was to start, not to change the pace.
            "every_seconds": None,
            "reason": "asked to be guided",
        }
    ]
    assert result["status"] == "changed"
    assert result["scene_narration"] == NARRATION_ON
    # The reply has to SAY it started: the listener cannot see a tile appear.
    assert "Say so" in result["message"]


@pytest.mark.asyncio
async def test_switching_off_sends_the_browser_one_frame(monkeypatch):
    sent = _browser_frames(monkeypatch)
    tool = build_scene_narration_tools(None, scene_narration="on")[0]
    result = await tool.ainvoke({"enabled": False})

    assert sent[0]["type"] == SCENE_NARRATION_EVENT and sent[0]["enabled"] is False
    assert result["status"] == "changed"
    assert result["scene_narration"] == NARRATION_OFF


@pytest.mark.asyncio
async def test_asking_for_a_mode_already_on_is_not_an_error(monkeypatch):
    # A person who cannot see the interface has no way to check; asking twice
    # is what anybody does when unsure. The browser is still told, so a browser
    # whose switch drifted from what it last reported comes back into line.
    sent = _browser_frames(monkeypatch)
    tool = build_scene_narration_tools(None, scene_narration="on")[0]
    result = await tool.ainvoke({"enabled": True})

    assert result["status"] == "unchanged"
    assert result["scene_narration"] == NARRATION_ON
    assert sent and sent[0]["enabled"] is True


@pytest.mark.asyncio
async def test_the_tool_works_with_no_stream_to_write_to():
    # Outside a graph run there is no writer; the call must still answer.
    tool = build_scene_narration_tools(None, scene_narration="off")[0]
    result = await tool.ainvoke({"enabled": True})
    assert result["status"] == "changed"


# --- The tag on a narrated observation ---------------------------------------


def test_a_narrated_observation_is_tagged_and_marked_on_its_header():
    kwargs = build_ambient_additional_kwargs(
        sources=["webcam"],
        captured_at="2026-09-11T01:00:00Z",
        voice_mode=True,
        observation_id="o-1",
        camera_facing="environment",
        narrate=True,
    )
    assert kwargs["hidden"] is True
    assert kwargs["ambient"]["narrate"] is True
    assert is_narration_observation(kwargs["ambient"])
    assert "narration=on" in observation_header(kwargs["ambient"])


def test_an_ordinary_observation_is_not_narrated_by_default():
    kwargs = build_ambient_additional_kwargs(
        sources=["webcam"], captured_at="2026-09-11T01:00:00Z", voice_mode=False
    )
    assert kwargs["ambient"]["narrate"] is False
    assert not is_narration_observation(kwargs["ambient"])
    assert "narration=" not in observation_header(kwargs["ambient"])
    assert not is_narration_observation(None)


def test_a_narrated_observation_carries_no_instruction_and_no_reason_line():
    ambient = {
        "observation_id": "o-1",
        "sources": ["webcam"],
        "captured_at": "2026-09-11T01:00:00Z",
        "narrate": True,
        "decision": "respond",
        "reason": "Scene narration is switched on",
    }
    body = "webcam: a kerb ahead, a bus on the left"
    text = compose_observation_text(ambient, body)
    # The description IS the reading and the browser speaks it directly, so no
    # avatar turn answers this and an instruction would be dead text stored on
    # the thread. The reason line is absent for the same reason it always was:
    # narration broke no silence that needs justifying.
    assert NARRATE_INSTRUCTION not in text
    assert REASON_LINE_PREFIX not in text
    assert text.endswith(body)
    assert "narration=on" in text.splitlines()[0]


# --- The triage bypass ------------------------------------------------------


class _FakeDescriber:
    prompts: list = []

    def __init__(self, system_prompt=None):
        type(self).prompts.append(system_prompt)

    async def describe(self, image_data, filename):
        return {
            "description": f"described {filename}",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "total_cost": 0.0001,
            "model_name": "vision-test",
            "latency_ms": 5.0,
        }


@pytest.fixture
def workflow(monkeypatch):
    from src.anubis.utils.ambient.observations import ambient_speech_cooldown

    ambient_speech_cooldown._last_spoken.clear()
    monkeypatch.setattr(nodes_module, "ImageDescriptionClass", _FakeDescriber)
    _FakeDescriber.prompts = []
    classifier_calls: list = []

    async def fake_classify(context, **kwargs):  # pragma: no cover - must not run
        classifier_calls.append(kwargs)
        raise AssertionError("the classifier must not run under scene narration")

    monkeypatch.setattr(triage_node_module, "classify_observation", fake_classify)
    avatar_runs = []

    async def fake_anubis(state):
        avatar_runs.append(state["messages"][-1])
        return {"messages": [AIMessage(content="A kerb ahead of you.")]}

    builder = StateGraph(GlobalState, context_schema=GlobalContext)
    builder.add_node(
        "resolve_human_message_images", nodes_module.resolve_human_message_images
    )
    builder.add_node(AMBIENT_TRIAGE_NODE, ambient_triage)
    builder.add_node("anubis", fake_anubis)
    builder.add_edge(START, "resolve_human_message_images")
    builder.add_conditional_edges(
        "resolve_human_message_images",
        route_after_image_resolution,
        {AMBIENT_TRIAGE_NODE: AMBIENT_TRIAGE_NODE, "anubis": "anubis"},
    )
    builder.add_conditional_edges(
        AMBIENT_TRIAGE_NODE, route_after_ambient_triage, {END: END, "anubis": "anubis"}
    )
    builder.add_edge("anubis", END)
    app = builder.compile(checkpointer=MemorySaver())
    return app, classifier_calls, avatar_runs


def _narrated_turn(message_id="narrated-message"):
    kwargs = build_ambient_additional_kwargs(
        sources=["webcam"],
        captured_at="2026-09-11T01:00:00Z",
        voice_mode=True,
        image_filenames=["webcam.jpg"],
        observation_id="obs-n",
        camera_facing="environment",
        narrate=True,
    )
    return HumanMessage(
        id=message_id,
        content=[{"type": "text", "text": ""}, IMAGE_BLOCK],
        additional_kwargs=kwargs,
    )


async def _run(app, message, thread_id, *, context=None):
    config = {"configurable": {"thread_id": thread_id}}
    events = []
    async for mode, payload in app.astream(
        {
            "messages": [message],
            "user_state": {"user_id": "u1"},
            "assistant_state": {"assistant_id": "a1", "assistant_name": "Ada"},
        },
        config,
        context=context or GlobalContext(),
        stream_mode=["custom", "updates"],
    ):
        events.append((mode, payload))
    state = await app.aget_state(config)
    custom = [payload for mode, payload in events if mode == "custom"]
    return state.values["messages"], custom


@pytest.mark.asyncio
async def test_a_narrated_observation_is_described_for_a_listener_and_always_spoken(
    workflow,
):
    app, classifier_calls, avatar_runs = workflow
    messages, custom = await _run(app, _narrated_turn(), "narration-thread")

    # Described with the narration prompt, not the triage one.
    assert _FakeDescriber.prompts and (
        "describe_scene_for_narration_spec" in _FakeDescriber.prompts[0]
    )
    # Never classified, and never rephrased: the description the vision pass
    # produced is already the sentence meant to be read out, so the run ends
    # here. Sending it through the avatar put a whole deep-agent turn between
    # the camera and the person's ears, which is what made readings arrive
    # half a minute apart, and billed a reply for every one of them.
    assert classifier_calls == []
    assert avatar_runs == []

    stored = messages[0]
    assert stored.id == "narrated-message"
    assert stored.additional_kwargs["hidden"] is True
    ambient = stored.additional_kwargs["ambient"]
    assert ambient["decision"] == "respond"
    assert ambient["narrate"] is True
    assert ambient["observation_kind"] == NARRATION_OBSERVATION_KIND
    assert ambient["salience"] == 1.0
    assert "narration=on" in stored.content.splitlines()[0]
    # No avatar turn was added; the observation stays as context on its own.
    assert not any(isinstance(message, AIMessage) for message in messages)

    decision = [event for event in custom if event.get("type") == "ambient_decision"]
    assert decision and decision[0]["decision"] == "respond"
    assert decision[0]["observation_kind"] == NARRATION_OBSERVATION_KIND
    # The words the browser reads out travel on the decision frame, in full and
    # untruncated: this IS the reading, and a description cut short is a hazard
    # left unsaid.
    assert decision[0]["narration"] == "webcam: described webcam.jpg"


@pytest.mark.asyncio
async def test_narration_ignores_the_quiet_period_after_the_avatar_last_spoke(
    workflow,
):
    # The cooldown exists to keep the avatar from speaking with nothing to say.
    # A standing request is the opposite situation, so two narrated frames in a
    # row are both spoken even with a long cooldown configured.
    from src.anubis.utils.ambient.observations import ambient_speech_cooldown

    app, _classifier_calls, _avatar_runs = workflow
    context = GlobalContext(
        ambient_respond_cooldown_seconds=3600.0,
        ambient_respond_salience_floor=0.99,
    )
    ambient_speech_cooldown.mark_spoken("cooldown-thread")
    _first, first_custom = await _run(
        app, _narrated_turn("first"), "cooldown-thread", context=context
    )
    _second, second_custom = await _run(
        app, _narrated_turn("second"), "cooldown-thread", context=context
    )
    spoken = [
        event
        for events in (first_custom, second_custom)
        for event in events
        if event.get("type") == "ambient_decision"
    ]
    assert len(spoken) == 2
    assert [event["decision"] for event in spoken] == ["respond", "respond"]
    assert all(event["narration"] for event in spoken)


# --- The prompt ----------------------------------------------------------------


def test_the_prompt_says_narration_is_on_even_before_the_first_frame():
    block = build_live_shares_block(
        [], [], can_look_now=False, scene_narration_on=True
    )
    assert block.startswith("\n<LIVE_SHARES>")
    assert "Scene narration is ON" in block
    assert "set_scene_narration with enabled=false" in block


def test_the_prompt_is_unchanged_for_a_browser_that_is_not_narrating():
    assert build_live_shares_block([], [], can_look_now=False) == ""
    assert (
        build_live_shares_block([], [], can_look_now=False, scene_narration_on=False)
        == ""
    )


def test_the_capability_prompt_explains_narrated_turns():
    from src.anubis.utils.prompts.system_prompts import (
        AMBIENT_VISION_CAPABILITY_PROMPT,
    )

    assert "narration=on" in AMBIENT_VISION_CAPABILITY_PROMPT
    assert "set_scene_narration" in AMBIENT_VISION_CAPABILITY_PROMPT


# --- The API edge --------------------------------------------------------------


def test_the_look_context_remembers_the_narration_report():
    from src.api.look_context import LookContext, LookContextRegistry

    registry = LookContextRegistry()
    registry.remember("t1", LookContext(scene_narration="on"), now=100.0)
    recalled = registry.recall("t1", now=101.0)
    assert recalled is not None and recalled.scene_narration == "on"
    # A report of nothing at all still forgets what the thread had.
    registry.remember("t1", LookContext(), now=102.0)
    assert registry.recall("t1", now=103.0) is None


def test_the_message_endpoint_accepts_the_narration_fields():
    import inspect

    from src.api import webapp

    message_fields = inspect.signature(webapp.message_avatar).parameters
    assert "narrate" in message_fields
    assert "scene_narration" in message_fields
    resume_fields = inspect.signature(webapp.resume_avatar_message).parameters
    assert "scene_narration" in resume_fields


def test_enforce_ambient_request_tags_a_narrated_capture(monkeypatch):
    from src.api import webapp

    class _Context:
        ambient_capture_enabled = "true"
        ambient_capture_max_image_bytes = 0
        ambient_capture_min_interval_seconds = 0.0
        scene_narration_min_interval_seconds = 0.0

    monkeypatch.setattr(webapp, "GlobalContext", lambda: _Context())
    monkeypatch.setattr(webapp, "is_anonymous_user", lambda user: False)

    class _Upload:
        filename = "webcam.jpg"
        size = 10

    kwargs = webapp.enforce_ambient_request(
        {"user_id": "u1"},
        files=[_Upload()],
        image_filenames=["webcam.jpg"],
        sources=None,
        captured_at="2026-09-11T01:00:00Z",
        voice_mode=True,
        thread_id="thread-n",
        camera_facing="environment",
        narrate=True,
    )
    assert kwargs["ambient"]["narrate"] is True
    assert kwargs["ambient"]["sources"] == ["webcam"]
    # The rear camera, as the browser names it, read as the world-facing one.
    assert kwargs["ambient"]["camera_facing"] == "world"


def test_narration_is_throttled_by_its_own_floor_not_the_ambient_one(monkeypatch):
    """A narrated look may come far more often than an ordinary ambient one.

    Narration paces itself for somebody walking through a place they cannot
    see. Holding it to the ambient floor — which is set for a person sitting at
    a desk whose room the avatar glances at — had the browser's own pacing
    refused with 429 on every second look.
    """
    from src.api import webapp

    class _Context:
        ambient_capture_enabled = "true"
        ambient_capture_max_image_bytes = 0
        ambient_capture_min_interval_seconds = 30.0
        scene_narration_min_interval_seconds = 3.0

    monkeypatch.setattr(webapp, "GlobalContext", lambda: _Context())
    monkeypatch.setattr(webapp, "is_anonymous_user", lambda user: False)

    marked: list[float] = []

    class _Throttle:
        def check_and_mark(self, thread_id, minimum_seconds):
            marked.append(minimum_seconds)
            return None

    import src.anubis.utils.ambient.observations as observations_module

    monkeypatch.setattr(observations_module, "ambient_throttle", _Throttle())

    class _Upload:
        filename = "webcam.jpg"
        size = 10

    def _call(narrate: bool) -> None:
        webapp.enforce_ambient_request(
            {"user_id": "u1"},
            files=[_Upload()],
            image_filenames=["webcam.jpg"],
            sources=None,
            captured_at="2026-09-11T01:00:00Z",
            voice_mode=False,
            thread_id="thread-floor",
            narrate=narrate,
        )

    _call(False)
    _call(True)

    assert marked == [30.0, 3.0]


# --- Asking for it faster or slower, in words ----------------------------------


@pytest.mark.asyncio
async def test_a_pace_asked_for_in_words_reaches_the_browser(monkeypatch):
    """"Describe more often" has to become a number the browser can act on."""
    sent = _browser_frames(monkeypatch)

    class _Context:
        scene_narration_min_interval_seconds = 3.0

    tool = build_scene_narration_tools(
        _Context(), scene_narration="on", scene_narration_seconds=12
    )[0]
    result = await tool.ainvoke(
        {"enabled": True, "every_seconds": 5, "reason": "asked for more often"}
    )

    assert sent == [
        {
            "type": SCENE_NARRATION_EVENT,
            "enabled": True,
            "every_seconds": 5.0,
            "reason": "asked for more often",
        }
    ]
    # A pace change on a mode already running is its own outcome; answering
    # "it was already on" would read as the request being ignored.
    assert result["status"] == "changed"
    assert result["every_seconds"] == 5.0
    assert "every 5 seconds" in result["message"]
    assert "instead of every 12" in result["message"]


@pytest.mark.asyncio
async def test_an_impossible_pace_is_brought_to_the_nearest_one_that_works(monkeypatch):
    """Clamped, never refused.

    "Describe much more often" is a reasonable thing to say, and an error is
    something a person listening to their phone cannot act on.
    """
    _browser_frames(monkeypatch)

    class _Context:
        scene_narration_min_interval_seconds = 3.0

    tool = build_scene_narration_tools(
        _Context(), scene_narration="on", scene_narration_seconds=10
    )[0]

    faster = await tool.ainvoke({"enabled": True, "every_seconds": 0.2})
    assert faster["every_seconds"] == 3.0

    slower = await tool.ainvoke({"enabled": True, "every_seconds": 9999})
    assert slower["every_seconds"] == SLOWEST_NARRATION_SECONDS


def test_the_tool_tells_the_model_the_pace_it_is_changing_from():
    class _Context:
        scene_narration_min_interval_seconds = 3.0

    tool = build_scene_narration_tools(
        _Context(), scene_narration="on", scene_narration_seconds=8
    )[0]
    assert "every 8 seconds" in tool.description
    assert "fastest this device will go is every 3 seconds" in tool.description
