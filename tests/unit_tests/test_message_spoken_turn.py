"""``diarize=true`` on ``POST /message``: the spoken-turn tag, frame and routing.

``label_spoken_turn_files`` turns the audio attachment into a speaker-labelled
script. An owner-only utterance is an ordinary (visible) turn tagged
``spoken_turn``; an utterance where someone else spoke is tagged like an
observation heard through the microphone so the graph triages the turn, while
staying visible. ``message_graph_sse`` announces the script as a
``spoken_turn`` frame right after ``turn_started``. Observation helpers pick
the speech instructions for microphone-only observations, and the triage node
keeps a visible spoken turn visible.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from starlette.datastructures import Headers, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils.ambient import observations  # noqa: E402
from src.anubis.utils.ambient.triage_node import route_after_image_resolution  # noqa: E402
from src.anubis.utils.voice import speakers as speakers_module  # noqa: E402
from src.anubis.utils.voice.speakers import LabelledSegment, SpokenTurn  # noqa: E402
from src.api import webapp as webapp_module  # noqa: E402


def _upload(name: str, content_type: str, payload: bytes = b"abc") -> UploadFile:
    return UploadFile(
        io.BytesIO(payload),
        filename=name,
        headers=Headers({"content-type": content_type}),
    )


def _turn(segments, *, owner_identified=True):
    others = sorted(
        {segment.speaker for segment in segments if not segment.is_owner and not segment.is_avatar}
    )
    return SpokenTurn(
        script=speakers_module.render_speaker_script(segments),
        segments=segments,
        owner_label="Evan",
        owner_identified=owner_identified,
        other_speakers=others,
        duration_seconds=3.0,
        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        total_cost=0.001,
        latency_ms=12.0,
        model="diarize-test",
    )


@pytest.fixture
def harness(monkeypatch):
    metrics = []

    async def fake_metrics_row(pool, **kwargs):
        metrics.append(kwargs)
        return True

    monkeypatch.setattr(webapp_module, "persist_api_metrics_row", fake_metrics_row)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(
            assistants=SimpleNamespace(
                get=_async_return({"name": "Evan Woods", "metadata": {}})
            )
        ),
    )
    return metrics


def _async_return(value):
    async def _call(**kwargs):
        return value

    return _call


def _install_fake_diarizer(monkeypatch, turn):
    seen = {}

    async def fake_diarize_spoken_turn(audio_bytes, **kwargs):
        seen["audio_bytes"] = audio_bytes
        seen.update(kwargs)
        return turn

    monkeypatch.setattr(speakers_module, "diarize_spoken_turn", fake_diarize_spoken_turn)
    return seen


CURRENT_USER = {
    "API_KEY": "k",
    "email": "owner@example.com",
    "identities": [{"user_id": "u1"}],
}


@pytest.mark.asyncio
async def test_owner_only_utterance_is_a_visible_spoken_turn(monkeypatch, harness):
    segments = [LabelledSegment("Evan", "What is on my calendar?", 0.0, 2.0, is_owner=True)]
    seen = _install_fake_diarizer(monkeypatch, _turn(segments))
    audio = _upload("utterance.webm", "audio/webm;codecs=opus", b"opus-bytes")
    image = _upload("photo.png", "image/png")

    remaining, message, kwargs, frame = await webapp_module.label_spoken_turn_files(
        SimpleNamespace(pool=None),
        CURRENT_USER,
        files=[audio, image],
        message="",
        assistant_id="a1",
        thread_id="t1",
        your_name=None,
        request_id="r1",
    )

    assert remaining == [image], "only the audio attachment is consumed"
    assert message == "Evan: What is on my calendar?"
    assert kwargs["kind"] == "spoken_turn" and "ambient" not in kwargs
    assert kwargs["speakers"]["owner_spoke"] is True
    assert kwargs["speakers"]["others_spoke"] is False
    assert frame == {"content": message, "speakers": kwargs["speakers"]}
    assert seen["audio_bytes"] == b"opus-bytes"
    assert seen["owner_label"] == "Evan Woods", "the avatar's name labels the owner"
    assert seen["thread_id"] == "t1"
    assert harness and harness[0]["inference_type"] == "diarization"
    assert harness[0]["total_tokens"] == 5 and harness[0]["cost_usd"] == 0.001

    human = HumanMessage(content=message, additional_kwargs=kwargs)
    assert route_after_image_resolution({"messages": [human]}) == "anubis"


@pytest.mark.asyncio
async def test_utterance_with_others_is_a_visible_microphone_observation(monkeypatch, harness):
    segments = [
        LabelledSegment("Evan", "Say hi to Maria.", 0.0, 1.0, is_owner=True),
        LabelledSegment("Speaker 2", "Hello, avatar.", 1.1, 2.0),
    ]
    _install_fake_diarizer(monkeypatch, _turn(segments))
    audio = _upload("utterance.webm", "audio/webm", b"x")

    remaining, message, kwargs, frame = await webapp_module.label_spoken_turn_files(
        SimpleNamespace(pool=None),
        CURRENT_USER,
        files=[audio],
        message="typed note",
        assistant_id="a1",
        thread_id=None,
        your_name="Evan",
        request_id="r1",
    )

    assert remaining == []
    assert message == "typed note\n\nEvan: Say hi to Maria.\nSpeaker 2: Hello, avatar."
    assert kwargs["kind"] == observations.AMBIENT_MESSAGE_KIND
    assert kwargs["hidden"] is False, "what was heard stays in the transcript"
    assert kwargs["ambient"]["sources"] == ["microphone"]
    assert kwargs["ambient"]["voice_mode"] is True
    assert kwargs["speakers"]["other_speakers"] == ["Speaker 2"]
    assert frame["speakers"]["others_spoke"] is True

    human = HumanMessage(content=message, additional_kwargs=kwargs)
    assert route_after_image_resolution({"messages": [human]}) == "ambient_triage"


@pytest.mark.asyncio
async def test_turn_without_audio_passes_through(monkeypatch, harness):
    image = _upload("photo.png", "image/png")
    remaining, message, kwargs, frame = await webapp_module.label_spoken_turn_files(
        SimpleNamespace(pool=None),
        CURRENT_USER,
        files=[image],
        message="hello",
        assistant_id="a1",
        thread_id="t1",
        your_name=None,
        request_id="r1",
    )
    assert remaining == [image] and message == "hello"
    assert kwargs is None and frame is None


@pytest.mark.asyncio
async def test_anonymous_callers_cannot_label_speakers(monkeypatch, harness):
    monkeypatch.setattr(webapp_module, "is_anonymous_user", lambda user: True)
    with pytest.raises(HTTPException) as raised:
        await webapp_module.label_spoken_turn_files(
            SimpleNamespace(pool=None),
            CURRENT_USER,
            files=[_upload("utterance.webm", "audio/webm")],
            message="",
            assistant_id="a1",
            thread_id=None,
            your_name=None,
            request_id="r1",
        )
    assert raised.value.status_code == 403


def test_speech_observations_get_the_speech_instructions():
    heard = {
        "observation_id": "o1",
        "sources": ["microphone"],
        "decision": "respond",
    }
    text = observations.compose_observation_text(heard, "Speaker 2: Hello, avatar.")
    assert text.endswith(observations.RESPOND_INSTRUCTION_SPEECH)
    assert observations.strip_instruction(text.split("\n", 1)[1]) == "Speaker 2: Hello, avatar."
    seen = {**heard, "sources": ["webcam", "microphone"]}
    assert observations.compose_observation_text(seen, "body").endswith(
        observations.RESPOND_INSTRUCTION
    )
    assert observations.is_speech_observation(heard)
    assert not observations.is_speech_observation(seen)
    notify = observations.compose_observation_text({**heard, "decision": "notify"}, "body")
    assert notify.endswith(observations.NOTIFY_INSTRUCTION_SPEECH)


def test_build_ambient_additional_kwargs_can_stay_visible():
    hidden = observations.build_ambient_additional_kwargs(
        sources=["webcam"], captured_at="now", voice_mode=False
    )
    visible = observations.build_ambient_additional_kwargs(
        sources=["microphone"], captured_at="now", voice_mode=True, hidden=False
    )
    assert hidden["hidden"] is True and visible["hidden"] is False
    assert observations.is_ambient_observation(HumanMessage(content="x", additional_kwargs=visible))
    assert not observations.is_hidden_message(HumanMessage(content="x", additional_kwargs=visible))


# --- the stream frame ------------------------------------------------------------


class _FakeGraph:
    def __init__(self, events):
        self._events = events

    async def astream(self, input, config, context, stream_mode, subgraphs):
        for event in self._events:
            yield event

    async def aget_state(self, config):
        return SimpleNamespace(next=(), tasks=(), interrupts=())


@pytest.mark.asyncio
async def test_message_graph_sse_announces_the_spoken_turn_before_tokens(monkeypatch):
    class _Threads:
        async def update(self, thread_id, metadata):
            return None

    monkeypatch.setattr(
        webapp_module, "get_client", lambda **kwargs: SimpleNamespace(threads=_Threads())
    )

    async def fake_message_meter(**kwargs):
        return None

    monkeypatch.setattr(webapp_module, "_meter_message_usage", fake_message_meter)
    frame_in = {
        "content": "Evan: Hello",
        "speakers": {"owner_label": "Evan", "segments": []},
    }
    generator = webapp_module.message_graph_sse(
        _FakeGraph([((), "custom", {"type": "assistant_token", "text": "Hi"})]),
        HumanMessage(content="Evan: Hello", additional_kwargs={"kind": "spoken_turn"}),
        {"configurable": {"thread_id": "t1"}},
        SimpleNamespace(),
        thread_id="t1",
        user_id="u1",
        assistant_id="a1",
        conversation_title_value="t1",
        start_time_ns=0,
        request_id="r1",
        langgraph_client_headers={},
        app_state=SimpleNamespace(),
        current_user=CURRENT_USER,
        include_usage_metrics=False,
        spoken_turn_frame=frame_in,
    )
    frames = []
    async for chunk in generator:
        if chunk.startswith("data: "):
            frames.append(json.loads(chunk[len("data: ") :].strip()))
    assert [frame["type"] for frame in frames] == [
        "turn_started",
        "spoken_turn",
        "assistant_token",
        "done",
    ]
    assert frames[1]["content"] == "Evan: Hello"
    assert frames[1]["speakers"]["owner_label"] == "Evan"


@pytest.mark.asyncio
async def test_echo_only_utterance_is_triaged_not_answered(monkeypatch, harness):
    segments = [
        LabelledSegment(
            "Evan (avatar)", "The project kicks off next Monday.", 0.0, 2.0, is_avatar=True
        )
    ]
    _install_fake_diarizer(monkeypatch, _turn(segments))
    remaining, message, kwargs, frame = await webapp_module.label_spoken_turn_files(
        SimpleNamespace(pool=None),
        CURRENT_USER,
        files=[_upload("utterance.webm", "audio/webm", b"x")],
        message="",
        assistant_id="a1",
        thread_id="t1",
        your_name="Evan",
        request_id="r1",
    )
    assert message == "Evan (avatar): The project kicks off next Monday."
    assert kwargs["kind"] == observations.AMBIENT_MESSAGE_KIND, "nothing to answer: triage it"
    assert kwargs["hidden"] is False
    assert kwargs["speakers"]["avatar_spoke"] is True
    assert kwargs["speakers"]["owner_spoke"] is False
    assert kwargs["speakers"]["other_speakers"] == []
    assert frame["speakers"]["avatar_label"] == "Evan (avatar)"


@pytest.mark.asyncio
async def test_recent_avatar_reply_texts_reads_the_last_ai_turns_newest_first():
    class _Threads:
        async def get_state(self, thread_id):
            return {
                "values": {
                    "messages": [
                        {"type": "human", "content": "hi"},
                        {"type": "ai", "content": "First reply."},
                        {"type": "human", "content": "more"},
                        {"type": "ai", "content": [{"type": "text", "text": "Second reply."}]},
                    ]
                }
            }

    client = SimpleNamespace(threads=_Threads())
    assert await webapp_module._recent_avatar_reply_texts(client, "t1") == [
        "Second reply.",
        "First reply.",
    ]
    assert await webapp_module._recent_avatar_reply_texts(client, None) == []

    class _Broken:
        async def get_state(self, thread_id):
            raise RuntimeError("down")

    assert await webapp_module._recent_avatar_reply_texts(SimpleNamespace(threads=_Broken()), "t1") == []
