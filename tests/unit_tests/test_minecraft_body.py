"""The latent Minecraft body: ``act_in_minecraft`` and the consciousness block.

The play contract belongs on the tool, not on the human message. These tests
are the gate, the closed command list, the as-is text, and the snapshot
staying off the conversation partner's words.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tools.minecraft.minecraft_body_tools import (
    ACT_IN_MINECRAFT_TOOL_NAME,
    MINECRAFT_ACT_EVENT,
    MINECRAFT_PLAY_COMMAND_NAMES,
    additional_as_is_text_of,
    build_minecraft_body_block,
    build_minecraft_body_tools,
    minecraft_body_is_enabled,
    minecraft_body_is_live,
    normalize_minecraft_commands,
)


def test_no_tool_is_attached_when_the_body_is_not_live():
    assert build_minecraft_body_tools(None, minecraft_body="") == []
    assert build_minecraft_body_tools(None, minecraft_body=None) == []
    assert build_minecraft_body_tools(None, minecraft_body=False) == []


def test_no_tool_is_attached_when_the_deployment_gate_is_off():
    context = SimpleNamespace(minecraft_body_enabled="false")
    assert build_minecraft_body_tools(context, minecraft_body="true") == []


def test_the_tool_is_attached_when_the_body_is_live():
    tools = build_minecraft_body_tools(None, minecraft_body="true")
    assert [tool.name for tool in tools] == [ACT_IN_MINECRAFT_TOOL_NAME]


def test_the_tool_description_carries_the_closed_list_but_not_the_snapshot():
    """The closed command list is fixed; the world snapshot is not.

    A tool description opens an OpenAI request, and the cached rate applies
    only to the longest identical opening stretch. Interpolating a snapshot
    that changes every play tick moved those opening tokens every turn and
    discarded the cached prefix for the whole system prompt behind them, so
    the snapshot reaches the model through the MINECRAFT_BODY prompt section
    instead and the description reads the same on every turn.
    """
    world = "position: 26.5, 73.0, -120.5\nheld: dark_oak_log"
    description = build_minecraft_body_tools(
        None, minecraft_body=True, minecraft_world=world
    )[0].description
    for name in MINECRAFT_PLAY_COMMAND_NAMES:
        assert f"!{name}(" in description
    assert "Mineflayer" not in description
    assert "JSON" not in description
    assert "{minecraft_world}" not in description
    assert "{command_list}" not in description
    assert "position: 26.5, 73.0, -120.5" not in description
    assert description == build_minecraft_body_tools(
        None, minecraft_body=True, minecraft_world="position: 0, 64, 0"
    )[0].description
    # The snapshot still reaches the model, in the section rebuilt each turn.
    block = build_minecraft_body_block(world_snapshot=world, tool_is_attached=True)
    assert "position: 26.5, 73.0, -120.5" in block
    assert "held: dark_oak_log" in block
    assert "<MINECRAFT_WORLD>" in block


def test_the_consciousness_block_holds_the_snapshot_not_a_human_message():
    block = build_minecraft_body_block(
        world_snapshot="position: 0, 64, 0",
        tool_is_attached=True,
    )
    assert "<MINECRAFT_BODY>" in block
    assert "act_in_minecraft" in block
    assert "<MINECRAFT_WORLD>position: 0, 64, 0</MINECRAFT_WORLD>" in block
    human = HumanMessage(content="follow me")
    assert "<MINECRAFT_BODY>" not in human.content
    assert "<LATENT_MINECRAFT_BODY>" not in human.content
    assert "<MINECRAFT_WORLD>" not in human.content


def test_as_is_text_is_never_rewritten():
    assert additional_as_is_text_of("come here, then get wood") == (
        "come here, then get wood"
    )
    assert additional_as_is_text_of(None) == ""
    assert additional_as_is_text_of("  keep the spaces  ") == "  keep the spaces  "


def test_invented_command_names_are_dropped():
    accepted = normalize_minecraft_commands(
        [
            {"name": "goto", "arguments": [100, 64, -20]},
            {"name": "newAction", "arguments": ["alert('x')"]},
            {"name": "follow", "arguments": []},
        ]
    )
    assert accepted == [
        {"name": "goto", "arguments": [100, 64, -20]},
        {"name": "follow", "arguments": []},
    ]


def test_an_empty_command_list_is_allowed():
    assert normalize_minecraft_commands([]) == []
    assert normalize_minecraft_commands(None) == []


@pytest.fixture
def _companion_frames(monkeypatch):
    import langgraph.config as langgraph_config

    sent: list[dict] = []
    monkeypatch.setattr(
        langgraph_config, "get_stream_writer", lambda: sent.append, raising=False
    )
    return sent


@pytest.mark.asyncio
async def test_calling_the_tool_emits_minecraft_act_with_allowed_names(
    _companion_frames,
):
    tool = build_minecraft_body_tools(
        None,
        minecraft_body="true",
        minecraft_world="position: 1, 2, 3",
    )[0]
    answer = await tool.ainvoke(
        {
            "commands": [
                {"name": "collectBlocks", "arguments": ["oak_log", 8]},
                {"name": "explodeWorld", "arguments": []},
            ],
            "additional_as_is_text": "stay near UncleEvan1337",
        }
    )
    assert answer["status"] == "sent"
    assert answer["commands"] == [
        {"name": "collectBlocks", "arguments": ["oak_log", 8]}
    ]
    assert answer["additional_as_is_text"] == "stay near UncleEvan1337"
    assert _companion_frames == [
        {
            "type": MINECRAFT_ACT_EVENT,
            "commands": [{"name": "collectBlocks", "arguments": ["oak_log", 8]}],
            "additional_as_is_text": "stay near UncleEvan1337",
        }
    ]


@pytest.mark.asyncio
async def test_additional_as_is_text_is_returned_on_the_frame_unchanged(
    _companion_frames,
):
    tool = build_minecraft_body_tools(None, minecraft_body="on")[0]
    await tool.ainvoke(
        {
            "commands": [],
            "additional_as_is_text": "Nobody just spoke. Keep gathering.",
        }
    )
    assert _companion_frames[0]["additional_as_is_text"] == (
        "Nobody just spoke. Keep gathering."
    )
    assert _companion_frames[0]["commands"] == []


def test_minecraft_body_enabled_reads_the_environment(monkeypatch):
    monkeypatch.delenv("MINECRAFT_BODY_ENABLED", raising=False)
    assert GlobalContext().minecraft_body_enabled == "true"
    monkeypatch.setenv("MINECRAFT_BODY_ENABLED", "false")
    assert GlobalContext().minecraft_body_enabled == "false"


def test_enabled_and_live_gates_read_the_usual_truthy_spellings():
    assert minecraft_body_is_enabled(None) is True
    assert minecraft_body_is_enabled(SimpleNamespace(minecraft_body_enabled="FALSE")) is False
    assert minecraft_body_is_live("true") is True
    assert minecraft_body_is_live("TRUE") is True
    assert minecraft_body_is_live("off") is False


def test_a_look_resume_keeps_the_remembered_minecraft_body():
    from src.api.look_context import LookContext, LookContextRegistry
    from src.api.webapp import _look_context_for_resume

    registry = LookContextRegistry()
    registry.remember(
        "t1",
        LookContext(
            live_shares="webcam",
            minecraft_body="true",
            minecraft_world="position: 1, 2, 3",
        ),
    )
    context = _look_context_for_resume(
        SimpleNamespace(look_contexts=registry),
        "t1",
        live_shares='["webcam","screen"]',
        peekable_shares="",
        may_control_shares=False,
        minecraft_body="",
        minecraft_world="",
    )
    assert context.minecraft_body == "true"
    assert context.minecraft_world == "position: 1, 2, 3"
    assert context.live_shares == '["webcam","screen"]'


def test_command_keys_the_model_guesses_are_still_read():
    # With an untyped dict the model sometimes wrote "command" / "args"; every
    # such item used to be dropped while the avatar announced the job anyway.
    assert normalize_minecraft_commands(
        [{"command": "collectBlocks", "args": ["oak_log", 8]}]
    ) == [{"name": "collectBlocks", "arguments": ["oak_log", 8]}]
    assert normalize_minecraft_commands([{"name": "!stop"}]) == [
        {"name": "stop", "arguments": []}
    ]


def test_the_tool_schema_names_the_command_keys():
    tool = build_minecraft_body_tools(None, minecraft_body=True)[0]
    schema_text = str(tool.tool_call_schema.model_json_schema())
    assert "name" in schema_text
    assert "arguments" in schema_text


@pytest.mark.asyncio
async def test_a_call_with_only_invented_commands_is_rejected(_companion_frames):
    tool = build_minecraft_body_tools(None, minecraft_body=True)[0]
    answer = await tool.ainvoke(
        {"commands": [{"name": "gatherWood", "arguments": []}]}
    )
    assert answer["status"] == "rejected"
    assert answer["rejected_names"] == ["gatherWood"]
    assert "again" in answer["message"]


@pytest.mark.asyncio
async def test_typed_command_objects_reach_the_companion(_companion_frames):
    tool = build_minecraft_body_tools(None, minecraft_body=True)[0]
    answer = await tool.ainvoke(
        {"commands": [{"name": "collectBlocks", "arguments": ["dark_oak_log", 8]}]}
    )
    assert answer["status"] == "sent"
    assert _companion_frames[-1]["commands"] == [
        {"name": "collectBlocks", "arguments": ["dark_oak_log", 8]}
    ]


def test_the_body_block_acts_first_instead_of_looking_first():
    block = build_minecraft_body_block(
        world_snapshot="position: 0, 64, 0", tool_is_attached=True
    )
    assert "before mining" not in block
    assert "Do not look before acting" in block
    assert "collectBlocks" in block


def test_look_now_is_offered_to_a_minecraft_body_only_for_sight_questions():
    from src.anubis.utils.tools.vision.look_tools import minecraft_turn_asks_to_see

    for command in ("gather wood", "dig", "look at me", "wait here", "stop following"):
        assert not minecraft_turn_asks_to_see([HumanMessage(content=command)])
    for question in (
        "what do you see?",
        "What’s around us",
        "look around",
        # The person's own player: the body turns to the player character,
        # then looks (LangSmith run 5056dbf7, 2026-10-02).
        "What do I look like?",
        "how do I look",
        "what am I wearing",
        "can you see me",
    ):
        assert minecraft_turn_asks_to_see([HumanMessage(content=question)])


def test_the_body_block_says_how_the_person_looks_is_the_player_character():
    block = build_minecraft_body_block(
        world_snapshot="position: 0, 64, 0", tool_is_attached=True
    )
    assert "player character" in block
    assert "Never answer that question from your own appearance" in block
    assert "minecraft_view.jpg" in block
    assert "call neither look_now nor act_in_minecraft to look again" in block
    assert "Describe the player character the picture shows" in block


def test_a_message_carrying_the_body_view_is_not_offered_a_second_look():
    """The companion attaches the view to a sight question; a look would repeat the picture."""
    from src.anubis.utils.tools.vision.look_tools import minecraft_turn_asks_to_see

    message_with_view = HumanMessage(
        content=(
            "What do I look like?\n\n---\nImage descriptions:\n"
            "[minecraft_view.jpg]\nA blocky figure with a tan head and a blue torso."
        )
    )
    assert not minecraft_turn_asks_to_see([message_with_view])
    assert minecraft_turn_asks_to_see([HumanMessage(content="What do I look like?")])


def test_the_system_prompt_never_gives_the_user_the_avatar_appearance():
    """An appearance question is answered from a look in a game, otherwise from user facts.

    LangSmith run 58c18df7-dc60-46a3-a2c8-d5c7d4347667 (2026-10-02) answered
    "What do I look like?" with the avatar's own reference-image description.
    """
    from src.anubis.utils.prompts.system_prompts import IDENTITY_SYSTEM_PROMPT_TEMPLATE

    assert "How to answer what the user looks like" in IDENTITY_SYSTEM_PROMPT_TEMPLATE
    assert "never the user's appearance" in IDENTITY_SYSTEM_PROMPT_TEMPLATE
    assert "ask the user to describe the user's appearance" in IDENTITY_SYSTEM_PROMPT_TEMPLATE


# An act_in_minecraft argument object written into the reply text. The dev API
# streamed this reply at 2026-10-04T01:50:03Z (gpt-5.6-luna); the JSON reached
# Minecraft chat as speech.
INLINE_ACT_REPLY = '{"commands":[{"name":"follow","arguments":[]}]} I’m coming, Marshall.'


def _feed_one_character_at_a_time(act_filter, reply_text):
    visible_text = "".join(act_filter.feed(character) for character in reply_text)
    return visible_text + act_filter.finish()


def test_an_inline_act_object_becomes_a_minecraft_act_frame_and_leaves_the_reply():
    from src.anubis.utils.tools.minecraft.minecraft_body_tools import (
        InlineMinecraftActFilter,
    )

    sent_frames = []
    act_filter = InlineMinecraftActFilter(on_act=sent_frames.append)
    visible_text = _feed_one_character_at_a_time(act_filter, INLINE_ACT_REPLY)
    assert visible_text.strip() == "I’m coming, Marshall."
    assert sent_frames == [
        {
            "type": MINECRAFT_ACT_EVENT,
            "commands": [{"name": "follow", "arguments": []}],
            "additional_as_is_text": "",
        }
    ]


def test_braces_in_speech_and_in_act_strings_are_handled():
    from src.anubis.utils.tools.minecraft.minecraft_body_tools import (
        InlineMinecraftActFilter,
    )

    sent_frames = []
    act_filter = InlineMinecraftActFilter(on_act=sent_frames.append)
    reply_text = (
        'A set {a, b} and {"note": 1}. '
        '{"commands": [{"name": "say_chat", "arguments": ["a } b"]}, '
        '{"name": "explodeWorld", "arguments": []}]} Done.'
    )
    visible_text = _feed_one_character_at_a_time(act_filter, reply_text)
    assert visible_text == 'A set {a, b} and {"note": 1}.  Done.'
    assert [frame["commands"] for frame in sent_frames] == [
        [{"name": "say_chat", "arguments": ["a } b"]}]
    ]


def test_an_unclosed_inline_act_object_is_dropped_at_the_end_of_the_reply():
    from src.anubis.utils.tools.minecraft.minecraft_body_tools import (
        InlineMinecraftActFilter,
    )

    act_filter = InlineMinecraftActFilter()
    assert _feed_one_character_at_a_time(
        act_filter, 'On it. {"commands": [{"name": "follow"'
    ) == "On it. "
    assert act_filter.acts == []


def test_the_saved_reply_has_no_inline_act_object():
    from src.anubis.utils.tools.minecraft.minecraft_body_tools import (
        strip_inline_minecraft_acts,
    )

    assert strip_inline_minecraft_acts(INLINE_ACT_REPLY) == "I’m coming, Marshall."
    assert strip_inline_minecraft_acts("Plain {speech}.") == "Plain {speech}."


class _ReplyStreamingDeepAgent:
    """A deep agent stand-in that streams one reply, one character per token."""

    def __init__(self, reply_text):
        self.reply_text = reply_text

    async def astream_events(self, agent_input, **keyword_arguments):
        from langchain_core.messages import AIMessageChunk

        for character in self.reply_text:
            yield {
                "event": "on_chat_model_stream",
                "run_id": "model-run",
                "data": {"chunk": AIMessageChunk(content=character)},
            }
        yield {"event": "on_chat_model_end", "run_id": "model-run", "data": {}}


@pytest.mark.asyncio
async def test_the_stream_sends_the_act_as_a_frame_while_a_body_is_live():
    from src.anubis.graph import _stream_deep_agent

    written_frames = []
    await _stream_deep_agent(
        _ReplyStreamingDeepAgent(INLINE_ACT_REPLY),
        {},
        {},
        None,
        written_frames.append,
        minecraft_body_is_live_this_turn=True,
    )
    streamed_text = "".join(
        frame["text"] for frame in written_frames if frame["type"] == "assistant_token"
    )
    assert streamed_text.strip() == "I’m coming, Marshall."
    assert [frame for frame in written_frames if frame["type"] == MINECRAFT_ACT_EVENT] == [
        {
            "type": MINECRAFT_ACT_EVENT,
            "commands": [{"name": "follow", "arguments": []}],
            "additional_as_is_text": "",
        }
    ]


@pytest.mark.asyncio
async def test_the_stream_is_untouched_without_a_live_body():
    from src.anubis.graph import _stream_deep_agent

    written_frames = []
    await _stream_deep_agent(
        _ReplyStreamingDeepAgent(INLINE_ACT_REPLY),
        {},
        {},
        None,
        written_frames.append,
    )
    assert "".join(frame["text"] for frame in written_frames) == INLINE_ACT_REPLY
