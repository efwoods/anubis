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
