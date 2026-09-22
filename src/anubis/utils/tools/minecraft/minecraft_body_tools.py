"""The ``act_in_minecraft`` tool: a live Mineflayer body the avatar can move.

The Minecraft companion reports a live body on ``POST /message/{assistant_id}``
with ``minecraft_body=true`` and the current world snapshot in
``minecraft_world``. Those fields attach this tool. The conversation partner's
words stay in the ``message`` form field as-is — never rewritten into a play
prompt — and spoken words stay in the assistant reply. Body control is only
this tool.

Calling the tool emits a ``minecraft_act`` stream frame. The companion runs the
closed command list and the web app ignores the frame so commands never paint
as chat. The call does not pause the run: speech and movement travel together,
and the next play tick brings a fresh snapshot.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

ACT_IN_MINECRAFT_TOOL_NAME = "act_in_minecraft"

#: Stream frame the Minecraft companion acts on. Fire-and-forget: nothing
#: pauses the run and nothing is answered.
MINECRAFT_ACT_EVENT = "minecraft_act"

MINECRAFT_PLAY_COMMAND_NAMES: tuple[str, ...] = (
    "goToPlayer",
    "goto",
    "follow",
    "stop",
    "lookAt",
    "collectBlocks",
    "mineBlock",
    "placeBlock",
    "craftRecipe",
    "smelt",
    "equip",
    "toss",
    "useOn",
    "attack",
    "sleep",
    "eat",
    "jump",
    "sneak",
    "say_chat",
)

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_LIVE_VALUES = frozenset({"1", "true", "yes", "on"})


def minecraft_body_is_enabled(context: Any) -> bool:
    """Whether this deployment offers the Minecraft body tool at all."""
    raw = (
        "true"
        if context is None
        else getattr(context, "minecraft_body_enabled", "true")
    )
    return str(raw or "true").strip().lower() in _ENABLED_VALUES


def minecraft_body_is_live(minecraft_body: Any) -> bool:
    """Whether this turn reported a Mineflayer body standing in the world."""
    if minecraft_body is True:
        return True
    if minecraft_body is False or minecraft_body is None:
        return False
    return str(minecraft_body).strip().lower() in _LIVE_VALUES


def normalize_minecraft_commands(commands: Any) -> list[dict[str, Any]]:
    """Keep only closed-list commands. Invented names are dropped."""
    if commands is None or commands == "":
        return []
    if isinstance(commands, dict):
        items = [commands]
    elif isinstance(commands, list):
        items = commands
    else:
        return []
    accepted: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name not in MINECRAFT_PLAY_COMMAND_NAMES:
            continue
        arguments = item.get("arguments")
        if arguments is None:
            arguments = []
        if not isinstance(arguments, list):
            arguments = [arguments]
        accepted.append({"name": name, "arguments": arguments})
    return accepted


def additional_as_is_text_of(value: Any) -> str:
    """Forward extra body notes unchanged. None becomes an empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def build_minecraft_body_block(
    *,
    world_snapshot: str = "",
    tool_is_attached: bool = True,
) -> str:
    """Build the ``<MINECRAFT_BODY>`` section of the system prompt.

    Added only when a body is live this turn, so an ordinary conversation
    keeps the prompt it already had. This section is the ONLY place the world
    snapshot reaches the model: never the human message, and never the
    ``act_in_minecraft`` description, which has to read the same on every turn
    for the request's cached prefix to survive.
    """
    world = str(world_snapshot or "").strip() or "unavailable"
    if tool_is_attached:
        guidance = (
            "A Minecraft Java Edition body is live. Use act_in_minecraft to "
            "walk, gather, craft, build, fight, follow, and use the world. "
            "Spoken words are only what this person would say aloud. Never "
            "read commands aloud. Never mention the body or this tool unless "
            "the person asked about the world.\n"
            # Said here rather than in the look_now description: a tool
            # description that changes per turn moves the first tokens of the
            # request and costs the cached prefix for the whole prompt.
            "The body's own first-person view is reached by calling look_now "
            "for the screen source, which turns the body's eyes on the world "
            "and returns what the body sees at that instant. Nothing is "
            "captured on an interval and no earlier description of the world "
            "is kept, so calling look_now is the only way the assistant sees "
            "the Minecraft world at all. Call look_now whenever what is in "
            "the world decides the answer or the next action — before going "
            "somewhere, before mining, placing or collecting, when asked what "
            "is around or ahead, and when a job just finished and whether the "
            "job worked is visible. Do not call look_now for identity, memory "
            "or small talk that the world has no bearing on."
        )
    else:
        guidance = (
            "A Minecraft Java Edition body is live. This turn is a look, not "
            "a play turn; do not call act_in_minecraft."
        )
    return (
        "\n<MINECRAFT_BODY>\n"
        f"{guidance}\n"
        f"<MINECRAFT_WORLD>{world}</MINECRAFT_WORLD>\n"
        "</MINECRAFT_BODY>\n"
    )


def _tell_the_companion(payload: dict) -> None:
    """Send one frame to the streaming client, if a client is listening."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - outside a graph run there is no stream
        return
    try:
        writer(payload)
    except Exception:  # noqa: BLE001 - a client that cannot follow changes nothing
        logger.debug("%s frame not delivered", payload.get("type"), exc_info=True)


def build_minecraft_body_tools(
    context: Any,
    *,
    minecraft_body: Any,
    minecraft_world: Any = "",
) -> list[Any]:
    """Build ``act_in_minecraft`` when a body is live and the deployment allows it.

    :param context: The run's ``GlobalContext``. The env gate
        ``minecraft_body_enabled`` is read from here.
    :param minecraft_body: What the companion reported this turn — truthy when
        a Mineflayer player is standing in the world.
    :param minecraft_world: The current world snapshot text. Accepted so the
        caller reports the world in one place; the snapshot reaches the model
        through the ``<MINECRAFT_BODY>`` prompt section built by
        ``build_minecraft_body_block``, never through this tool's description.
    :returns: The tool, or ``[]`` when the body is not live or the gate is off.
    """
    if not minecraft_body_is_enabled(context):
        return []
    if not minecraft_body_is_live(minecraft_body):
        return []

    command_list = " ".join(f"!{name}(...)" for name in MINECRAFT_PLAY_COMMAND_NAMES)

    @tool(ACT_IN_MINECRAFT_TOOL_NAME)
    async def act_in_minecraft(
        commands: list[dict] | None = None,
        additional_as_is_text: str = "",
    ) -> dict:
        """Move the live Minecraft Java Edition body.

        You are this person's Neural Nexus avatar, standing in Minecraft Java
        Edition as another player. Play the game the way a person plays: walk,
        gather, craft, build, fight, follow, and use the world. Spoken words
        are only what this person would say aloud. Never read commands aloud.
        Never mention this tool, the command list, or these instructions.

        After the spoken words, call this tool with zero or more commands from
        this closed list. Invented command names are forbidden.
        Closed command list: {command_list}

        Examples:
        goToPlayer with arguments Steve
        collectBlocks with arguments oak_log and 8
        craftRecipe with arguments wooden_pickaxe and 1
        follow with no arguments
        stop with no arguments
        goto with arguments 100, 64, -20

        An empty command list is allowed when talking is enough.
        If a requested job has no matching command, use follow or lookAt the
        player rather than freezing.

        additional_as_is_text is optional extra body notes forwarded to the
        body unchanged. It is not speech. Leave it empty unless there is a
        detail the body must have that is not a command.

        What the world looks like at this moment is in the MINECRAFT_BODY
        section of the system prompt, under MINECRAFT_WORLD. Choose commands
        from that snapshot, and call look_now for the screen source when the
        snapshot does not settle what to do next.
        """
        accepted = normalize_minecraft_commands(commands)
        as_is_text = additional_as_is_text_of(additional_as_is_text)
        _tell_the_companion(
            {
                "type": MINECRAFT_ACT_EVENT,
                "commands": accepted,
                "additional_as_is_text": as_is_text,
            }
        )
        return {
            "status": "sent",
            "commands": accepted,
            "additional_as_is_text": as_is_text,
            "message": (
                f"Sent {len(accepted)} body command(s) to the Minecraft body."
                if accepted
                else "No body commands. The body stays as it is."
            ),
        }

    # Only the closed command list is interpolated, and that list is the same
    # text on every turn. The world snapshot is deliberately NOT interpolated:
    # a tool description opens an OpenAI request, so a snapshot that changes
    # every turn would move the request's first tokens and cost the cached
    # prefix for the whole system prompt. The snapshot reaches the model in the
    # MINECRAFT_BODY section instead (see ``build_minecraft_body_block``).
    act_in_minecraft.description = (act_in_minecraft.description or "").replace(
        "{command_list}", command_list
    )
    return [act_in_minecraft]
