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

import json
import logging
import re
from typing import Any

from langchain.tools import tool
from pydantic import AliasChoices, BaseModel, Field

logger = logging.getLogger(__name__)

ACT_IN_MINECRAFT_TOOL_NAME = "act_in_minecraft"

#: Stream frame the Minecraft companion acts on. Fire-and-forget: nothing
#: pauses the run and nothing is answered.
MINECRAFT_ACT_EVENT = "minecraft_act"

# The body's first-person view picture the companion attaches to a typed sight
# question (MINECRAFT_VIEW_FILE_NAME in the companion's bot/lookIntent.js). The
# picture rides the first request, so the question is answered in one model
# call instead of a look_now pause, a vision call and a second model call.
MINECRAFT_VIEW_FILE_NAME = "minecraft_view.jpg"

# How the body's view picture is described. The companion draws the view
# itself: flat block colours, with player characters drawn as head, torso,
# arms and legs in the player's skin and armor colours. Short, because the
# description is on the reply path.
DESCRIBE_MINECRAFT_VIEW_PROMPT = (
    "The image is the first-person view of a player body in Minecraft Java "
    "Edition, drawn in flat colours at low resolution: each block is one flat "
    "colour, and a player character is drawn as a blocky figure with a head, "
    "a torso, two arms and two legs. Describe the image in at most 50 words. "
    "When a player character is in view, describe the player character first: "
    "the colour of the head, the torso, the arms, and the legs, and where the "
    "figure stands. Then name the main blocks in view. Left and right are as "
    "the viewer sees the image. Write only the description."
)


def human_message_carries_minecraft_view(message_text: str) -> bool:
    """Whether a resolved human message already holds the body's view picture."""
    return f"[{MINECRAFT_VIEW_FILE_NAME}]" in str(message_text or "")

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
    "giveCollected",
    "useOn",
    "attack",
    "sleep",
    "eat",
    "jump",
    "sneak",
    "say_chat",
    "stfu",
    "followPlayer",
    "goToCoordinates",
    "searchForBlock",
    "searchForEntity",
    "moveAway",
    "goToSurface",
    "digDown",
    "stay",
    "rememberHere",
    "goToRememberedPlace",
    "givePlayer",
    "consume",
    "discard",
    "putInChest",
    "takeFromChest",
    "viewChest",
    "smeltItem",
    "clearFurnace",
    "placeHere",
    "attackPlayer",
    "goToBed",
    "lookAtPlayer",
    "lookAtPosition",
    "showVillagerTrades",
    "tradeWithVillager",
    "goal",
    "endGoal",
    "setMode",
    "startConversation",
    "endConversation",
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


class MinecraftCommand(BaseModel):
    """One body command from the closed command list.

    A typed schema, rather than a bare ``dict``, is what tells the model the
    exact keys. With ``list[dict]`` the model could guess keys such as
    ``command`` and ``args``, every item was silently dropped, and the avatar
    announced "I'm gathering dark oak wood now" while the body stood still.
    """

    name: str = Field(
        validation_alias=AliasChoices("name", "command", "command_name"),
        description=(
            "One name from the closed command list, for example collectBlocks, "
            "follow, stop, goto, lookAt."
        ),
    )
    arguments: list[str | int | float] = Field(
        default_factory=list,
        validation_alias=AliasChoices("arguments", "args", "parameters"),
        description=(
            'Positional arguments in order, for example ["oak_log", 8] for '
            "collectBlocks or [100, 64, -20] for goto. Empty for follow and stop."
        ),
    )


def _command_item_as_dict(item: Any) -> dict[str, Any] | None:
    """Read one command item from a model instance or a plain mapping."""
    if isinstance(item, BaseModel):
        return item.model_dump()
    if isinstance(item, dict):
        return {
            "name": item.get("name", item.get("command", item.get("command_name"))),
            "arguments": item.get(
                "arguments", item.get("args", item.get("parameters"))
            ),
        }
    return None


def split_minecraft_commands(
    commands: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split command items into closed-list commands and rejected names.

    :returns: The accepted commands as ``{"name", "arguments"}`` mappings, and
        the name of every rejected item (an empty string for an item with no
        readable name), so the tool can tell the model which commands never
        reached the body.
    """
    if commands is None or commands == "":
        return [], []
    if isinstance(commands, (dict, BaseModel)):
        items = [commands]
    elif isinstance(commands, list):
        items = commands
    else:
        return [], [str(commands)]
    accepted: list[dict[str, Any]] = []
    rejected: list[str] = []
    for item in items:
        command_mapping = _command_item_as_dict(item)
        if command_mapping is None:
            rejected.append(str(item))
            continue
        name = str(command_mapping.get("name") or "").strip().lstrip("!")
        if name not in MINECRAFT_PLAY_COMMAND_NAMES:
            rejected.append(name)
            continue
        arguments = command_mapping.get("arguments")
        if arguments is None:
            arguments = []
        if not isinstance(arguments, list):
            arguments = [arguments]
        accepted.append({"name": name, "arguments": arguments})
    return accepted, rejected


def normalize_minecraft_commands(commands: Any) -> list[dict[str, Any]]:
    """Keep only closed-list commands. Invented names are dropped."""
    accepted, _rejected = split_minecraft_commands(commands)
    return accepted


def additional_as_is_text_of(value: Any) -> str:
    """Forward extra body notes unchanged. None becomes an empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


#: The opening of an act_in_minecraft argument object, with whitespace removed.
#: A reply that holds this object as text instead of a tool call puts raw JSON
#: into chat and speech (2026-10-04, gpt-5.6-luna replied
#: '{"commands":[{"name":"follow","arguments":[]}]} I'm coming, Marshall.').
_INLINE_ACT_OPENING = '{"commands":'


def _closing_brace_index(text: str) -> int:
    """Index of the brace closing the JSON object that opens at index 0.

    Braces inside JSON strings are skipped. Returns -1 when the object has not
    closed yet.
    """
    depth = 0
    inside_string = False
    escaped = False
    for index, character in enumerate(text):
        if inside_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                inside_string = False
            continue
        if character == '"':
            inside_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _compact_prefix(text: str) -> str:
    """Remove whitespace from the text and cut the text to the act opening length."""
    return "".join(text.split())[: len(_INLINE_ACT_OPENING)]


class InlineMinecraftActFilter:
    """Cut act_in_minecraft argument objects out of streamed reply text.

    The reply text is fed one token delta at a time. Text that cannot start an
    inline act object is returned at once. Text that may start an inline act
    object is held until the object closes or turns out to be speech. A closed
    inline act object is never returned: the object's accepted commands are
    handed to ``on_act`` as a ``minecraft_act`` frame, so the body still moves
    and the JSON never reaches chat, speech, or the saved conversation.
    """

    def __init__(self, on_act: Any = None) -> None:
        """Start with no held text; ``on_act`` receives each act frame."""
        self._held_text = ""
        self._on_act = on_act
        self.acts: list[dict[str, Any]] = []

    def feed(self, text_delta: str) -> str:
        """Add one token delta; return the text that is safe to show now."""
        self._held_text += text_delta
        visible_parts: list[str] = []
        while self._held_text:
            brace_index = self._held_text.find("{")
            if brace_index == -1:
                visible_parts.append(self._held_text)
                self._held_text = ""
                break
            visible_parts.append(self._held_text[:brace_index])
            self._held_text = self._held_text[brace_index:]
            compact_prefix = _compact_prefix(self._held_text)
            if compact_prefix == _INLINE_ACT_OPENING:
                closing_index = _closing_brace_index(self._held_text)
                if closing_index == -1:
                    break
                self._take_act(self._held_text[: closing_index + 1])
                self._held_text = self._held_text[closing_index + 1 :]
                continue
            if _INLINE_ACT_OPENING.startswith(compact_prefix):
                # Too short to tell yet: '{' alone, or '{"comm'.
                break
            visible_parts.append("{")
            self._held_text = self._held_text[1:]
        return "".join(visible_parts)

    def finish(self) -> str:
        """Return the held text at the end of the reply.

        An inline act object that never closed is dropped, not shown.
        """
        held_text = self._held_text
        self._held_text = ""
        if _compact_prefix(held_text) == _INLINE_ACT_OPENING:
            logger.warning(
                "Reply text ended inside an unclosed act_in_minecraft object; "
                "dropped %d characters.",
                len(held_text),
            )
            return ""
        return held_text

    def _take_act(self, object_text: str) -> None:
        """Turn one closed inline act object into a ``minecraft_act`` frame."""
        try:
            act_arguments = json.loads(object_text)
        except json.JSONDecodeError:
            logger.warning(
                "Reply text held an unparseable act_in_minecraft object; "
                "dropped %d characters.",
                len(object_text),
            )
            return
        accepted, rejected = split_minecraft_commands(act_arguments.get("commands"))
        logger.warning(
            "Reply text carried act_in_minecraft arguments instead of a tool "
            "call; accepted=%s rejected=%s.",
            [command["name"] for command in accepted],
            rejected,
        )
        if not accepted:
            return
        act_frame = {
            "type": MINECRAFT_ACT_EVENT,
            "commands": accepted,
            "additional_as_is_text": additional_as_is_text_of(
                act_arguments.get("additional_as_is_text")
            ),
        }
        self.acts.append(act_frame)
        if self._on_act is not None:
            self._on_act(act_frame)


def strip_inline_minecraft_acts(reply_text: str) -> str:
    """Remove every inline act_in_minecraft object from a finished reply text."""
    act_filter = InlineMinecraftActFilter()
    visible_text = act_filter.feed(reply_text) + act_filter.finish()
    if visible_text == reply_text:
        return reply_text
    return re.sub(r"[ \t]{2,}", " ", visible_text).strip()


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
            # Mindcraft-style play: a command is carried out, not researched.
            # Said here rather than in the tool descriptions: a description
            # that changes per turn moves the first tokens of the request and
            # costs the cached prefix for the whole prompt.
            "When the person asks the body to do something, call "
            "act_in_minecraft in this same turn, choosing the command and the "
            "block names from MINECRAFT_WORLD below. Never say the body is "
            "doing, has started, or will do something unless act_in_minecraft "
            "was called for that job in this turn and returned status sent. "
            "The body carries out commands after the spoken words are said, "
            "so speak of a job as starting, never as finished: 'I'll make a "
            "pickaxe', 'Getting wood now', not 'I made a pickaxe' or 'I gave "
            "you everything'. "
            "When act_in_minecraft returns status rejected, call "
            "act_in_minecraft again with corrected commands before replying. "
            "Do not look before acting: MINECRAFT_WORLD already holds what the "
            "body needs, and the body's skills find the nearest matching block "
            "on their own.\n"
            "Common requests: gather wood or get logs is collectBlocks with the "
            "nearest log type listed in MINECRAFT_WORLD (oak_log when none is "
            "listed) and a count such as 8; dig or mine is collectBlocks with "
            "the named block, or dirt when no block is named; wait here, stay, "
            "or stop following is stop; come here or follow me is follow; look "
            "at me is lookAt with the argument player; go to x y z is goto.\n"
            "Answer questions about the weather, the time of day, where the "
            "body is, and what the body carries from the weather, time, "
            "position, and inventory lines of MINECRAFT_WORLD; never guess "
            "them.\n"
            "look_now is attached only when the person asks what the body "
            "sees or how something in the world looks; when attached, "
            "look_now for the screen source returns the body's first-person "
            "view at that instant, with blocks and nearby player characters.\n"
            # Latency first: the companion turns the body to the person and
            # attaches the view picture to the first request, so the question
            # is answered from that picture in one model call (2026-10-02:
            # turn, look and answer as three model calls cost 5,811 ms).
            "When the person's message carries the body's view picture "
            f"({MINECRAFT_VIEW_FILE_NAME} under Image descriptions), the body "
            "has already looked: answer from that picture description alone, "
            "and call neither look_now nor act_in_minecraft to look again.\n"
            "In the game, a question about how the person looks ('what do I "
            "look like', 'how do I look', 'what am I wearing') is about the "
            "person's player character. Describe the player character the "
            "picture shows, in one or two short sentences, and nothing else. "
            "When the picture does not show a player character, say in one "
            "sentence that the player character is not in view. Without a "
            "picture, call look_now for the screen source and describe what "
            "the look shows. Never answer that question from your own "
            "appearance or from any description of a real person."
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
        commands: list[MinecraftCommand] | None = None,
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

        Each command is an object with the keys name and arguments, such as:
        {"name": "goToPlayer", "arguments": ["Steve"]}
        {"name": "collectBlocks", "arguments": ["oak_log", 8]}
        {"name": "collectBlocks", "arguments": ["dirt", 4]}
        {"name": "craftRecipe", "arguments": ["wooden_pickaxe", 1]}
        {"name": "follow", "arguments": []}
        {"name": "stop", "arguments": []}
        {"name": "lookAt", "arguments": ["player"]}
        {"name": "goto", "arguments": [100, 64, -20]}

        Arguments of each command, in order (square brackets mark optional):
        Movement: goToPlayer: player name, [closeness]. followPlayer or
        follow: [player name], [distance]. goToCoordinates or goto: x, y, z.
        searchForBlock: block name, [range]. searchForEntity: entity type,
        [range]. moveAway: distance. goToSurface: none. digDown: distance.
        stop: none. stay: seconds, or -1 to stay until told otherwise.
        lookAt or lookAtPlayer: player name or "player", [at or with].
        lookAtPosition: x, y, z.
        Places: rememberHere: name. goToRememberedPlace: name.
        Gathering and building: collectBlocks: block name, count. mineBlock:
        block name. placeBlock: block name, then x, y, z of the spot to fill,
        or only the block name to place beside the body. placeHere: block name.
        digDown: distance.
        Crafting: craftRecipe: item name, count. smelt or smeltItem: item name,
        [count]. clearFurnace: none.
        Items: equip: item name. eat: none. consume: item name. toss or
        discard: item name, [count]. givePlayer: player name, item name,
        count. giveCollected: player name.
        Chests: putInChest: item name, [count]. takeFromChest: item name,
        [count]. viewChest: none.
        Combat: attack: mob type (fights until the mob is gone). attackPlayer:
        player name, only when the person asks for a fight in the game.
        Life: sleep or goToBed: none. jump, sneak: none.
        Use: useOn: tool name or "hand", then target name (an entity or a
        block), or "nothing" to use the held item.
        Villagers: showVillagerTrades: villager id. tradeWithVillager:
        villager id, trade index, count.
        Autonomy: goal: a sentence describing a longer job, worked on over
        several turns. endGoal: none, once the goal is done. setMode: mode
        name, true or false. stfu: none, to stop unprompted talk.
        Talking: say_chat: text. startConversation: player name, message.
        endConversation: player name.
        Coordinates are whole numbers read from MINECRAFT_WORLD; never guess a
        coordinate that is not there.
        Commands run one after another in the order given, so a plan that
        places or uses an item missing from the inventory in MINECRAFT_WORLD
        includes the gathering and the craftRecipe steps first, for example
        collectBlocks birch_log, then craftRecipe birch_planks, then
        placeBlock birch_planks.

        When the person asks the body to do a job, this tool must be called in
        the same turn; saying the job is under way without calling this tool
        leaves the body standing still. An empty command list is allowed only
        when talking is enough. If a requested job has no matching command,
        use follow or lookAt the player rather than freezing.

        additional_as_is_text is optional extra body notes forwarded to the
        body unchanged. It is not speech. Leave it empty unless there is a
        detail the body must have that is not a command.

        What the world looks like at this moment is in the MINECRAFT_BODY
        section of the system prompt, under MINECRAFT_WORLD. Choose commands
        and block names from that snapshot and act at once; do not look first.
        """
        accepted, rejected = split_minecraft_commands(commands)
        as_is_text = additional_as_is_text_of(additional_as_is_text)
        _tell_the_companion(
            {
                "type": MINECRAFT_ACT_EVENT,
                "commands": accepted,
                "additional_as_is_text": as_is_text,
            }
        )
        # Every command rejected: the body received nothing, so the avatar must
        # not announce the job. The message tells the model to try again.
        if rejected and not accepted:
            return {
                "status": "rejected",
                "commands": [],
                "rejected_names": rejected,
                "additional_as_is_text": as_is_text,
                "message": (
                    "No command reached the body. These names are not on the "
                    f"closed command list: {', '.join(rejected) or 'unnamed'}. "
                    "Call act_in_minecraft again with names from the closed "
                    "command list before telling the person the job started."
                ),
            }
        message = (
            f"Sent {len(accepted)} body command(s) to the Minecraft body."
            if accepted
            else "No body commands. The body stays as it is."
        )
        if rejected:
            message += (
                " Dropped names not on the closed command list: "
                f"{', '.join(rejected) or 'unnamed'}."
            )
        return {
            "status": "sent",
            "commands": accepted,
            "rejected_names": rejected,
            "additional_as_is_text": as_is_text,
            "message": message,
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
