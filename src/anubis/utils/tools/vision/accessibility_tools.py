"""Scene narration: the accessibility mode the avatar can switch on by being asked.

A person who cannot see holds or wears a phone with the rear camera pointed at
whatever is in front of them, and the avatar tells them what is there —
continuously, for as long as the mode is on. The camera aimed outward is
already a standing request to be told what is in view (see
``src/anubis/utils/ambient/``); scene narration is that request made explicit
and made continuous, and it is the one mode in which the avatar describes every
observation instead of noticing almost all of them silently.

The mode belongs to the browser, because the browser is what holds the camera,
paces the captures and reads each description aloud. This module is the other
half: the way the conversation partner can turn the mode on and off **by saying
so**, which for a blind person in voice mode is the only control that is really
within reach. The avatar calls ``set_scene_narration`` and the browser acts on
the ``scene_narration`` frame the call emits.

Nothing here pauses the run. Switching the mode needs no answer from the
browser — the browser either has a camera to point or it does not, and it says
so on the next turn — so the reply keeps its latency. That matters more here
than anywhere else in the product: the conversation partner is often standing
in a place they cannot see, waiting.

The tool is attached only to a turn whose client reported the ``scene_narration``
form field, which is how the web app says it can narrate at all. Every other
client — the Discord bot, the Slack bot, an API caller — never sends the field,
never sees the tool, and so can never promise a person something that will not
happen.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

SET_SCENE_NARRATION_TOOL_NAME = "set_scene_narration"

#: The stream frame the browser acts on. Like ``share_stop``, it asks nothing
#: of the conversation partner and pauses nothing.
SCENE_NARRATION_EVENT = "scene_narration"

#: What the browser reports about the mode on every turn.
NARRATION_ON = "on"
NARRATION_OFF = "off"
#: This client cannot narrate at all (or never learned the field). The avatar is
#: given no tool, so it can neither promise nor deny something it cannot do.
NARRATION_UNSUPPORTED = "unsupported"

#: The paces offered, in seconds — the same five the Accessibility page shows.
#:
#: Five named choices rather than any number of seconds. Each one changes how
#: LONG a reading is as well as how often one arrives (see
#: ``narration_word_budget``), so moving between them is audible; a free scale
#: would have given a hundred settings, almost all indistinguishable from their
#: neighbours and none of them nameable back to the person.
NARRATION_PACE_OPTIONS = (5.0, 10.0, 15.0, 30.0, 60.0)

#: The slowest pace worth calling narration. Past this the person is better
#: served by asking for a look when they want one.
SLOWEST_NARRATION_SECONDS = NARRATION_PACE_OPTIONS[-1]

_NARRATION_STATES = {
    "on": NARRATION_ON,
    "true": NARRATION_ON,
    "1": NARRATION_ON,
    "enabled": NARRATION_ON,
    "off": NARRATION_OFF,
    "false": NARRATION_OFF,
    "0": NARRATION_OFF,
    "disabled": NARRATION_OFF,
}


def normalize_scene_narration_state(value: Any) -> str:
    """Read the browser's ``scene_narration`` report.

    :param value: The form value, or anything a configurable carries.
    :returns: ``on``, ``off``, or ``unsupported`` for a client that said
        nothing — which is every client that was written before this mode
        existed, and every client with no camera to point.
    """
    key = str(value if value is not None else "").strip().casefold()
    if not key:
        return NARRATION_UNSUPPORTED
    return _NARRATION_STATES.get(key, NARRATION_UNSUPPORTED)


def normalize_narration_seconds(value: Any, **_ignored: Any) -> float | None:
    """Read a requested pace as the nearest pace actually offered.

    Snapped rather than refused. "Describe much more often" is a reasonable
    thing to say, and it should land on the fastest pace there is rather than
    come back as an error a person listening to their phone cannot act on. It
    also has to land on one of the five the browser offers, or the avatar and
    the Accessibility page would disagree about how often it is reading.

    :param value: Seconds between readings, as the model or browser gave it.
    :returns: One of ``NARRATION_PACE_OPTIONS``, or ``None`` when nothing
        usable was given.
    """
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return min(NARRATION_PACE_OPTIONS, key=lambda option: abs(option - seconds))


def narration_word_budget(seconds: Any) -> int:
    """How many words one reading may run to, at a given pace.

    This is what makes the pace worth changing. A reading has to finish before
    the next one starts, and speech runs at roughly two and a half words a
    second — so eighty words is a twenty-second reading, which at a five-second
    pace means the person hears a permanent backlog of a scene they have
    already walked out of. Faster therefore has to mean SHORTER: at five
    seconds one clause about the one thing that matters, at a minute the fuller
    picture.

    :param seconds: The pace, in seconds between readings.
    :returns: A word budget for the description.
    """
    pace = normalize_narration_seconds(seconds) or NARRATION_PACE_OPTIONS[0]
    # Roughly 60% of the gap spent speaking, at about 2.5 words a second, so
    # there is silence between readings rather than a continuous monologue.
    return max(8, min(80, int(pace * 0.6 * 2.5)))


def _tell_the_browser(payload: dict) -> None:
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


def build_scene_narration_tools(
    context: Any, *, scene_narration: Any, scene_narration_seconds: Any = None
) -> list[Any]:
    """Build the tool that switches scene narration on and off, when a client can.

    :param context: The run's ``GlobalContext``. Held for symmetry with the
        other tool factories; switching the mode costs no configuration.
    :param scene_narration: What the browser reported this turn — ``on``,
        ``off``, or nothing at all.
    :param scene_narration_seconds: How often the browser is describing the
        scene right now, so the avatar can make a request to go faster or
        slower into a number rather than a guess.
    :returns: The tool, or ``[]`` for a client that cannot narrate.
    """
    state = normalize_scene_narration_state(scene_narration)
    if state == NARRATION_UNSUPPORTED:
        return []
    narrating_now = state == NARRATION_ON
    pace_seconds = normalize_narration_seconds(scene_narration_seconds)

    @tool(SET_SCENE_NARRATION_TOOL_NAME)
    async def set_scene_narration(
        enabled: bool, every_seconds: float | None = None, reason: str = ""
    ) -> dict:
        """Switch continuous scene narration on or off.

        {state_line} Scene narration is the accessibility mode: while it is on, the
        conversation partner's camera is pointed at whatever is in front of
        them, the scene is described every few seconds, and each description is
        read aloud to them. It is built for a conversation partner who cannot
        see the scene, so it is the difference between them knowing what is
        around them and not knowing.

        Call this with enabled=true whenever the conversation partner asks to be
        told what is around them from now on, to have their surroundings
        described, to be guided or walked through a place, or asks for the
        accessibility or narration mode — in any words, including words that
        only approximately name it, and including a request made because they
        say they cannot see. Call it with enabled=false when they ask for the
        describing to stop, say they are done, or ask for quiet.

        Switching the mode on takes effect immediately and needs nothing from
        the conversation partner beyond allowing the camera if they have not
        already; the browser opens the outward-facing camera itself. Say in one
        short sentence that the descriptions are starting (or stopping) — a
        conversation partner who cannot see the screen has only what is said to
        tell them the mode changed.

        Also call this, with enabled=true, to change HOW OFTEN the
        conversation partner is told what is in view. {pace_line} Asking for
        more often, faster, more detail as they move, or saying the gaps are
        too long means a SMALLER every_seconds; less often, slower, quieter,
        too much talking, or wanting room to think means a LARGER one. Change
        it by a real step rather than a token one — roughly half or double the
        current pace is what "more often" and "less often" mean to somebody
        listening — and leave every_seconds out entirely when the request is
        only to start or stop.

        This is NOT how to take a single look at something: one look at what is
        in view right now is look_now, and it leaves the mode alone. Use this
        only for the standing mode.

        :param enabled: True to start narrating the scene continuously, false
            to stop. Pass true when changing only the pace.
        :param every_seconds: How many seconds between readings. Leave unset to
            keep the current pace. A value outside what this device can do is
            brought to the nearest pace that works rather than refused.
        :param reason: A few words on what the conversation partner asked for,
            for the conversation's record.
        """
        wanted = bool(enabled)
        asked_pace = normalize_narration_seconds(every_seconds)
        _tell_the_browser(
            {
                "type": SCENE_NARRATION_EVENT,
                "enabled": wanted,
                "every_seconds": asked_pace,
                "reason": str(reason or "").strip(),
            }
        )
        if asked_pace is not None and wanted:
            # A pace change on a mode that is already running is its own
            # outcome, and saying "it was already on" to somebody who just
            # asked for it to go faster would read as the request being
            # ignored.
            return {
                "status": "changed",
                "scene_narration": NARRATION_ON,
                "every_seconds": asked_pace,
                "message": (
                    f"Scene narration is on and now describes the scene every "
                    f"{asked_pace:g} seconds"
                    + (
                        f", instead of every {pace_seconds:g}."
                        if pace_seconds is not None and pace_seconds != asked_pace
                        else "."
                    )
                    + " Say so in a few words, naming the new pace, so the "
                    "conversation partner knows the change took effect."
                ),
            }
        if wanted == narrating_now:
            # Not an error: a conversation partner who cannot see the interface
            # has no way to check, and asking twice is what anybody does when
            # they are not sure. Say it is on rather than correcting them.
            return {
                "status": "unchanged",
                "scene_narration": NARRATION_ON if wanted else NARRATION_OFF,
                "message": (
                    "Scene narration was already on and is still on. Tell the "
                    "conversation partner it is running, briefly."
                    if wanted
                    else "Scene narration was already off and is still off."
                ),
            }
        return {
            "status": "changed",
            "scene_narration": NARRATION_ON if wanted else NARRATION_OFF,
            "message": (
                (
                    "Scene narration is now on. The camera is being opened and "
                    "descriptions of what is in front of the conversation "
                    "partner will begin arriving within a few seconds, each one "
                    "read aloud to them. Say so in one short sentence, and tell "
                    "them they can ask for it to stop at any time. If the "
                    "camera cannot be opened the next turns will simply carry "
                    "no observations, and it is worth asking them to allow the "
                    "camera when that happens."
                )
                if wanted
                else (
                    "Scene narration is now off; the camera is closed and no "
                    "more descriptions will arrive. Say so in a few words."
                )
            ),
        }

    # What the model reads has to say which way the switch is currently set:
    # without it the avatar answers "please start describing" by calling the
    # tool that is already on, or announces a change it did not make.
    state_line = (
        "Scene narration is ON right now: the scene is being described to the "
        "conversation partner continuously."
        if narrating_now
        else "Scene narration is OFF right now."
    )
    pace_line = (
        (
            f"Right now a reading arrives every {pace_seconds:g} seconds. "
            if pace_seconds is not None
            else ""
        )
        + "The paces available are every "
        + ", ".join(f"{option:g}" for option in NARRATION_PACE_OPTIONS[:-1])
        + f" or {SLOWEST_NARRATION_SECONDS:g} seconds, and nothing in between — "
        "a value between two of them is taken as the nearer one. A faster pace "
        "also makes each reading SHORTER, down to a single clause about the one "
        "thing that matters most, so ask for faster when the conversation "
        "partner needs to keep up with somewhere they are moving through and "
        "slower when they want the fuller picture."
    )
    set_scene_narration.description = (
        (set_scene_narration.description or "")
        .replace("{state_line}", state_line)
        .replace("{pace_line}", pace_line)
    )
    return [set_scene_narration]
