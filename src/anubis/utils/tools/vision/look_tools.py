"""The ``look_now`` tool: one fresh look at what is being shared this moment.

Ambient vision describes the webcam and the shared screen on a fixed interval,
as background context (see ``src/anubis/utils/ambient/``). Those descriptions
are the only thing the avatar knows about the scene, and by the time the
conversation partner asks "what is on my screen" the newest one can be minutes
old — the ambient loop skips a capture whenever a turn is in flight, and drops
a frame that has not changed since the frame before it. Worse, an observation
of a screen the conversation partner has since stopped sharing stays in the
thread forever, and reads exactly like a description of what is on the screen
right now.

``look_now`` closes both gaps. The tool is attached ONLY on a turn where the
browser reported a live webcam or screen share, or reported that it can open
one for a single look, so an ordinary conversation neither sees the tool nor
pays for it. When the avatar calls the tool the run pauses on an ``interrupt``;
the browser captures one frame per requested source and resumes the run through
``POST /message/{assistant_id}/resume``, which describes the frames and hands
the descriptions back as the resume value. The pause is marked ``silent``: no
card is raised and nobody is asked to approve anything — the conversation
partner sees only that the reply took a moment.

The camera and the desktop are two different views and the avatar has to
choose between them: the camera answers questions about the conversation
partner and their room, the desktop answers questions about what they are
working on. Both can be *peeked* at — opened for one frame and closed again —
when the browser says so, but they get there differently. A granted camera
permission persists on the origin, so the browser reopens the camera itself. A
desktop cannot be opened by a page at all (``getDisplayMedia`` needs a real
gesture every time and keeps no standing grant), so a peekable desktop is one
the person granted once and the browser is still holding; a desktop that was
never granted is asked for with a button instead.

Because the resumed run re-enters this module and rebuilds these tools from the
resume request's own configuration, the resume MUST carry the same
``live_shares`` / ``peekable_shares`` / ``may_control_shares`` the paused turn
carried. If it does not, the rebuilt tool takes a different path through
``look_now``, never reaches the ``interrupt`` that collects the answer, and the
frame the browser just captured is silently thrown away — which reads, to the
person, as an avatar that opened their camera and then said it could not see
anything. ``resume_avatar_message`` restores that context from the request and,
failing that, from what the paused turn recorded.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool
from langgraph.types import interrupt

logger = logging.getLogger(__name__)

LOOK_NOW_TOOL_NAME = "look_now"
STOP_SHARING_TOOL_NAME = "stop_sharing"

#: The ``kind`` the browser matches to answer this pause without asking anyone.
LOOK_NOW_INTERRUPT_KIND = "look_now"

#: Stream frames the browser acts on. Neither pauses the run: stopping a share
#: needs no answer, and a screen share the browser cannot start on its own is
#: offered to the person as a button rather than waited for.
SHARE_REQUEST_EVENT = "share_request"
SHARE_STOP_EVENT = "share_stop"


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

SOURCE_WEBCAM = "webcam"
SOURCE_SCREEN = "screen"

#: The sources a look may ask for, in the order they are offered to the model.
LOOKABLE_SOURCES = (SOURCE_WEBCAM, SOURCE_SCREEN)

#: What each source answers, in the words the model reads. The two are not
#: interchangeable and the avatar has to pick between them: a question about
#: the conversation partner themselves is answered by the camera, a question
#: about what they are working on by the desktop, and answering one with the
#: other is a wrong answer delivered confidently.
SOURCE_PURPOSE = {
    SOURCE_WEBCAM: (
        "the camera — the conversation partner themselves, their face and "
        "posture, who else is with them, and the room they are in"
    ),
    SOURCE_SCREEN: (
        "the desktop — what is on the conversation partner's screen: the "
        "application in front of them, the text, the code, the page, the error"
    ),
}

#: A held desktop grant is a capture stream the browser can end; a camera that
#: may be opened for a look is a standing browser permission with nothing
#: running to switch off. Only the first can be stopped while it is merely
#: peekable rather than shared.
STOPPABLE_WHILE_PEEKABLE = (SOURCE_SCREEN,)

_SOURCE_ALIASES = {
    "webcam": SOURCE_WEBCAM,
    "camera": SOURCE_WEBCAM,
    "cam": SOURCE_WEBCAM,
    "screen": SOURCE_SCREEN,
    "screenshare": SOURCE_SCREEN,
    "screen_share": SOURCE_SCREEN,
    "screenshot": SOURCE_SCREEN,
    "display": SOURCE_SCREEN,
}


def normalize_live_shares(value: Any) -> list[str]:
    """Read the browser's live-share report into known source names.

    Accepts the form field as a JSON list, a comma-separated string, or an
    already-parsed sequence. Unknown names are dropped rather than trusted: the
    field decides whether a tool that pauses the run is attached at all, so a
    client typo must not open a pause the browser will never answer.

    :param value: The ``live_shares`` form value, or any sequence of names.
    :returns: The live sources, de-duplicated, in ``LOOKABLE_SOURCES`` order.
    """
    raw: list[Any]
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return []
            raw = list(parsed) if isinstance(parsed, list) else []
        else:
            raw = text.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        return []

    found: set[str] = set()
    for item in raw:
        name = _SOURCE_ALIASES.get(str(item).strip().casefold())
        if name:
            found.add(name)
    return [source for source in LOOKABLE_SOURCES if source in found]


def _observations_of(answer: Any) -> list[dict[str, str]]:
    """Read the descriptions the browser's resume carried, defensively."""
    if not isinstance(answer, dict):
        return []
    observations = answer.get("observations")
    if not isinstance(observations, list):
        return []
    read: list[dict[str, str]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        description = str(observation.get("description") or "").strip()
        if not description:
            continue
        read.append(
            {
                "source": str(observation.get("source") or "image").strip(),
                "description": description,
            }
        )
    return read


def peekable_sources(
    peekable_shares: Any, *, may_control_shares: bool = False
) -> list[str]:
    """What the browser can open for one look, from what the browser reported.

    ``peekable_shares`` is the browser's own report: the camera when the owner
    granted the camera-peek permission in this browser, the desktop when the
    owner granted a desktop peek and the browser is still holding that grant.
    A client that never sends the field is read the way clients were read
    before the field existed — the camera is peekable exactly when the avatar
    may control shares — so an older browser keeps the behaviour it had.

    :param peekable_shares: The ``peekable_shares`` form value, or a sequence.
    :param may_control_shares: The avatar-settings permission, used only to
        read a client that sends no ``peekable_shares`` at all.
    :returns: The peekable sources, in ``LOOKABLE_SOURCES`` order.
    """
    reported_nothing = peekable_shares is None or (
        isinstance(peekable_shares, str) and not peekable_shares.strip()
    )
    if reported_nothing:
        return [SOURCE_WEBCAM] if may_control_shares else []
    return normalize_live_shares(peekable_shares)


def build_look_tools(
    context: Any,
    *,
    live_shares: Any,
    conversation_has_scene_observations: bool = False,
    may_control_shares: bool = False,
    peekable_shares: Any = None,
) -> list[Any]:
    """Build the ``look_now`` tool for a turn where looking could matter.

    The tool is attached in two situations, and in neither does an ordinary
    typed conversation pay for it. Something is live, so a look can be taken.
    Or nothing is live but this conversation already holds webcam / screen
    observations — the situation where the avatar would otherwise describe a
    screen that stopped being shared as though the avatar could still see it.
    In that second case the tool takes no pause and costs no capture: it
    answers, truthfully, that the source is not being shared. Checking beats
    asserting, and this is what the avatar checks with.

    What the browser reported is read at the start of the turn, so a share
    started in the second between that report and this call is not seen. That
    is deliberate: pausing every such turn on the chance a share just began
    would cost a round trip on the many turns where none did.

    :param context: The run's ``GlobalContext``. Held for symmetry with the
        other tool factories; the look itself costs no configuration.
    :param live_shares: What the browser reported as live on this turn — the
        ``live_shares`` form value, or a sequence of source names.
    :param conversation_has_scene_observations: Whether the thread already
        holds a webcam / screen observation.
    :param may_control_shares: Whether the owner has allowed this avatar, in
        this browser, to look on its own and to switch a share off. This is the
        avatar-settings permission; with it off the avatar can look only at
        what the person has already chosen to share.
    :param peekable_shares: What the browser can open for ONE look right now —
        the camera when the camera-peek permission is granted in this browser,
        the desktop when the owner granted a desktop peek and the browser is
        holding that grant. Absent from a client that does not report it, which
        is read as "the camera, when the avatar may control shares".
    :returns: The tools for this turn, or ``[]`` when looking could never matter.
    """
    live = normalize_live_shares(live_shares)
    # What can be opened for a single look and closed again. The camera rides a
    # standing browser permission, so a granted camera reopens without asking.
    # The desktop cannot: no browser lets a page call getDisplayMedia without a
    # real gesture and none keeps a standing grant, so a peekable desktop is a
    # grant the person gave once and the browser is still holding — the gesture
    # happened, and the held stream is what the peek captures from. A desktop
    # that is neither shared nor granted is still asked for with a button.
    peekable = peekable_sources(
        peekable_shares, may_control_shares=may_control_shares
    )
    if (
        not live
        and not conversation_has_scene_observations
        and not may_control_shares
        and not peekable
    ):
        return []

    live_description = " and ".join(live) if live else "nothing"
    openable = [source for source in peekable if source not in live]

    @tool(LOOK_NOW_TOOL_NAME)
    async def look_now(sources: list[str] | None = None, reason: str = "") -> dict:
        """Take one fresh look at what the conversation partner is sharing right now.

        {sharing_line} Ambient vision describes a shared webcam or screen on an
        interval, but an ambient description can be minutes old and a scene
        changes; this takes a new look this instant and reports what is not
        being shared as not being shared.

        Call this whenever what is happening NOW decides the answer — the
        conversation partner asks what is on the screen, what the assistant can
        see, how something looks, what the assistant makes of what is in front
        of the conversation partner, or asks about anything the assistant would
        have to be looking at to answer. Prefer calling this over describing an
        earlier observation: an observation from before is what the scene used
        to be, and reading it back as though it were the present is the one
        mistake this tool exists to prevent.

        THE TWO SOURCES ARE DIFFERENT VIEWS AND ARE NOT INTERCHANGEABLE. The
        camera shows the conversation partner and the room they are in. The
        desktop shows what is on their screen — the application, the page, the
        code, the error in front of them. Pick the one the question is about:
        "what do you see", "how do I look", "who is with me", "what is behind
        me" is the camera; "what is on my screen", "what am I looking at",
        "read this", "what is this error", "what do you make of this" is the
        desktop. Ask for both only when the answer genuinely needs both, and
        never answer a question about one of them from a look at the other.

        Do not call this on a turn where nothing about the live scene matters.

        :param sources: Which to look at — "webcam" for the camera, "screen"
            for the desktop, or both. Leave empty only when the question does
            not distinguish them; that looks at everything available. A source
            that is neither being shared nor openable is reported back as not
            shared rather than guessed at.
        :param reason: A few words on what the look is meant to settle, for the
            conversation's record.
        """
        # With no source named, a look covers what is already shared and — when
        # the avatar may open the camera — the camera too, since a person who
        # asked what the assistant can see is asking about themselves.
        requested = (
            normalize_live_shares(sources)
            if sources
            else normalize_live_shares(list(live) + openable)
        )
        wanted = [source for source in requested if source in live]
        # Not shared, but the browser can open it for one look.
        to_open = [
            source
            for source in requested
            if source not in live and source in openable
        ]
        # Not shared and not openable: a screen share, or any source with the
        # permission switched off.
        refused = [
            source
            for source in requested
            if source not in live and source not in openable
        ]

        if not wanted and not to_open:
            # The one case that asks the person for something: a screen the
            # browser cannot start by itself. The button is offered here rather
            # than after a pause, because there is nothing to wait for — and it
            # is deliberately NOT gated on ``may_control_shares``. That
            # permission governs the assistant acting on the person's own
            # device; putting a button in front of someone is asking them, and
            # asking never needed permission. Gating it meant a person who had
            # never granted the camera could not even be offered a look at
            # their screen.
            if SOURCE_SCREEN in refused:
                _tell_the_browser(
                    {
                        "type": SHARE_REQUEST_EVENT,
                        "sources": [SOURCE_SCREEN],
                        "reason": str(reason or "").strip(),
                    }
                )
            return {
                "status": "not_shared",
                "live_sources": live,
                "requested": requested,
                "offered_to_share": (
                    [SOURCE_SCREEN] if SOURCE_SCREEN in refused else []
                ),
                "message": (
                    "The conversation partner is not sharing "
                    f"{' or '.join(refused) if refused else 'that'} right now, "
                    "and it cannot be opened for a look either. "
                    + (
                        f"Being shared right now: {live_description}. "
                        if live
                        else "Nothing at all is being shared right now. "
                    )
                    + "Say plainly what is not being shared instead of "
                    "describing it from an earlier observation, which is what "
                    "that source used to show and not what it shows now."
                    + (
                        " A button has just been put in front of the "
                        "conversation partner that lets the assistant take one "
                        "look at their screen, so ask them to press it rather "
                        "than asking them to go and find the share control."
                        if SOURCE_SCREEN in refused
                        else ""
                    )
                ),
            }

        # Nothing above this line has a side effect, which matters: a resumed
        # run re-enters the tool from the top and runs all of it a second time.
        answer = interrupt(
            {
                "kind": LOOK_NOW_INTERRUPT_KIND,
                # The browser answers this pause on its own — no card, nobody
                # asked to approve anything.
                "silent": True,
                "sources": wanted,
                # Not being shared, but the browser may open it for this one
                # look and close it again straight after. The camera light is
                # the person's signal; nothing here starts a standing watch.
                "open": to_open,
                "reason": str(reason or "").strip(),
            }
        )

        observations = _observations_of(answer)
        if not observations:
            message = (
                str((answer or {}).get("message") or "").strip()
                if isinstance(answer, dict)
                else ""
            )
            return {
                "status": "unavailable",
                "live_sources": live,
                "message": message
                or (
                    "The fresh look did not come back. Say that the current view "
                    "could not be checked; do not fall back on an earlier "
                    "observation as though it were the present."
                ),
            }

        # Asked for after the pause, never before: everything above the
        # ``interrupt`` runs a second time when the run resumes, and the person
        # must not be handed the same button twice for one look.
        if SOURCE_SCREEN in refused:
            _tell_the_browser(
                {
                    "type": SHARE_REQUEST_EVENT,
                    "sources": [SOURCE_SCREEN],
                    "reason": str(reason or "").strip(),
                }
            )

        looked_at = [observation["source"] for observation in observations]
        opened = [source for source in to_open if source in looked_at]
        # Each description says which of the two views it is. Without this the
        # descriptions arrive as two paragraphs of prose and the avatar has no
        # way to say "on your screen" rather than "in front of you" — or, worse,
        # answers a question about the desktop out of the camera frame.
        views = "; ".join(
            f"{observation['source']} = "
            f"{SOURCE_PURPOSE.get(observation['source'], observation['source'])}"
            for observation in observations
        )
        return {
            "status": "looked",
            "looked_at": looked_at,
            "opened_for_this_look": opened,
            "not_shared": refused,
            "observations": observations,
            "message": (
                "THIS IS THE CURRENT VIEW of "
                f"{' and '.join(looked_at)}, captured just now. This supersedes "
                "every earlier observation of "
                f"{' and '.join(looked_at)} in this conversation: any of those "
                "marked EARLIER VIEW describe what was in view before and are "
                "history now. Answer from what is here, in the avatar's own "
                "voice, without reading the description back word for word. "
                f"Each observation names the view it came from ({views}); keep "
                "them apart in the answer and say which one is being described "
                "when both were looked at."
                + (
                    " The "
                    + " and ".join(opened)
                    + " was opened for this one look and has been closed again."
                    if opened
                    else ""
                )
                + (
                    " The screen is neither being shared nor open to a look; a "
                    "button that lets the assistant take one look at it has "
                    "just been put in front of the conversation partner."
                    if SOURCE_SCREEN in refused
                    else ""
                )
            ),
        }

    # The docstring is what the model reads to decide whether to call the tool,
    # so the sources that are actually live are named in it rather than left as
    # a general description of the capability.
    # What the model reads to decide whether to call the tool has to say, per
    # source, whether that view can be had at all — shared already, openable
    # for one look, or out of reach. A single sentence about "what is being
    # shared" left the avatar unable to tell a desktop it may glance at from a
    # desktop it cannot see.
    availability = []
    for source in LOOKABLE_SOURCES:
        purpose = SOURCE_PURPOSE[source]
        if source in live:
            availability.append(f"{source} ({purpose}) is being shared right now")
        elif source in openable:
            availability.append(
                f"{source} ({purpose}) is not being shared, but the conversation "
                "partner has allowed this avatar to open it for a single look "
                "and close it again, which is what asking for it does"
            )
        elif source == SOURCE_SCREEN:
            availability.append(
                f"{source} ({purpose}) cannot be seen — asking for it puts a "
                "button in front of the conversation partner instead of taking "
                "a look"
            )
        else:
            availability.append(f"{source} ({purpose}) cannot be seen at all")
    if live or openable:
        sharing_line = "Right now: " + "; ".join(availability) + "."
    else:
        sharing_line = (
            "Right now the conversation partner is sharing NOTHING — no camera "
            "and no desktop — and neither can be opened for a look. This "
            "conversation holds descriptions of a camera or a desktop from "
            "earlier, and calling this is how to confirm that none of them is "
            "what is in view now."
        )
    look_now.description = (look_now.description or "").replace(
        "{sharing_line}", sharing_line
    )
    if not may_control_shares:
        return [look_now]

    @tool(STOP_SHARING_TOOL_NAME)
    async def stop_sharing(sources: list[str] | None = None) -> dict:
        """Switch the conversation partner's camera or screen share off.

        Call this when the conversation partner asks for the camera or the
        screen share to be turned off, and when the assistant has finished with
        a share the conversation partner would plainly rather not leave running
        — after looking at something they asked about, at the end of what the
        share was for, or whenever leaving a camera on would serve nobody.

        Switching a share off never needs the conversation partner to do
        anything and takes effect at once. Switching one back ON is not
        something this can do: a camera or a desktop is opened only as part of
        look_now, for the one look that needs it, and a desktop that was never
        granted can be started only by the conversation partner pressing the
        button that look_now puts in front of them.

        :param sources: Which to switch off — "webcam" for the camera, "screen"
            for the desktop, or both. Leave empty to switch off everything the
            conversation partner currently has open.
        """
        # A desktop the browser is merely holding open for looks is a running
        # capture the person can see in their browser's sharing bar, so asking
        # for it to stop is a real request with a real effect. A camera that is
        # only peekable is a standing permission with nothing running, so there
        # is nothing there to switch off.
        stoppable = list(live) + [
            source
            for source in openable
            if source in STOPPABLE_WHILE_PEEKABLE and source not in live
        ]
        requested = normalize_live_shares(sources) if sources else stoppable
        switching_off = [source for source in requested if source in stoppable]
        if not switching_off:
            return {
                "status": "nothing_to_stop",
                "live_sources": live,
                "message": (
                    "None of those is open right now, so there was nothing to "
                    "switch off."
                ),
            }
        # No pause: the browser needs to answer nothing, so the turn does not
        # wait on this and the reply keeps its latency.
        _tell_the_browser(
            {"type": SHARE_STOP_EVENT, "sources": switching_off}
        )
        return {
            "status": "stopped",
            "stopped": switching_off,
            "message": (
                "The " + " and ".join(switching_off) + " has been switched off. "
                "Say so plainly and briefly; do not make more of it than that."
            ),
        }

    return [look_now, stop_sharing]
