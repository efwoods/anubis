"""What the browser could see when a turn paused, remembered for the resume.

A ``look_now`` pause is answered by a second HTTP request — ``POST
/message/{assistant_id}/resume`` — and the resumed run rebuilds every tool from
THAT request's configuration. So the resume has to arrive carrying the same
report of what is shared, what can be opened for a look, and whether this
avatar may open anything, that the paused turn carried. If it does not, the
rebuilt ``look_now`` sees a browser sharing nothing, takes the branch that
returns ``not_shared`` without pausing, and never reaches the ``interrupt``
that would have collected the frame the browser just captured and described.
The person sees their camera light come on and then hears the avatar say it
cannot see anything — the exact failure ``look_now`` was built to prevent,
arriving through the back door.

The browser sends the report again on the resume, and this registry is the
second line: the paused turn records what it reported, keyed by thread, and the
resume falls back to that when the request did not carry one (an older client,
a reload between the pause and the answer, a client that only ever learned the
first half of this protocol). Entries are in-process, expire on their own, and
are capped — a look is answered within seconds, and a context nobody came back
for is worth nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

#: How long a recorded context is worth falling back to. A browser answers a
#: look in the time it takes to grab one frame; well beyond that, what the
#: browser could see has almost certainly changed.
LOOK_CONTEXT_MEMORY_SECONDS = 600.0

#: Ceiling on remembered threads, so a server nobody resumes on cannot grow
#: this table without bound. Oldest recorded goes first.
LOOK_CONTEXT_MEMORY_MAX_THREADS = 2048


@dataclass(frozen=True)
class LookContext:
    """One turn's report of what the browser could see.

    The two share fields are kept as the raw form values the browser sent,
    because that is exactly what the configurable carries and what
    ``normalize_live_shares`` / ``peekable_sources`` are written to read.
    """

    live_shares: str = ""
    peekable_shares: str = ""
    may_control_shares: bool = False
    #: What the browser said about scene narration (``on`` / ``off``), or
    #: ``""`` for a client that cannot narrate. Kept raw for the same reason
    #: as the share fields: it is what the configurable carries.
    scene_narration: str = ""

    def says_anything(self) -> bool:
        """Whether this context reports any way for the avatar to see."""
        return bool(
            (self.live_shares or "").strip()
            or (self.peekable_shares or "").strip()
            or self.may_control_shares
            or (self.scene_narration or "").strip()
        )


class LookContextRegistry:
    """In-process table of what each thread's last turn could see."""

    def __init__(
        self,
        *,
        memory_seconds: float = LOOK_CONTEXT_MEMORY_SECONDS,
        max_threads: int = LOOK_CONTEXT_MEMORY_MAX_THREADS,
    ) -> None:
        """Start with nothing remembered."""
        self._memory_seconds = float(memory_seconds)
        self._max_threads = int(max_threads)
        self._by_thread_id: dict[str, tuple[float, LookContext]] = {}

    def remember(
        self, thread_id: str | None, context: LookContext, *, now: float | None = None
    ) -> None:
        """Record what a turn reported, so its own pause can be answered.

        A context reporting nothing is not recorded but does forget whatever
        the thread had: a browser that has stopped sharing and stopped allowing
        looks must not be answered out of what it allowed ten minutes ago.
        """
        identifier = str(thread_id or "").strip()
        if not identifier:
            return
        if not context.says_anything():
            self._by_thread_id.pop(identifier, None)
            return
        moment = time.time() if now is None else float(now)
        self._by_thread_id[identifier] = (moment, context)
        self._forget_the_expired(now=moment)
        while len(self._by_thread_id) > self._max_threads:
            oldest = min(
                self._by_thread_id,
                key=lambda key: self._by_thread_id[key][0],
            )
            self._by_thread_id.pop(oldest, None)

    def recall(
        self, thread_id: str | None, *, now: float | None = None
    ) -> LookContext | None:
        """What that thread's last turn reported, or ``None`` if too old."""
        identifier = str(thread_id or "").strip()
        if not identifier:
            return None
        recorded = self._by_thread_id.get(identifier)
        if recorded is None:
            return None
        moment = time.time() if now is None else float(now)
        recorded_at, context = recorded
        if moment - recorded_at > self._memory_seconds:
            self._by_thread_id.pop(identifier, None)
            return None
        return context

    def forget(self, thread_id: str | None) -> None:
        """Drop one thread's context."""
        self._by_thread_id.pop(str(thread_id or "").strip(), None)

    def _forget_the_expired(self, *, now: float) -> None:
        for identifier, (recorded_at, _context) in list(self._by_thread_id.items()):
            if now - recorded_at > self._memory_seconds:
                self._by_thread_id.pop(identifier, None)
