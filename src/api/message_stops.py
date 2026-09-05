"""Stopping a reply that is still being generated.

A reply streams over ``POST /message/{assistant_id}`` (and its ``/resume``
sibling) as server-sent events. Two things can end that stream before the
avatar finishes on its own:

* the person presses Stop, and the browser calls
  ``POST /message/{assistant_id}/stop`` with the ``request_id`` the stream
  announced in its first frame (or the ``thread_id`` when the request id was
  never seen);
* the connection drops — the tab closed, the network failed, or the browser
  aborted the fetch because the stop route could not be reached.

Both paths funnel through the objects here. ``ActiveMessageTurnRegistry`` is
the in-process table of replies being generated right now, so the stop route
can find the running turn and wake the generator that is streaming it.
``GraphStreamPump`` drains ``graph.astream`` on its own task and hands items
to the generator through a queue, so a stop request can be injected as a
sentinel between two frames instead of waiting for the graph to produce the
next one, and so the graph run itself can be cancelled from outside the
generator that is consuming it.

The registry is per process. A stop request that reaches a process other
than the one streaming the reply finds no turn and answers 404; the browser
then falls back to aborting the fetch, which the streaming process sees as a
disconnect and finalizes the same way.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from time import time_ns
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, RemoveMessage

logger = logging.getLogger(__name__)

# How long a typed turn waits for an in-flight ambient observation on the same
# thread to wind down after being told to stop, before streaming anyway. The
# wait exists so the two runs never write the thread's checkpoint at the same
# time; the bound exists so a stuck observation cannot hold a typed turn.
AMBIENT_OBSERVATION_YIELD_TIMEOUT_SECONDS = 15.0

# What an ambient observation that arrives while the thread is busy with another
# turn is told to wait before trying again (the ``Retry-After`` of the 409).
AMBIENT_BUSY_RETRY_AFTER_SECONDS = 5

# Rough characters-per-token ratio used to estimate the completion tokens a
# stopped reply consumed. The provider's stream was cut before it reported
# usage, so the estimate is what gets metered; four characters per token is
# the customary English approximation and errs slightly high for prose.
CHARACTERS_PER_TOKEN_ESTIMATE = 4

# The outer graph node whose completion a stopped reply is recorded as. The
# ``anubis`` node wraps the whole think loop; writing the partial reply as
# that node's output leaves the thread with a human turn followed by the
# avatar's truncated answer, and nothing pending.
STOPPED_REPLY_GRAPH_NODE = "anubis"


class StopRequested:
    """Sentinel queued to the streaming generator when a stop is requested."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Name the sentinel in logs and debugger output."""
        return "<StopRequested>"


class StreamEnded:
    """Sentinel queued when ``graph.astream`` finishes on its own."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Name the sentinel in logs and debugger output."""
        return "<StreamEnded>"


@dataclass
class StreamFailed:
    """Queued when ``graph.astream`` raised; the generator re-raises it."""

    error: BaseException


STOP_REQUESTED = StopRequested()
STREAM_ENDED = StreamEnded()


@dataclass
class ActiveMessageTurn:
    """One reply being generated right now."""

    request_id: str
    thread_id: str | None
    assistant_id: str | None
    user_id: str | None
    started_at_ns: int = field(default_factory=time_ns)
    # An ambient observation (a hidden webcam / screen snapshot) rather than a
    # turn the person typed or spoke. Observations yield to typed turns: a
    # typed turn arriving on the same thread stops them and waits for
    # ``finished`` before streaming.
    ambient: bool = False
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    # Set once the graph run behind this turn has fully wound down and any
    # record of the stop has been written, so a turn waiting on this thread can
    # start without racing the checkpoint.
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    _wake: Any = None

    def mark_finished(self) -> None:
        """Record that the graph run behind this turn is over, however it ended."""
        self.finished.set()

    def attach_wake(self, wake) -> None:
        """Register the callable that interrupts the streaming generator.

        The generator sets this to the pump's ``request_stop`` once the pump
        exists. A stop that arrives before then is remembered on
        ``stop_requested`` and honoured as soon as the pump is attached.
        """
        self._wake = wake
        if self.stop_requested.is_set():
            wake()

    def request_stop(self) -> None:
        """Ask the generator streaming this turn to end the reply now."""
        if self.stop_requested.is_set():
            return
        self.stop_requested.set()
        if self._wake is not None:
            self._wake()


class ActiveMessageTurnRegistry:
    """In-process table of the replies being generated right now."""

    def __init__(self) -> None:
        """Start with no replies in flight."""
        self._turns_by_request_id: dict[str, ActiveMessageTurn] = {}

    def register(
        self,
        *,
        request_id: str,
        thread_id: str | None,
        assistant_id: str | None,
        user_id: str | None,
        ambient: bool = False,
    ) -> ActiveMessageTurn:
        """Record a reply that just started streaming."""
        turn = ActiveMessageTurn(
            request_id=request_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            user_id=user_id,
            ambient=bool(ambient),
        )
        self._turns_by_request_id[request_id] = turn
        return turn

    def unregister(self, request_id: str) -> None:
        """Forget a reply whose stream has ended, however it ended."""
        self._turns_by_request_id.pop(request_id, None)

    def find(
        self,
        *,
        request_id: str | None = None,
        thread_id: str | None = None,
    ) -> ActiveMessageTurn | None:
        """Find a running turn by request id, else the newest one on a thread."""
        if request_id:
            turn = self._turns_by_request_id.get(request_id)
            if turn is not None:
                return turn
        if thread_id:
            candidates = [
                turn
                for turn in self._turns_by_request_id.values()
                if turn.thread_id == thread_id
            ]
            if candidates:
                return max(candidates, key=lambda turn: turn.started_at_ns)
        return None

    def ambient_turns_on_thread(self, thread_id: str | None) -> list[ActiveMessageTurn]:
        """Return the ambient observations being processed on a thread right now."""
        if not thread_id:
            return []
        return [
            turn
            for turn in self._turns_by_request_id.values()
            if turn.ambient and turn.thread_id == thread_id
        ]

    def has_turn_on_thread(
        self, thread_id: str | None, *, except_request_id: str | None = None
    ) -> bool:
        """Whether any turn (typed, spoken, or ambient) is running on a thread."""
        if not thread_id:
            return False
        return any(
            turn.thread_id == thread_id and turn.request_id != except_request_id
            for turn in self._turns_by_request_id.values()
        )

    def __len__(self) -> int:
        """Count the replies being generated right now."""
        return len(self._turns_by_request_id)

    def active_request_ids(self) -> list[str]:
        """List the request ids of the replies being generated right now."""
        return list(self._turns_by_request_id)


class GraphStreamPump:
    """Drain an async iterator on its own task and expose it through a queue.

    ``next_item`` returns the iterator's items in order, then ``STREAM_ENDED``.
    ``request_stop`` cancels the draining task and queues ``STOP_REQUESTED``
    so a waiting consumer wakes immediately, even when the graph is in the
    middle of a long tool call and would not produce another item for a while.
    ``aclose`` cancels the draining task and waits for the iterator to close;
    the consumer calls this once the reply is over however the reply ended.
    """

    def __init__(self, source: AsyncIterator[Any]) -> None:
        """Start draining ``source`` immediately on a task of its own."""
        self._source = source
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._stopped = False
        self._task: asyncio.Task[None] = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        try:
            async for item in self._source:
                if self._stopped:
                    break
                self._queue.put_nowait(item)
        except asyncio.CancelledError:
            # Cancelled by request_stop / aclose: the consumer already knows.
            raise
        except BaseException as error:  # noqa: BLE001 - re-raised by the consumer
            if not self._stopped:
                self._queue.put_nowait(StreamFailed(error))
            return
        finally:
            aclose = getattr(self._source, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except BaseException:  # noqa: BLE001 - closing must not mask the turn
                    logger.debug("Closing the graph stream raised", exc_info=True)
        if not self._stopped:
            self._queue.put_nowait(STREAM_ENDED)

    async def next_item(self) -> Any:
        """Return the next item, ``STREAM_ENDED``, or ``STOP_REQUESTED``.

        Raises the source's exception when the source failed.
        """
        item = await self._queue.get()
        if isinstance(item, StreamFailed):
            raise item.error
        return item

    def request_stop(self) -> None:
        """Wake the consumer with ``STOP_REQUESTED`` and cancel the drain."""
        if self._stopped:
            return
        self._stopped = True
        self._task.cancel()
        self._queue.put_nowait(STOP_REQUESTED)

    def cancel(self) -> None:
        """Cancel the drain without waiting; safe from a cancelled scope."""
        self._stopped = True
        self._task.cancel()

    async def aclose(self) -> None:
        """Cancel the drain and wait for the graph run to wind down."""
        self.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            # Either the pump task was cancelled (expected) or the caller is
            # being cancelled; the latter must propagate.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except BaseException:  # noqa: BLE001 - the run is over either way
            logger.debug(
                "The graph stream ended with an error while stopping", exc_info=True
            )

    @property
    def done(self) -> bool:
        """Whether the draining task has finished, by any route."""
        return self._task.done()


async def yield_in_flight_ambient_observations(
    registry: ActiveMessageTurnRegistry | None,
    thread_id: str | None,
    *,
    timeout_seconds: float = AMBIENT_OBSERVATION_YIELD_TIMEOUT_SECONDS,
) -> list[str]:
    """Stop the ambient observations running on a thread and wait for them to end.

    Called before a typed or spoken turn starts its graph run. An observation is
    disposable context; the person's message is not. Stopping the observation
    first, and waiting for the run behind it to wind down, means the two never
    write the thread's checkpoint at the same time — which is how a typed
    exchange could otherwise be lost to a snapshot that finished later.

    Returns the request ids of the observations that were told to stop. The
    wait is bounded by ``timeout_seconds``; an observation still winding down
    after that is logged and the typed turn proceeds.
    """
    if registry is None or not thread_id:
        return []
    observations = registry.ambient_turns_on_thread(thread_id)
    if not observations:
        return []
    for observation in observations:
        observation.request_stop()
    waits = [observation.finished.wait() for observation in observations]
    try:
        await asyncio.wait_for(asyncio.gather(*waits), timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(
            "Ambient observation(s) %s on thread %s did not wind down within %.1fs; "
            "the typed turn proceeds",
            [observation.request_id for observation in observations],
            thread_id,
            timeout_seconds,
        )
    return [observation.request_id for observation in observations]


async def discard_ambient_observation(
    graph, config: dict, message_id: str | None
) -> bool:
    """Remove an observation that was stopped before the graph triaged it.

    An observation stopped mid-way may still hold the raw snapshot bytes (the
    images are described into text only once ``resolve_human_message_images``
    runs) or a description nobody decided on. Neither is worth keeping as
    context, and the raw bytes would be fed to the model on every later turn.
    Returns whether a removal was written. Never raises.
    """
    if not message_id:
        return False
    update_state = getattr(graph, "aupdate_state", None)
    if update_state is None:
        return False
    try:
        await update_state(
            config,
            {"messages": [RemoveMessage(id=message_id)]},
            as_node=STOPPED_REPLY_GRAPH_NODE,
        )
        return True
    except Exception:  # noqa: BLE001 - see docstring
        logger.warning(
            "Could not discard the stopped ambient observation %s on thread %s",
            message_id,
            (config.get("configurable") or {}).get("thread_id"),
            exc_info=True,
        )
        return False


def estimate_completion_tokens(text: str) -> int:
    """Estimate the completion tokens a partial reply consumed."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARACTERS_PER_TOKEN_ESTIMATE))


def build_stopped_reply_metadata(
    partial_text: str,
    *,
    estimated_prompt_tokens: int,
    model_name: str | None,
    stopped_by: str,
) -> dict[str, Any]:
    """``response_metadata`` for a reply that was cut short.

    The provider never reported usage for the aborted stream, so the token
    counts are estimates: the request's pre-call prompt estimate plus a
    character-based guess at the completion. ``stopped_by`` is ``"user"``
    when the stop route ended the reply and ``"disconnect"`` when the client
    went away.
    """
    prompt_tokens = max(0, int(estimated_prompt_tokens or 0))
    completion_tokens = estimate_completion_tokens(partial_text)
    metadata: dict[str, Any] = {
        "stopped": True,
        "stopped_by": stopped_by,
        "stopped_by_user": stopped_by == "user",
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
        },
    }
    if model_name:
        metadata["model_name"] = model_name
    return metadata


def build_stopped_reply_message(
    partial_text: str, *, request_id: str, response_metadata: dict[str, Any]
) -> AIMessage:
    """Build the truncated avatar reply as it is recorded on the thread."""
    return AIMessage(
        id=f"stopped-{request_id}",
        content=partial_text,
        response_metadata=dict(response_metadata),
    )


async def persist_stopped_reply(
    graph, config: dict, partial_text: str, *, request_id: str, response_metadata: dict
) -> bool:
    """Record the partial reply on the thread so the transcript keeps it.

    A reply stopped before its first token is not recorded: an empty avatar
    turn would only confuse the next model call. Returns whether a message
    was written. Never raises — a thread that keeps only the human turn is
    an acceptable outcome of a stop, and the stream must still end cleanly.
    """
    if not (partial_text or "").strip():
        return False
    update_state = getattr(graph, "aupdate_state", None)
    if update_state is None:
        return False
    try:
        await update_state(
            config,
            {
                "messages": [
                    build_stopped_reply_message(
                        partial_text,
                        request_id=request_id,
                        response_metadata=response_metadata,
                    )
                ]
            },
            as_node=STOPPED_REPLY_GRAPH_NODE,
        )
        return True
    except Exception:  # noqa: BLE001 - see docstring
        logger.warning(
            "Could not record the stopped reply on thread %s",
            (config.get("configurable") or {}).get("thread_id"),
            exc_info=True,
        )
        return False


# Background finalizers for turns whose client disconnected. Held so the tasks
# are not garbage-collected before they finish.
_background_finalizers: set[asyncio.Task[Any]] = set()


def schedule_background(coroutine) -> asyncio.Task[Any]:
    """Run ``coroutine`` to completion independently of the current task.

    Used from inside a request task that is being cancelled: the request's
    own awaits would be cancelled again, so the work is handed to a fresh
    task that the cancellation does not reach.
    """
    task = asyncio.create_task(coroutine)
    _background_finalizers.add(task)
    task.add_done_callback(_background_finalizers.discard)
    return task
