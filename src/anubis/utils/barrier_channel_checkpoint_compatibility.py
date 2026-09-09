"""Barrier-channel checkpoints that are read back as lists instead of sets.

``message_workflow`` fans out from ``chat`` into ``resolve_human_message_images``
and ``observe_user`` and fans back in with

    add_edge(["resolve_human_message_images", "observe_user"], "join_user_observation")

which LangGraph compiles into a ``NamedBarrierValue`` channel. That channel's
checkpoint is the ``set`` of branch names that have arrived so far, and
``NamedBarrierValue.update`` calls ``self.seen.add(...)``.

When the LangGraph platform reads a thread's state back through its own
checkpoint path, the stored set is handed to ``from_checkpoint`` as a ``list``.
Nothing notices until the platform has to replay ``apply_writes`` on read — which
happens exactly when the thread carries a pending write to the join channel,
i.e. when a run was cancelled or interrupted between the fan-out and the fan-in.
``NamedBarrierValue.update`` then raises ``AttributeError: 'list' object has no
attribute 'add'``, ``GET /threads/{thread_id}/state`` answers 500, and
``GET /conversations/{thread_id}/messages`` turns that into the
``Error loading messages`` the browser shows. The conversation stays unreadable
for as long as the pending write is there, which is forever.

Restoring ``seen`` to a ``set`` when a checkpoint is loaded gives the channel back
its own documented contract. Nothing else about the channel changes: an already
correct ``set`` checkpoint is left exactly as it is.
"""

from __future__ import annotations

import functools
from typing import Any

from langgraph.channels.named_barrier_value import (
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
)

_PATCH_MARKER_ATTRIBUTE = "_anubis_restores_seen_as_set"


def _with_seen_restored_to_set(channel: Any) -> Any:
    """Return ``channel`` with the ``seen`` attribute held as a ``set``."""
    seen = getattr(channel, "seen", None)
    if seen is not None and not isinstance(seen, set):
        channel.seen = set(seen)
    return channel


def _patch_barrier_channel_class(barrier_channel_class: type) -> None:
    """Make one barrier channel class load a list checkpoint as a ``set``."""
    original_from_checkpoint = barrier_channel_class.from_checkpoint
    if getattr(original_from_checkpoint, _PATCH_MARKER_ATTRIBUTE, False):
        return

    @functools.wraps(original_from_checkpoint)
    def from_checkpoint(self: Any, checkpoint: Any) -> Any:
        return _with_seen_restored_to_set(
            original_from_checkpoint(self, checkpoint)
        )

    setattr(from_checkpoint, _PATCH_MARKER_ATTRIBUTE, True)
    barrier_channel_class.from_checkpoint = from_checkpoint  # type: ignore[method-assign]


def apply_barrier_channel_checkpoint_compatibility_patch() -> None:
    """Apply the patch to both barrier channel classes. Safe to call twice."""
    _patch_barrier_channel_class(NamedBarrierValue)
    _patch_barrier_channel_class(NamedBarrierValueAfterFinish)
