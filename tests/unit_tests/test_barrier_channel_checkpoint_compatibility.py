"""A barrier channel whose checkpoint comes back as a list still accepts writes.

``add_edge(["resolve_human_message_images", "observe_user"], "join_user_observation")``
compiles to a ``NamedBarrierValue``. The LangGraph platform reads that channel's
``set`` checkpoint back as a ``list``, and the first write to the channel then
raised ``AttributeError: 'list' object has no attribute 'add'`` — the
``Error loading messages`` a browser saw for every thread whose run was cancelled
between the fan-out and the fan-in.
"""

from langgraph.channels.named_barrier_value import (
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
)

from src.anubis.utils.barrier_channel_checkpoint_compatibility import (
    apply_barrier_channel_checkpoint_compatibility_patch,
)

BRANCH_NAMES = {"resolve_human_message_images", "observe_user"}


def test_named_barrier_value_accepts_a_list_checkpoint() -> None:
    apply_barrier_channel_checkpoint_compatibility_patch()
    channel = NamedBarrierValue(str, BRANCH_NAMES).from_checkpoint(
        ["resolve_human_message_images"]
    )

    assert channel.seen == {"resolve_human_message_images"}
    assert channel.update(["observe_user"]) is True
    assert channel.is_available() is True


def test_named_barrier_value_after_finish_accepts_a_list_checkpoint() -> None:
    apply_barrier_channel_checkpoint_compatibility_patch()
    channel = NamedBarrierValueAfterFinish(str, BRANCH_NAMES).from_checkpoint(
        (["resolve_human_message_images"], False)
    )

    assert channel.seen == {"resolve_human_message_images"}
    assert channel.update(["observe_user"]) is True
    assert channel.finish() is True
    assert channel.is_available() is True


def test_a_set_checkpoint_is_left_alone() -> None:
    apply_barrier_channel_checkpoint_compatibility_patch()
    channel = NamedBarrierValue(str, BRANCH_NAMES).from_checkpoint(
        {"resolve_human_message_images"}
    )

    assert channel.seen == {"resolve_human_message_images"}


def test_applying_the_patch_twice_does_not_stack_wrappers() -> None:
    apply_barrier_channel_checkpoint_compatibility_patch()
    from_checkpoint_after_first_call = NamedBarrierValue.from_checkpoint
    apply_barrier_channel_checkpoint_compatibility_patch()

    assert NamedBarrierValue.from_checkpoint is from_checkpoint_after_first_call
