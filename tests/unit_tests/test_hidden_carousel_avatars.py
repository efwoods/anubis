import pytest

from src.anubis.utils.hidden_carousel_avatars import (
    MAXIMUM_HIDDEN_CAROUSEL_AVATARS,
    normalize_hidden_assistant_ids,
)


def test_hidden_assistant_ids_are_trimmed_distinct_and_in_request_order():
    assert normalize_hidden_assistant_ids([" ava-2 ", "ava-1", "", None, "ava-2"]) == [
        "ava-2",
        "ava-1",
    ]


def test_hidden_assistant_ids_must_be_a_list():
    with pytest.raises(ValueError):
        normalize_hidden_assistant_ids("ava-1")
    with pytest.raises(ValueError):
        normalize_hidden_assistant_ids(None)


def test_hidden_assistant_ids_are_capped():
    with pytest.raises(ValueError):
        normalize_hidden_assistant_ids(
            [f"ava-{i}" for i in range(MAXIMUM_HIDDEN_CAROUSEL_AVATARS + 1)]
        )
