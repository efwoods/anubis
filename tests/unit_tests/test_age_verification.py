"""Age verification policy and the in-memory repository."""

from datetime import date

import pytest

from src.anubis.utils.age_verification.policy import (
    date_of_birth_verification_error,
    filter_discoverable_avatars,
    parse_date_of_birth,
    years_elapsed_since,
)
from src.anubis.utils.age_verification.repository import (
    InMemoryAgeVerificationRepository,
    verification_public_view,
)


TODAY = date(2026, 9, 14)


def test_years_elapsed_since_counts_full_years_only():
    assert years_elapsed_since(date(2008, 9, 14), TODAY) == 18
    assert years_elapsed_since(date(2008, 9, 15), TODAY) == 17


def test_a_date_of_birth_under_the_minimum_is_refused():
    refusal = date_of_birth_verification_error(
        date(2012, 1, 1), minimum_years=18, on_date=TODAY
    )
    assert refusal is not None
    assert "18" in refusal


def test_a_date_of_birth_on_the_eighteenth_birthday_is_accepted():
    assert (
        date_of_birth_verification_error(
            date(2008, 9, 14), minimum_years=18, on_date=TODAY
        )
        is None
    )


def test_a_future_date_of_birth_is_refused():
    assert (
        date_of_birth_verification_error(
            date(2026, 9, 15), minimum_years=18, on_date=TODAY
        )
        == "Date of birth cannot be in the future."
    )


def test_parse_date_of_birth_requires_a_real_calendar_day():
    assert parse_date_of_birth("2000-01-02") == date(2000, 1, 2)
    try:
        parse_date_of_birth("2000-13-01")
    except ValueError as refused:
        assert "YYYY-MM-DD" in str(refused)
    else:
        raise AssertionError("an impossible month must be refused")


def test_the_public_view_never_includes_the_date_of_birth():
    view = verification_public_view(
        {
            "date_of_birth": date(1990, 1, 1),
            "verified_at": "2026-01-01T00:00:00+00:00",
        },
        minimum_years=18,
        on_date=TODAY,
    )
    assert view["verified"] is True
    assert "date_of_birth" not in view


@pytest.mark.asyncio
async def test_the_in_memory_repository_records_and_rechecks_age():
    repository = InMemoryAgeVerificationRepository()
    await repository.set_verification("user-1", date(2000, 1, 1), source="account_settings")
    assert await repository.is_verified("user-1", minimum_years=18, on_date=TODAY)
    assert not await repository.is_verified("user-1", minimum_years=40, on_date=TODAY)


def test_adult_only_avatars_stay_out_of_search_until_verified():
    adult = {
        "assistant_id": "adult-1",
        "name": "Adult Avatar",
        "metadata": {"user_id": "owner-1", "adult_only": True, "is_public": True},
    }
    ordinary = {
        "assistant_id": "ordinary-1",
        "name": "Guide",
        "metadata": {"user_id": "owner-2", "is_public": True},
    }
    hidden = filter_discoverable_avatars(
        [adult, ordinary],
        viewer_user_id="visitor-1",
        viewer_is_admin=False,
        viewer_age_verified=False,
    )
    assert [avatar["assistant_id"] for avatar in hidden] == ["ordinary-1"]

    shown = filter_discoverable_avatars(
        [adult, ordinary],
        viewer_user_id="visitor-1",
        viewer_is_admin=False,
        viewer_age_verified=True,
    )
    assert [avatar["assistant_id"] for avatar in shown] == ["adult-1", "ordinary-1"]


def test_the_owner_does_not_unlock_adult_only_search():
    adult = {
        "assistant_id": "adult-1",
        "metadata": {"user_id": "owner-1", "adult_only": True},
    }
    owner_listing = filter_discoverable_avatars(
        [adult],
        viewer_user_id="owner-1",
        viewer_is_admin=False,
        viewer_age_verified=False,
    )
    assert owner_listing == []


def test_the_administrator_sees_adult_only_avatars_in_search():
    adult = {
        "assistant_id": "adult-1",
        "metadata": {"user_id": "owner-1", "adult_only": True},
    }
    admin_listing = filter_discoverable_avatars(
        [adult],
        viewer_user_id="admin-1",
        viewer_is_admin=True,
        viewer_age_verified=False,
    )
    assert admin_listing == [adult]


def test_a_direct_lookup_does_not_hide_an_adult_only_avatar():
    adult = {
        "assistant_id": "adult-1",
        "metadata": {"user_id": "owner-1", "adult_only": True},
    }
    assert filter_discoverable_avatars(
        [adult],
        viewer_user_id=None,
        viewer_is_admin=False,
        viewer_age_verified=False,
        allow_direct_lookup=True,
    ) == [adult]
