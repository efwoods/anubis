# tests/unit_tests/test_usage_period_math.py

"""Unit tests for the pure usage-period and rate-limit arithmetic in
``src/anubis/utils/billing/metering.py``.

These functions decide WHEN an allotment resets and WHEN a token rate limit
clears — pure datetime arithmetic with no Stripe or database access, so every
boundary case is pinned here.
"""

from datetime import UTC, datetime, timedelta

from src.anubis.utils.billing.metering import (
    GLOBAL_USAGE_PERIOD_ANCHOR,
    resolve_usage_period_end,
    resolve_usage_period_start,
    token_rate_limit_retry_after_seconds,
)


class TestCalendarMonthPeriods:
    """``usage_period_days == 0`` — calendar-month semantics."""

    def test_without_anchor_period_starts_first_of_current_month(self) -> None:
        now = datetime(2026, 7, 15, 13, 45, tzinfo=UTC)
        assert resolve_usage_period_start(now, 0) == datetime(2026, 7, 1, tzinfo=UTC)

    def test_naive_now_is_interpreted_as_utc(self) -> None:
        now = datetime(2026, 7, 15, 13, 45)
        assert resolve_usage_period_start(now, 0) == datetime(2026, 7, 1, tzinfo=UTC)

    def test_anchor_day_of_month_governs_the_boundary(self) -> None:
        anchor = datetime(2026, 5, 10, 9, 30, tzinfo=UTC)
        now = datetime(2026, 7, 15, tzinfo=UTC)
        assert resolve_usage_period_start(now, 0, anchor) == datetime(
            2026, 7, 10, 9, 30, tzinfo=UTC
        )

    def test_before_this_months_boundary_uses_previous_month(self) -> None:
        anchor = datetime(2026, 5, 20, tzinfo=UTC)
        now = datetime(2026, 7, 15, tzinfo=UTC)
        assert resolve_usage_period_start(now, 0, anchor) == datetime(
            2026, 6, 20, tzinfo=UTC
        )

    def test_january_rolls_back_to_december_of_previous_year(self) -> None:
        anchor = datetime(2025, 11, 20, tzinfo=UTC)
        now = datetime(2026, 1, 10, tzinfo=UTC)
        assert resolve_usage_period_start(now, 0, anchor) == datetime(
            2025, 12, 20, tzinfo=UTC
        )

    def test_day_31_anchor_clamps_to_short_months(self) -> None:
        # An anchor on January 31 yields February 28 in a non-leap year —
        # the same clamping Stripe applies to billing_cycle_anchor.
        anchor = datetime(2026, 1, 31, tzinfo=UTC)
        now = datetime(2026, 3, 5, tzinfo=UTC)
        assert resolve_usage_period_start(now, 0, anchor) == datetime(
            2026, 2, 28, tzinfo=UTC
        )

    def test_period_never_starts_before_the_anchor_itself(self) -> None:
        # First period: the anchor was written mid-month, so the window
        # begins at the anchor, not at the boundary before the anchor.
        anchor = datetime(2026, 7, 10, tzinfo=UTC)
        now = datetime(2026, 7, 15, tzinfo=UTC)
        assert resolve_usage_period_start(now, 0, anchor) == anchor

    def test_future_anchor_is_returned_as_the_period_start(self) -> None:
        anchor = datetime(2026, 8, 1, tzinfo=UTC)
        now = datetime(2026, 7, 15, tzinfo=UTC)
        assert resolve_usage_period_start(now, 0, anchor) == anchor

    def test_period_end_is_next_monthly_boundary(self) -> None:
        period_start = datetime(2026, 7, 10, 9, 30, tzinfo=UTC)
        assert resolve_usage_period_end(period_start, 0) == datetime(
            2026, 8, 10, 9, 30, tzinfo=UTC
        )

    def test_period_end_rolls_december_into_january(self) -> None:
        period_start = datetime(2026, 12, 5, tzinfo=UTC)
        assert resolve_usage_period_end(period_start, 0) == datetime(
            2027, 1, 5, tzinfo=UTC
        )

    def test_period_end_clamps_day_31_start_to_short_next_month(self) -> None:
        period_start = datetime(2026, 1, 31, tzinfo=UTC)
        assert resolve_usage_period_end(period_start, 0) == datetime(
            2026, 2, 28, tzinfo=UTC
        )


class TestFixedLengthPeriods:
    """``usage_period_days > 0`` — fixed windows counted from an anchor."""

    def test_windows_count_from_the_global_anchor_without_a_personal_one(
        self,
    ) -> None:
        now = GLOBAL_USAGE_PERIOD_ANCHOR + timedelta(days=75)
        period_start = resolve_usage_period_start(now, 30)
        assert period_start == GLOBAL_USAGE_PERIOD_ANCHOR + timedelta(days=60)

    def test_personal_anchor_overrides_the_global_anchor(self) -> None:
        anchor = datetime(2026, 6, 1, tzinfo=UTC)
        now = anchor + timedelta(days=45)
        assert resolve_usage_period_start(now, 30, anchor) == anchor + timedelta(
            days=30
        )

    def test_now_exactly_on_a_boundary_starts_the_new_window(self) -> None:
        anchor = datetime(2026, 6, 1, tzinfo=UTC)
        now = anchor + timedelta(days=30)
        assert resolve_usage_period_start(now, 30, anchor) == now

    def test_fixed_window_end_adds_the_window_length(self) -> None:
        period_start = datetime(2026, 6, 1, tzinfo=UTC)
        assert resolve_usage_period_end(period_start, 30) == datetime(
            2026, 7, 1, tzinfo=UTC
        )


class TestTokenRateLimitRetryAfter:
    """Pure decision math behind the HTTP 429 token rate limit."""

    def test_zero_cap_disables_the_limit(self) -> None:
        assert token_rate_limit_retry_after_seconds(10_000, 0, 60, None) is None

    def test_zero_window_disables_the_limit(self) -> None:
        assert token_rate_limit_retry_after_seconds(10_000, 100, 0, None) is None

    def test_usage_under_the_cap_is_allowed(self) -> None:
        assert token_rate_limit_retry_after_seconds(99, 100, 60, None) is None

    def test_usage_at_the_cap_is_refused(self) -> None:
        assert token_rate_limit_retry_after_seconds(100, 100, 60, None) is not None

    def test_missing_oldest_timestamp_waits_the_full_window(self) -> None:
        assert token_rate_limit_retry_after_seconds(100, 100, 60, None) == 60

    def test_retry_after_is_time_until_the_oldest_row_ages_out(self) -> None:
        now = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)
        oldest = now - timedelta(seconds=45)
        retry_after = token_rate_limit_retry_after_seconds(
            100, 100, 60, oldest, now=now
        )
        # The oldest row expires in 15 seconds; +1 rounds the hint up.
        assert retry_after == 16

    def test_retry_after_is_clamped_to_at_least_one_second(self) -> None:
        now = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)
        oldest = now - timedelta(seconds=200)
        assert token_rate_limit_retry_after_seconds(100, 100, 60, oldest, now=now) == 1

    def test_retry_after_is_clamped_to_the_window_length(self) -> None:
        now = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)
        oldest = now + timedelta(seconds=500)  # skewed future timestamp
        assert (
            token_rate_limit_retry_after_seconds(100, 100, 60, oldest, now=now) == 60
        )
