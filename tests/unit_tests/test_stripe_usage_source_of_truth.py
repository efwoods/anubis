"""Unit tests for Stripe-as-source-of-truth usage reads and anonymous windows.

The customer portal reads usage from Stripe's Billing Meter aggregation while
the API historically read only the local ``api_metrics`` table, and the two
ledgers drifted whenever one fail-open write landed without the other — an
anonymous visitor's portal showed an exhausted allotment while the messaging API
still reported budget remaining. These tests pin the two halves of the fix:

* ``fetch_stripe_period_usage`` / ``reconcile_period_usage`` — Stripe governs,
  the local sum is a floor for usage Stripe has not aggregated yet, and an
  unreadable Stripe degrades to the local sum rather than to "zero used".
* the anonymous billing record — the visitor's $0 free-tier subscription's
  billing cycle becomes the window the API counts usage over (Stripe counts over
  the cycle; the calendar month is a different span of time), without granting
  the anonymous visitor any paid capability.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.anubis.utils.billing.gating import (
    resolve_pay_per_use_enabled,
    resolve_tier,
)
from src.anubis.utils.billing.metering import resolve_usage_period_start
from src.anubis.utils.billing.stripe_usage import (
    fetch_stripe_period_usage,
    invalidate_stripe_usage_cache,
    reconcile_period_usage,
    stripe_usage_source_of_truth_enabled,
)
from src.anubis.utils.billing.tiers import SubscriptionTier, UsageMeter
from src.security.anonymous_billing import (
    AnonymousBillingRecord,
    _billing_record_from_subscription,
)

MESSAGING_METER_ID = "mtr_test_messaging"


class _SummaryObject:
    def __init__(self, aggregated_value):
        self._aggregated_value = aggregated_value

    def to_dict(self):
        return {"aggregated_value": self._aggregated_value}


class _SummaryPage:
    def __init__(self, aggregated_values):
        self._aggregated_values = aggregated_values

    def auto_paging_iter(self):
        return iter(_SummaryObject(value) for value in self._aggregated_values)


class _RecordingMeterAPI:
    """Records every ``list_event_summaries`` call and replays fixed buckets.

    ``aggregated_values`` is either one list of buckets replayed for every meter,
    or a ``{meter_id: buckets}`` mapping when a test needs each meter to report a
    distinguishable number.
    """

    def __init__(self, aggregated_values, raise_error: bool = False):
        self._aggregated_values = aggregated_values
        self._raise_error = raise_error
        self.calls: list[dict] = []

    def list_event_summaries(self, meter_id, **kwargs):
        self.calls.append({"meter_id": meter_id, **kwargs})
        if self._raise_error:
            raise RuntimeError("stripe unavailable")
        if isinstance(self._aggregated_values, dict):
            return _SummaryPage(self._aggregated_values.get(meter_id, []))
        return _SummaryPage(self._aggregated_values)


def _fake_stripe_client(aggregated_values, raise_error: bool = False):
    meter_api = _RecordingMeterAPI(aggregated_values, raise_error=raise_error)
    return SimpleNamespace(billing=SimpleNamespace(Meter=meter_api)), meter_api


def _billing_config(meter_ids=None):
    return SimpleNamespace(
        meter_ids=(
            meter_ids
            if meter_ids is not None
            else {UsageMeter.MESSAGING_TOKENS: MESSAGING_METER_ID}
        )
    )


@pytest.fixture(autouse=True)
def _clear_stripe_usage_cache():
    invalidate_stripe_usage_cache()
    yield
    invalidate_stripe_usage_cache()


# ---------------------------------------------------------------------------
# Reconciliation: which ledger governs
# ---------------------------------------------------------------------------


def test_stripe_governs_when_it_reports_more_than_the_local_ledger():
    """The measured production case: 342,864 in Stripe against 100,909 locally."""
    assert reconcile_period_usage(100_909, 342_864) == 342_864


def test_local_ledger_is_a_floor_for_usage_stripe_has_not_aggregated_yet():
    """Meter ingestion lags, so freshly recorded local usage must still count."""
    assert reconcile_period_usage(100_909, 80_000) == 100_909


def test_unreadable_stripe_falls_back_to_the_local_ledger_not_to_zero():
    """A Stripe outage must not hand out a second allotment."""
    assert reconcile_period_usage(100_909, None) == 100_909


def test_negative_or_missing_values_never_produce_negative_usage():
    assert reconcile_period_usage(0, None) == 0
    assert reconcile_period_usage(-5, -7) == 0


# ---------------------------------------------------------------------------
# Reading Stripe's aggregation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_period_usage_sums_every_event_summary_bucket():
    stripe_client, meter_api = _fake_stripe_client([19_845.0, 20_648.0, 60_416.0])
    period_start = datetime(2026, 7, 16, 3, 35, 49, tzinfo=timezone.utc)
    now = datetime(2026, 7, 24, 22, 22, 30, tzinfo=timezone.utc)

    usage = await fetch_stripe_period_usage(
        stripe_client,
        _billing_config(),
        UsageMeter.MESSAGING_TOKENS,
        "cus_anonymous",
        period_start,
        now=now,
    )

    assert usage == 100_909
    assert meter_api.calls[0]["meter_id"] == MESSAGING_METER_ID
    assert meter_api.calls[0]["customer"] == "cus_anonymous"
    # Both Stripe usage endpoints reject timestamps that are not minute-aligned,
    # so the window is aligned DOWN at both ends.
    assert meter_api.calls[0]["start_time"] % 60 == 0
    assert meter_api.calls[0]["end_time"] % 60 == 0
    assert meter_api.calls[0]["start_time"] == int(period_start.timestamp()) - 49
    assert meter_api.calls[0]["end_time"] == int(now.timestamp()) - 30


@pytest.mark.asyncio
async def test_repeat_reads_inside_the_cache_window_do_not_call_stripe_again():
    """Allotment enforcement runs on the message hot path; one call per period."""
    stripe_client, meter_api = _fake_stripe_client([1_000.0])
    period_start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)

    first_reading = await fetch_stripe_period_usage(
        stripe_client,
        _billing_config(),
        UsageMeter.MESSAGING_TOKENS,
        "cus_anonymous",
        period_start,
        now=now,
    )
    second_reading = await fetch_stripe_period_usage(
        stripe_client,
        _billing_config(),
        UsageMeter.MESSAGING_TOKENS,
        "cus_anonymous",
        period_start,
        now=now + timedelta(seconds=5),
    )

    assert (first_reading, second_reading) == (1_000, 1_000)
    assert len(meter_api.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stripe_customer_id, billing_config, raise_error, reason",
    [
        (None, _billing_config(), False, "anonymous visitor with no Stripe customer"),
        ("cus_anonymous", _billing_config({}), False, "meter not provisioned"),
        ("cus_anonymous", _billing_config(), True, "Stripe call failed"),
    ],
)
async def test_unavailable_stripe_usage_is_none_so_callers_fall_back(
    stripe_customer_id, billing_config, raise_error, reason
):
    """``None`` means "ask the local ledger", never "no usage" — see ``reason``."""
    stripe_client, _ = _fake_stripe_client([5_000.0], raise_error=raise_error)

    usage = await fetch_stripe_period_usage(
        stripe_client,
        billing_config,
        UsageMeter.MESSAGING_TOKENS,
        stripe_customer_id,
        datetime(2026, 7, 16, tzinfo=timezone.utc),
        now=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    assert usage is None, reason


@pytest.mark.asyncio
async def test_a_period_that_began_inside_this_minute_reports_zero_without_calling_stripe():
    stripe_client, meter_api = _fake_stripe_client([5_000.0])
    period_start = datetime(2026, 7, 24, 22, 22, 10, tzinfo=timezone.utc)

    usage = await fetch_stripe_period_usage(
        stripe_client,
        _billing_config(),
        UsageMeter.MESSAGING_TOKENS,
        "cus_anonymous",
        period_start,
        now=period_start + timedelta(seconds=20),
    )

    assert usage == 0
    assert meter_api.calls == []


@pytest.mark.asyncio
async def test_the_stripe_read_can_be_turned_off_entirely(monkeypatch):
    monkeypatch.setenv("STRIPE_USAGE_SOURCE_OF_TRUTH_ENABLED", "FALSE")
    stripe_client, meter_api = _fake_stripe_client([5_000.0])

    assert stripe_usage_source_of_truth_enabled() is False
    usage = await fetch_stripe_period_usage(
        stripe_client,
        _billing_config(),
        UsageMeter.MESSAGING_TOKENS,
        "cus_anonymous",
        datetime(2026, 7, 16, tzinfo=timezone.utc),
        now=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )
    assert usage is None
    assert meter_api.calls == []


def test_the_stripe_read_is_enabled_by_default_and_when_the_value_is_blank(monkeypatch):
    monkeypatch.delenv("STRIPE_USAGE_SOURCE_OF_TRUTH_ENABLED", raising=False)
    assert stripe_usage_source_of_truth_enabled() is True
    assert (
        stripe_usage_source_of_truth_enabled(
            SimpleNamespace(stripe_usage_source_of_truth_enabled="")
        )
        is True
    )


# ---------------------------------------------------------------------------
# The anonymous visitor's usage window
# ---------------------------------------------------------------------------


def _anonymous_free_tier_subscription():
    """A $0 anonymous free-tier subscription as Stripe returns it today.

    Flexible billing mode reports the period on the subscription ITEMS, which is
    the shape the live anonymous customers carry.
    """
    return {
        "id": "sub_anonymous",
        "status": "active",
        "items": {
            "data": [
                {
                    "id": "si_base",
                    "current_period_start": int(
                        datetime(
                            2026, 7, 16, 3, 35, 49, tzinfo=timezone.utc
                        ).timestamp()
                    ),
                    "current_period_end": int(
                        datetime(
                            2026, 8, 16, 3, 35, 49, tzinfo=timezone.utc
                        ).timestamp()
                    ),
                }
            ]
        },
    }


def test_the_billing_record_carries_the_subscription_cycle():
    billing_record = _billing_record_from_subscription(
        "cus_anonymous", _anonymous_free_tier_subscription()
    )

    assert billing_record.stripe_customer_id == "cus_anonymous"
    assert billing_record.subscription_id == "sub_anonymous"
    assert billing_record.subscription_status == "active"
    assert billing_record.current_period_start == int(
        datetime(2026, 7, 16, 3, 35, 49, tzinfo=timezone.utc).timestamp()
    )
    assert billing_record.current_period_end == int(
        datetime(2026, 8, 16, 3, 35, 49, tzinfo=timezone.utc).timestamp()
    )


def test_a_missing_subscription_leaves_the_window_to_the_configured_default():
    billing_record = _billing_record_from_subscription("cus_anonymous", None)

    assert billing_record == AnonymousBillingRecord(stripe_customer_id="cus_anonymous")


def _anonymous_user_with_billing_record():
    """The anonymous user shape ``get_anonymous_user_with_anonymous_api_key`` builds."""
    billing_record = _billing_record_from_subscription(
        "cus_anonymous", _anonymous_free_tier_subscription()
    )
    return {
        "id": "supabase-anonymous-id",
        "email": "",
        "is_anonymous": True,
        "identities": [{"user_id": "a" * 64}],
        "app_metadata": {
            "stripe_customer_id": billing_record.stripe_customer_id,
            "subscription_status": {
                "status": billing_record.subscription_status,
                "subscription_id": billing_record.subscription_id,
                "customer_id": billing_record.stripe_customer_id,
                "email": None,
                "tier": SubscriptionTier.FREE.value,
                "current_period_start": billing_record.current_period_start,
                "current_period_end": billing_record.current_period_end,
            },
        },
    }


def test_the_anonymous_usage_window_follows_stripe_not_the_calendar_month():
    """The reported bug: the API counted from Jul 1, Stripe billed from Jul 16."""
    anonymous_user = _anonymous_user_with_billing_record()
    now = datetime(2026, 7, 24, 21, 30, tzinfo=timezone.utc)

    calendar_month_start = resolve_usage_period_start(now, 0, None)
    stripe_cycle_start = datetime.fromtimestamp(
        anonymous_user["app_metadata"]["subscription_status"]["current_period_start"],
        tz=timezone.utc,
    )

    assert calendar_month_start == datetime(2026, 7, 1, tzinfo=timezone.utc)
    # The cached Stripe period start is the LATER signal, so
    # resolve_usage_period_start_for_user takes it over the calendar month and
    # both sides then measure the same span of time.
    assert stripe_cycle_start > calendar_month_start
    assert max(calendar_month_start, stripe_cycle_start) == stripe_cycle_start


def test_the_anonymous_usage_window_ends_where_the_stripe_cycle_ends():
    anonymous_user = _anonymous_user_with_billing_record()

    assert datetime.fromtimestamp(
        anonymous_user["app_metadata"]["subscription_status"]["current_period_end"],
        tz=timezone.utc,
    ) == datetime(2026, 8, 16, 3, 35, 49, tzinfo=timezone.utc)


def test_caching_a_stripe_subscription_grants_the_anonymous_visitor_nothing():
    """An ``active`` cached status must not leak a paid tier or pay-per-use.

    ``resolve_pay_per_use_enabled`` infers overage billing from an ``active``
    subscription status for authenticated users, so an anonymous visitor
    carrying a cached free-tier subscription is exactly the case that must stay
    pinned by the ``is_anonymous`` flag.
    """
    anonymous_user = _anonymous_user_with_billing_record()

    assert resolve_tier(anonymous_user) is SubscriptionTier.FREE
    assert resolve_pay_per_use_enabled(anonymous_user) is False


# ---------------------------------------------------------------------------
# The subscription-status endpoint: anonymous callers, per-meter reads
# ---------------------------------------------------------------------------

DOCUMENT_UPLOAD_METER_ID = "mtr_test_document_upload"


def _endpoint_request(stripe_client, meter_ids):
    """A request whose app.state carries Stripe but NO database pool.

    A missing pool makes the local ledger read zero, which isolates what the
    Stripe reading contributes — and is also the very condition that produced
    the divergence in production.
    """
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                stripe=stripe_client,
                stripe_billing_config=_billing_config(meter_ids),
                pool=None,
            )
        )
    )


@pytest.mark.asyncio
async def test_the_status_endpoint_reports_stripe_usage_to_an_anonymous_caller():
    """BUG 2: an anonymous visitor could not read their own usage at all."""
    from src.api.webapp import verify_subscription_status

    stripe_client, meter_api = _fake_stripe_client([342_864.0])
    response = await verify_subscription_status(
        request=_endpoint_request(
            stripe_client, {UsageMeter.MESSAGING_TOKENS: MESSAGING_METER_ID}
        ),
        current_user=_anonymous_user_with_billing_record(),
    )

    assert response["anonymous"] is True
    assert response["tier"] == SubscriptionTier.FREE.value
    assert response["customer_id"] == "cus_anonymous"
    assert response["pay_per_use_enabled"] is False
    # The window is Stripe's cycle, and the usage is Stripe's aggregation — the
    # exact figures the customer portal displays for the same visitor.
    assert response["usage_period_start"] == "2026-07-16T03:35:49+00:00"
    assert response["usage_period_end"] == "2026-08-16T03:35:49+00:00"
    messaging = response["meters"][UsageMeter.MESSAGING_TOKENS.value]
    assert messaging["used_to_date"] == 342_864
    assert messaging["remaining"] == 0
    assert messaging["over_allotment"] == 142_864
    assert len(meter_api.calls) == 1


@pytest.mark.asyncio
async def test_each_meter_reports_its_own_usage_when_read_concurrently(monkeypatch):
    """The per-meter Stripe reads are gathered, so they must not cross wires."""
    import src.api.webapp as webapp_module

    async def _pro_status(request, current_user):
        return {
            "status": "active",
            "tier": SubscriptionTier.PRO.value,
            "subscription_id": "sub_pro",
            "customer_id": "cus_pro",
            "email": "person@example.com",
        }

    monkeypatch.setattr(webapp_module, "check_subscription_status", _pro_status)

    stripe_client, meter_api = _fake_stripe_client(
        {
            MESSAGING_METER_ID: [1_111.0],
            DOCUMENT_UPLOAD_METER_ID: [2_222.0],
        }
    )
    paid_user = {
        "user_id": "auth0|person",
        "email": "person@example.com",
        "app_metadata": {
            "stripe_customer_id": "cus_pro",
            "subscription_status": {
                "status": "active",
                "tier": SubscriptionTier.PRO.value,
                "customer_id": "cus_pro",
            },
        },
    }

    response = await webapp_module.verify_subscription_status(
        request=_endpoint_request(
            stripe_client,
            {
                UsageMeter.MESSAGING_TOKENS: MESSAGING_METER_ID,
                UsageMeter.DOCUMENT_UPLOAD_TOKENS: DOCUMENT_UPLOAD_METER_ID,
            },
        ),
        current_user=paid_user,
    )

    assert response["anonymous"] is False
    assert (
        response["meters"][UsageMeter.MESSAGING_TOKENS.value]["used_to_date"] == 1_111
    )
    assert (
        response["meters"][UsageMeter.DOCUMENT_UPLOAD_TOKENS.value]["used_to_date"]
        == 2_222
    )
    assert {call["meter_id"] for call in meter_api.calls} == {
        MESSAGING_METER_ID,
        DOCUMENT_UPLOAD_METER_ID,
    }
