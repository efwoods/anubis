"""Unit tests for the shared Stripe subscription-lifecycle helpers.

Covers the billing-period bounds reader (items-first with top-level fallback),
the pending-schedule release, the pending-cancellation clearing used by both
the reactivation endpoint and delete-and-re-signup adoption, and the
one-trial-ever Checkout guard.
"""

import pytest

from src.anubis.utils.billing.subscription_lifecycle import (
    clear_pending_cancellation,
    release_pending_subscription_schedule,
    resolve_checkout_trial_period_days,
    subscription_period_bounds,
)


class _RecordingSubscriptionAPI:
    def __init__(self, fail_on_modify: bool = False):
        self.modify_calls = []
        self._fail_on_modify = fail_on_modify

    def modify(self, subscription_id, **kwargs):
        if self._fail_on_modify:
            raise RuntimeError("stripe unavailable")
        self.modify_calls.append((subscription_id, kwargs))


class _RecordingSubscriptionScheduleAPI:
    def __init__(self):
        self.released_schedule_ids = []

    def release(self, schedule_id):
        self.released_schedule_ids.append(schedule_id)


class _DictResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _RecordingCustomerAPI:
    def __init__(self, customer_payload=None, fail_on_retrieve: bool = False):
        self.retrieved_customer_ids = []
        self._customer_payload = customer_payload or {}
        self._fail_on_retrieve = fail_on_retrieve

    def retrieve(self, customer_id):
        self.retrieved_customer_ids.append(customer_id)
        if self._fail_on_retrieve:
            raise RuntimeError("stripe unavailable")
        return _DictResult(self._customer_payload)


class _FakeStripeClient:
    def __init__(
        self,
        customer_payload=None,
        fail_on_retrieve: bool = False,
        fail_on_modify: bool = False,
    ):
        self.Subscription = _RecordingSubscriptionAPI(fail_on_modify=fail_on_modify)
        self.SubscriptionSchedule = _RecordingSubscriptionScheduleAPI()
        self.Customer = _RecordingCustomerAPI(
            customer_payload=customer_payload, fail_on_retrieve=fail_on_retrieve
        )


# ---------------------------------------------------------------------------
# subscription_period_bounds
# ---------------------------------------------------------------------------


def test_subscription_period_bounds_reads_items_first_then_top_level():
    items_first = {
        "items": {
            "data": [{"current_period_start": 100, "current_period_end": 200}]
        },
        "current_period_start": 1,
        "current_period_end": 2,
    }
    assert subscription_period_bounds(items_first) == (100, 200)

    top_level_only = {"current_period_start": 300, "current_period_end": 400}
    assert subscription_period_bounds(top_level_only) == (300, 400)

    assert subscription_period_bounds({}) == (None, None)


# ---------------------------------------------------------------------------
# release_pending_subscription_schedule / clear_pending_cancellation
# ---------------------------------------------------------------------------


def test_clear_pending_cancellation_releases_the_schedule_then_clears_the_flag():
    stripe_client = _FakeStripeClient()
    subscription = {
        "id": "sub_1",
        "cancel_at_period_end": True,
        "schedule": "sched_1",
    }
    clear_pending_cancellation(stripe_client, subscription)
    assert stripe_client.SubscriptionSchedule.released_schedule_ids == ["sched_1"]
    assert stripe_client.Subscription.modify_calls == [
        ("sub_1", {"cancel_at_period_end": False})
    ]


def test_clear_pending_cancellation_without_a_schedule_only_clears_the_flag():
    stripe_client = _FakeStripeClient()
    subscription = {"id": "sub_2", "cancel_at_period_end": True}
    clear_pending_cancellation(stripe_client, subscription)
    assert stripe_client.SubscriptionSchedule.released_schedule_ids == []
    assert stripe_client.Subscription.modify_calls == [
        ("sub_2", {"cancel_at_period_end": False})
    ]


def test_release_pending_subscription_schedule_accepts_an_expanded_schedule_object():
    stripe_client = _FakeStripeClient()
    subscription = {"id": "sub_3", "schedule": {"id": "sched_3"}}
    release_pending_subscription_schedule(stripe_client, subscription)
    assert stripe_client.SubscriptionSchedule.released_schedule_ids == ["sched_3"]


def test_clear_pending_cancellation_propagates_stripe_errors_to_the_caller():
    stripe_client = _FakeStripeClient(fail_on_modify=True)
    subscription = {"id": "sub_4", "cancel_at_period_end": True}
    with pytest.raises(RuntimeError):
        clear_pending_cancellation(stripe_client, subscription)


# ---------------------------------------------------------------------------
# resolve_checkout_trial_period_days (one free trial per customer, ever)
# ---------------------------------------------------------------------------


def test_checkout_trial_kept_for_a_customer_without_trial_history():
    stripe_client = _FakeStripeClient(customer_payload={"metadata": {}})
    assert resolve_checkout_trial_period_days(stripe_client, "cus_1", 30) == 30
    assert stripe_client.Customer.retrieved_customer_ids == ["cus_1"]


def test_checkout_trial_withheld_when_the_customer_already_used_a_trial():
    stripe_client = _FakeStripeClient(
        customer_payload={"metadata": {"neural_nexus_trial_used": "true"}}
    )
    assert resolve_checkout_trial_period_days(stripe_client, "cus_1", 30) == 0


def test_checkout_trial_withheld_when_the_customer_record_is_unreadable():
    stripe_client = _FakeStripeClient(fail_on_retrieve=True)
    assert resolve_checkout_trial_period_days(stripe_client, "cus_1", 30) == 0


def test_checkout_trial_kept_when_there_is_no_customer_history_to_consult():
    stripe_client = _FakeStripeClient()
    assert resolve_checkout_trial_period_days(stripe_client, None, 30) == 30
    assert stripe_client.Customer.retrieved_customer_ids == []


def test_checkout_trial_zero_for_a_tier_without_a_trial():
    stripe_client = _FakeStripeClient()
    assert resolve_checkout_trial_period_days(stripe_client, "cus_1", 0) == 0
    assert stripe_client.Customer.retrieved_customer_ids == []
