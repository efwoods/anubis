"""Unit tests for post-verification enrollment and delete-and-re-signup adoption.

Exercises ``ensure_initial_subscription_after_verification`` against a recording
fake Stripe client: adoption of a still-live subscription after a delete-and-
re-signup (clearing the pending period-end cancellation written by
``delete_user``, retaining the original trial end, rebuilding the local
usage-period anchor from the real billing period), the free-tier enrollment for
customers whose trial was already used, and the first-ever PRO trial grant.
"""

from types import SimpleNamespace

import pytest

from src.security import auth as auth_module
from src.security.auth import ensure_initial_subscription_after_verification


class _DictResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _RecordingSubscriptionAPI:
    def __init__(self, listed_subscriptions, fail_on_modify: bool = False):
        self._listed_subscriptions = listed_subscriptions
        self._fail_on_modify = fail_on_modify
        self.list_calls = []
        self.modify_calls = []
        self.create_calls = []
        self.create_result = {"id": "sub_created", "status": "active"}

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return _DictResult({"data": self._listed_subscriptions})

    def modify(self, subscription_id, **kwargs):
        if self._fail_on_modify:
            raise RuntimeError("stripe unavailable")
        self.modify_calls.append((subscription_id, kwargs))

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _DictResult(dict(self.create_result))


class _RecordingSubscriptionScheduleAPI:
    def __init__(self):
        self.released_schedule_ids = []

    def release(self, schedule_id):
        self.released_schedule_ids.append(schedule_id)


class _RecordingCustomerAPI:
    def __init__(self, customer_payload):
        self._customer_payload = customer_payload
        self.modify_calls = []

    def retrieve(self, customer_id):
        return _DictResult(dict(self._customer_payload))

    def modify(self, customer_id, **kwargs):
        self.modify_calls.append((customer_id, kwargs))


class _FakeStripeClient:
    def __init__(
        self,
        listed_subscriptions=None,
        customer_metadata=None,
        fail_on_modify: bool = False,
    ):
        self.Subscription = _RecordingSubscriptionAPI(
            listed_subscriptions or [], fail_on_modify=fail_on_modify
        )
        self.SubscriptionSchedule = _RecordingSubscriptionScheduleAPI()
        self.Customer = _RecordingCustomerAPI(
            {"id": "cus_1", "metadata": customer_metadata or {}}
        )


class _FakeBillingConfig:
    def identifiers_for_tier(self, tier):
        return SimpleNamespace(
            base_price_id=f"price_base_{tier.value}",
            metered_price_ids={"messaging_tokens": f"price_metered_{tier.value}"},
        )


def _fake_request(stripe_client):
    app_state = SimpleNamespace(
        stripe=stripe_client, stripe_billing_config=_FakeBillingConfig()
    )
    return SimpleNamespace(app=SimpleNamespace(state=app_state))


def _verified_user():
    return {
        "email": "person@example.com",
        "user_id": "auth0|new-account",
        "app_metadata": {"stripe_customer_id": "cus_1"},
    }


def _pro_subscription(**overrides):
    subscription = {
        "id": "sub_live",
        "status": "active",
        "cancel_at_period_end": True,
        "items": {
            "data": [
                {
                    "current_period_start": 1_750_000_000,
                    "current_period_end": 1_752_600_000,
                    "price": {
                        "product": {"metadata": {"neural_nexus_tier": "pro"}}
                    },
                }
            ]
        },
    }
    subscription.update(overrides)
    return subscription


@pytest.fixture
def recorded_app_metadata_writes(monkeypatch):
    writes = []

    async def _record_write(request, auth0_user_id, fields):
        writes.append((auth0_user_id, fields))
        return True

    monkeypatch.setattr(
        auth_module, "update_user_app_metadata_fields", _record_write
    )
    return writes


@pytest.mark.asyncio
async def test_adoption_clears_cancel_at_period_end_and_releases_the_pending_schedule(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(
        listed_subscriptions=[_pro_subscription(schedule="sched_1")]
    )
    user = _verified_user()
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), user
    )
    assert stripe_client.SubscriptionSchedule.released_schedule_ids == ["sched_1"]
    assert stripe_client.Subscription.modify_calls == [
        ("sub_live", {"cancel_at_period_end": False})
    ]
    ((auth0_user_id, fields),) = recorded_app_metadata_writes
    assert auth0_user_id == "auth0|new-account"
    assert fields["subscription_status"]["subscription_id"] == "sub_live"
    assert fields["subscription_status"]["tier"] == "pro"
    assert fields["initial_subscription_provisioned"] is True
    # No new subscription was created: the running one was adopted, so the
    # already-paid period is never charged twice.
    assert stripe_client.Subscription.create_calls == []


@pytest.mark.asyncio
async def test_adoption_retains_the_original_trial_end_in_trial_context(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(
        listed_subscriptions=[
            _pro_subscription(status="trialing", trial_end=1_752_000_000)
        ]
    )
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), _verified_user()
    )
    ((_, fields),) = recorded_app_metadata_writes
    assert fields["trial_context"] == {"tier": "pro", "trial_end": 1_752_000_000}


@pytest.mark.asyncio
async def test_adoption_rebuilds_the_usage_period_anchor_from_the_period_start(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(listed_subscriptions=[_pro_subscription()])
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), _verified_user()
    )
    ((_, fields),) = recorded_app_metadata_writes
    assert fields["usage_period_anchor"].startswith("2025-06-15")


@pytest.mark.asyncio
async def test_adoption_survives_a_reactivation_failure_best_effort(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(
        listed_subscriptions=[_pro_subscription()], fail_on_modify=True
    )
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), _verified_user()
    )
    # The adoption still completes: the account is linked to the running
    # subscription even though the cancellation flag could not be cleared.
    ((_, fields),) = recorded_app_metadata_writes
    assert fields["subscription_status"]["subscription_id"] == "sub_live"


@pytest.mark.asyncio
async def test_adoption_skips_reactivation_when_nothing_is_pending(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(
        listed_subscriptions=[_pro_subscription(cancel_at_period_end=False)]
    )
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), _verified_user()
    )
    assert stripe_client.Subscription.modify_calls == []
    assert stripe_client.SubscriptionSchedule.released_schedule_ids == []


@pytest.mark.asyncio
async def test_used_trial_enrolls_free_and_never_creates_a_paid_subscription(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(
        listed_subscriptions=[
            {"id": "sub_old", "status": "canceled", "cancel_at_period_end": False}
        ],
        customer_metadata={"neural_nexus_trial_used": "true"},
    )
    stripe_client.Subscription.create_result = {
        "id": "sub_free",
        "status": "active",
    }
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), _verified_user()
    )
    (create_call,) = stripe_client.Subscription.create_calls
    assert create_call["items"][0]["price"] == "price_base_free"
    assert "trial_period_days" not in create_call
    ((_, fields),) = recorded_app_metadata_writes
    assert fields["subscription_status"]["tier"] == "free"
    assert fields["subscription_status"]["subscription_id"] == "sub_free"


@pytest.mark.asyncio
async def test_first_verified_account_still_receives_the_pro_trial(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(listed_subscriptions=[])
    stripe_client.Subscription.create_result = {
        "id": "sub_trial",
        "status": "trialing",
        "trial_end": 1_753_000_000,
    }
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), _verified_user()
    )
    (create_call,) = stripe_client.Subscription.create_calls
    assert create_call["trial_period_days"] == 30
    assert create_call["items"][0]["price"] == "price_base_pro"
    # The one-trial-ever flag is stamped at grant time on the customer record,
    # which survives account deletion.
    assert stripe_client.Customer.modify_calls == [
        ("cus_1", {"metadata": {"neural_nexus_trial_used": "true"}})
    ]
    ((_, fields),) = recorded_app_metadata_writes
    assert fields["subscription_status"]["tier"] == "pro"
    assert fields["trial_context"] == {"tier": "pro", "trial_end": 1_753_000_000}


@pytest.mark.asyncio
async def test_already_provisioned_account_is_left_alone(
    recorded_app_metadata_writes,
):
    stripe_client = _FakeStripeClient(listed_subscriptions=[_pro_subscription()])
    user = _verified_user()
    user["app_metadata"]["initial_subscription_provisioned"] = True
    await ensure_initial_subscription_after_verification(
        _fake_request(stripe_client), user
    )
    assert recorded_app_metadata_writes == []
    assert stripe_client.Subscription.list_calls == []
