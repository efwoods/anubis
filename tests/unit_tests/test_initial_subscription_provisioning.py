"""Unit tests for post-verification enrollment and delete-and-re-signup adoption.

Exercises ``ensure_initial_subscription_after_verification`` against a recording
fake Stripe client: adoption of a still-live subscription after a delete-and-
re-signup (clearing the pending period-end cancellation written by
``delete_user``, retaining the original trial end, rebuilding the local
usage-period anchor from the real billing period), the free-tier enrollment for
customers whose trial was already used, and the first-ever PRO trial grant.

Also exercises the ``/login`` route as an enrollment trigger. Sign-in is the
only trigger the customer portal reaches — it authenticates with email +
password and never holds an API key — so these tests are what keep a portal
signup from silently ending up with a Stripe customer and no subscription.

The last group asserts that an account's billing identity stays separate from an
anonymous visitor's: anonymous usage and account usage are separate allotments
reported separately, which holds only while the two never share one Stripe
customer.
"""

from types import SimpleNamespace

import pytest

from src.security import auth as auth_module
from src.security.auth import (
    _provision_stripe_customer_and_default_tier,
    ensure_initial_subscription_after_verification,
    login,
)


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
    def __init__(self, customer_payload, listed_customers=None):
        self._customer_payload = customer_payload
        self._listed_customers = listed_customers or []
        self.retrieve_calls = []
        self.modify_calls = []
        self.list_calls = []
        self.create_calls = []

    def retrieve(self, customer_id):
        self.retrieve_calls.append(customer_id)
        return _DictResult(dict(self._customer_payload))

    def modify(self, customer_id, **kwargs):
        self.modify_calls.append((customer_id, kwargs))

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return _DictResult({"data": list(self._listed_customers)})

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "cus_new_account"}


class _FakeStripeClient:
    def __init__(
        self,
        listed_subscriptions=None,
        customer_metadata=None,
        fail_on_modify: bool = False,
        listed_customers=None,
    ):
        self.Subscription = _RecordingSubscriptionAPI(
            listed_subscriptions or [], fail_on_modify=fail_on_modify
        )
        self.SubscriptionSchedule = _RecordingSubscriptionScheduleAPI()
        self.Customer = _RecordingCustomerAPI(
            {"id": "cus_1", "metadata": customer_metadata or {}},
            listed_customers=listed_customers,
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


# ── Sign-in as an enrollment trigger ────────────────────────────────────────
#
# The customer portal authenticates with email + password and never holds an
# API key, so ``get_user_with_api_key`` — the other enrollment trigger — never
# runs for a portal-only account. Sign-in is the only trigger it reaches.


class _FakeLoginResponse:
    def __init__(self, payload, status_code: int = 200):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def login_environment(monkeypatch):
    """Wire ``/login``'s Auth0 collaborators to fakes and expose the Auth0 record.

    Only the identity-provider calls are faked. The enrollment itself runs for
    real against the fake Stripe client, so these tests cover the whole path
    from sign-in through to the Stripe subscription.
    """
    auth0_record = {
        "user_id": "auth0|new-account",
        "email": "person@example.com",
        "email_verified": True,
        "app_metadata": {"stripe_customer_id": "cus_1"},
    }
    environment = SimpleNamespace(
        auth0_record=auth0_record,
        login_status_calls=[],
        get_user_calls=[],
        get_user_error=None,
    )

    async def _fake_login_user(email, password, request):
        return _FakeLoginResponse({"id_token": "fake.id.token"})

    async def _fake_set_login_status(user_id, logged_in, request):
        environment.login_status_calls.append((user_id, logged_in))

    async def _fake_get_user(user_id, request):
        environment.get_user_calls.append(user_id)
        if environment.get_user_error is not None:
            raise environment.get_user_error
        return environment.auth0_record

    monkeypatch.setattr(auth_module, "login_user", _fake_login_user)
    monkeypatch.setattr(auth_module, "set_login_status", _fake_set_login_status)
    monkeypatch.setattr(auth_module, "get_user", _fake_get_user)
    monkeypatch.setattr(
        auth_module.jwt,
        "get_unverified_claims",
        lambda id_token: {"sub": "auth0|new-account"},
    )
    return environment


async def _sign_in(stripe_client):
    return await login(
        auth_module.LoginRequest(email="person@example.com", password="Secret!1"),
        _fake_request(stripe_client),
    )


@pytest.mark.asyncio
async def test_signing_in_enrolls_a_verified_account_into_the_pro_trial(
    login_environment, recorded_app_metadata_writes
):
    stripe_client = _FakeStripeClient(listed_subscriptions=[])
    stripe_client.Subscription.create_result = {
        "id": "sub_trial",
        "status": "trialing",
        "trial_end": 1_753_000_000,
    }

    data = await _sign_in(stripe_client)

    assert data == {"id_token": "fake.id.token"}
    (create_call,) = stripe_client.Subscription.create_calls
    assert create_call["trial_period_days"] == 30
    assert create_call["items"][0]["price"] == "price_base_pro"
    assert create_call["trial_settings"] == {
        "end_behavior": {"missing_payment_method": "cancel"}
    }
    ((_, fields),) = recorded_app_metadata_writes
    assert fields["subscription_status"]["tier"] == "pro"
    assert fields["initial_subscription_provisioned"] is True
    assert fields["trial_context"] == {"tier": "pro", "trial_end": 1_753_000_000}
    assert fields["usage_period_anchor"]


@pytest.mark.asyncio
async def test_signing_in_unverified_creates_no_subscription_but_still_signs_in(
    login_environment, recorded_app_metadata_writes
):
    login_environment.auth0_record["email_verified"] = False
    stripe_client = _FakeStripeClient(listed_subscriptions=[])

    data = await _sign_in(stripe_client)

    # The sign-in itself is unaffected — only enrollment waits for verification.
    assert data == {"id_token": "fake.id.token"}
    assert login_environment.login_status_calls == [("auth0|new-account", True)]
    assert stripe_client.Subscription.create_calls == []
    assert stripe_client.Subscription.list_calls == []
    assert recorded_app_metadata_writes == []


@pytest.mark.asyncio
async def test_signing_in_again_after_enrollment_makes_no_stripe_call(
    login_environment, recorded_app_metadata_writes
):
    login_environment.auth0_record["app_metadata"][
        "initial_subscription_provisioned"
    ] = True
    stripe_client = _FakeStripeClient(listed_subscriptions=[_pro_subscription()])

    await _sign_in(stripe_client)

    # The marker short-circuits before any Stripe call, so the multi-call
    # enrollment cost is paid exactly once per account rather than every login.
    assert stripe_client.Customer.retrieve_calls == []
    assert stripe_client.Subscription.list_calls == []
    assert stripe_client.Subscription.create_calls == []
    assert recorded_app_metadata_writes == []


@pytest.mark.asyncio
async def test_a_stripe_failure_during_enrollment_never_fails_the_sign_in(
    login_environment, recorded_app_metadata_writes
):
    stripe_client = _FakeStripeClient(listed_subscriptions=[])

    def _raise(**kwargs):
        raise RuntimeError("stripe unavailable")

    stripe_client.Subscription.list = _raise

    data = await _sign_in(stripe_client)

    assert data == {"id_token": "fake.id.token"}
    # No marker was written, so the next sign-in retries the enrollment.
    assert recorded_app_metadata_writes == []


@pytest.mark.asyncio
async def test_an_auth0_lookup_failure_never_fails_the_sign_in(
    login_environment, recorded_app_metadata_writes
):
    login_environment.get_user_error = RuntimeError("auth0 unreachable")
    stripe_client = _FakeStripeClient(listed_subscriptions=[])

    data = await _sign_in(stripe_client)

    assert data == {"id_token": "fake.id.token"}
    assert stripe_client.Subscription.create_calls == []


# ── Anonymous and account billing identities stay separate ──────────────────
#
# Anonymous usage and account usage are separate allotments reported
# separately. That holds only while an anonymous visitor's Stripe customer and
# an account's Stripe customer never become the same record.


@pytest.mark.asyncio
async def test_an_anonymous_subscription_never_denies_an_account_its_trial(
    recorded_app_metadata_writes,
):
    # A $0 subscription that exists only to meter an anonymous visitor is not
    # subscription history. If it were counted, the account would be enrolled
    # straight into the free tier and silently lose its pro trial.
    stripe_client = _FakeStripeClient(
        listed_subscriptions=[
            {
                "id": "sub_anonymous",
                "status": "active",
                "cancel_at_period_end": False,
                "metadata": {
                    "anonymous_hashed_ip": "245c0ffc0f6a0215471542b9add1fa53"
                    "31647f4af18c431f039c66dbee92732e",
                    "neural_nexus_tier": "free",
                },
            }
        ]
    )
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
    ((_, fields),) = recorded_app_metadata_writes
    assert fields["subscription_status"]["tier"] == "pro"


@pytest.mark.asyncio
async def test_a_signup_never_reuses_a_customer_that_has_no_matching_email(
    monkeypatch,
):
    # An anonymous visitor's customer carries a hashed IP and no email, so a
    # signup from that same address must mint its own customer rather than
    # attach to the anonymous one and merge the two usage ledgers.
    stripe_client = _FakeStripeClient(listed_customers=[])
    patched_app_metadata = {}

    async def _fake_mgmt_headers(request):
        return {}

    async def _fake_patch(method, url, headers, json):
        patched_app_metadata.update(json["app_metadata"])
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(auth_module, "_mgmt_headers", _fake_mgmt_headers)
    monkeypatch.setattr(auth_module, "retry_async_httpx_request", _fake_patch)

    customer_id = await _provision_stripe_customer_and_default_tier(
        request=_fake_request(stripe_client),
        user_id="auth0|new-account",
        email="person@example.com",
    )

    # The lookup is by email only — a hashed-IP customer can never match it.
    assert stripe_client.Customer.list_calls == [
        {"email": "person@example.com", "limit": 1}
    ]
    assert stripe_client.Customer.modify_calls == []
    assert customer_id == "cus_new_account"
    (create_call,) = stripe_client.Customer.create_calls
    assert create_call["metadata"] == {"auth0_user_id": "auth0|new-account"}
    assert "anonymous_hashed_ip" not in create_call["metadata"]
    assert patched_app_metadata["stripe_customer_id"] == "cus_new_account"
