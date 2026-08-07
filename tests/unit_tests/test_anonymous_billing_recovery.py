"""Unit tests for recovering an anonymous visitor whose Stripe customer is gone.

The ``anonymous_billing_customers`` Postgres row outlives the Stripe account it
points into. When the recorded customer is deleted — test data wiped, or removed
by hand — trusting the row is silently fatal: every meter event is rejected with
``No such customer`` so nothing reaches Stripe, every usage read falls back to
the local ``api_metrics`` table, and the customer portal (which finds the visitor
by searching Stripe for ``metadata.anonymous_hashed_ip``) finds nothing and
reports no anonymous usage at all.

These tests pin the recovery: a dead recorded customer is discarded and replaced,
a live one is reused, and a transient Stripe failure never orphans a good
customer into a duplicate.
"""

from types import SimpleNamespace

import pytest

from src.security import anonymous_billing as anonymous_billing_module
from src.security.anonymous_billing import (
    invalidate_anonymous_billing_cache,
    resolve_or_create_anonymous_billing_record,
)

HASHED_IP = "245c0ffc0f6a0215471542b9add1fa5331647f4af18c431f039c66dbee92732e"


class _DictResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _StripeResourceMissing(Exception):
    """Stands in for stripe.error.InvalidRequestError on a deleted customer."""

    code = "resource_missing"


class _RecordingCustomerAPI:
    def __init__(self, retrieve_result, searched_customer_id=None):
        self._retrieve_result = retrieve_result
        self._searched_customer_id = searched_customer_id
        self.retrieve_calls = []
        self.search_calls = []
        self.create_calls = []

    def retrieve(self, customer_id):
        self.retrieve_calls.append(customer_id)
        if isinstance(self._retrieve_result, Exception):
            raise self._retrieve_result
        return _DictResult(dict(self._retrieve_result))

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        data = (
            [{"id": self._searched_customer_id}] if self._searched_customer_id else []
        )
        return _DictResult({"data": data})

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _DictResult({"id": "cus_replacement"})


class _RecordingSubscriptionAPI:
    def __init__(self):
        self.list_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return _DictResult({"data": []})


class _FakeStripeClient:
    def __init__(self, retrieve_result, searched_customer_id=None):
        self.Customer = _RecordingCustomerAPI(
            retrieve_result, searched_customer_id=searched_customer_id
        )
        self.Subscription = _RecordingSubscriptionAPI()


def _fake_request(stripe_client):
    app_state = SimpleNamespace(
        stripe=stripe_client,
        stripe_billing_config=SimpleNamespace(),
        pool=SimpleNamespace(),
    )
    return SimpleNamespace(app=SimpleNamespace(state=app_state))


@pytest.fixture
def anonymous_billing_environment(monkeypatch):
    """Enable anonymous billing and record what is read from / written to Postgres."""
    invalidate_anonymous_billing_cache()

    environment = SimpleNamespace(
        persisted_customer_id="cus_recorded", persisted_writes=[]
    )

    monkeypatch.setattr(
        anonymous_billing_module,
        "GlobalContext",
        lambda: SimpleNamespace(anonymous_billing_enabled="TRUE"),
    )

    async def _fake_fetch(pool, hashed_ip):
        return environment.persisted_customer_id

    async def _fake_persist(pool, hashed_ip, stripe_customer_id):
        environment.persisted_writes.append((hashed_ip, stripe_customer_id))
        return True

    monkeypatch.setattr(
        anonymous_billing_module, "fetch_anonymous_stripe_customer_id", _fake_fetch
    )
    monkeypatch.setattr(
        anonymous_billing_module, "persist_anonymous_stripe_customer_id", _fake_persist
    )

    # create_free_tier_subscription is imported inside the function (circular
    # import with src.security.auth), so it is patched on its defining module.
    from src.security import auth as auth_module

    created_subscriptions = []

    def _fake_create_free_tier_subscription(
        stripe_client, billing_config, customer_id, *args, **kwargs
    ):
        created_subscriptions.append((customer_id, kwargs.get("extra_metadata")))
        return {"id": "sub_anonymous_free", "status": "active"}

    monkeypatch.setattr(
        auth_module,
        "create_free_tier_subscription",
        _fake_create_free_tier_subscription,
    )
    environment.created_subscriptions = created_subscriptions

    yield environment

    invalidate_anonymous_billing_cache()


@pytest.mark.asyncio
async def test_a_deleted_recorded_customer_is_replaced(anonymous_billing_environment):
    # Stripe answers a deleted customer with {"deleted": true}, not an error.
    stripe_client = _FakeStripeClient({"id": "cus_recorded", "deleted": True})

    record = await resolve_or_create_anonymous_billing_record(
        _fake_request(stripe_client), HASHED_IP
    )

    assert record is not None
    assert record.stripe_customer_id == "cus_replacement"
    assert record.subscription_id == "sub_anonymous_free"
    # A replacement customer AND its $0 free-tier billing vehicle were created,
    # so meter events for this visitor reach Stripe again.
    (create_call,) = stripe_client.Customer.create_calls
    assert create_call["metadata"] == {"anonymous_hashed_ip": HASHED_IP}
    assert anonymous_billing_environment.created_subscriptions == [
        (
            "cus_replacement",
            {"anonymous_hashed_ip": HASHED_IP, "neural_nexus_tier": "free"},
        )
    ]
    # The stale row is overwritten, so the dead id is never handed out again.
    assert anonymous_billing_environment.persisted_writes == [
        (HASHED_IP, "cus_replacement")
    ]


@pytest.mark.asyncio
async def test_a_resource_missing_recorded_customer_is_replaced(
    anonymous_billing_environment,
):
    stripe_client = _FakeStripeClient(_StripeResourceMissing("No such customer"))

    record = await resolve_or_create_anonymous_billing_record(
        _fake_request(stripe_client), HASHED_IP
    )

    assert record is not None
    assert record.stripe_customer_id == "cus_replacement"
    assert anonymous_billing_environment.persisted_writes == [
        (HASHED_IP, "cus_replacement")
    ]


@pytest.mark.asyncio
async def test_a_deleted_recorded_customer_prefers_a_searchable_survivor(
    anonymous_billing_environment,
):
    # Customer Search never returns deleted customers, so a hit here is a live
    # customer already carrying this hashed ip — reuse it instead of duplicating.
    stripe_client = _FakeStripeClient(
        {"id": "cus_recorded", "deleted": True},
        searched_customer_id="cus_survivor",
    )

    record = await resolve_or_create_anonymous_billing_record(
        _fake_request(stripe_client), HASHED_IP
    )

    assert record.stripe_customer_id == "cus_survivor"
    assert stripe_client.Customer.create_calls == []
    assert anonymous_billing_environment.persisted_writes == [
        (HASHED_IP, "cus_survivor")
    ]


@pytest.mark.asyncio
async def test_a_live_recorded_customer_is_reused_without_creating_anything(
    anonymous_billing_environment,
):
    stripe_client = _FakeStripeClient({"id": "cus_recorded"})

    record = await resolve_or_create_anonymous_billing_record(
        _fake_request(stripe_client), HASHED_IP
    )

    assert record.stripe_customer_id == "cus_recorded"
    assert stripe_client.Customer.create_calls == []
    assert stripe_client.Customer.search_calls == []
    assert anonymous_billing_environment.created_subscriptions == []


@pytest.mark.asyncio
async def test_a_transient_stripe_failure_keeps_the_recorded_customer(
    anonymous_billing_environment,
):
    # Network/auth/rate-limit failures carry no resource_missing code. Treating
    # them as "deleted" would fan out a duplicate customer per outage and split
    # the visitor's usage across two ledgers.
    stripe_client = _FakeStripeClient(RuntimeError("connection reset"))

    record = await resolve_or_create_anonymous_billing_record(
        _fake_request(stripe_client), HASHED_IP
    )

    assert record.stripe_customer_id == "cus_recorded"
    assert stripe_client.Customer.create_calls == []


@pytest.mark.asyncio
async def test_a_visitor_with_no_recorded_customer_still_creates_one(
    anonymous_billing_environment,
):
    anonymous_billing_environment.persisted_customer_id = None
    stripe_client = _FakeStripeClient({"id": "unused"})

    record = await resolve_or_create_anonymous_billing_record(
        _fake_request(stripe_client), HASHED_IP
    )

    assert record.stripe_customer_id == "cus_replacement"
    # No recorded id means no liveness check to make.
    assert stripe_client.Customer.retrieve_calls == []
