"""One Stripe customer per Stripe mode for every signed-in account.

Regression for the 2026-09-28 prod failure: dev (test key) wrote a test-mode
customer into the shared Auth0 ``app_metadata.stripe_customer_id``, prod sent
that customer to Stripe with the live key, and every meter event failed with
``No such customer … a similar object exists in test mode``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.anubis.utils.billing import stripe_mode
from src.anubis.utils.billing.gating import resolve_stripe_customer_id


class _MissingCustomerError(Exception):
    code = "resource_missing"


class _StripeObject(dict):
    def to_dict(self):
        return dict(self)


class _FakeCustomerApi:
    def __init__(self, customers_in_mode: dict[str, dict], customers_by_email: dict[str, str]):
        self.customers_in_mode = customers_in_mode
        self.customers_by_email = customers_by_email
        self.created_customers: list[dict] = []

    def retrieve(self, customer_id):
        if customer_id not in self.customers_in_mode:
            raise _MissingCustomerError(f"No such customer: '{customer_id}'")
        return _StripeObject(self.customers_in_mode[customer_id])

    def list(self, email, limit):
        customer_id = self.customers_by_email.get(email)
        return _StripeObject({"data": [{"id": customer_id}] if customer_id else []})

    def create(self, **customer_fields):
        self.created_customers.append(customer_fields)
        return _StripeObject({"id": f"cus_created_{len(self.created_customers)}"})


def _request_with(customer_api):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(stripe=SimpleNamespace(Customer=customer_api))))


@pytest.fixture
def live_mode(monkeypatch):
    stripe_mode._reconciled_customer_ids.clear()
    stripe_mode._reconcile_locks.clear()
    monkeypatch.setitem(stripe_mode._current_stripe_mode_cache, "mode", stripe_mode.STRIPE_MODE_LIVE)
    auth0_patches: list[tuple[str, dict]] = []

    async def _record_patch(request, auth0_user_id, fields):
        auth0_patches.append((auth0_user_id, fields))
        return True

    import src.security.auth as auth_module

    monkeypatch.setattr(auth_module, "update_user_app_metadata_fields", _record_patch)
    yield auth0_patches
    stripe_mode._current_stripe_mode_cache.clear()
    stripe_mode._reconciled_customer_ids.clear()


@pytest.mark.parametrize(
    ("secret_key", "expected_mode"),
    [
        ("sk_live_abc", "live"),
        ('"sk_test_abc"', "test"),
        ("rk_live_abc", "live"),
        ("rk_test_abc", "test"),
        ("", None),
        (None, None),
        ("pk_live_abc", None),
    ],
)
def test_mode_is_read_from_the_key_prefix(secret_key, expected_mode):
    assert stripe_mode.stripe_mode_of_secret_key(secret_key) == expected_mode


@pytest.mark.asyncio
async def test_test_mode_customer_is_replaced_for_live_requests(live_mode):
    customer_api = _FakeCustomerApi(customers_in_mode={}, customers_by_email={})
    user = {
        "user_id": "auth0|evan",
        "email": "evan@example.com",
        "app_metadata": {
            "stripe_customer_id": "cus_Ux1G8zxlKL1GFV",
            "subscription_status": {"customer_id": "cus_Ux1G8zxlKL1GFV", "tier": "free"},
        },
    }
    customer_id = await stripe_mode.reconcile_stripe_customer_for_current_mode(
        _request_with(customer_api), user
    )
    assert customer_id == "cus_created_1"
    assert customer_api.created_customers[0]["metadata"]["neural_nexus_stripe_mode"] == "live"
    assert user["app_metadata"]["stripe_customer_id"] == "cus_created_1"
    assert user["app_metadata"]["subscription_status"]["customer_id"] == "cus_created_1"
    assert live_mode == [
        ("auth0|evan", {"stripe_customer_ids": {"test": "cus_Ux1G8zxlKL1GFV", "live": "cus_created_1"}})
    ]
    assert resolve_stripe_customer_id(user) == "cus_created_1"


@pytest.mark.asyncio
async def test_legacy_customer_that_exists_in_this_mode_is_kept(live_mode):
    customer_api = _FakeCustomerApi(customers_in_mode={"cus_live_legacy": {}}, customers_by_email={})
    user = {"user_id": "auth0|a", "email": "a@example.com", "app_metadata": {"stripe_customer_id": "cus_live_legacy"}}
    assert (
        await stripe_mode.reconcile_stripe_customer_for_current_mode(_request_with(customer_api), user)
        == "cus_live_legacy"
    )
    assert customer_api.created_customers == []
    assert live_mode == [("auth0|a", {"stripe_customer_ids": {"live": "cus_live_legacy"}})]


@pytest.mark.asyncio
async def test_existing_live_customer_with_the_same_email_is_reused(live_mode):
    customer_api = _FakeCustomerApi(customers_in_mode={}, customers_by_email={"b@example.com": "cus_live_by_email"})
    user = {"user_id": "auth0|b", "email": "b@example.com", "app_metadata": {"stripe_customer_id": "cus_test_only"}}
    assert (
        await stripe_mode.reconcile_stripe_customer_for_current_mode(_request_with(customer_api), user)
        == "cus_live_by_email"
    )
    assert customer_api.created_customers == []


@pytest.mark.asyncio
async def test_recorded_map_answers_without_calling_stripe(live_mode):
    class _RefusingCustomerApi:
        def __getattr__(self, name):
            raise AssertionError("Stripe must not be called when the map already holds the mode")

    user = {
        "user_id": "auth0|c",
        "app_metadata": {
            "stripe_customer_id": "cus_test_c",
            "stripe_customer_ids": {"test": "cus_test_c", "live": "cus_live_c"},
        },
    }
    assert (
        await stripe_mode.reconcile_stripe_customer_for_current_mode(
            _request_with(_RefusingCustomerApi()), user
        )
        == "cus_live_c"
    )
    assert live_mode == []


@pytest.mark.asyncio
async def test_a_stripe_outage_creates_nothing_and_retries_later(live_mode):
    class _NetworkError(Exception):
        code = None

    class _UnreachableCustomerApi(_FakeCustomerApi):
        def retrieve(self, customer_id):
            raise _NetworkError("connection reset")

    customer_api = _UnreachableCustomerApi(customers_in_mode={}, customers_by_email={})
    user = {"user_id": "auth0|d", "email": "d@example.com", "app_metadata": {"stripe_customer_id": "cus_d"}}
    assert await stripe_mode.reconcile_stripe_customer_for_current_mode(_request_with(customer_api), user) is None
    assert customer_api.created_customers == []
    assert live_mode == []
    assert user["app_metadata"]["stripe_customer_id"] == "cus_d"


@pytest.mark.asyncio
async def test_anonymous_and_unconfigured_requests_are_left_alone(live_mode, monkeypatch):
    customer_api = _FakeCustomerApi(customers_in_mode={}, customers_by_email={})
    anonymous_user = {"user_id": "anon", "is_anonymous": True, "app_metadata": {}}
    assert await stripe_mode.reconcile_stripe_customer_for_current_mode(_request_with(customer_api), anonymous_user) is None
    monkeypatch.setitem(stripe_mode._current_stripe_mode_cache, "mode", None)
    signed_in_user = {"user_id": "auth0|e", "app_metadata": {"stripe_customer_id": "cus_e"}}
    assert await stripe_mode.reconcile_stripe_customer_for_current_mode(_request_with(customer_api), signed_in_user) is None
    assert customer_api.created_customers == []
