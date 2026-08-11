"""Unit tests for mutating a Stripe price from the tier definitions.

Editing a number in ``tiers.py`` and re-running
``scripts/provision_stripe_billing.py`` used to do nothing at all: prices were
matched by lookup key and reused without the amount ever being compared. These
tests cover the machinery that makes the edit land — the exact decimal
arithmetic a rate needs to survive a round trip through Stripe, the comparison
that decides a live price has changed, and the replace-and-migrate path itself.

The end-to-end cases drive the real script (loaded by path, as an operator runs
it) against an in-memory Stripe, because the failures that matter here are
silent: a price that is never replaced, or one that is replaced on every single
run.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.anubis.utils.billing.config import load_stripe_billing_config
from src.anubis.utils.billing.tiers import (
    TIER_DEFINITIONS,
    MeterAllotment,
    SubscriptionTier,
    UsageMeter,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROVISIONING_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "provision_stripe_billing.py"


# ---------------------------------------------------------------------------
# Exact decimal arithmetic — what keeps a rate from being "changed" every run.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate_per_million, expected",
    [
        (1.50, "0.00015"),
        (3.00, "0.0003"),
        (2.00, "0.0002"),
        (1.25, "0.000125"),
        # Rates whose float division produces a representation artifact:
        # 1.10 / 10_000 is 0.00011000000000000002 in binary floating point —
        # twenty decimal places where Stripe permits twelve.
        (1.10, "0.00011"),
        (0.33, "0.000033"),
        (19.99, "0.001999"),
        (0.07, "0.000007"),
    ],
)
def test_overage_rate_is_exact_and_within_stripe_precision(
    rate_per_million: float, expected: str
) -> None:
    allotment = MeterAllotment(
        meter=UsageMeter.MESSAGING_TOKENS,
        monthly_allotment=1_000,
        overage_price_per_million=rate_per_million,
    )
    rendered = allotment.stripe_unit_amount_decimal()
    assert rendered == expected
    assert "e" not in rendered.lower(), "Stripe rejects exponent notation"
    _, _, fractional_digits = rendered.partition(".")
    assert len(fractional_digits) <= 12


def test_per_unit_overage_rate_is_rendered_in_cents() -> None:
    allotment = MeterAllotment(
        meter=UsageMeter.ADAPTER_TRAINING_UNITS,
        monthly_allotment=5,
        overage_price_per_unit_usd=5.00,
    )
    assert allotment.stripe_unit_amount_decimal() == "500"


def test_base_fee_cents_survive_float_representation() -> None:
    definition = replace(
        TIER_DEFINITIONS[SubscriptionTier.PRO], monthly_base_fee_usd=19.99
    )
    assert definition.stripe_base_unit_amount_cents() == 1999


def test_every_configured_rate_is_within_stripe_precision() -> None:
    for definition in TIER_DEFINITIONS.values():
        for allotment in definition.meter_allotments.values():
            rendered = allotment.stripe_unit_amount_decimal()
            _, _, fractional_digits = rendered.partition(".")
            assert len(fractional_digits) <= 12, (
                f"{allotment.meter.value} rate {rendered} exceeds Stripe's precision"
            )


# ---------------------------------------------------------------------------
# The script, loaded the way an operator runs it.
# ---------------------------------------------------------------------------


class _StripeObject:
    """Stand-in for a StripeObject (not a dict subclass in stripe-python 15)."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class _StripeList:
    def __init__(self, data: list) -> None:
        self._data = data

    def auto_paging_iter(self):
        return iter(self._data)

    def to_dict(self) -> dict:
        return {"data": [entry.to_dict() for entry in self._data]}


class FakeStripe:
    """An in-memory Stripe good enough to exercise price mutation."""

    def __init__(self) -> None:
        self.products: dict[str, dict] = {}
        self.prices: dict[str, dict] = {}
        self.meters: dict[str, dict] = {}
        self.subscriptions: dict[str, dict] = {}
        self.portal_configurations: dict[str, dict] = {}
        self.created_price_calls: list[tuple[str | None, bool]] = []
        self.archived_price_ids: list[str] = []
        self.subscription_modifications: list[tuple[str, str | None]] = []
        self._counter = 0
        outer = self

        class Product:
            @staticmethod
            def list(active=None, limit=None, **_kwargs):
                return _StripeList(
                    [
                        _StripeObject(product)
                        for product in outer.products.values()
                        if product["active"] == active
                    ]
                )

            @staticmethod
            def create(**parameters):
                outer._counter += 1
                product_id = f"prod_{outer._counter}"
                outer.products[product_id] = {
                    "id": product_id,
                    "active": True,
                    "created": 1_000 + outer._counter,
                    **parameters,
                }
                return _StripeObject(outer.products[product_id])

            @staticmethod
            def modify(product_id, **parameters):
                outer.products[product_id].update(parameters)
                return _StripeObject(outer.products[product_id])

        class Price:
            @staticmethod
            def list(lookup_keys=None, active=None, limit=None, expand=None, **_kwargs):
                return _StripeList(
                    [
                        _StripeObject(price)
                        for price in outer.prices.values()
                        if (active is None or price["active"] == active)
                        and (
                            lookup_keys is None
                            or price.get("lookup_key") in lookup_keys
                        )
                    ]
                )

            @staticmethod
            def create(transfer_lookup_key=False, **parameters):
                outer._counter += 1
                price_id = f"price_{outer._counter}"
                if transfer_lookup_key:
                    # Stripe moves the key off whichever price currently holds it.
                    for price in outer.prices.values():
                        if price.get("lookup_key") == parameters.get("lookup_key"):
                            price["lookup_key"] = None
                recurring = dict(parameters.get("recurring") or {})
                recurring.setdefault("usage_type", "licensed")
                stored = {
                    "id": price_id,
                    "active": True,
                    **parameters,
                    "recurring": recurring,
                }
                if "tiers" in parameters:
                    stored["tiers"] = [
                        {
                            "up_to": None if tier["up_to"] == "inf" else tier["up_to"],
                            "unit_amount_decimal": Decimal(tier["unit_amount_decimal"]),
                        }
                        for tier in parameters["tiers"]
                    ]
                outer.prices[price_id] = stored
                outer.created_price_calls.append(
                    (parameters.get("lookup_key"), transfer_lookup_key)
                )
                return _StripeObject(stored)

            @staticmethod
            def modify(price_id, **parameters):
                outer.prices[price_id].update(parameters)
                if parameters.get("active") is False:
                    outer.archived_price_ids.append(price_id)
                return _StripeObject(outer.prices[price_id])

        class Meter:
            @staticmethod
            def list(status=None, limit=None):
                return _StripeList(
                    [_StripeObject(meter) for meter in outer.meters.values()]
                )

            @staticmethod
            def create(**parameters):
                outer._counter += 1
                meter_id = f"mtr_{outer._counter}"
                outer.meters[meter_id] = {"id": meter_id, **parameters}
                return _StripeObject(outer.meters[meter_id])

        class Subscription:
            @staticmethod
            def list(status=None, limit=None, **_kwargs):
                return _StripeList(
                    [
                        _StripeObject(subscription)
                        for subscription in outer.subscriptions.values()
                    ]
                )

            @staticmethod
            def modify(subscription_id, **parameters):
                subscription = outer.subscriptions[subscription_id]
                deleted_item_ids = {
                    item["id"] for item in parameters["items"] if item.get("deleted")
                }
                retained = [
                    item
                    for item in subscription["items"]["data"]
                    if item["id"] not in deleted_item_ids
                ]
                added = [
                    {"id": f"si_new_{index}", "price": {"id": item["price"]}}
                    for index, item in enumerate(parameters["items"])
                    if item.get("price")
                ]
                subscription["items"]["data"] = retained + added
                outer.subscription_modifications.append(
                    (subscription_id, parameters.get("proration_behavior"))
                )
                return _StripeObject(subscription)

        class Configuration:
            @staticmethod
            def list(active=None, limit=None):
                return _StripeList(
                    [
                        _StripeObject(configuration)
                        for configuration in outer.portal_configurations.values()
                    ]
                )

            @staticmethod
            def create(**parameters):
                outer._counter += 1
                configuration_id = f"bpc_{outer._counter}"
                outer.portal_configurations[configuration_id] = {
                    "id": configuration_id,
                    "active": True,
                    **parameters,
                }
                return _StripeObject(outer.portal_configurations[configuration_id])

        class _BillingPortalNamespace:
            pass

        class _BillingNamespace:
            pass

        # Class bodies do not close over enclosing function scope, so the nested
        # namespaces are assembled after the classes exist.
        _BillingPortalNamespace.Configuration = Configuration
        _BillingNamespace.Meter = Meter

        self.Product = Product
        self.Price = Price
        self.Subscription = Subscription
        self.billing = _BillingNamespace
        self.billing_portal = _BillingPortalNamespace

    def active_product_count(self) -> int:
        return len([p for p in self.products.values() if p["active"]])

    def active_price_count(self) -> int:
        return len([p for p in self.prices.values() if p["active"]])

    def active_lookup_keys(self) -> list[str]:
        return [
            price["lookup_key"]
            for price in self.prices.values()
            if price["active"] and price.get("lookup_key")
        ]


def _load_provisioning_script():
    specification = importlib.util.spec_from_file_location(
        "provision_stripe_billing_under_test", PROVISIONING_SCRIPT_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def provisioning_script(monkeypatch: pytest.MonkeyPatch):
    """The real script with an in-memory Stripe injected."""
    module = _load_provisioning_script()
    fake_stripe = FakeStripe()
    monkeypatch.setattr(module, "stripe", fake_stripe)
    return module, fake_stripe


def _edit_pro_tier(monkeypatch: pytest.MonkeyPatch, **changes: Any) -> None:
    """Apply an edit to the pro tier definition, as an operator edits tiers.py."""
    monkeypatch.setitem(
        TIER_DEFINITIONS,
        SubscriptionTier.PRO,
        replace(TIER_DEFINITIONS[SubscriptionTier.PRO], **changes),
    )


# ---------------------------------------------------------------------------
# Drift detection.
# ---------------------------------------------------------------------------


def _pro_messaging_allotment() -> MeterAllotment:
    return TIER_DEFINITIONS[SubscriptionTier.PRO].meter_allotments[
        UsageMeter.MESSAGING_TOKENS
    ]


def _metered_price_echo(overage_rendering: Any) -> dict:
    """The price dictionary Stripe returns for the current pro messaging price."""
    return {
        "id": "price_existing",
        "product": "prod_messaging",
        "recurring": {"interval": "month", "usage_type": "metered", "meter": "mtr_1"},
        "billing_scheme": "tiered",
        "tiers_mode": "graduated",
        "tiers": [
            {"up_to": 5_000_000, "unit_amount_decimal": Decimal("0")},
            # Stripe reports the final, unbounded tier's bound as null.
            {"up_to": None, "unit_amount_decimal": overage_rendering},
        ],
    }


@pytest.mark.parametrize(
    "overage_rendering",
    [
        Decimal("0.00015"),
        # Trailing zeros: the same number, a different representation.
        Decimal("0.000150"),
        # Raw HTTP returns a string; fixtures naturally use floats.
        "0.00015",
        0.00015,
    ],
)
def test_unchanged_metered_price_is_not_treated_as_changed(
    provisioning_script, overage_rendering: Any
) -> None:
    """A representation difference must never be mistaken for a price change."""
    module, _ = provisioning_script
    assert (
        module.metered_price_differences(
            _metered_price_echo(overage_rendering),
            _pro_messaging_allotment(),
            "prod_messaging",
            "mtr_1",
        )
        == []
    )


def test_changed_overage_rate_is_detected(provisioning_script) -> None:
    module, _ = provisioning_script
    differences = module.metered_price_differences(
        _metered_price_echo(Decimal("0.0002")),
        _pro_messaging_allotment(),
        "prod_messaging",
        "mtr_1",
    )
    assert len(differences) == 1 and "overage rate" in differences[0]


def test_changed_allotment_is_detected(provisioning_script) -> None:
    module, _ = provisioning_script
    echoed = _metered_price_echo(Decimal("0.00015"))
    echoed["tiers"][0]["up_to"] = 4_000_000
    differences = module.metered_price_differences(
        echoed, _pro_messaging_allotment(), "prod_messaging", "mtr_1"
    )
    assert differences == ["allotment 4000000 -> 5,000,000"]


def test_unexpanded_tiers_are_reported_rather_than_assumed_equal(
    provisioning_script,
) -> None:
    """Treating an uncomparable price as matching would freeze a stale allotment.

    Every later run would make the same assumption, so the edit could never land.
    """
    module, _ = provisioning_script
    echoed = _metered_price_echo(Decimal("0.00015"))
    del echoed["tiers"]
    assert module.metered_price_differences(
        echoed, _pro_messaging_allotment(), "prod_messaging", "mtr_1"
    ) == ["graduated tiers were not expanded on the existing price"]


def test_unchanged_base_price_is_not_treated_as_changed(provisioning_script) -> None:
    module, _ = provisioning_script
    definition = TIER_DEFINITIONS[SubscriptionTier.PRO]
    echoed = {
        "id": "price_base",
        "product": "prod_base",
        "currency": "usd",
        "unit_amount": 2_000,
        "recurring": {"interval": "month", "usage_type": "licensed"},
    }
    assert module.base_price_differences(echoed, definition, "prod_base") == []

    edited = replace(definition, monthly_base_fee_usd=21.0)
    assert module.base_price_differences(echoed, edited, "prod_base") == [
        "base fee $20.00 -> $21.00"
    ]


# ---------------------------------------------------------------------------
# End to end: mutating a price.
# ---------------------------------------------------------------------------


def test_first_run_creates_the_catalog(provisioning_script) -> None:
    module, fake_stripe = provisioning_script
    config = module.provision()

    # free: base + 1 meter; pro: base + 2; premium: base + 4.
    assert fake_stripe.active_product_count() == 10
    assert fake_stripe.active_price_count() == 10
    assert not fake_stripe.archived_price_ids
    assert set(config["tiers"]) == {"free", "pro", "premium"}


def test_rerunning_without_an_edit_changes_nothing(provisioning_script) -> None:
    module, fake_stripe = provisioning_script
    first_config = module.provision()
    price_count_before = len(fake_stripe.prices)

    second_config = module.provision()

    assert second_config == first_config
    assert len(fake_stripe.prices) == price_count_before
    assert not fake_stripe.archived_price_ids


def test_editing_a_price_replaces_it_and_archives_the_old_one(
    provisioning_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: an edited number reaches Stripe."""
    module, fake_stripe = provisioning_script
    original_config = module.provision()
    superseded_price_id = original_config["tiers"]["pro"]["base_price"]

    _edit_pro_tier(monkeypatch, monthly_base_fee_usd=21.0)
    fake_stripe.created_price_calls.clear()
    edited_config = module.provision()

    # Exactly one new price, carrying the lookup key away from the old one.
    assert fake_stripe.created_price_calls == [("nn_pro_base_v2", True)]
    assert fake_stripe.archived_price_ids == [superseded_price_id]
    assert not fake_stripe.prices[superseded_price_id]["active"]

    new_price_id = edited_config["tiers"]["pro"]["base_price"]
    assert new_price_id != superseded_price_id
    assert fake_stripe.prices[new_price_id]["unit_amount"] == 2_100
    assert fake_stripe.prices[new_price_id]["lookup_key"] == "nn_pro_base_v2"
    # No product churn, and still one active price per lookup key.
    assert fake_stripe.active_product_count() == 10
    assert fake_stripe.active_price_count() == 10
    lookup_keys = fake_stripe.active_lookup_keys()
    assert len(lookup_keys) == len(set(lookup_keys))


def test_editing_an_allotment_replaces_the_metered_price(
    provisioning_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, fake_stripe = provisioning_script
    original_config = module.provision()
    superseded_price_id = original_config["tiers"]["pro"]["metered_prices"][
        "messaging_tokens"
    ]

    pro = TIER_DEFINITIONS[SubscriptionTier.PRO]
    edited_allotments = dict(pro.meter_allotments)
    edited_allotments[UsageMeter.MESSAGING_TOKENS] = replace(
        edited_allotments[UsageMeter.MESSAGING_TOKENS], monthly_allotment=7_000_000
    )
    _edit_pro_tier(monkeypatch, meter_allotments=edited_allotments)
    edited_config = module.provision()

    new_price_id = edited_config["tiers"]["pro"]["metered_prices"]["messaging_tokens"]
    assert fake_stripe.archived_price_ids == [superseded_price_id]
    assert fake_stripe.prices[new_price_id]["tiers"][0]["up_to"] == 7_000_000
    assert fake_stripe.active_price_count() == 10


def test_rerunning_after_an_edit_does_not_thrash(
    provisioning_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate that round-trips inexactly would mint a new price on every run."""
    module, fake_stripe = provisioning_script
    module.provision()

    pro = TIER_DEFINITIONS[SubscriptionTier.PRO]
    edited_allotments = dict(pro.meter_allotments)
    edited_allotments[UsageMeter.MESSAGING_TOKENS] = replace(
        edited_allotments[UsageMeter.MESSAGING_TOKENS],
        overage_price_per_million=1.10,
    )
    _edit_pro_tier(monkeypatch, meter_allotments=edited_allotments)

    module.provision()
    fake_stripe.archived_price_ids.clear()
    fake_stripe.created_price_calls.clear()

    settled_config = module.provision()
    again_config = module.provision()

    assert settled_config == again_config
    assert not fake_stripe.created_price_calls
    assert not fake_stripe.archived_price_ids
    assert fake_stripe.active_price_count() == 10


def test_live_subscription_is_migrated_onto_the_new_price(
    provisioning_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise Stripe bills the old rate while the API grants the new allotment."""
    module, fake_stripe = provisioning_script
    config = module.provision()

    pro_price_ids = [config["tiers"]["pro"]["base_price"]] + list(
        config["tiers"]["pro"]["metered_prices"].values()
    )
    fake_stripe.subscriptions["sub_live"] = {
        "id": "sub_live",
        "status": "active",
        "metadata": {"neural_nexus_tier": "pro"},
        "items": {
            "data": [
                {"id": f"si_{index}", "price": {"id": price_id}}
                for index, price_id in enumerate(pro_price_ids)
            ]
        },
    }

    _edit_pro_tier(monkeypatch, monthly_base_fee_usd=21.0)
    edited_config = module.provision()

    migrated_price_ids = {
        item["price"]["id"]
        for item in fake_stripe.subscriptions["sub_live"]["items"]["data"]
    }
    expected_price_ids = {edited_config["tiers"]["pro"]["base_price"]} | set(
        edited_config["tiers"]["pro"]["metered_prices"].values()
    )
    assert migrated_price_ids == expected_price_ids
    # A price edit is the operator's doing, not the customer's: no proration.
    assert fake_stripe.subscription_modifications == [("sub_live", "none")]


def test_subscriptions_without_the_replaced_price_are_untouched(
    provisioning_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, fake_stripe = provisioning_script
    config = module.provision()
    fake_stripe.subscriptions["sub_premium"] = {
        "id": "sub_premium",
        "status": "active",
        "metadata": {"neural_nexus_tier": "premium"},
        "items": {
            "data": [
                {
                    "id": "si_0",
                    "price": {"id": config["tiers"]["premium"]["base_price"]},
                }
            ]
        },
    }

    _edit_pro_tier(monkeypatch, monthly_base_fee_usd=21.0)
    module.provision()

    assert fake_stripe.subscription_modifications == []


def test_canceled_subscriptions_are_left_alone(
    provisioning_script, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, fake_stripe = provisioning_script
    config = module.provision()
    fake_stripe.subscriptions["sub_dead"] = {
        "id": "sub_dead",
        "status": "canceled",
        "metadata": {"neural_nexus_tier": "pro"},
        "items": {
            "data": [
                {"id": "si_0", "price": {"id": config["tiers"]["pro"]["base_price"]}}
            ]
        },
    }

    _edit_pro_tier(monkeypatch, monthly_base_fee_usd=21.0)
    module.provision()

    assert fake_stripe.subscription_modifications == []


def test_scheduled_subscription_is_reported_rather_than_migrated(
    provisioning_script, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Releasing a schedule to swap items would cancel a pending downgrade."""
    module, fake_stripe = provisioning_script
    config = module.provision()
    fake_stripe.subscriptions["sub_scheduled"] = {
        "id": "sub_scheduled",
        "status": "active",
        "schedule": "sub_sched_1",
        "metadata": {"neural_nexus_tier": "pro"},
        "items": {
            "data": [
                {"id": "si_0", "price": {"id": config["tiers"]["pro"]["base_price"]}}
            ]
        },
    }

    _edit_pro_tier(monkeypatch, monthly_base_fee_usd=21.0)
    module.provision()

    assert fake_stripe.subscription_modifications == []
    assert "sub_scheduled" in capsys.readouterr().out


def test_emitted_config_document_matches_the_api_parser(provisioning_script) -> None:
    """The printed JSON must load through the API's own config parser."""
    module, _ = provisioning_script
    config = module.provision()

    parsed = load_stripe_billing_config(json.dumps(config))
    assert parsed is not None
    pro_identifiers = parsed.identifiers_for_tier(SubscriptionTier.PRO)
    assert pro_identifiers.base_price_id == config["tiers"]["pro"]["base_price"]
    assert pro_identifiers.metered_price_ids[UsageMeter.MESSAGING_TOKENS] == (
        config["tiers"]["pro"]["metered_prices"]["messaging_tokens"]
    )
    assert parsed.portal_configuration_id == config["portal_configuration"]


# ---------------------------------------------------------------------------
# Which Stripe account a run touches.
# ---------------------------------------------------------------------------


def _write_environment_files(
    directory: Path, test_key: str | None, live_key: str | None
) -> None:
    scripts_directory = directory / "scripts"
    scripts_directory.mkdir(parents=True, exist_ok=True)
    if test_key is not None:
        (directory / ".env.dev").write_text(f"STRIPE_SECRET_KEY={test_key}\n")
    if live_key is not None:
        (directory / ".env").write_text(f"STRIPE_SECRET_KEY={live_key}\n")


@pytest.fixture
def script_in_temporary_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The script with its repository root pointed at a temporary directory.

    ``resolve_stripe_secret_key`` locates the env files relative to the script's
    own path, so relocating that path is what lets the env-selection tests use
    fixture env files instead of the real ones.
    """
    module = _load_provisioning_script()
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / "script.py"))
    return module, tmp_path


def test_default_run_takes_the_test_key_from_the_dev_environment_file(
    script_in_temporary_repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, repository_root = script_in_temporary_repository
    _write_environment_files(repository_root, "sk_test_from_file", "sk_live_from_file")
    # An ambient live key must not win: this is exactly how a run intended for
    # test would otherwise rewrite live prices.
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_from_shell")

    secret_key, source = module.resolve_stripe_secret_key(use_live=False)

    assert secret_key == "sk_test_from_file"
    assert source == ".env.dev"


def test_live_run_takes_the_live_key_from_the_live_environment_file(
    script_in_temporary_repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, repository_root = script_in_temporary_repository
    _write_environment_files(repository_root, "sk_test_from_file", "sk_live_from_file")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_from_shell")

    secret_key, source = module.resolve_stripe_secret_key(use_live=True)

    assert secret_key == "sk_live_from_file"
    assert source == ".env"


def test_a_key_that_contradicts_its_environment_is_refused(
    script_in_temporary_repository,
) -> None:
    module, repository_root = script_in_temporary_repository
    # A live key sitting in the test env file, or the reverse.
    _write_environment_files(repository_root, "sk_live_wrong", "sk_test_wrong")

    with pytest.raises(SystemExit, match="sk_test_"):
        module.resolve_stripe_secret_key(use_live=False)
    with pytest.raises(SystemExit, match="sk_live_"):
        module.resolve_stripe_secret_key(use_live=True)


def test_process_environment_is_the_fallback_when_the_file_has_no_key(
    script_in_temporary_repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers a container handed a key without a mounted env file."""
    module, repository_root = script_in_temporary_repository
    _write_environment_files(repository_root, None, None)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_from_shell")

    secret_key, source = module.resolve_stripe_secret_key(use_live=False)

    assert secret_key == "sk_test_from_shell"
    assert source == "the process environment"


def test_a_missing_key_fails_with_the_file_named(
    script_in_temporary_repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, repository_root = script_in_temporary_repository
    _write_environment_files(repository_root, None, None)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)

    with pytest.raises(SystemExit, match=r"\.env\.dev"):
        module.resolve_stripe_secret_key(use_live=False)


def test_provisioning_script_path_exists() -> None:
    """Guards the by-path import above against the script being moved."""
    assert PROVISIONING_SCRIPT_PATH.is_file()
    assert os.access(PROVISIONING_SCRIPT_PATH, os.R_OK)


# ---------------------------------------------------------------------------
# The API picking up the new price ids without a restart.
# ---------------------------------------------------------------------------


def _billing_config_document(base_price_id: str) -> str:
    return json.dumps(
        {
            "meters": {"messaging_tokens": "mtr_messaging"},
            "tiers": {
                "free": {
                    "base_price": base_price_id,
                    "metered_prices": {"messaging_tokens": "price_free_messaging"},
                }
            },
        }
    )


class _ApplicationState:
    """Stands in for the FastAPI ``app.state`` the accessors cache onto."""

    def __init__(self, context: Any) -> None:
        self.context = context


def _context(config_file: Path, environment_json: str = "") -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        stripe_billing_config_json=environment_json,
        stripe_billing_config_file=str(config_file),
    )


def _base_price_of(config: Any) -> str:
    return config.identifiers_for_tier(SubscriptionTier.FREE).base_price_id


def test_replaced_price_ids_are_picked_up_without_a_restart(tmp_path: Path) -> None:
    """After a mutation the cached ids name an ARCHIVED price.

    Stripe refuses to open a Checkout Session on an archived price, so without
    this reload every new subscription fails until someone restarts the API.
    """
    from src.anubis.utils.billing.config import (
        current_stripe_billing_config,
        initialize_stripe_billing_config,
    )

    config_file = tmp_path / "billing_config.json"
    config_file.write_text(_billing_config_document("price_before_edit"))
    app_state = _ApplicationState(_context(config_file))

    initialize_stripe_billing_config(app_state)
    assert _base_price_of(app_state.stripe_billing_config) == "price_before_edit"
    # Unchanged file: the cached config is returned as-is.
    assert _base_price_of(current_stripe_billing_config(app_state)) == "price_before_edit"

    config_file.write_text(_billing_config_document("price_after_edit"))
    os.utime(config_file, (1_900_000_000, 1_900_000_000))

    assert _base_price_of(current_stripe_billing_config(app_state)) == "price_after_edit"


def test_a_corrupt_config_file_keeps_the_last_good_prices(tmp_path: Path) -> None:
    """A half-written config must not demote every paying customer to free tier."""
    from src.anubis.utils.billing.config import (
        current_stripe_billing_config,
        initialize_stripe_billing_config,
    )

    config_file = tmp_path / "billing_config.json"
    config_file.write_text(_billing_config_document("price_good"))
    app_state = _ApplicationState(_context(config_file))
    initialize_stripe_billing_config(app_state)

    config_file.write_text('{"tiers": ')
    os.utime(config_file, (1_900_000_000, 1_900_000_000))

    assert _base_price_of(current_stripe_billing_config(app_state)) == "price_good"


def test_environment_json_still_wins_over_the_file(tmp_path: Path) -> None:
    from src.anubis.utils.billing.config import (
        current_stripe_billing_config,
        initialize_stripe_billing_config,
    )

    config_file = tmp_path / "billing_config.json"
    config_file.write_text(_billing_config_document("price_from_file"))
    app_state = _ApplicationState(
        _context(config_file, environment_json=_billing_config_document("price_from_env"))
    )

    initialize_stripe_billing_config(app_state)
    assert _base_price_of(app_state.stripe_billing_config) == "price_from_env"

    config_file.write_text(_billing_config_document("price_reprovisioned"))
    os.utime(config_file, (1_900_000_000, 1_900_000_000))
    assert _base_price_of(current_stripe_billing_config(app_state)) == "price_from_env"


def test_environment_json_shadowing_a_different_file_is_reported(tmp_path: Path) -> None:
    """The stale-blob-in-.env trap that makes a reprovision look like a no-op."""
    from src.anubis.utils.billing.config import billing_config_source_conflict

    config_file = tmp_path / "billing_config.json"
    config_file.write_text(_billing_config_document("price_from_file"))

    warning = billing_config_source_conflict(
        _context(config_file, environment_json=_billing_config_document("price_from_env"))
    )
    assert warning is not None
    assert "STRIPE_BILLING_CONFIG_JSON" in warning

    # No environment value, or an identical one, is not a conflict.
    assert billing_config_source_conflict(_context(config_file)) is None
    assert (
        billing_config_source_conflict(
            _context(
                config_file,
                environment_json=_billing_config_document("price_from_file"),
            )
        )
        is None
    )
