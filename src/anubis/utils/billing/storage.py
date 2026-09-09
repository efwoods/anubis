"""Per-avatar storage allotments and add-on packs. INERT SALVAGE — nothing imports this yet.

Salvaged from the ``z-anubis`` branch (commits ``c8a958e`` and ``6ed92cb``)
on 2026-09-09. The module is complete and importable, but no live code path
calls it, so importing or not importing this file changes no behaviour. See
``src/api/salvage/README.md`` for the activation checklist.

Every avatar's identity lives in the LangGraph store (documents, quotes,
transcripts, analysis findings, and their embeddings). Each tier grants a
number of bytes per avatar and the account may buy one-time add-on packs
(``STORAGE_ADDON_PACK_BYTES`` per pack), recorded per avatar in the Auth0
``app_metadata.storage_addons`` mapping by the Stripe webhook. Uploads and
research reads are refused with ``402 storage_exhausted`` once an avatar's
measured usage plus the request's estimated bytes would exceed the allotment.

Measurement sums the on-disk size of every ``store`` row (and the matching
``store_vectors`` embeddings) whose namespace mentions the avatar — the same
segment predicate ``avatar_deletion.py`` deletes by — and is cached briefly
because the query walks the whole store table.

DIVERGENCE FROM THE z-anubis ORIGINAL: on ``z-anubis`` the per-tier byte
allotment was a ``storage_bytes_per_avatar`` field added to ``TierDefinition``
and the pack constants lived in ``tiers.py``. Both are kept local to this
module instead, so salvaging this file edits no live billing code. On
activation, move ``TIER_STORAGE_BYTES_PER_AVATAR`` onto ``TierDefinition`` and
the four ``STORAGE_ADDON_*`` constants into ``tiers.py``, exactly as the
z-anubis diff has them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from src.anubis.utils.billing.gating import resolve_tier
from src.anubis.utils.billing.tiers import SubscriptionTier

logger = logging.getLogger(__name__)

STORAGE_ADDONS_METADATA_KEY = "storage_addons"
STORAGE_EXHAUSTED_REASON = "storage_exhausted"

MEBIBYTE = 1024 * 1024
GIBIBYTE = 1024 * MEBIBYTE

# Store bytes each avatar of a tier may hold (identity documents, quotes,
# transcripts, analysis, embeddings). A level, not a flow, so this is NOT a
# Stripe meter; more storage is bought as one-time add-on packs.
TIER_STORAGE_BYTES_PER_AVATAR: dict[SubscriptionTier, int] = {
    SubscriptionTier.FREE: 50 * MEBIBYTE,
    SubscriptionTier.PRO: 1 * GIBIBYTE,
    SubscriptionTier.PREMIUM: 5 * GIBIBYTE,
}

# One-time storage add-on pack: bytes granted per pack and the price of one
# pack. Provisioned as a single one-time Stripe price by
# scripts/provision_stripe_billing.py (lookup key STORAGE_ADDON_LOOKUP_KEY);
# purchases are recorded on the account as app_metadata.storage_addons
# {assistant_id: packs} by the checkout.session.completed webhook.
STORAGE_ADDON_PACK_BYTES = 1 * GIBIBYTE
STORAGE_ADDON_PACK_PRICE_USD = 2.00
STORAGE_ADDON_PRODUCT_NAME = "Neural Nexus Storage Add-on Pack"
STORAGE_ADDON_LOOKUP_KEY = "neural_nexus_storage_addon_pack_v1"

# Bytes assumed for a URL upload whose size is unknown at request time.
URL_UPLOAD_ESTIMATED_BYTES = 256 * 1024

_MEASUREMENT_CACHE_TTL_SECONDS = 60.0
_measurement_cache: dict[str, tuple[float, int]] = {}

# The row's ``value`` column plus the vector rows that cascade from it.
_MEASURE_STORE_BYTES_SQL = """
SELECT COALESCE(SUM(pg_column_size(store.*)), 0)
  FROM store
 WHERE EXISTS (
     SELECT 1
       FROM unnest(string_to_array(prefix, '.')) AS namespace_segment
      WHERE trim(namespace_segment) = %s
 );
"""
_MEASURE_VECTOR_BYTES_SQL = """
SELECT COALESCE(SUM(pg_column_size(store_vectors.*)), 0)
  FROM store_vectors
 WHERE EXISTS (
     SELECT 1
       FROM unnest(string_to_array(store_vectors.prefix, '.')) AS namespace_segment
      WHERE trim(namespace_segment) = %s
 );
"""


def storage_addon_pack_unit_amount_cents() -> int:
    """The add-on pack price as an integer number of US cents, for Stripe."""
    return int(
        (Decimal(str(STORAGE_ADDON_PACK_PRICE_USD)) * Decimal(100)).quantize(Decimal(1))
    )


def invalidate_storage_measurement(assistant_id: str) -> None:
    _measurement_cache.pop(assistant_id, None)


async def measure_avatar_storage_bytes(
    pool: Any, assistant_id: str, *, use_cache: bool = True
) -> int:
    """Bytes the avatar currently occupies in the store (cached one minute)."""
    if pool is None or not assistant_id:
        return 0
    now = time.monotonic()
    if use_cache:
        cached = _measurement_cache.get(assistant_id)
        if cached is not None and now - cached[0] < _MEASUREMENT_CACHE_TTL_SECONDS:
            return cached[1]
    total = 0
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_MEASURE_STORE_BYTES_SQL, (assistant_id,))
                row = await cursor.fetchone()
                total += int((row or [0])[0] or 0)
                try:
                    await cursor.execute(_MEASURE_VECTOR_BYTES_SQL, (assistant_id,))
                    row = await cursor.fetchone()
                    total += int((row or [0])[0] or 0)
                except Exception as vector_error:  # noqa: BLE001 - vectors table may differ
                    logger.debug("store_vectors measurement skipped: %s", vector_error)
    except Exception as measurement_error:  # noqa: BLE001 - fail open on a database hiccup
        logger.error(
            "Could not measure storage for %s: %s", assistant_id, measurement_error
        )
        return 0
    _measurement_cache[assistant_id] = (now, total)
    return total


def storage_addon_packs(user: Mapping[str, Any] | None, assistant_id: str) -> int:
    app_metadata = (user or {}).get("app_metadata") or {}
    addons = app_metadata.get(STORAGE_ADDONS_METADATA_KEY) or {}
    try:
        return max(0, int(addons.get(assistant_id) or 0))
    except (TypeError, ValueError):
        return 0


def tier_storage_bytes(tier: SubscriptionTier) -> int:
    return int(TIER_STORAGE_BYTES_PER_AVATAR.get(tier, 0) or 0)


@dataclass(frozen=True)
class StorageAllotment:
    tier: SubscriptionTier
    used_bytes: int
    tier_bytes: int
    addon_packs: int
    addon_bytes: int

    @property
    def allotment_bytes(self) -> int:
        return self.tier_bytes + self.addon_bytes

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.allotment_bytes - self.used_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "used_bytes": self.used_bytes,
            "tier_bytes": self.tier_bytes,
            "addon_packs": self.addon_packs,
            "addon_bytes": self.addon_bytes,
            "allotment_bytes": self.allotment_bytes,
            "remaining_bytes": self.remaining_bytes,
            "pack_bytes": STORAGE_ADDON_PACK_BYTES,
        }


async def resolve_storage_allotment(
    pool: Any, user: Mapping[str, Any] | None, assistant_id: str
) -> StorageAllotment:
    tier = resolve_tier(user)
    packs = storage_addon_packs(user, assistant_id)
    used = await measure_avatar_storage_bytes(pool, assistant_id)
    return StorageAllotment(
        tier=tier,
        used_bytes=used,
        tier_bytes=tier_storage_bytes(tier),
        addon_packs=packs,
        addon_bytes=packs * STORAGE_ADDON_PACK_BYTES,
    )


def storage_exhausted_detail(
    allotment: StorageAllotment, requested_bytes: int
) -> dict[str, Any]:
    """The 402 body: the same ``reason``-keyed shape the allotment refusals use."""
    return {
        "reason": STORAGE_EXHAUSTED_REASON,
        "message": (
            "This avatar's storage is full: "
            f"{allotment.used_bytes:,} of {allotment.allotment_bytes:,} bytes are used and this "
            f"request needs about {requested_bytes:,} more. Buy a storage add-on pack "
            f"({STORAGE_ADDON_PACK_BYTES:,} bytes) or remove documents from the avatar."
        ),
        "storage": allotment.as_dict(),
        "requested_bytes": requested_bytes,
    }


def estimate_upload_bytes(file_sizes: list[int], url_count: int) -> int:
    return sum(
        max(0, int(size)) for size in file_sizes
    ) + URL_UPLOAD_ESTIMATED_BYTES * max(0, url_count)


__all__ = [
    "GIBIBYTE",
    "MEBIBYTE",
    "STORAGE_ADDONS_METADATA_KEY",
    "STORAGE_ADDON_LOOKUP_KEY",
    "STORAGE_ADDON_PACK_BYTES",
    "STORAGE_ADDON_PACK_PRICE_USD",
    "STORAGE_ADDON_PRODUCT_NAME",
    "STORAGE_EXHAUSTED_REASON",
    "TIER_STORAGE_BYTES_PER_AVATAR",
    "URL_UPLOAD_ESTIMATED_BYTES",
    "StorageAllotment",
    "estimate_upload_bytes",
    "invalidate_storage_measurement",
    "measure_avatar_storage_bytes",
    "resolve_storage_allotment",
    "storage_addon_pack_unit_amount_cents",
    "storage_addon_packs",
    "storage_exhausted_detail",
    "tier_storage_bytes",
]
