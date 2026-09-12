"""Per-avatar storage endpoints. INERT SALVAGE — this router is never registered.

Salvaged from the ``z-anubis`` branch (commits ``c8a958e`` and ``6ed92cb``) on
2026-09-09, where these two handlers were declared inline in
``src/api/webapp.py`` with the ``@app.get`` / ``@app.post`` decorators. They
are moved onto a standalone ``APIRouter`` here so the salvage costs the live
application nothing: ``webapp.py`` does not import this module and never calls
``include_router`` on ``storage_route``, so neither path is served.

The handlers carry their own dependencies as imports INSIDE each function, the
way the live routes already lazy-import (see ``record_message_feedback_route``
in ``webapp.py``). That keeps this module importable on its own for review and
testing without pulling ``webapp`` in at import time.

Activation is one line in ``webapp.py`` plus the two pieces of billing
plumbing listed in ``README.md``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.anubis.utils.billing.storage import (
    STORAGE_ADDON_PACK_BYTES,
    STORAGE_ADDON_PACK_PRICE_USD,
    STORAGE_ADDONS_METADATA_KEY,
    estimate_upload_bytes,
    invalidate_storage_measurement,
    resolve_storage_allotment,
    storage_addon_packs,
)
from src.security.auth import get_current_user

logger = logging.getLogger(__name__)

storage_route = APIRouter(tags=["storage (inert salvage)"])


def estimate_media_upload_bytes(media_items: list[Any]) -> int:
    """Bytes an upload batch is expected to add, for the pre-flight storage gate.

    Salvaged from the helper that sat immediately above the storage routes on
    ``z-anubis``. Files contribute their actual length; a URL contributes the
    fixed ``URL_UPLOAD_ESTIMATED_BYTES`` guess because the size is unknown
    until the fetch happens.
    """
    file_sizes: list[int] = []
    url_count = 0
    for entry in media_items or []:
        content = getattr(entry, "content", None)
        if content is not None:
            file_sizes.append(len(content))
        elif isinstance(entry, dict) and (
            entry.get("page_url")
            or entry.get("audio_url")
            or entry.get("image_url")
            or entry.get("video_url")
        ):
            url_count += 1
    return estimate_upload_bytes(file_sizes, url_count)


@storage_route.get("/avatar/{assistant_id}/storage")
async def get_avatar_storage(
    request: Request, assistant_id: str, current_user: dict = Depends(get_current_user)
) -> JSONResponse:
    """The avatar's storage: bytes used, the tier allotment, purchased packs, and the pack offer."""
    from src.anubis.utils.billing.config import current_stripe_billing_config
    from src.api.webapp import resolve_assistant_for_creator

    await resolve_assistant_for_creator(
        assistant_id, current_user, action_description="read that avatar's storage"
    )
    allotment = await resolve_storage_allotment(
        request.app.state.pool, current_user, assistant_id
    )
    billing_config = current_stripe_billing_config(request.app.state)
    return JSONResponse(
        {
            "assistant_id": assistant_id,
            **allotment.as_dict(),
            "pack_price_usd": STORAGE_ADDON_PACK_PRICE_USD,
            # ``storage_addon_price_id`` does not exist on f-anubis's
            # StripeBillingConfig yet — it is one of the activation steps in
            # README.md — so it is read defensively here.
            "purchase_available": bool(
                billing_config
                and getattr(billing_config, "storage_addon_price_id", None)
            ),
        }
    )


class StoragePurchaseRequest(BaseModel):
    packs: int = 1


@storage_route.post("/avatar/{assistant_id}/storage/purchase")
async def purchase_avatar_storage(
    request: Request,
    assistant_id: str,
    purchase: StoragePurchaseRequest,
    current_user: dict = Depends(get_current_user),
) -> JSONResponse:
    """Start a Stripe Checkout session for one-time storage add-on packs on this avatar.

    The packs are granted by the ``checkout.session.completed`` webhook, which
    records them in the account's ``storage_addons`` per avatar.
    """
    from src.anubis.utils.billing.config import current_stripe_billing_config
    from src.anubis.utils.billing.gating import resolve_stripe_customer_id
    from src.api.webapp import resolve_assistant_for_creator

    await resolve_assistant_for_creator(
        assistant_id, current_user, action_description="buy storage for that avatar"
    )
    packs = max(1, min(int(purchase.packs or 1), 100))
    billing_config = current_stripe_billing_config(request.app.state)
    storage_addon_price_id = getattr(billing_config, "storage_addon_price_id", None)
    if not billing_config or not storage_addon_price_id:
        raise HTTPException(
            status_code=503,
            detail="Storage add-on packs are not provisioned in Stripe yet.",
        )
    stripe_client = request.app.state.stripe
    base_url = str(request.base_url).rstrip("/")
    checkout_kwargs: dict = {
        "mode": "payment",
        "line_items": [{"price": storage_addon_price_id, "quantity": packs}],
        "success_url": f"{base_url}/avatar/{assistant_id}/storage",
        "cancel_url": f"{base_url}/docs",
        "metadata": {
            "auth0_user_id": current_user.get("user_id", ""),
            "assistant_id": assistant_id,
            "neural_nexus_storage_packs": str(packs),
        },
    }
    customer_id = resolve_stripe_customer_id(current_user)
    if customer_id:
        checkout_kwargs["customer"] = customer_id
    elif current_user.get("email"):
        checkout_kwargs["customer_email"] = current_user["email"]
    try:
        session = stripe_client.checkout.Session.create(**checkout_kwargs)
    except Exception as checkout_error:  # noqa: BLE001
        logger.error("Could not create storage Checkout session: %s", checkout_error)
        raise HTTPException(
            status_code=502, detail="Could not start checkout. Please try again."
        ) from checkout_error
    return JSONResponse(
        {
            "url": session["url"],
            "packs": packs,
            "pack_bytes": STORAGE_ADDON_PACK_BYTES,
            "message": f"Follow this link to buy {packs} storage pack{'s' if packs != 1 else ''}.",
        }
    )


async def grant_storage_packs_from_checkout(
    request: Request, data_object: dict
) -> bool:
    """Record purchased storage packs on the account; True when the session was a storage purchase.

    On ``z-anubis`` this was ``_grant_storage_packs_from_checkout``, called
    from the ``checkout.session.completed`` branch of the Stripe webhook. It is
    public here because the webhook lives in ``webapp.py`` and will import it
    by name on activation.
    """
    from src.api.webapp import _auth0_user_id_for_customer, _read_user_app_metadata
    from src.security.auth import update_user_app_metadata_fields

    metadata = data_object.get("metadata") or {}
    raw_packs = metadata.get("neural_nexus_storage_packs")
    assistant_id = metadata.get("assistant_id")
    if not raw_packs or not assistant_id:
        return False
    try:
        packs = int(raw_packs)
    except ValueError:
        return True
    auth0_user_id = metadata.get("auth0_user_id") or _auth0_user_id_for_customer(
        request.app.state.stripe, data_object.get("customer")
    )
    if not auth0_user_id:
        logger.error(
            "Storage purchase with no resolvable user: %s", data_object.get("id")
        )
        return True
    app_metadata = await _read_user_app_metadata(request, auth0_user_id)
    addons = dict((app_metadata or {}).get(STORAGE_ADDONS_METADATA_KEY) or {})
    addons[assistant_id] = (
        storage_addon_packs({"app_metadata": app_metadata}, assistant_id) + packs
    )
    await update_user_app_metadata_fields(
        request, auth0_user_id, {STORAGE_ADDONS_METADATA_KEY: addons}
    )
    invalidate_storage_measurement(assistant_id)
    logger.info(
        "Granted %s storage pack(s) on avatar %s for %s",
        packs,
        assistant_id,
        auth0_user_id,
    )
    return True


__all__ = [
    "StoragePurchaseRequest",
    "estimate_media_upload_bytes",
    "get_avatar_storage",
    "grant_storage_packs_from_checkout",
    "purchase_avatar_storage",
    "storage_route",
]
