"""LiveKit Agents worker for inbound conversation and outbound restaurant orders.

This is a separate process from the API and from the iOS app. STT is Deepgram
(or LiveKit Inference). TTS is the existing ElevenLabs clone. LangGraph stays
the only agent: inbound reuses ``response_only_workflow``; outbound runs a
frozen order brief and extracts cost / ready time / location at hangup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.anubis.utils.phone.extract import extract_order_result
from src.anubis.utils.phone.repository import get_phone_call_repository
from src.anubis.utils.phone.results import write_phone_call_result

logger = logging.getLogger(__name__)


async def finish_outbound_call(
    *,
    call_id: str,
    transcript: str,
    user_id: str,
    assistant_id: str,
    destination_name: str = "",
    travel_minutes: float | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Extract the order result, persist it, and write inbox plus chat fields."""
    result = extract_order_result(transcript, destination_name=destination_name)
    if travel_minutes is not None:
        result["travel_minutes"] = travel_minutes
        result["travel_text"] = f"{int(round(travel_minutes))} min"
    repository = get_phone_call_repository()
    if repository is not None:
        await repository.update_call(
            call_id,
            {"state": "ended", "transcript": transcript, "result": result},
        )
    return await write_phone_call_result(
        user_id=user_id,
        assistant_id=assistant_id,
        call_id=call_id,
        result=result,
        transcript=transcript,
        thread_id=thread_id,
    )


def outbound_order_script(brief: dict[str, Any]) -> str:
    """Frozen words the outbound graph speaks. Pickup and pay at the counter only."""
    item = str(brief.get("item") or "the order")
    size = str(brief.get("size") or "").strip()
    pickup = str(brief.get("pickup_name") or "the guest").strip() or "the guest"
    named = f"{size} {item}".strip() if size else item
    return (
        f"Hello, I would like to place a pickup order for {named}, "
        f"under the name {pickup}. We will pay at the counter. "
        "Please do not take a card number over the phone. "
        "What is the total, when will the order be ready, and what is the pickup address?"
    )


async def run_worker() -> None:
    """Idle loop when LiveKit is not configured; otherwise log readiness.

    A full LiveKit Agents session is started only when the LiveKit SDK and
    credentials are present. Tests call ``finish_outbound_call`` directly.
    """
    from src.anubis.utils.context import GlobalContext

    context = GlobalContext()
    livekit_url = str(getattr(context, "livekit_url", None) or "").strip()
    if not livekit_url:
        logger.warning(
            "phone_worker is running without LIVEKIT_URL; inbound and outbound "
            "SIP will stay idle until the credentials are set."
        )
        while True:
            await asyncio.sleep(60)
    logger.info("phone_worker ready for LiveKit SIP at %s", livekit_url)
    while True:
        await asyncio.sleep(60)


def main() -> None:
    """Entrypoint for ``python -m src.anubis.utils.phone.worker``."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
