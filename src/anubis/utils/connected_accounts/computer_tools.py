"""Tools that pause the run so the owner can take over the avatar's computer."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool
from langgraph.types import interrupt

from src.anubis.utils.connected_accounts.agent_computer import (
    COMPUTER_HANDOFF_INTERRUPT_KIND,
    VENDOR_DASHBOARD_QUEUE,
    DashboardStep,
    agent_computer_is_enabled,
    build_handoff_card,
    computer_for_user,
    current_step,
    finish_handoff,
    page_looks_like_login,
    skip_handoff,
    start_computer,
)

logger = logging.getLogger(__name__)

COMPUTER_TOOL_NAMES: tuple[str, ...] = (
    "request_computer_handoff",
    "walk_vendor_dashboards",
)

_RESUME_CREDENTIAL_KEYS = frozenset(
    {
        "password",
        "pass",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "code",
        "otp",
        "two_factor",
    }
)


def _resume_type(decision: Any) -> str:
    """Read only the decision type. Credential-looking keys are ignored."""
    payload = decision if isinstance(decision, dict) else {}
    for key in _RESUME_CREDENTIAL_KEYS:
        if key in payload:
            logger.warning("Ignored a credential-looking key on a computer resume")
    return str(payload.get("type") or "").strip().lower()


def _handoff_and_wait(
    session: Any,
    context: Any,
    *,
    message: str | None = None,
) -> dict[str, Any]:
    """Raise the Computer card and return the owner's decision type."""
    card = build_handoff_card(session, context=context, message=message)
    decision = interrupt(card)
    return {"type": _resume_type(decision), "card": card}


def build_computer_tools(
    context: Any,
    *,
    store: Any,
    pool: Any,
    user_id: str,
    assistant_id: str,
    connected_accounts: list[dict[str, Any]],
    allow_interrupt: bool = True,
    is_admin: bool = False,
) -> list[Any]:
    """Build the computer-handoff tools for the administrator's personal avatar."""
    if not is_admin or not agent_computer_is_enabled(context) or not allow_interrupt:
        return []

    async def _raise_until_resolved(
        session: Any,
        *,
        message: str | None = None,
        run_recipe: bool = True,
    ) -> dict[str, Any]:
        while True:
            outcome = _handoff_and_wait(session, context, message=message)
            decision_type = outcome["type"]
            if decision_type == "takeover":
                message = (
                    "The owner is on the computer. Wait until they press "
                    "I'm done, continue — do not assume two-factor finished."
                )
                continue
            if decision_type in ("skip", "cancel"):
                skipped = await skip_handoff(session)
                return {
                    "status": "skipped",
                    "kind": COMPUTER_HANDOFF_INTERRUPT_KIND,
                    "context_closed": False,
                    **skipped,
                }
            finished = await finish_handoff(
                context,
                store,
                session,
                existing_records=connected_accounts,
                pool=pool,
                run_recipe=run_recipe,
            )
            if finished.get("status") == "login_required":
                message = finished.get("message") or (
                    "The page is still a sign-in wall. Take over again, finish "
                    "signing in including any two-factor step, then press I'm done."
                )
                continue
            return {
                "kind": COMPUTER_HANDOFF_INTERRUPT_KIND,
                "context_closed": False,
                **finished,
            }

    @tool
    async def request_computer_handoff(
        task: str,
        start_url: str,
        provider: str,
    ) -> dict[str, Any]:
        """Pause so the owner can take over the avatar's hosted computer and sign in.

        Call this when a vendor dashboard is behind a login wall and there is no
        API key: Cursor, Claude.ai, the OpenAI usage page, ElevenLabs, or xAI.
        The owner takes over that computer, completes any two-factor step on the
        vendor's own page, and presses I'm done. Never ask the owner to type a
        password or a two-factor code into this chat. After I'm done the same
        browser stays open so the next dashboard can be opened immediately.

        Args:
            task: One sentence the Computer card shows (what to sign in to).
            start_url: The dashboard or login page to open.
            provider: cursor, claude_app, openai, elevenlabs, or xai.
        """
        session = await start_computer(
            context,
            user_id=user_id,
            assistant_id=assistant_id,
            start_url=start_url,
            provider=str(provider or "").strip().lower(),
            task=task,
            queue=[
                DashboardStep(
                    provider=str(provider or "").strip().lower(),
                    url=start_url,
                    recipe="",
                    task=task,
                )
            ],
        )
        return await _raise_until_resolved(session, run_recipe=True)

    @tool
    async def walk_vendor_dashboards() -> dict[str, Any]:
        """Walk Cursor spending, Cursor usage, Claude.ai, OpenAI spend-categories, ElevenLabs, and xAI on one computer.

        Use when the owner asks for live vendor spend and several dashboards
        are unsigned. One Google sign-in on this computer is reused where those
        products share it. Skip advances to the next site. I'm done persists
        cookies and pulls that provider immediately, then continues. A login
        wall raises the Computer card again. The browser context stays open
        for the whole walk.
        """
        session = computer_for_user(user_id)
        if session is None:
            session = await start_computer(
                context,
                user_id=user_id,
                assistant_id=assistant_id,
                queue=list(VENDOR_DASHBOARD_QUEUE),
            )
        results: list[dict[str, Any]] = []
        while current_step(session) is not None:
            step = current_step(session)
            needs_login = await page_looks_like_login(session)
            if needs_login:
                outcome = await _raise_until_resolved(
                    session,
                    message=(
                        f"The {step.provider} dashboard is blocked. Take over the "
                        "computer, sign in (including any two-factor step), then "
                        "press I'm done, continue."
                    ),
                    run_recipe=True,
                )
                results.append(outcome)
                if outcome.get("status") == "skipped" and outcome.get("queue_finished"):
                    break
                if outcome.get("status") == "skipped":
                    continue
                if outcome.get("status") == "done":
                    from src.anubis.utils.connected_accounts.agent_computer import (
                        advance_queue,
                        navigate_computer,
                    )

                    next_step = advance_queue(session)
                    if next_step is None:
                        break
                    await navigate_computer(
                        session,
                        next_step.url,
                        provider=next_step.provider,
                        task=next_step.task,
                    )
                    continue
            else:
                from src.anubis.utils.connected_accounts.agent_computer import (
                    advance_queue,
                    navigate_computer,
                    run_recipe_on_computer,
                )

                pulled = await run_recipe_on_computer(session, pool=pool)
                results.append(pulled)
                if pulled.get("status") == "login_required":
                    continue
                next_step = advance_queue(session)
                if next_step is None:
                    break
                await navigate_computer(
                    session,
                    next_step.url,
                    provider=next_step.provider,
                    task=next_step.task,
                )
        return {
            "status": "ok",
            "kind": COMPUTER_HANDOFF_INTERRUPT_KIND,
            "context_closed": False,
            "steps": results,
            "providers_waiting": [],
        }

    return [request_computer_handoff, walk_vendor_dashboards]
