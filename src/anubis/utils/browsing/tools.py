"""The tool that lets the owner ask what their browsing says about them.

The background sweeper keeps the avatar current on its own. This is the other
half: the owner asking directly — "what has my browsing said about me lately",
"read the last month and tell me what you learned", "analyse my browsing on
the desktop" — and getting an answer in the same turn.

The tool runs the same pass the sweeper runs, with the thresholds waived,
because a person who asked has already decided the pass is worth running.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)


def build_browsing_insight_tools(
    context: Any,
    *,
    store: Any,
    user_id: str,
    assistant_id: str,
    target_name: str = "",
) -> list[Any]:
    """Build the browsing-analysis tool for one turn, scoped to one owner and avatar."""

    @tool("analyze_browsing_history")
    async def analyze_browsing_history(days: int = 30, device_label: str = "") -> dict:
        """Read the owner's recent web browsing and learn what it says about them.

        Use when the owner asks what their browsing says about them, asks the
        avatar to catch up on what they have been reading or working on, or
        asks for a report on their own browsing. The findings are stored: facts
        about the owner become part of what the avatar knows, and traits fold
        into the owner's psychological profile.

        Args:
            days: How far back to read when this machine has never been read
                before. Ignored once a machine has a watermark, because then
                only genuinely new browsing is read.
            device_label: One machine to read. Omitted, every machine of the
                owner's that is online is read.
        """
        from src.anubis.utils.browsing.history_client import online_connections
        from src.anubis.utils.browsing.sweeper import analyse_machine

        connections = await online_connections(store, user_id, assistant_id)
        if not connections:
            return {
                "analysed": False,
                "message": (
                    "None of your machines are connected right now, so there is no "
                    "browsing history to read. Start the Neural Nexus connector on "
                    "the machine you browse with."
                ),
            }
        wanted = str(device_label or "").strip().lower()
        if wanted:
            connections = [
                connection
                for connection in connections
                if str(connection.device_label or "").lower() == wanted
            ]
            if not connections:
                return {
                    "analysed": False,
                    "message": f"No connected machine is named {device_label!r}.",
                }
        # ``days`` only matters for a machine with no watermark; a machine that
        # has been read before continues from where the last pass stopped.
        pass_context = _context_with_backfill(context, days)
        outcomes = []
        for connection in connections:
            outcomes.append(
                await analyse_machine(
                    store,
                    connection,
                    context=pass_context,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    target_name=target_name,
                    force=True,
                )
            )
        analysed = [outcome for outcome in outcomes if outcome.get("analysed")]
        if not analysed:
            return {
                "analysed": False,
                "machines": outcomes,
                "message": _nothing_to_report(outcomes),
            }
        return {
            "analysed": True,
            "machines": outcomes,
            "facts_learned": [
                fact for outcome in analysed for fact in (outcome.get("facts_written") or [])
            ],
            "summary": "\n\n".join(
                outcome.get("summary_markdown", "") for outcome in analysed
            ).strip(),
        }

    return [analyze_browsing_history]


class _BackfillOverride:
    """The global context, with the backfill window the caller asked for.

    A shallow proxy rather than a copy: the context is a dataclass with many
    fields, and only one of them is being overridden for the length of one
    tool call.
    """

    def __init__(self, context: Any, backfill_days: int) -> None:
        self._context = context
        self._backfill_days = backfill_days

    def __getattr__(self, name: str) -> Any:
        if name == "browsing_insights_backfill_days":
            return self._backfill_days
        return getattr(self._context, name)


def _context_with_backfill(context: Any, days: int) -> Any:
    """Return the context to run one asked-for pass with."""
    try:
        window = max(1, min(int(days or 30), 3650))
    except (TypeError, ValueError):
        window = 30
    return _BackfillOverride(context, window)


def _nothing_to_report(outcomes: list[dict[str, Any]]) -> str:
    """One sentence saying why a pass found nothing, in the owner's terms."""
    reasons = [
        f"{outcome.get('device_label') or 'a machine'}: {outcome.get('reason')}"
        for outcome in outcomes
        if outcome.get("reason")
    ]
    if not reasons:
        return "There was no new browsing to read."
    return "Nothing new to read — " + "; ".join(reasons) + "."
