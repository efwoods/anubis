"""Render the ``=== HOW YOU MOVE ===`` text, read once per turn.

One primary-key read of ``avatar_motion_profile`` through the published
repository. Anything that goes wrong yields an empty section: the avatar
must never fail to answer because its movement could not be read.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _flag(context: Any, name: str, default: bool = True) -> bool:
    value = getattr(context, name, None) if context is not None else None
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


async def read_how_you_move_section(assistant_id: str | None, context: Any = None) -> str:
    """Return the rendered ROLE section for an avatar, or ``""``."""
    if not assistant_id:
        return ""
    if not _flag(context, "motion_prompt_enabled", True):
        return ""
    from src.anubis.utils.motion.repository import get_motion_repository

    repository = get_motion_repository()
    if repository is None:
        return ""
    try:
        profile = await repository.get_profile(str(assistant_id))
    except Exception as read_error:  # noqa: BLE001 - never fail a turn on this
        logger.info("Could not read the motion profile for %s: %s", assistant_id, read_error)
        return ""
    return str((profile or {}).get("role_section") or "")


async def read_motion_blocks(assistant_id: str | None, context: Any = None) -> dict[str, str]:
    """Return the per-emotion behavioural blocks for generation prompts, or ``{}``."""
    if not assistant_id or not _flag(context, "motion_prompt_enabled", True):
        return {}
    from src.anubis.utils.motion.repository import get_motion_repository

    repository = get_motion_repository()
    if repository is None:
        return {}
    try:
        profile = await repository.get_profile(str(assistant_id))
    except Exception as read_error:  # noqa: BLE001
        logger.info("Could not read the motion blocks for %s: %s", assistant_id, read_error)
        return {}
    blocks = (profile or {}).get("blocks") or {}
    return {str(key): str(value) for key, value in blocks.items() if str(value or "").strip()}


__all__ = ["read_how_you_move_section", "read_motion_blocks"]
