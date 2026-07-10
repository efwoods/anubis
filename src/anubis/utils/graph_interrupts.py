"""Helpers for LangGraph human-in-the-loop interrupt collection and resume."""

from __future__ import annotations

from typing import Any

from langgraph.types import Command


def collect_pending_interrupts(snapshot) -> list:
    """Return any ``Interrupt`` objects pending on a graph ``StateSnapshot``.

    Newer LangGraph exposes ``snapshot.interrupts``; older surfaces them per task.
    """
    interrupts = list(getattr(snapshot, "interrupts", None) or [])
    if interrupts:
        return interrupts
    for task in getattr(snapshot, "tasks", None) or []:
        interrupts.extend(getattr(task, "interrupts", None) or [])
    return interrupts


def build_interrupt_resume_command(
    pending_interrupts: list[Any], decision: Any
) -> Command:
    """Build a ``Command(resume=...)`` for one or more pending interrupts.

    LangGraph requires ``Command(resume={interrupt_id: value, ...})`` when more
    than one interrupt is pending (e.g. parallel ``edit_identity_fact`` tool
    calls). A bare ``Command(resume=value)`` only works for a single interrupt.
    """
    if len(pending_interrupts) > 1:
        return Command(resume={intr.id: decision for intr in pending_interrupts})
    if len(pending_interrupts) == 1:
        return Command(resume={pending_interrupts[0].id: decision})
    return Command(resume=decision)
