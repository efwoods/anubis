"""Unit tests for the suggested-context handling in ``_suggest_correction``.

The HITL panel's editable window shows a suggested edit for BOTH the fact content and the
fact context. The context suggestion has a known failure mode: a stored provenance context
such as 'The user said: "You don't have a favorite color."' reads as "still true" under a
minimal-edit instruction, so the suggestion model returns the context unchanged — leaving a
suggested context that contradicts the suggested fact. ``ProposedFactEdit`` now carries a
``context_asserts_inaccurate_fact`` flag and ``_suggest_correction`` applies a deterministic
backstop: a flagged-but-unchanged context is replaced with the fresh correction context from
the user's most recent message. The suggestion model is stubbed so the backstop plumbing is
exercised without a live model.
"""

from types import SimpleNamespace

import pytest

import src.anubis.utils.tools.identity.identity_tools as identity_tools
from src.anubis.utils.tools.identity.identity_tools import (
    FactMatch,
    ProposedFactEdit,
    _suggest_correction,
    wrap_fact_with_context,
)

STORED_FACT = "I don't have a favorite color."
STALE_CONTEXT = 'The user said: "You don\'t have a favorite color."'
CORRECTED_FACT = "My favorite color is yellow."
FRESH_CONTEXT = "The user told me my favorite color is yellow."
REWRITTEN_CONTEXT = 'The user said: "Your favorite color is yellow."'


def _make_match() -> FactMatch:
    item = SimpleNamespace(
        value={
            "document": {
                "kwargs": {
                    "page_content": wrap_fact_with_context(STORED_FACT, STALE_CONTEXT),
                    "metadata": {"fact": STORED_FACT, "fact_context": STALE_CONTEXT},
                }
            }
        },
        namespace=("creator-1", "asst-1", "identity_memory"),
        key="doc-1",
    )
    return FactMatch(
        item=item,
        namespace=item.namespace,
        key=item.key,
        kind="fact",
        matched_text=STORED_FACT,
        score=0.99,
    )


class _FakeModel:
    def __init__(self, result: ProposedFactEdit):
        self._result = result
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._result


def _stub_model(monkeypatch, result: ProposedFactEdit) -> _FakeModel:
    model = _FakeModel(result)
    monkeypatch.setattr(identity_tools, "init_model", lambda **kwargs: model)
    return model


async def _suggest(match):
    return await _suggest_correction(
        match,
        inaccurate_information=STORED_FACT,
        corrected_information=CORRECTED_FACT,
        correction_context=FRESH_CONTEXT,
        is_deletion=False,
    )


@pytest.mark.asyncio
async def test_rewritten_context_passes_through(monkeypatch):
    _stub_model(
        monkeypatch,
        ProposedFactEdit(
            asserts_inaccurate_fact=True,
            corrected_text=CORRECTED_FACT,
            corrected_context=REWRITTEN_CONTEXT,
            context_asserts_inaccurate_fact=True,
        ),
    )
    corrected_text, corrected_context, asserts_flag = await _suggest(_make_match())
    assert corrected_text == CORRECTED_FACT
    assert corrected_context == REWRITTEN_CONTEXT
    assert asserts_flag is True


@pytest.mark.asyncio
async def test_flagged_unchanged_context_is_replaced_with_fresh_context(monkeypatch):
    """The model says the stored context asserts the inaccurate fact yet returns the context
    unchanged — the backstop swaps in the fresh correction context so the suggested context
    never contradicts the suggested fact."""
    _stub_model(
        monkeypatch,
        ProposedFactEdit(
            asserts_inaccurate_fact=True,
            corrected_text=CORRECTED_FACT,
            corrected_context=STALE_CONTEXT,
            context_asserts_inaccurate_fact=True,
        ),
    )
    _corrected_text, corrected_context, _asserts_flag = await _suggest(_make_match())
    assert corrected_context == FRESH_CONTEXT


@pytest.mark.asyncio
async def test_unflagged_unchanged_context_is_kept(monkeypatch):
    """A context the model did NOT flag (the fact was edited inside a long story context
    that never mentions the inaccurate claim) stays as returned — the backstop only fires
    on the flagged-but-unchanged combination."""
    _stub_model(
        monkeypatch,
        ProposedFactEdit(
            asserts_inaccurate_fact=True,
            corrected_text=CORRECTED_FACT,
            corrected_context=STALE_CONTEXT,
            context_asserts_inaccurate_fact=False,
        ),
    )
    _corrected_text, corrected_context, _asserts_flag = await _suggest(_make_match())
    assert corrected_context == STALE_CONTEXT


@pytest.mark.asyncio
async def test_empty_model_context_falls_back_to_fresh_context(monkeypatch):
    _stub_model(
        monkeypatch,
        ProposedFactEdit(
            asserts_inaccurate_fact=True,
            corrected_text=CORRECTED_FACT,
            corrected_context="",
            context_asserts_inaccurate_fact=False,
        ),
    )
    _corrected_text, corrected_context, _asserts_flag = await _suggest(_make_match())
    assert corrected_context == FRESH_CONTEXT


@pytest.mark.asyncio
async def test_model_receives_the_fresh_correction_context(monkeypatch):
    model = _stub_model(
        monkeypatch,
        ProposedFactEdit(
            asserts_inaccurate_fact=True,
            corrected_text=CORRECTED_FACT,
            corrected_context=REWRITTEN_CONTEXT,
            context_asserts_inaccurate_fact=True,
        ),
    )
    await _suggest(_make_match())
    (_system_message, human_message) = model.calls[0]
    assert f"USER_CORRECTION_CONTEXT: {FRESH_CONTEXT}" in human_message.content
    assert f"ORIGINAL_CONTEXT: {STALE_CONTEXT}" in human_message.content


@pytest.mark.asyncio
async def test_model_error_falls_back_to_global_correction(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(identity_tools, "init_model", _boom)
    corrected_text, corrected_context, asserts_flag = await _suggest(_make_match())
    assert corrected_text == CORRECTED_FACT
    assert corrected_context == FRESH_CONTEXT
    assert asserts_flag is True
