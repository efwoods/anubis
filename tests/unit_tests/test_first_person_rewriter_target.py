"""Unit tests for target-name plumbing through the first-person identity rewriter.

The media pipeline's :class:`FactRewriterClass` deliberately harvests statements
made ABOUT the target by the other people in a recording (see its module
docstring). :class:`FirstPersonRewriterClass` then converts each harvested
statement into first person. When the rewriter did not know the target's name it
applied its "the speaker is always the target" pronoun convention to those
harvested statements, so a castmate's "I have worked with <target> for years and
consult him on electronics" was stored as the target's OWN identity fact — and
the avatar went on to introduce itself as its own colleague.

These tests cover the deterministic plumbing only: that the target name reaches
the system prompt, that the prompt carries the Case D rules that use it, and
that ``_build_identity_documents_from_facts`` forwards the name it already holds
for metadata. The model itself is stubbed, so no live inference runs.
"""

import pytest

from src.anubis.utils.classes.FirstPersonRewriterClass import FirstPersonStatement
from src.anubis.utils.prompts.first_person_rewriter_prompt import (
    FIRST_PERSON_REWRITER_SYSTEM_PROMPT,
)


class _FakeModel:
    """Records every message list it is invoked with and returns a canned statement."""

    def __init__(self, first_person_statement: str = "I wired the electronics."):
        self._statement = first_person_statement
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        # The real structured-output type, so the caller's token accounting
        # (which calls ``model_dump()``) exercises the same path it does live.
        return FirstPersonStatement(first_person_statement=self._statement)


def _build_rewriter(monkeypatch, statement: str = "I wired the electronics."):
    """Return a ``FirstPersonRewriterClass`` whose model is the fake above."""
    import src.anubis.utils.classes.FirstPersonRewriterClass as rewriter_module

    fake_model = _FakeModel(statement)
    monkeypatch.setattr(
        rewriter_module, "init_model", lambda *args, **kwargs: fake_model
    )
    rewriter = rewriter_module.FirstPersonRewriterClass()
    return rewriter, fake_model


def _system_prompt_text(fake_model: _FakeModel) -> str:
    """The rendered system prompt from the fake model's first recorded call."""
    assert fake_model.calls, "the rewriter never invoked the model"
    return fake_model.calls[0][0].content


# --------------------------------------------------------------------------
# The prompt template itself
# --------------------------------------------------------------------------


def test_prompt_template_carries_both_placeholders():
    """The template must interpolate the context summary AND the target name."""
    assert "{concise_context_summary}" in FIRST_PERSON_REWRITER_SYSTEM_PROMPT
    assert "{target_name}" in FIRST_PERSON_REWRITER_SYSTEM_PROMPT


def test_prompt_template_renders_without_stray_braces():
    """``str.format`` must consume every brace — a stray one raises at call time."""
    rendered = FIRST_PERSON_REWRITER_SYSTEM_PROMPT.format(
        concise_context_summary="a colleague describes the working relationship",
        target_name="Grant Imahara",
    )
    assert "Grant Imahara" in rendered
    assert "{" not in rendered
    assert "}" not in rendered


def test_prompt_carries_the_case_d_rules_that_use_the_target_name():
    """The target name is only useful alongside the rules that consult it."""
    assert "<target_identity>" in FIRST_PERSON_REWRITER_SYSTEM_PROMPT
    assert "TARGET_NAME" in FIRST_PERSON_REWRITER_SYSTEM_PROMPT
    assert "Case D" in FIRST_PERSON_REWRITER_SYSTEM_PROMPT
    # The rule that forbids the avatar from naming itself as a third party.
    assert (
        "NEVER produce a statement in which TARGET_NAME refers to somebody"
        in FIRST_PERSON_REWRITER_SYSTEM_PROMPT
    )


# --------------------------------------------------------------------------
# The rewriter class
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_target_name_reaches_the_system_prompt(monkeypatch):
    """The name the caller passes must appear in the prompt the model sees."""
    rewriter, fake_model = _build_rewriter(monkeypatch)

    await rewriter.rewrite(
        ["The speaker has worked with Grant Imahara for years."],
        concise_context_summary="a colleague describes the working relationship",
        target_name="Grant Imahara",
    )

    system_prompt = _system_prompt_text(fake_model)
    assert "Grant Imahara" in system_prompt
    assert "a colleague describes the working relationship" in system_prompt


@pytest.mark.asyncio
async def test_missing_target_name_falls_back_to_a_readable_placeholder(monkeypatch):
    """A blank name must not render an empty section that reads as "no name"."""
    rewriter, fake_model = _build_rewriter(monkeypatch)

    await rewriter.rewrite(
        ["The speaker holds a degree in electrical engineering."],
        concise_context_summary="a biographical profile",
    )

    system_prompt = _system_prompt_text(fake_model)
    assert "target name not supplied" in system_prompt


@pytest.mark.asyncio
async def test_blank_target_name_is_treated_as_missing(monkeypatch):
    """Whitespace is as absent as ``None`` — both take the placeholder path."""
    rewriter, fake_model = _build_rewriter(monkeypatch)

    await rewriter.rewrite(
        ["The speaker holds a degree in electrical engineering."],
        target_name="   ",
    )

    assert "target name not supplied" in _system_prompt_text(fake_model)


@pytest.mark.asyncio
async def test_every_statement_in_a_batch_sees_the_same_target_name(monkeypatch):
    """One model call per statement, each carrying the shared target identity."""
    rewriter, fake_model = _build_rewriter(monkeypatch)

    await rewriter.rewrite(
        [
            "The speaker has worked with Grant Imahara for years.",
            "The speaker consults Grant Imahara about electronics.",
            "Grant Imahara built a robot in under a day.",
        ],
        target_name="Grant Imahara",
    )

    assert len(fake_model.calls) == 3
    for messages in fake_model.calls:
        assert "Grant Imahara" in messages[0].content


@pytest.mark.asyncio
async def test_rewrite_still_pairs_outputs_with_their_source_statements(monkeypatch):
    """Adding the target name must not disturb the provenance pairing."""
    rewriter, fake_model = _build_rewriter(
        monkeypatch, statement="A colleague has worked with me for years."
    )

    response = await rewriter.rewrite(
        ["The speaker has worked with Grant Imahara for years."],
        target_name="Grant Imahara",
    )

    statements = response["statements"]
    assert len(statements) == 1
    assert (
        statements[0]["first_person_statement"]
        == "A colleague has worked with me for years."
    )
    assert (
        statements[0]["original_statement"]
        == "The speaker has worked with Grant Imahara for years."
    )


# --------------------------------------------------------------------------
# The media-pipeline call site
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identity_document_builder_forwards_the_target_name(monkeypatch):
    """``_build_identity_documents_from_facts`` held the name only for metadata.

    It has always written ``target_name`` into each Document's metadata; the
    defect was that it never handed the same value to the rewriter that decides
    whose "I" the stored fact speaks with.
    """
    import src.subgraphs.process_media_graph.utils.helper_functions as helper_functions

    recorded: dict = {}

    class _RecordingRewriter:
        async def rewrite(
            self, statements, concise_context_summary="", target_name=None
        ):
            recorded["target_name"] = target_name
            recorded["statements"] = statements
            return {
                "statements": [
                    {"first_person_statement": "A colleague has worked with me for years."}
                ]
            }

    monkeypatch.setattr(
        helper_functions, "FirstPersonRewriterClass", lambda: _RecordingRewriter()
    )

    documents = await helper_functions._build_identity_documents_from_facts(
        [{"rewritten_statement": "The speaker has worked with Grant Imahara for years."}],
        user_id="user-1",
        assistant_id="assistant-1",
        media_item={"metadata": {"filename": "mythbusters.mp4"}},
        concise_context_summary="a colleague describes the working relationship",
        target_name="Grant Imahara",
    )

    assert recorded["target_name"] == "Grant Imahara"
    assert len(documents) == 1
    # The stored fact speaks as the target, and never names the target as
    # somebody the target worked with.
    assert "A colleague has worked with me for years." in documents[0].page_content
    assert documents[0].metadata["target_name"] == "Grant Imahara"
