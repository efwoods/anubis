"""Passive psychological analysis of the target on upload.

Covers what the feature actually promises: that a dimension only ever reads the
target's own words, that a second upload reinforces the first rather than erasing
it, that the graph fans every dimension out concurrently, and that what it learns
reaches the avatar's system prompt.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.psycho.profile import (
    empty_profile,
    merge_findings_into_profile,
    render_profile,
)
from src.subgraphs.psycho_analysis_graph.graph import build_psycho_analysis_graph
from src.subgraphs.psycho_analysis_graph.utils.dimensions import (
    PSYCHOLOGICAL_DIMENSIONS,
    selected_dimensions,
)
from src.subgraphs.psycho_analysis_graph.utils.nodes import select_target_documents


class MemoryStore:
    """The smallest store the psycho nodes need: get and put by namespace and key."""

    def __init__(self):
        self.data: dict = {}

    async def aput(self, namespace, key, value):
        self.data[(tuple(namespace), key)] = value

    async def aget(self, namespace, key):
        value = self.data.get((tuple(namespace), key))
        if value is None:
            return None
        return type("Item", (), {"value": value})()

    async def asearch(self, namespace, query="", limit=10):
        return []


def _long_text(text: str = "I always just fix things for people. ") -> str:
    return text * 20


class FakeTrait:
    first_person_statement = "I fix things for people instead of saying how I feel."
    supporting_evidence = "'let me just sort that out for you'"

    def __init__(self, trait: str, score: float, confidence: float):
        self.trait = trait
        self.score = score
        self.confidence = confidence


class FakeGradedResponse:
    summary_statement = "I show love by doing."

    def __init__(self, traits):
        self.traits = traits


def fake_graded_model(traits):
    class FakeModel:
        async def ainvoke(self, messages):
            return FakeGradedResponse(traits)

    return lambda **kwargs: FakeModel()


# ── target scoping ──────────────────────────────────────────────────────────


def test_only_the_targets_own_words_are_analyzed():
    """Another speaker's words must never become the target's psychology."""
    documents = [
        Document(
            page_content=_long_text(),
            metadata={"target_name": "Evan", "is_target": True},
        ),
        # Somebody else speaking in the same transcript.
        Document(
            page_content=_long_text(),
            metadata={"target_name": "Evan", "is_target": False},
        ),
        # No target identified at all.
        Document(page_content=_long_text(), metadata={}),
        # Reference material the classifier already ruled is not about the target.
        Document(
            page_content=_long_text(),
            metadata={
                "target_name": "Evan",
                "classified_situation": "proprietary_content",
            },
        ),
    ]
    selected = select_target_documents({"documents": documents, "max_documents": 10})[
        "selected_documents"
    ]
    assert len(selected) == 1
    assert selected[0].metadata["is_target"] is True


def test_documents_too_short_to_read_a_person_are_skipped():
    documents = [Document(page_content="hi", metadata={"target_name": "Evan"})]
    assert select_target_documents({"documents": documents})["selected_documents"] == []


def test_selection_is_capped_and_prefers_the_richest_documents():
    documents = [
        Document(page_content="x" * (500 + index), metadata={"target_name": "Evan"})
        for index in range(10)
    ]
    selected = select_target_documents({"documents": documents, "max_documents": 3})[
        "selected_documents"
    ]
    assert len(selected) == 3
    lengths = [len(document.page_content) for document in selected]
    assert lengths == sorted(lengths, reverse=True)


# ── accumulation across uploads ─────────────────────────────────────────────


def test_a_second_upload_reinforces_the_first_rather_than_replacing_it():
    first = {
        "dimension": "love_languages",
        "kind": "graded",
        "traits": {
            "expressing_acts_of_service": {
                "score": 0.9,
                "confidence": 0.8,
                "statement": "I fix things.",
                "evidence": "a",
            }
        },
    }
    second = {
        "dimension": "love_languages",
        "kind": "graded",
        "traits": {
            "expressing_acts_of_service": {
                "score": 0.5,
                "confidence": 0.4,
                "statement": "I help before I speak.",
                "evidence": "b",
            }
        },
    }
    profile = merge_findings_into_profile(None, [first])
    profile = merge_findings_into_profile(profile, [second])
    trait = profile["dimensions"]["love_languages"]["traits"][
        "expressing_acts_of_service"
    ]
    # Confidence-weighted mean: (0.9*0.8 + 0.5*0.4) / 1.2
    assert trait["score"] == pytest.approx((0.9 * 0.8 + 0.5 * 0.4) / 1.2)
    assert trait["observations"] == 2
    assert profile["upload_count"] == 2


def test_a_disagreeing_reading_moves_the_score_without_erasing_it():
    high = {
        "dimension": "attachment_style",
        "kind": "graded",
        "traits": {
            "secure": {
                "score": 1.0,
                "confidence": 1.0,
                "statement": "s",
                "evidence": "",
            }
        },
    }
    low = {
        "dimension": "attachment_style",
        "kind": "graded",
        "traits": {
            "secure": {
                "score": 0.0,
                "confidence": 1.0,
                "statement": "s",
                "evidence": "",
            }
        },
    }
    profile = merge_findings_into_profile(None, [high])
    profile = merge_findings_into_profile(profile, [low])
    score = profile["dimensions"]["attachment_style"]["traits"]["secure"]["score"]
    assert 0.0 < score < 1.0


def test_narrative_statements_accumulate_and_deduplicate():
    finding = {
        "dimension": "defense_mechanisms",
        "kind": "narrative",
        "statements": [
            {
                "statement": "When someone criticizes my work I agree immediately.",
                "evidence": "x",
            },
            {
                "statement": "when someone criticizes my work I agree immediately!",
                "evidence": "y",
            },
        ],
    }
    profile = merge_findings_into_profile(None, [finding])
    profile = merge_findings_into_profile(profile, [finding])
    assert len(profile["dimensions"]["defense_mechanisms"]["statements"]) == 1


def test_a_reading_with_no_confidence_cannot_dominate_the_running_score():
    confident = {
        "dimension": "schwartz_values",
        "kind": "graded",
        "traits": {
            "power": {"score": 0.1, "confidence": 1.0, "statement": "s", "evidence": ""}
        },
    }
    guessed = {
        "dimension": "schwartz_values",
        "kind": "graded",
        "traits": {
            "power": {"score": 1.0, "confidence": 0.0, "statement": "s", "evidence": ""}
        },
    }
    profile = merge_findings_into_profile(None, [confident])
    profile = merge_findings_into_profile(profile, [guessed])
    assert profile["dimensions"]["schwartz_values"]["traits"]["power"]["score"] < 0.2


# ── rendering into the prompt ───────────────────────────────────────────────


def test_traits_the_target_does_not_have_are_not_rendered():
    profile = merge_findings_into_profile(
        None,
        [
            {
                "dimension": "love_languages",
                "kind": "graded",
                "traits": {
                    "expressing_gifts": {
                        "score": 0.02,
                        "confidence": 0.5,
                        "statement": "I give presents.",
                        "evidence": "",
                    },
                    "expressing_acts_of_service": {
                        "score": 0.9,
                        "confidence": 0.9,
                        "statement": "I fix things for people.",
                        "evidence": "",
                    },
                },
            }
        ],
    )
    rendered = profile["value"]
    assert "I fix things for people." in rendered
    assert "I give presents." not in rendered


def test_the_rendered_profile_respects_the_character_ceiling():
    findings = [
        {
            "dimension": f"dimension_{index}",
            "kind": "narrative",
            "statements": [{"statement": "s" * 400, "evidence": ""}],
        }
        for index in range(10)
    ]
    profile = merge_findings_into_profile(None, findings)
    assert len(render_profile(profile, max_characters=800)) <= 800


def test_an_empty_profile_renders_nothing():
    assert render_profile(empty_profile()) == ""


# ── the dimension registry ──────────────────────────────────────────────────


def test_every_registered_dimension_carries_a_target_name_placeholder():
    """A prompt with no target name would analyze whoever the model decided on."""
    for dimension in PSYCHOLOGICAL_DIMENSIONS.values():
        assert "{target_name}" in dimension.system_prompt, dimension.name


def test_every_registered_dimension_forbids_attributing_another_speaker():
    for dimension in PSYCHOLOGICAL_DIMENSIONS.values():
        assert "<RESTRICTIONS>" in dimension.system_prompt, dimension.name
        assert "Never" in dimension.system_prompt, dimension.name


def test_dark_traits_are_off_unless_explicitly_enabled():
    context = GlobalContext()
    context.psychological_analysis_dimensions = ""
    context.enable_dark_trait_analysis = "FALSE"
    assert "dark_traits" not in [d.name for d in selected_dimensions(context)]
    context.enable_dark_trait_analysis = "TRUE"
    assert "dark_traits" in [d.name for d in selected_dimensions(context)]


def test_an_unknown_dimension_name_is_skipped_rather_than_raising():
    context = GlobalContext()
    context.psychological_analysis_dimensions = "love_languages,not_a_dimension"
    assert [d.name for d in selected_dimensions(context)] == ["love_languages"]


# ── the graph ───────────────────────────────────────────────────────────────


def test_the_graph_gives_every_dimension_its_own_concurrent_node():
    graph = build_psycho_analysis_graph()
    nodes = set(graph.get_graph().nodes)
    for name in PSYCHOLOGICAL_DIMENSIONS:
        assert f"analyze_{name}" in nodes


def test_an_upload_produces_documents_a_profile_and_an_emotional_baseline():
    store = MemoryStore()
    document = Document(
        page_content=_long_text(),
        metadata={"target_name": "Evan", "filename": "chat.txt", "is_target": True},
    )
    traits = [FakeTrait("joy", 0.8, 0.7), FakeTrait("trust", 0.6, 0.7)]
    with patch("src.anubis.utils.model.init_model", fake_graded_model(traits)):
        graph = build_psycho_analysis_graph(["emotional_baseline"])
        result = asyncio.run(
            graph.ainvoke(
                {
                    "documents": [document],
                    "max_documents": 5,
                    "creator_id": "creator-1",
                    "assistant_id": "avatar-1",
                    "store": store,
                },
                context=GlobalContext(),
            )
        )
    assert result["psychological_documents"]
    profile = store.data[
        (("creator-1", "avatar-1", "psychological_profile"), "current")
    ]
    assert "emotional_baseline" in profile["dimensions"]
    emotion = store.data[(("creator-1", "avatar-1", "current_emotion"), "current")]
    assert emotion["baseline_wheel"]["joy"] == pytest.approx(0.8)


def test_one_failing_dimension_does_not_fail_the_upload():
    store = MemoryStore()
    document = Document(
        page_content=_long_text(),
        metadata={"target_name": "Evan", "is_target": True},
    )

    class ExplodingModel:
        async def ainvoke(self, messages):
            raise RuntimeError("the provider is down")

    with patch("src.anubis.utils.model.init_model", lambda **kwargs: ExplodingModel()):
        graph = build_psycho_analysis_graph(["love_languages"])
        result = asyncio.run(
            graph.ainvoke(
                {
                    "documents": [document],
                    "max_documents": 5,
                    "creator_id": "creator-1",
                    "assistant_id": "avatar-1",
                    "store": store,
                },
                context=GlobalContext(),
            )
        )
    assert result.get("psychological_documents") in (None, [])


def test_analysis_outputs_are_never_queued_for_analysis_again():
    """An analysis output that stayed analysis_acceptable would loop the pipeline."""
    store = MemoryStore()
    document = Document(
        page_content=_long_text(),
        metadata={
            "target_name": "Evan",
            "is_target": True,
            "analysis_acceptable": True,
        },
    )
    traits = [FakeTrait("expressing_acts_of_service", 0.9, 0.8)]
    with patch("src.anubis.utils.model.init_model", fake_graded_model(traits)):
        graph = build_psycho_analysis_graph(["love_languages"])
        result = asyncio.run(
            graph.ainvoke(
                {
                    "documents": [document],
                    "max_documents": 5,
                    "creator_id": "creator-1",
                    "assistant_id": "avatar-1",
                    "store": store,
                },
                context=GlobalContext(),
            )
        )
    for produced in result["psychological_documents"]:
        assert produced.metadata["analysis_acceptable"] is False
        assert produced.metadata["vectorstore_acceptable"] is True
        assert produced.metadata["namespace"] == "analysis"


def test_one_dimension_cannot_flood_the_profile_from_a_single_upload():
    """Chunks of one transcript describe the same habit in slightly different words."""
    from src.subgraphs.psycho_analysis_graph.utils.nodes import (
        MAX_STATEMENTS_PER_DIMENSION_PER_UPLOAD,
        _merge_dimension_findings,
    )

    findings = [
        {"statements": [{"statement": f"I do habit number {index}.", "evidence": ""}]}
        for index in range(20)
    ]
    merged = _merge_dimension_findings("conversation_subtleties", "narrative", findings)
    assert len(merged["statements"]) == MAX_STATEMENTS_PER_DIMENSION_PER_UPLOAD


def test_the_same_habit_worded_identically_across_chunks_is_kept_once():
    from src.subgraphs.psycho_analysis_graph.utils.nodes import (
        _merge_dimension_findings,
    )

    statement = {
        "statement": "I take the blame so the conversation ends.",
        "evidence": "",
    }
    findings = [{"statements": [statement]} for _ in range(5)]
    merged = _merge_dimension_findings("defense_mechanisms", "narrative", findings)
    assert len(merged["statements"]) == 1


# ── trigger dimensions read the conversation, not the target's chunks ────────
#
# A trigger is a pair: something happens, the target reacts. The analysis queue
# holds the target's answers cut into separate few-hundred-character chunks, and
# an isolated answer almost never contains both halves — measured on one
# interview, the chunked queue yielded a single trigger where the same interview
# as one document yielded eight.


def test_the_trigger_dimension_is_the_one_that_reads_the_whole_conversation():
    assert PSYCHOLOGICAL_DIMENSIONS["dialogue_emotional_triggers"].reads_full_dialogue
    for name, dimension in PSYCHOLOGICAL_DIMENSIONS.items():
        if name == "dialogue_emotional_triggers":
            continue
        # Trait dimensions keep the target-only chunks on purpose: the isolation
        # is what stops another speaker's values being read as the target's.
        assert not dimension.reads_full_dialogue, name


def test_dialogue_documents_are_selected_separately_from_target_documents():
    from src.subgraphs.psycho_analysis_graph.utils.nodes import select_target_documents

    body = "word " * 100
    result = select_target_documents(
        {
            "documents": [
                Document(
                    page_content=body,
                    metadata={"target_name": "Evan", "is_target": True},
                )
            ],
            "dialogue_documents": [
                Document(page_content=body, metadata={"target_name": "Evan"})
            ],
            "max_documents": 5,
        }
    )
    assert len(result["selected_documents"]) == 1
    assert len(result["selected_dialogue_documents"]) == 1


def test_a_dialogue_document_without_a_target_is_not_selected():
    from src.subgraphs.psycho_analysis_graph.utils.nodes import select_target_documents

    body = "word " * 100
    result = select_target_documents(
        {
            "documents": [
                Document(
                    page_content=body,
                    metadata={"target_name": "Evan", "is_target": True},
                )
            ],
            "dialogue_documents": [Document(page_content=body, metadata={})],
            "max_documents": 5,
        }
    )
    assert result["selected_dialogue_documents"] == []


def test_an_upload_with_no_dialogue_falls_back_to_the_target_documents():
    """A monologue or a tweet series carries no exchange; read it anyway."""
    store = MemoryStore()
    document = Document(
        page_content=_long_text(),
        metadata={"target_name": "Evan", "is_target": True},
    )

    class FakeTrigger:
        emotion = "angry"
        trigger_description = "somebody questions whether I earned what I have"
        trigger_occurrence = "'did you really earn that'"
        trigger_speaker = "an interviewer"
        target_response = "I go cold and start listing facts."
        feature_statement = (
            "When somebody questions whether I earned what I have, I go cold."
        )
        supporting_reason = "'I go cold. Real cold.'"

    class FakeResponse:
        triggers = [FakeTrigger()]

    class FakeModel:
        async def ainvoke(self, messages):
            return FakeResponse()

    with patch("src.anubis.utils.model.init_model", lambda **kwargs: FakeModel()):
        graph = build_psycho_analysis_graph(["dialogue_emotional_triggers"])
        result = asyncio.run(
            graph.ainvoke(
                {
                    "documents": [document],
                    "dialogue_documents": [],
                    "max_documents": 5,
                    "creator_id": "creator-fallback",
                    "assistant_id": "avatar-fallback",
                    "store": store,
                },
                context=GlobalContext(),
            )
        )
    produced = result.get("psychological_documents") or []
    assert produced, "a monologue upload must still yield what the target self-reports"
    assert produced[0].metadata["namespace"] == "emotional_trigger"


# ── every prompt is a real prompt ────────────────────────────────────────────
#
# Ten of the twelve pre-existing trait analyzers shipped on a generic
# fill-in-the-blank template whose own docstring called itself a stub. They fed
# the same ANALYZED TRAITS section the avatar reads, so ten of the identity
# dimensions were being read by a placeholder.


def test_no_trait_analyzer_still_runs_on_the_generic_stub():
    from src.anubis.utils.analysis.analysis_methods import _NARRATIVE_ANALYZER_SPECS
    from src.anubis.utils.prompts.psycho_analysis.latent_feature_analysis_prompts import (
        build_stub_feature_prompt,
    )

    skeleton = build_stub_feature_prompt("X", "d", "e").split("<task>")[1][:200]
    stubbed = [
        name
        for name, (noun, prompt) in _NARRATIVE_ANALYZER_SPECS.items()
        if skeleton.replace("X", noun) in prompt
    ]
    assert stubbed == [], f"still on the generic stub: {stubbed}"


def test_every_trait_analyzer_names_the_target_and_guards_attribution():
    from src.anubis.utils.analysis.analysis_methods import _NARRATIVE_ANALYZER_SPECS

    for name, (_noun, prompt) in _NARRATIVE_ANALYZER_SPECS.items():
        assert "{target_name}" in prompt, name
        assert "anti_patterns" in prompt, name
        # Every one must forbid reading another speaker's trait as the target's;
        # that is the failure that quietly corrupts an avatar's identity.
        assert "another" in prompt or "another person" in prompt, name


def test_cognitive_style_is_registered_and_covers_reasoning_and_bias():
    dimension = PSYCHOLOGICAL_DIMENSIONS["cognitive_style"]
    prompt = dimension.system_prompt
    for trait in (
        "analytical_over_intuitive",
        "tolerance_for_ambiguity",
        "revises_under_evidence",
        "confirmation_bias",
        "sunk_cost",
    ):
        assert trait in prompt, trait
