"""Unit tests for the judging stage of signature key-phrase discovery.

The judge is the precision stage: the statistics cannot tell a stance marker
from a subject word, so a model classifies each shortlisted candidate. These
tests pin the three properties that make that call safe to run inside an upload
calibration — it fails open, it caches, and it is capped — plus the filtering
rules that decide what actually reaches the avatar's voice profile.

No test here makes a real model call; ``init_model`` is faked throughout.
"""

import asyncio

import pytest

from src.anubis.utils.dataset.key_phrase_candidates import KeyPhraseCandidate
from src.anubis.utils.dataset.key_phrase_judgement import (
    MODEL_JUDGEMENT_SOURCE,
    STAGE_ONE_FALLBACK_SOURCE,
    SignaturePhraseJudgement,
    SignaturePhraseJudgementResponse,
    build_signature_key_phrase_profile,
    judge_key_phrase_candidates,
)


def _candidate(phrase, *, log_ratio=5.0, occurrences=10):
    """A candidate with plausible statistics; only the phrase usually matters."""
    return KeyPhraseCandidate(
        phrase=phrase,
        ngram_size=len(phrase.split()),
        occurrence_count=occurrences,
        dispersion_unit_count=occurrences,
        target_relative_frequency=0.001,
        reference_expected_count=0.5,
        dunning_log_likelihood=25.0,
        log_ratio_over_reference=log_ratio,
        concordance_lines=(f"some words [{phrase}] more words",),
    )


class _FakeModel:
    """Stands in for the structured-output runnable ``init_model`` returns."""

    def __init__(
        self, judgements_by_phrase, *, raise_error=None, delay=0.0, answer_limit=None
    ):
        self.judgements_by_phrase = judgements_by_phrase
        self.raise_error = raise_error
        self.delay = delay
        # When set, the model answers only this many of the phrases it was
        # asked about — the real truncation behaviour that made most of a
        # shortlist fall open unjudged.
        self.answer_limit = answer_limit
        self.invocation_count = 0
        self.batch_sizes = []

    async def ainvoke(self, input):  # noqa: A002 - matches the runnable signature
        self.invocation_count += 1
        self.batch_sizes.append(len(getattr(self, "_last_requested", []) or []))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_error is not None:
            raise self.raise_error
        records = self.judgements_by_phrase
        if self.answer_limit is not None:
            records = records[: self.answer_limit]
        return SignaturePhraseJudgementResponse(
            judgements=[SignaturePhraseJudgement(**record) for record in records]
        )


@pytest.fixture
def fake_model(monkeypatch):
    """Install a fake ``init_model`` and hand the test the model it returns."""
    holder = {}

    def _install(
        judgements_by_phrase, *, raise_error=None, delay=0.0, answer_limit=None
    ):
        model = _FakeModel(
            judgements_by_phrase,
            raise_error=raise_error,
            delay=delay,
            answer_limit=answer_limit,
        )
        holder["model"] = model

        import src.anubis.utils.model as model_module

        monkeypatch.setattr(
            model_module, "init_model", lambda **keyword_arguments: model
        )
        return model

    return _install


# ---------------------------------------------------------------------------
# Filtering: what the judge lets through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_signature_style_survives(fake_model):
    """Subject matter, generic English and boilerplate are all discarded."""
    candidates = [
        _candidate("concerning"),
        _candidate("civilization"),
        _candidate("great"),
        _candidate("sign up via web"),
    ]
    fake_model(
        [
            {"phrase": "concerning", "classification": "signature_style",
             "confidence": "high", "reason": "stance marker"},
            {"phrase": "civilization", "classification": "topic_or_content",
             "confidence": "high", "reason": "names a subject"},
            {"phrase": "great", "classification": "generic_english",
             "confidence": "high", "reason": "anyone says this"},
            {"phrase": "sign up via web", "classification": "boilerplate",
             "confidence": "high", "reason": "advertising copy"},
        ]
    )

    judgements = await judge_key_phrase_candidates(
        candidates, speaker_name="Someone"
    )
    kept = [
        phrase
        for phrase, judgement in judgements.items()
        if judgement["classification"] == "signature_style"
    ]
    assert kept == ["concerning"]


@pytest.mark.asyncio
async def test_low_confidence_style_is_not_written_into_the_profile(
    fake_model, monkeypatch
):
    """The prompt asks for "low" when unsure, and an unsure inclusion is costly."""
    fake_model(
        [
            {"phrase": "concerning", "classification": "signature_style",
             "confidence": "low", "reason": "not sure"},
            {"phrase": "sigh", "classification": "signature_style",
             "confidence": "high", "reason": "clear interjection"},
        ]
    )
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [
            _candidate("concerning"), _candidate("sigh")
        ],
    )

    profile = await build_signature_key_phrase_profile(["some quotes"])
    assert profile.phrases == ["sigh"]


@pytest.mark.asyncio
async def test_a_phrase_returned_with_stray_whitespace_still_counts(fake_model):
    """The judge pads its echo of the phrase; an exact-match guard loses the verdict.

    Observed against a live model: asked about "want you to", it answered
    "want you to " with a trailing space. Exact equality discarded every verdict
    in the batch, the batch was reported as entirely unjudged, and a real avatar
    fell back to its raw statistical shortlist while the model had in fact
    classified all forty-five of its candidates correctly.
    """
    fake_model(
        [
            {"phrase": "want you to ", "classification": "signature_style",
             "confidence": "high", "reason": "trailing space"},
            {"phrase": "  SIGH  ", "classification": "signature_style",
             "confidence": "high", "reason": "padded and upper-cased"},
        ]
    )
    judgements = await judge_key_phrase_candidates(
        [_candidate("want you to"), _candidate("sigh")], speaker_name=None
    )

    # Keyed by the CANDIDATE's spelling, never the model's echo of it.
    assert set(judgements) == {"want you to", "sigh"}
    assert all(
        judgement["judgement_source"] == MODEL_JUDGEMENT_SOURCE
        for judgement in judgements.values()
    )


@pytest.mark.asyncio
async def test_a_phrase_the_batch_never_asked_about_is_discarded(fake_model):
    """A hallucinated phrase must never reach the voice profile."""
    fake_model(
        [
            {"phrase": "sigh", "classification": "signature_style",
             "confidence": "high", "reason": "real candidate"},
            {"phrase": "never asked about this", "classification": "signature_style",
             "confidence": "high", "reason": "invented"},
        ]
    )
    judgements = await judge_key_phrase_candidates(
        [_candidate("sigh")], speaker_name=None
    )
    assert set(judgements) == {"sigh"}


# ---------------------------------------------------------------------------
# Failing open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_exception_falls_back_to_the_statistical_shortlist(fake_model):
    """A classifier outage must degrade the list, never empty it."""
    fake_model([], raise_error=RuntimeError("provider is down"))
    candidates = [_candidate("sigh"), _candidate("concerning")]

    judgements = await judge_key_phrase_candidates(candidates, speaker_name=None)

    assert set(judgements) == {"sigh", "concerning"}
    assert all(
        judgement["judgement_source"] == STAGE_ONE_FALLBACK_SOURCE
        for judgement in judgements.values()
    )


@pytest.mark.asyncio
async def test_a_timeout_falls_back_to_the_statistical_shortlist(fake_model):
    """The judge must not consume the much larger calibration budget."""
    fake_model([], delay=5.0)
    judgements = await judge_key_phrase_candidates(
        [_candidate("sigh")], speaker_name=None, timeout_seconds=0.05
    )
    assert judgements["sigh"]["judgement_source"] == STAGE_ONE_FALLBACK_SOURCE


@pytest.mark.asyncio
async def test_a_failed_judgement_is_never_cached(fake_model, monkeypatch):
    """A fail-open record must be retried next time, not remembered as a verdict."""
    fake_model([], raise_error=RuntimeError("provider is down"))
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [_candidate("sigh")],
    )

    profile = await build_signature_key_phrase_profile(["some quotes"])
    # The phrase still reaches the profile, because failing open keeps the
    # statistical shortlist...
    assert profile.phrases == ["sigh"]
    # ...but nothing was learned, so the next calibration asks again.
    assert "sigh" not in profile.judgement_cache


@pytest.mark.asyncio
async def test_model_confirmed_phrases_outrank_fail_open_ones(fake_model, monkeypatch):
    """An outage degrades the tail of the list rather than displacing verdicts."""
    confirmed = _candidate("sigh", log_ratio=1.0)
    unconfirmed = _candidate("hmm", log_ratio=9.0)
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [unconfirmed, confirmed],
    )
    fake_model([])  # returns no judgements at all

    profile = await build_signature_key_phrase_profile(
        ["some quotes"],
        previous_profile_detail={
            "phrases": [],
            "judgement_cache": {
                "sigh": {
                    "classification": "signature_style",
                    "confidence": "high",
                    "reason": "cached verdict",
                    "judgement_source": MODEL_JUDGEMENT_SOURCE,
                }
            },
        },
    )
    # "hmm" scores far higher but was never confirmed, so it ranks below "sigh".
    assert profile.phrases == ["sigh", "hmm"]


# ---------------------------------------------------------------------------
# Caching, the gate, and the cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_fully_cached_shortlist_makes_no_model_call(fake_model):
    """Re-judging a stable corpus would make the stored phrase set drift."""
    model = fake_model([])
    cache = {
        "sigh": {
            "classification": "signature_style",
            "confidence": "high",
            "reason": "cached",
            "judgement_source": MODEL_JUDGEMENT_SOURCE,
        }
    }
    judgements = await judge_key_phrase_candidates(
        [_candidate("sigh")], speaker_name=None, previously_judged=cache
    )
    assert model.invocation_count == 0
    assert judgements["sigh"]["reason"] == "cached"


@pytest.mark.asyncio
async def test_the_environment_gate_skips_judging_entirely(
    fake_model, monkeypatch
):
    """FALSE is the rollback: keep the shortlist, make no calls."""
    model = fake_model([])
    monkeypatch.setenv("KEY_PHRASE_JUDGEMENT_ENABLED", "FALSE")
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [_candidate("sigh")],
    )

    profile = await build_signature_key_phrase_profile(["some quotes"])
    assert model.invocation_count == 0
    assert profile.phrases == ["sigh"]
    assert profile.entries[0]["judgement_source"] == STAGE_ONE_FALLBACK_SOURCE


@pytest.mark.asyncio
async def test_the_stored_set_is_capped(fake_model, monkeypatch):
    """The cap is what stops the list growing by an upload's worth every time."""
    monkeypatch.setenv("KEY_PHRASE_PROFILE_MAXIMUM_PHRASES", "25")
    many = [_candidate(f"phrase{index}", log_ratio=10.0 - index * 0.01) for index in range(200)]
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: many,
    )
    fake_model([])  # nothing confirmed; everything falls open and still gets capped

    profile = await build_signature_key_phrase_profile(["some quotes"])
    assert len(profile.phrases) == 25


@pytest.mark.asyncio
async def test_an_empty_result_preserves_the_previous_profile(monkeypatch):
    """Never destroy a working voice profile on a run that found nothing."""
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [],
    )
    profile = await build_signature_key_phrase_profile(
        [""], previous_profile_detail={"phrases": ["tricky", "sigh"]}
    )
    assert profile.phrases == ["tricky", "sigh"]


@pytest.mark.asyncio
async def test_incumbents_win_a_tie_but_do_not_win_outright(fake_model, monkeypatch):
    """The bonus stops churn at the cap without freezing the list."""
    monkeypatch.setenv("KEY_PHRASE_PROFILE_MAXIMUM_PHRASES", "1")
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [
            _candidate("newcomer", log_ratio=5.1),
            _candidate("incumbent", log_ratio=5.0),
        ],
    )
    fake_model(
        [
            {"phrase": "newcomer", "classification": "signature_style",
             "confidence": "high", "reason": "yes"},
            {"phrase": "incumbent", "classification": "signature_style",
             "confidence": "high", "reason": "yes"},
        ]
    )

    # A 0.1 gap is inside the 0.25 bonus, so the incumbent holds its slot.
    profile = await build_signature_key_phrase_profile(
        ["some quotes"], previous_profile_detail={"phrases": ["incumbent"]}
    )
    assert profile.phrases == ["incumbent"]


@pytest.mark.asyncio
async def test_a_clearly_better_newcomer_displaces_an_incumbent(
    fake_model, monkeypatch
):
    """The bonus damps churn; it must not freeze the list against real evidence."""
    monkeypatch.setenv("KEY_PHRASE_PROFILE_MAXIMUM_PHRASES", "1")
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [
            _candidate("newcomer", log_ratio=8.0),
            _candidate("incumbent", log_ratio=5.0),
        ],
    )
    fake_model(
        [
            {"phrase": "newcomer", "classification": "signature_style",
             "confidence": "high", "reason": "yes"},
            {"phrase": "incumbent", "classification": "signature_style",
             "confidence": "high", "reason": "yes"},
        ]
    )

    # A 3.0 gap dwarfs the 0.25 bonus, so the better phrase takes the slot.
    profile = await build_signature_key_phrase_profile(
        ["some quotes"], previous_profile_detail={"phrases": ["incumbent"]}
    )
    assert profile.phrases == ["newcomer"]


# ---------------------------------------------------------------------------
# Incomplete answers
#
# A structured-output call does not reliably return one record per input. On a
# real 125-candidate shortlist a batch of forty came back covering only part of
# itself, and because a missing record looks exactly like a refusal, most of the
# shortlist was falling open unjudged and never reaching the profile.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidates_a_batch_leaves_out_are_asked_about_again(fake_model):
    """A skipped phrase must be re-asked, not silently discarded."""
    candidates = [_candidate("sigh"), _candidate("tbh"), _candidate("concerning")]
    model = fake_model(
        [
            {"phrase": "sigh", "classification": "signature_style",
             "confidence": "high", "reason": "interjection"},
            {"phrase": "tbh", "classification": "signature_style",
             "confidence": "high", "reason": "shorthand"},
            {"phrase": "concerning", "classification": "signature_style",
             "confidence": "high", "reason": "stance marker"},
        ],
        answer_limit=1,  # the model answers one phrase per call and stops
    )

    judgements = await judge_key_phrase_candidates(
        candidates, speaker_name=None, batch_size=3
    )

    assert model.invocation_count > 1, "the unjudged remainder was never re-asked"
    assert judgements["sigh"]["judgement_source"] == MODEL_JUDGEMENT_SOURCE


@pytest.mark.asyncio
async def test_a_round_that_judges_nothing_stops_the_retry_loop(fake_model):
    """Retrying a call that answers nothing would just burn the budget."""
    model = fake_model([], answer_limit=0)
    candidates = [_candidate(f"phrase{index}") for index in range(6)]

    judgements = await judge_key_phrase_candidates(
        candidates, speaker_name=None, batch_size=3
    )

    # Two batches in the first round, then the loop gives up.
    assert model.invocation_count == 2
    assert all(
        judgement["judgement_source"] == STAGE_ONE_FALLBACK_SOURCE
        for judgement in judgements.values()
    )


@pytest.mark.asyncio
async def test_a_judge_that_rejects_everything_clears_the_previous_phrases(
    fake_model, monkeypatch
):
    """Rejecting every candidate is a real verdict and must replace the old set.

    An earlier version preserved the previous phrases here. On live data that
    pinned the worst profiles to their worst phrases: avatars whose stored lists
    were film-credit fragments and bare grammar were re-judged, had every
    candidate correctly rejected, and had the junk written straight back — the
    profiles that most needed clearing were the only ones that never could be.
    """
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [
            _candidate("and he"), _candidate("credits director taylor")
        ],
    )
    fake_model(
        [
            {"phrase": "and he", "classification": "generic_english",
             "confidence": "high", "reason": "bare grammar"},
            {"phrase": "credits director taylor", "classification": "topic_or_content",
             "confidence": "high", "reason": "names people"},
        ]
    )

    profile = await build_signature_key_phrase_profile(
        ["some quotes"],
        previous_profile_detail={"phrases": ["and he", "credits director taylor"]},
    )
    assert profile.phrases == []


@pytest.mark.asyncio
async def test_an_unreadable_corpus_still_preserves_the_previous_phrases(monkeypatch):
    """The transient-failure guard stays: no candidates means we could not judge."""
    monkeypatch.setattr(
        "src.anubis.utils.dataset.key_phrase_judgement.discover_key_phrase_candidates",
        lambda documents, **keyword_arguments: [],
    )
    profile = await build_signature_key_phrase_profile(
        [""], previous_profile_detail={"phrases": ["tricky"]}
    )
    assert profile.phrases == ["tricky"]
