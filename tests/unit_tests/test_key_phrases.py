"""Unit tests for the signature key-phrase helpers.

These cover the deterministic, offline stylometry helpers in
:mod:`src.anubis.utils.dataset.key_phrases`:

* ``discover_key_phrases`` surfaces recurring multi-word expressions that are
  over-represented versus the bundled generic-English baseline, ranked by keyness;
* ``key_phrase_occurrence_rate`` counts contiguous phrase matches per total word
  (the scalar behind the ``key_phrase_rate`` stylometric feature).
"""

import pytest

from src.anubis.utils.dataset.key_phrases import (
    discover_key_phrases,
    key_phrase_occurrence_rate,
)


def _casual_quotes():
    return [
        "You know, I just think it works. You know what I mean?",
        "Got it. You know, that is a really good point, got it.",
        "What do ya mean by that? You know, I got it now.",
        "Honestly you know I got it, what do ya mean though.",
        "You know, got it, you know, what do ya mean.",
    ]


def test_discover_key_phrases_surfaces_recurring_markers():
    phrases = discover_key_phrases(_casual_quotes(), min_count=2, top_k=40)
    surfaced = {item["phrase"] for item in phrases}

    # The intended discourse markers are all discovered. (Longer phrases built
    # from rarer words score HIGHER keyness, so bare common-word bigrams like
    # "you know" sit lower in the ranking — hence the generous top_k here.)
    assert "you know" in surfaced
    assert "got it" in surfaced
    assert "what do ya mean" in surfaced


def test_discover_key_phrases_ranked_by_keyness_descending():
    phrases = discover_key_phrases(_casual_quotes(), min_count=2, top_k=15)
    keyness = [item["keyness_log2_over_generic_english"] for item in phrases]
    assert keyness == sorted(keyness, reverse=True)
    # Over-represented phrases have positive keyness (more frequent than a generic
    # writer chaining the same words independently would produce).
    assert keyness[0] > 0


def test_discover_key_phrases_respects_min_count():
    # With min_count above every phrase's frequency, nothing qualifies.
    assert discover_key_phrases(_casual_quotes(), min_count=99, top_k=15) == []


def test_discover_key_phrases_empty_corpus():
    assert discover_key_phrases([], min_count=2) == []
    assert discover_key_phrases(["", "   "], min_count=2) == []


def test_key_phrase_occurrence_rate_counts_matches_per_word():
    # 10 tokens; "you know" occurs twice -> rate 2/10.
    text = "you know I got it, you know what I mean."
    rate = key_phrase_occurrence_rate(text, ["you know"])
    assert rate == pytest.approx(2 / 10)


def test_key_phrase_occurrence_rate_sums_across_phrases():
    # "you know" twice + "got it" once over the same 10 tokens -> 3/10.
    text = "you know I got it, you know what I mean."
    rate = key_phrase_occurrence_rate(text, ["you know", "got it"])
    assert rate == pytest.approx(3 / 10)


def test_key_phrase_occurrence_rate_no_match_is_zero():
    assert key_phrase_occurrence_rate("completely unrelated text here", ["you know"]) == 0.0


def test_key_phrase_occurrence_rate_empty_inputs():
    # Empty/None phrase set or empty text is the neutral 0.0, never an error.
    assert key_phrase_occurrence_rate("some text", None) == 0.0
    assert key_phrase_occurrence_rate("some text", []) == 0.0
    assert key_phrase_occurrence_rate("", ["you know"]) == 0.0
    # Phrases that split to nothing are skipped defensively.
    assert key_phrase_occurrence_rate("some text", ["", "   "]) == 0.0


def test_key_phrase_occurrence_rate_phrase_longer_than_text():
    assert key_phrase_occurrence_rate("you know", ["you know what I mean"]) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
