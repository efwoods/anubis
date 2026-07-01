"""Unit tests for the capture-only key-phrase + function-word features.

These cover the deterministic, offline stylometry helpers in
:mod:`src.anubis.utils.dataset.key_phrases`:

* ``discover_key_phrases`` surfaces recurring multi-word expressions that are
  over-represented versus the bundled generic-English baseline, ranked by keyness;
* ``function_word_frequencies`` reports closed-class function-word rates.
"""

import pytest

from src.anubis.utils.dataset.key_phrases import (
    discover_key_phrases,
    function_word_frequencies,
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


def test_function_word_frequencies_basic_shape():
    result = function_word_frequencies("I think you know it is fine.")
    assert result["token_total"] == 7
    assert 0.0 <= result["function_word_token_share"] <= 1.0
    # Function words (i, you, it, is) are counted; a content word ("think") is not.
    assert "think" not in result["function_word_rates"]
    assert "you" in result["function_word_rates"]


def test_function_word_frequencies_empty_text():
    result = function_word_frequencies("")
    assert result["token_total"] == 0
    assert result["function_word_rates"] == {}
    assert result["function_word_token_share"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
