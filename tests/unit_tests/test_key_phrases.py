"""Unit tests for the signature key-phrase helpers.

These cover the deterministic, offline stylometry helpers in
:mod:`src.anubis.utils.dataset.key_phrases`:

* ``discover_key_phrases`` surfaces recurring multi-word expressions that are
  over-represented versus the bundled generic-English baseline, ranked by keyness;
* ``key_phrase_occurrence_rate`` counts contiguous phrase matches per total word
  (the scalar behind the ``key_phrase_rate`` stylometric feature).
"""

import pytest

from src.anubis.utils.dataset.burrows_delta import tokenize
from src.anubis.utils.dataset.key_phrases import (
    discover_key_phrases,
    key_phrase_occurrence_rate,
    phrase_is_well_formed,
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


""" ------------------------------------------------------------------ """
""" Markup-debris hygiene of the discovered phrase set                 """
""" ------------------------------------------------------------------ """


def test_tokenize_keeps_curly_apostrophe_words_whole():
    # ChatGPT prose uses the Unicode curly apostrophe; both variants must
    # produce the identical single token, or discovered phrases degrade to
    # apostrophe-less shrapnel like "don t" that then never matches.
    assert tokenize("Don’t overthink it") == tokenize("Don't overthink it")
    assert tokenize("don’t")[0] == "don't"


def test_discover_key_phrases_ignores_urls_mentions_and_entities():
    # Raw tweets: t.co links, @mention chains, and &amp; previously dominated
    # discovery as top-keyness junk ("https t co ...", "amp ...").
    tweets = [
        "Great launch today &amp; more to come https://t.co/AbC123 @SpaceX @flcnhvy",
        "Great launch today &amp; more to come https://t.co/XyZ789 @SpaceX @flcnhvy",
        "Great launch today &amp; more to come https://t.co/QrS456 @SpaceX @flcnhvy",
    ]
    phrases = discover_key_phrases(tweets, min_count=2, top_k=40)
    for item in phrases:
        phrase_tokens = set(item["phrase"].split())
        assert not phrase_tokens & {"https", "t", "co", "amp", "spacex", "flcnhvy"}, item
    # The genuine recurring speech survives.
    assert any("great launch" in item["phrase"] for item in phrases)


def test_discovered_phrases_score_on_cleaned_text():
    # A phrase discovered from raw tweets must MATCH when the rate is later
    # measured on the clean_text()'d version of the same writing — this is the
    # discovery/measurement consistency the avatar rate depends on.
    from src.anubis.utils.dataset.style_features import clean_text

    tweets = [
        "you know it works https://t.co/AbC @someone",
        "you know it works https://t.co/DeF @someone",
        "you know it works https://t.co/GhI @someone",
    ]
    phrases = [item["phrase"] for item in discover_key_phrases(tweets, min_count=2)]
    assert phrases
    rate = key_phrase_occurrence_rate(clean_text(tweets[0]), phrases)
    assert rate > 0.0


def test_corpus_attestation_drops_mention_chain_artifacts():
    # Phrases mined from RAW text before discovery cleaned its corpus include
    # @mention chains whose tokens look like real words ("cb doge tesla
    # mayemusk"). No shape filter can reject those — but they never occur in
    # the CLEANED corpus, so attestation removes them while keeping phrases
    # the avatar actually says.
    from src.anubis.utils.dataset.key_phrases import (
        build_corpus_phrase_attestation_set,
    )

    corpus = [
        "you know it works @cb_doge @Tesla @MayeMusk",
        "you know it works, honestly",
    ]
    attested = build_corpus_phrase_attestation_set(corpus)
    assert "you know" in attested
    assert "you know it works" in attested
    assert "cb doge tesla mayemusk" not in attested
    assert "doge tesla" not in attested


def test_phrase_is_well_formed():
    assert phrase_is_well_formed("you know")
    assert phrase_is_well_formed("don't have personal favorites")
    assert phrase_is_well_formed("a clear next step")   # "a" is a real word
    assert not phrase_is_well_formed("https t co")
    assert not phrase_is_well_formed("amp more to come")
    assert not phrase_is_well_formed("b https t co")
    assert not phrase_is_well_formed("video https t co")
    assert not phrase_is_well_formed("x marks the spot")  # lone "x" is shrapnel
    assert not phrase_is_well_formed("")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
