"""Unit tests for the statistical stage of signature key-phrase discovery.

These cover :mod:`src.anubis.utils.dataset.key_phrase_candidates`, which is
deterministic, offline and free of model calls. The behaviours asserted here are
the ones the previous scoring got wrong: it could not represent a single
distinctive word at all, it ranked rare subject matter above verbal habits, it
let one repeated string occupy many slots, and it had no notion of whether a
phrase was spread through the speaker's media or trapped in one document.
"""

import pytest

from src.anubis.utils.dataset.key_phrase_candidates import (
    KeyPhraseDiscoveryConfiguration,
    build_dispersion_units,
    build_probable_proper_noun_token_set,
    build_reference_language_model,
    discover_key_phrase_candidates,
    load_default_reference_language_model,
    phrase_is_entirely_grammatical,
)


def _reference_model():
    """The bundled reference, which every production call uses."""
    return load_default_reference_language_model()


def _phrases(candidates):
    return [candidate.phrase for candidate in candidates]


# ---------------------------------------------------------------------------
# The owner's own examples: the acceptance test for the whole change.
# ---------------------------------------------------------------------------


def test_single_distinctive_word_and_fixed_expression_are_both_discovered():
    """"tricky" and "what do ya mean" must both be reachable.

    The old discovery mined two-to-four-word phrases only, so a single word
    could never be found no matter how characteristic it was. Both of these are
    the owner's own examples of what the list is supposed to contain.
    """
    documents = [
        "That part is tricky, honestly. What do ya mean by scaling it up?",
        "Tricky problem. What do ya mean exactly, walk me through the tricky bit.",
        "It gets tricky fast. What do ya mean, the whole thing?",
        "What do ya mean by that? Tricky either way.",
        "Pretty tricky. What do ya mean when you say that is done?",
    ]
    candidates = discover_key_phrase_candidates(documents)
    phrases = _phrases(candidates)

    assert "tricky" in phrases
    assert any(phrase.startswith("what do ya") for phrase in phrases)


def test_a_verbal_habit_outranks_rare_subject_matter():
    """Frequency of a habit must beat rarity of a topic phrase.

    Under the old scoring every out-of-table word was assigned the same tiny
    floor frequency, so a phrase built from rare words always won. That made the
    score a proxy for unusual vocabulary — which is subject matter — rather than
    for style.
    """
    habit_sentences = [f"Honestly that is tricky, sentence number {index}." for index in range(40)]
    topic_sentences = [
        "The heterogeneous catalytic reformer failed again.",
        "Another heterogeneous catalytic reformer incident today.",
        "We rebuilt the heterogeneous catalytic reformer overnight.",
    ]
    candidates = discover_key_phrase_candidates(habit_sentences + topic_sentences)
    phrases = _phrases(candidates)

    # Every sentence here wraps "tricky" in the same template, so the template is
    # the expression and the collapse keeps it rather than the bare word; what
    # matters is that the habit outranks the rare technical phrase.
    habit_ranks = [
        index for index, phrase in enumerate(phrases) if "tricky" in phrase
    ]
    topic_ranks = [
        index
        for index, phrase in enumerate(phrases)
        if "reformer" in phrase or "catalytic" in phrase
    ]
    assert habit_ranks, f"no habit phrase discovered in {phrases}"
    assert all(
        min(habit_ranks) < topic_rank for topic_rank in topic_ranks
    ), phrases


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_probable_proper_nouns_are_rejected_but_sentence_initial_words_are_not():
    """Capitalisation mid-sentence marks a name; at a sentence start it marks nothing."""
    documents = [
        "I went to Tesla today and Tesla was busy. Really good though.",
        "Everyone at Tesla agrees. Tesla keeps growing, which is good.",
        "Good is what I said about Tesla. Tesla again tomorrow.",
    ]
    proper_nouns = build_probable_proper_noun_token_set(
        documents,
        capitalisation_ratio_threshold=0.5,
        minimum_observation_count=2,
    )

    assert "tesla" in proper_nouns
    # "good" appears capitalised only at a sentence start, so the capitalisation
    # carries no naming information and the word must survive.
    assert "good" not in proper_nouns


def test_shouted_words_are_not_treated_as_names():
    """A run of capitals is emphasis, which is style, not a name."""
    documents = [
        "this is REALLY REALLY important to me and i mean that",
        "REALLY REALLY good outcome here honestly",
        "i said REALLY REALLY carefully that it matters",
    ]
    proper_nouns = build_probable_proper_noun_token_set(
        documents,
        capitalisation_ratio_threshold=0.5,
        minimum_observation_count=2,
    )
    assert "really" not in proper_nouns


def test_all_grammatical_phrases_are_rejected_but_mixed_phrases_survive():
    """Bare grammar carries no lexical signal and is measured by Burrows' Delta."""
    assert phrase_is_entirely_grammatical(["the"])
    assert phrase_is_entirely_grammatical(["will", "be"])
    assert phrase_is_entirely_grammatical(["out", "of", "the"])

    assert not phrase_is_entirely_grammatical(["tricky"])
    assert not phrase_is_entirely_grammatical(["what", "do", "ya", "mean"])
    assert not phrase_is_entirely_grammatical(["you", "know"])
    assert not phrase_is_entirely_grammatical(["got", "it"])


def test_bare_function_words_never_reach_the_candidate_list():
    documents = [
        "The thing is that the answer is the answer and the rest will be fine.",
        "The other thing is that the answer will be the same, will be fine.",
        "The answer will be the answer. The rest of the rest will be fine.",
    ]
    phrases = _phrases(discover_key_phrase_candidates(documents))
    for grammatical_phrase in ("the", "is", "that", "will be", "of the"):
        assert grammatical_phrase not in phrases


# ---------------------------------------------------------------------------
# Dispersion — "a phrase may persist only once"
# ---------------------------------------------------------------------------


def test_a_phrase_trapped_in_one_document_is_rejected():
    """Ten repeats inside one document is that document's subject, not a habit."""
    concentrated = " ".join(["the quarterly synergy briefing matters"] * 10)
    other_documents = [f"Just an ordinary sentence number {index}." for index in range(9)]
    phrases = _phrases(discover_key_phrase_candidates([concentrated] + other_documents))
    assert not any("synergy" in phrase for phrase in phrases)


def test_the_same_phrase_spread_across_documents_is_kept():
    """Spread through the media is exactly what makes a phrase a signature."""
    documents = [
        "Honestly the quarterly synergy briefing matters a lot here.",
        "We discussed how the quarterly synergy briefing matters again.",
        "Again the quarterly synergy briefing matters more than expected.",
        "Plainly the quarterly synergy briefing matters to everyone.",
        "Once more the quarterly synergy briefing matters today.",
    ]
    phrases = _phrases(discover_key_phrase_candidates(documents))
    assert any("synergy" in phrase for phrase in phrases)


def test_long_documents_are_split_into_many_dispersion_units():
    """One long transcript is one document but many occasions.

    Without windowing, a person who uploaded a single recording could never
    satisfy a dispersion floor expressed in documents.
    """
    long_documents = [" ".join(["word"] * 5000) for _ in range(3)]
    units = build_dispersion_units(long_documents, window_size_tokens=400)
    assert len(units) >= 30
    # Windows never straddle a document, so every unit is fully inside one.
    assert all(len(unit) <= 400 for unit in units)


def test_short_documents_stay_one_unit_each():
    units = build_dispersion_units(["one two three", "four five"], window_size_tokens=400)
    assert len(units) == 2


# ---------------------------------------------------------------------------
# Overlap collapse
# ---------------------------------------------------------------------------


def test_a_repeated_string_does_not_occupy_many_slots():
    """One advertisement previously took eight of forty slots as eight n-grams."""
    advertisement = "sign up via web browser at the site"
    documents = [
        f"Great stuff today. {advertisement}",
        f"Another update here. {advertisement}",
        f"More news for you. {advertisement}",
        f"Something else entirely. {advertisement}",
        f"Final note of the day. {advertisement}",
    ]
    phrases = _phrases(discover_key_phrase_candidates(documents))
    advertisement_fragments = [
        phrase
        for phrase in phrases
        if "browser" in phrase or "via web" in phrase or "sign up" in phrase
    ]
    assert len(advertisement_fragments) <= 1


# ---------------------------------------------------------------------------
# Determinism and degenerate input
# ---------------------------------------------------------------------------


def test_discovery_is_byte_stable_across_runs():
    """An unstable set forces a full recomputation of every stored feature row."""
    documents = [
        "Honestly that is tricky and pretty good, you know.",
        "Pretty tricky honestly, you know what I mean.",
        "You know it gets tricky, pretty much always, honestly.",
        "Tricky and pretty good, you know, honestly.",
        "Honestly, you know, pretty tricky again.",
    ]
    first = _phrases(discover_key_phrase_candidates(documents))
    second = _phrases(discover_key_phrase_candidates(documents))
    assert first == second


@pytest.mark.parametrize("documents", [[], [""], ["   ", "\n"]])
def test_degenerate_corpora_return_no_candidates(documents):
    assert discover_key_phrase_candidates(documents) == []


def test_no_candidates_when_the_reference_corpus_is_unavailable():
    """A trimmed data directory must degrade, never raise."""
    empty_reference = build_reference_language_model(())
    candidates = discover_key_phrase_candidates(
        ["tricky tricky tricky", "tricky again", "so tricky"],
        reference_language_model=empty_reference,
    )
    assert candidates == []


# ---------------------------------------------------------------------------
# The back-off reference model
# ---------------------------------------------------------------------------


def test_expected_frequency_is_positive_for_an_unseen_long_phrase():
    """A zero expectation would make the log ratio unbounded."""
    model = _reference_model()
    unseen = model.expected_relative_frequency(
        ["zzqx", "wvbt", "kkpr", "mmld"]
    )
    assert unseen > 0.0


def test_a_phrase_the_reference_uses_scores_a_higher_expectation():
    model = _reference_model()
    seen = model.expected_relative_frequency(["the", "same", "way"])
    unseen = model.expected_relative_frequency(["zzqx", "wvbt", "kkpr"])
    assert seen > unseen


def test_concordance_lines_are_attached_and_mark_the_phrase():
    documents = [
        "Honestly that is tricky when the deadline moves like this.",
        "The plan is tricky because nobody agreed on the order.",
        "Everything about it is tricky once the numbers land.",
        "Which makes it tricky to promise anything at all.",
        "Still tricky, even after we simplified the whole thing.",
    ]
    candidates = discover_key_phrase_candidates(documents)
    tricky = next(
        candidate for candidate in candidates if candidate.phrase == "tricky"
    )
    assert tricky.concordance_lines
    assert all("[tricky]" in line for line in tricky.concordance_lines)


def test_adaptive_floors_scale_with_corpus_size():
    """Three hits in fifty thousand tokens is a burst; in two thousand it is evidence."""
    from src.anubis.utils.dataset.key_phrase_candidates import (
        resolve_minimum_occurrence_count,
    )

    configuration = KeyPhraseDiscoveryConfiguration()
    assert resolve_minimum_occurrence_count(2000, configuration) == 3
    assert resolve_minimum_occurrence_count(50000, configuration) == 5
