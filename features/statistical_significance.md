what if I wanted to say given 1000 generated texts there is no statistically significant difference and arbitrary generated and the corpus given a list of these features 95% of the time?

# Detailed Features yet to be used:
character n-grams
word n-grams
function word frequencies
punctuation frequencies
lexical richness (vocab size): unique_words vs. total_words
(FIND AND STORE KEY PHRASES) such as "you know" "got it." "what do ya mean?"
average sentence length (words_per_sentence)
average word length (characters per word)



# Detailed Features — implementation status

Scalar features (now part of the fixed Mahalanobis vector in
`src/anubis/utils/dataset/style_features.py`, `STYLE_FEATURE_VECTOR_VERSION = 2`,
width 33 → 36):
- average sentence length (words per sentence) — `mean_sentence_length_words` (pre-existing).
- average word length (characters per word) — `average_word_length_characters` ✅ added.
- lexical richness, unique vs total words — `vocabulary_size_unique_words` +
  `total_word_count` ✅ added (the ratio is the pre-existing `type_token_ratio`).
- punctuation frequencies — the seven `*_rate_per_1k` marks (pre-existing).

Vector-valued features (CAPTURE-ONLY for now — persisted in the nested profile
built by `src/anubis/utils/dataset/stylistic_profile.py`; NOT yet wired into the
authenticity evaluator):
- character n-grams — `_character_ngram_features` (top char 2/3/4-grams) ✅ added.
- word n-grams — `_lexical_features` top uni/bi/trigrams (pre-existing).
- function word frequencies — `key_phrases.function_word_frequencies`
  (closed-class function-word rates) ✅ added.
- key phrases such as "you know", "got it", "what do ya mean" —
  `key_phrases.discover_key_phrases`, auto-discovered statistically as recurring
  multi-word expressions over-represented vs a bundled generic-English baseline
  (a pointwise-mutual-information keyness score) ✅ added.

Migration note: growing the scalar vector bumped `STYLE_FEATURE_VECTOR_VERSION`
to 2. The bundled ChatGPT baseline artifacts were regenerated at the new width
(`data/build_baseline_features_arr.py`); stored per-avatar corpora and cached
baseline arrays/models at the old width are dropped/reloaded on read so existing
deployments self-heal.

# Best to implement open-set attribution: requires a rejection threshold to determine if zero or more authors wrote this document



# Background similar works

---

#### Mosteller and Wallace (inference and disputed authorship): Situation: Given anonymous documents in the Federalist Papers, what is the probablity that a document was written from a particular individual. bayesian probability; what is the probability a document d came from a corpus? 

Goal: P(corpus | D); use few function words instead of topics or content words. 

identify the P(corpus) (count frequencies of function words, aggregate, estimate probabilities (ratio of function words to total function words)); 

use same indicators on the document: count the function words, compute the likelihood using the sum of products of the log of probabilities of frequencies of the same function words to determine P(D | corpus_1 ). 

Finally: identify P(corpus | D) = P(D | corpus)*P(corpus) where P (corpus) is the estimation of the corpus and P(D | corpus) is the likelihood computation (observation of the function words; computed likelihood: sum of count per log of probabilities of a given frequency of a word for all words); 

# Koppel et all: list features to create a vector; use a model to train on this vector of features; use a threshold for rejection. 
