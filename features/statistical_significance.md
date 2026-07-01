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

The fixed Mahalanobis vector in `src/anubis/utils/dataset/style_features.py` is
now **28 scalars** (`STYLE_FEATURE_VECTOR_VERSION = 3`, width 36 → 28). Version 3
pruned nine multicollinear features — `type_token_ratio`, `maas_lexical_diversity`,
`yule_characteristic_k`, `pos_sequence_compressibility`, `flesch_kincaid_grade`,
`gunning_fog_index`, `smog_index`, `vocabulary_size_unique_words`,
`total_word_count` — because each was definitionally derived from (or measured
the same signal as) features already in the vector, inflating the covariance the
Mahalanobis distance must invert while adding no signal. Diversity is carried by
the length-robust MATTR / MTLD / HD-D trio plus `lexical_entropy_bits`.

Scalar features in the vector:
- average sentence length (words per sentence) — `mean_sentence_length_words`.
- average word length (characters per word) — `average_word_length_characters`.
- punctuation frequencies — the seven `*_rate_per_1k` marks.
- **signature key-phrase rate** — `key_phrase_rate` ✅ added in v3: occurrences of
  the avatar's auto-discovered signature phrases (contiguous token matches,
  overlaps counted) divided by total words. Computed by
  `key_phrases.key_phrase_occurrence_rate`; the one avatar-relative feature in
  the vector. The bundled ChatGPT baseline rows are measured against the
  baseline's OWN self-discovered phrases (`baseline_key_phrases.json`) so the
  column has real variance.

Signature key phrases (separate from the scalar vector):
- discovered by `key_phrases.discover_key_phrases` — recurring 2–4-word
  expressions over-represented vs a bundled generic-English baseline
  (a pointwise-mutual-information keyness score).
- on each media upload with direct quotes, `calibrate_ground_truth` re-discovers
  the phrases over the FULL quote corpus, upserts one Document per phrase into
  the `(creator_id, assistant_id, "key_phrase")` vectorstore namespace (deduped
  by phrase), and stores a `key_phrase_profile` blob (raw phrase list + rendered
  string) at `(assistant_id, "key_phrase_profile")` — parallel to `style_profile`.
- the rendered list is prompt-injected as its own `<SIGNATURE PHRASES>` system
  prompt section; the raw list feeds `key_phrase_rate` for both the ground-truth
  cloud rows and each evaluated message.

Dropped entirely (previously capture-only, never scored): character n-grams and
function-word frequency vectors.

Migration note: shrinking/reshaping the vector bumped
`STYLE_FEATURE_VECTOR_VERSION` to 3. The bundled ChatGPT baseline artifacts
(feature matrix, IsolationForest, and the now-persisted SHAP explainer) were
regenerated at the new width (`data/build_baseline_features_arr.py`); stored
per-avatar corpora and cached baseline arrays/models/explainers at the old width
are dropped/reloaded on read so existing deployments self-heal, and per-avatar
ground-truth models rebuild on the next media upload.

# Best to implement open-set attribution: requires a rejection threshold to determine if zero or more authors wrote this document



# Background similar works

---

#### Mosteller and Wallace (inference and disputed authorship): Situation: Given anonymous documents in the Federalist Papers, what is the probablity that a document was written from a particular individual. bayesian probability; what is the probability a document d came from a corpus? 

Goal: P(corpus | D); use few function words instead of topics or content words. 

identify the P(corpus) (count frequencies of function words, aggregate, estimate probabilities (ratio of function words to total function words)); 

use same indicators on the document: count the function words, compute the likelihood using the sum of products of the log of probabilities of frequencies of the same function words to determine P(D | corpus_1 ). 

Finally: identify P(corpus | D) = P(D | corpus)*P(corpus) where P (corpus) is the estimation of the corpus and P(D | corpus) is the likelihood computation (observation of the function words; computed likelihood: sum of count per log of probabilities of a given frequency of a word for all words); 

# Koppel et all: list features to create a vector; use a model to train on this vector of features; use a threshold for rejection. 
