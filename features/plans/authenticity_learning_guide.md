# Measuring & Improving Text Authenticity — a follow-able learning guide for `pystylometry`

**Goal:** learn to (1) *measure* the stylometric features `pystylometry` 1.4.3 already ships (50+ metrics, 11 modules), and (2) *improve* an avatar's generated text against a quantitative "does this sound like the real person?" score. Everything here is Python-implementable. Each entry: **URL → what it teaches → which `pystylometry` module it maps to → type.**

The package surface this guide targets (from PyPI 1.4.3): `lexical`, `readability`, `syntactic`, `authorship`, `stylistic`, `character`, `ngrams`, `dialect`, `consistency`, `prosody`, `viz`.
Repo: https://github.com/craigtrim/pystylometry · PyPI: https://pypi.org/project/pystylometry/

---

## Section A — Foundational hands-on tutorials (start here)

**⭐ Best starting point: The Programming Historian — "Introduction to Stylometry with Python"**
https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python
Walks through three escalating stylometric tests on the Federalist-Papers problem entirely in Python: (1) Mendenhall's word-length curves, (2) Kilgarriff's chi-squared, (3) **John Burrows' Delta** with step-by-step math. Maps directly to `authorship` (Burrows' Delta, Kilgarriff chi-squared) and `lexical` (word frequency, function words). *Type: tutorial. Difficulty: beginner→intermediate.* This is the single best on-ramp because it explains the *reasoning* behind the metrics, not just an API.

**faststylometry — library + "Burrows Delta Walkthrough" notebook**
- Repo: https://github.com/fastdatascience/faststylometry
- Walkthrough notebook (one-click Colab): https://github.com/fastdatascience/faststylometry/blob/main/Burrows%20Delta%20Walkthrough.ipynb
- Tutorial: https://fastdatascience.com/natural-language-processing/fast-stylometry-python-library/
- Forensic-stylometry primer: https://fastdatascience.com/natural-language-processing/forensic-stylometry-linguistics-authorship-analysis/

Calculates Burrows' Delta across candidate authors **and** — the part most libraries skip — calibrates it into a *probability* that two texts share an author via `calibrate()` + `predict_proba()`. That probability is exactly the kind of single-number authenticity score you want for the avatar. Maps to `authorship`. *Type: library + notebook. Difficulty: beginner.* The notebook requires Python ≥3.12.

**The `stylo` ecosystem (canonical methods reference, even though it's R)**
- R Journal paper (Eder, Rybicki, Kestemont 2016, RJ-2016-007): https://journal.r-project.org/articles/RJ-2016-007/
- Repo: https://github.com/computationalstylistics/stylo
- Group hub + datasets + tutorials: https://computationalstylistics.github.io/
- Christof Schöch's curated Stylometry Bibliography (linked from the repo) — the master reading list.
- Calling `stylo` *from Python* (José Calvo Tello's post, linked in the repo README) to cross-check `pystylometry` output against the field-standard implementation.

`stylo` is the de-facto reference for Delta variants, MFW (most-frequent-word) feature selection, and clustering/bootstrap-consensus-tree visualization. Use it to validate your `pystylometry` numbers and to learn feature-selection discipline. Maps to `authorship`, `lexical`, `viz`. *Type: paper + library. Difficulty: intermediate.*

**Towards Data Science — "Linguistic Fingerprinting with Python"**
https://towardsdatascience.com/linguistic-fingerprinting-with-python-5b128ae7a9fc/
Practical feature-extraction walkthrough (punctuation habits, lexical richness, function-word frequencies → similarity). Maps to `stylistic`, `character`, `lexical`. *Type: tutorial. Difficulty: beginner.*

---

## Section B — Per-metric deep references (the math + a Python implementation)

### Lexical diversity (`lexical`: TTR, MTLD, Yule's K/I, HD-D, VocD-D, MATTR, MSTTR, Hapax)

**⭐ McCarthy & Jarvis (2010), "MTLD, vocd-D, and HD-D: A validation study"** — the citation that justifies *which* diversity metric to trust.
DOI/Springer: https://link.springer.com/article/10.3758/BRM.42.2.381 · PubMed: https://pubmed.ncbi.nlm.nih.gov/20479170/ · Semantic Scholar (PDF): https://www.semanticscholar.org/paper/70085f28eb99e5bbba6f5abf7ea964f17eba26ea
Key finding: **MTLD is the only common diversity index that does *not* vary with text length**, and MTLD + vocd-D (or HD-D) + Maas each capture unique information — so report several, not just TTR. This rule should govern how your scorecard weights the `lexical` module. *Type: paper. Difficulty: intermediate.*

**Python implementations to read/borrow from:**
- `LexicalRichness`: https://github.com/LSYS/LexicalRichness — clean implementations of TTR, MTLD, vocd-D, HD-D, Herdan, Maas, MSTTR, MATTR with docstrings citing the formulas. Best for verifying `pystylometry.lexical`.
- `lexical_diversity` (Kristopher Kyle): https://github.com/kristopherkyle/lexical_diversity — MTLD, MATTR, HD-D from an applied-linguistics author. *Type: libraries. Difficulty: beginner.*

### Readability (`readability`: Flesch, FK, SMOG, Gunning Fog, Coleman-Liau, ARI, Dale-Chall, Linsear Write, FORCAST, Fry, Powers-Sumner-Kearl)

**`textstat`** — reference Python implementations of essentially every formula in `pystylometry.readability`, with docstrings explaining each.
https://github.com/textstat/textstat
Use it to unit-test your readability module and to understand each formula's normed range (e.g., SMOG is only valid on ≥30-sentence samples — a caveat the docs spell out). *Type: library. Difficulty: beginner.*

### Delta family & feature scaling (`authorship`: Burrows' Delta, Cosine Delta, John's Delta, MinMax)

**⭐ Evert et al. (2017), "Understanding and explaining Delta measures for authorship attribution"** — explains *why* Cosine Delta beats classic Burrows' Delta.
Open Access: https://academic.oup.com/dsh/article/32/suppl_2/ii4/3865676 · Code/data repo: https://github.com/schtepf/ExplainingDelta
Decisive result: **L2 vector normalization (implicit in cosine distance) is the single factor that makes Delta work better** — feature *scaling* matters more than the distance metric itself. Prefer `pystylometry`'s Cosine Delta and z-score/normalize features before any distance computation in your scorecard. *Type: paper + R code. Difficulty: advanced.*

Background it builds on: Burrows (2002) "'Delta': a measure of stylistic difference" (LLC 17:267) and Argamon "Interpreting Burrows' Delta: geometric and probabilistic foundations" (LLC 23:131) — both cited and linked in the Evert references.

### Function words as the core signal (`lexical`, `stylistic`)

**Mosteller & Wallace, Federalist Papers (1964)** — the founding study showing *function words* (not content words) carry authorial identity. Accessible retelling + code is the Programming Historian lesson in Section A. Principle: strip pronouns/content, keep "the/of/but/upon" frequencies. This is why `pystylometry` separates function-word features — weight them heavily.

### Comparing corpora / Kilgarriff chi-squared (`authorship`, `ngrams`)

**Kilgarriff (2001), "Comparing Corpora"** — basis for the chi-squared word-frequency comparison and the word-frequency-distribution thinking behind n-gram profiles. Searchable on Semantic Scholar; the method itself is implemented and explained in the Programming Historian lesson (test #2). Maps to `authorship` (Kilgarriff chi-squared), `consistency` (sliding-window chi-squared drift). *Type: paper. Difficulty: intermediate.*

### Compression distance (`authorship`: NCD)

**Cilibrasi & Vitányi, "Clustering by Compression"** (IEEE Trans. Inf. Theory 2005) — theory behind Normalized Compression Distance. arXiv: https://arxiv.org/abs/cs/0312044
NCD needs no feature engineering — it compares raw byte streams — so it's a useful *independent* check against your engineered-feature score. Maps to `authorship` (NCD). *Type: paper. Difficulty: advanced.*

### N-gram entropy (`ngrams`: Shannon entropy, char/word/POS n-grams, skipgrams)

Character n-grams are repeatedly shown (Stamatatos survey, Section C) to be among the most robust authorship features. For a hands-on Python treatment of Shannon entropy over n-gram distributions, NLTK's frequency-distribution chapter is the standard reference: https://www.nltk.org/book/ch02.html plus the `nltk.lm` module docs. Maps to `ngrams`. *Type: docs. Difficulty: intermediate.*

### Syntax & POS (`syntactic`, `prosody` — both require spaCy)

spaCy's free course covers tokenization, POS tagging, and dependency parsing — the primitives behind POS ratios, parse-tree depth, dependency distance, clausal density, and syllable/stress rhythm: https://course.spacy.io/ and linguistic-features docs: https://spacy.io/usage/linguistic-features. Maps to `syntactic`, `prosody`. *Type: course + docs. Difficulty: beginner→intermediate.*

---

## Section C — Turning metrics into a *validated* score (authorship verification & evaluation)

**⭐ Stamatatos (2009), "A Survey of Modern Authorship Attribution Methods"** — the canonical map of the field: which features (character n-grams, function words, syntactic), which classifiers, how to evaluate.
Semantic Scholar: https://www.semanticscholar.org/paper/d25c27c7a3e9f41f150e8eadbad34c1c05d67510 · ResearchGate: https://www.researchgate.net/publication/220435062
Read before designing your scorecard — it tells you which `pystylometry` modules carry the most attribution signal and warns about topic-vs-style confounds. *Type: survey. Difficulty: intermediate.*

**Koppel, Schler & Argamon (2009), "Computational Methods in Authorship Attribution"** (JASIST 60:9) — companion survey; introduces the **"unmasking"** technique (degradation curves to verify same-author) and the **impostors method**, both of which give a *verification* score (same author yes/no + confidence) rather than just a distance. Reachable via the PAN reference list: https://pan.webis.de/clef15/pan15-web/authorship-verification.html *Type: survey. Difficulty: advanced.*

**PAN @ CLEF — Authorship Verification shared tasks** (datasets, baselines, and the *evaluation metrics* to adopt)
Hub: https://pan.webis.de/ · 2015 task https://pan.webis.de/clef15/pan15-web/authorship-verification.html · 2023 overview https://ceur-ws.org/Vol-3497/paper-199.pdf
PAN standardizes the metrics that turn a stylometric comparison into a defensible authenticity score: **AUC, c@1, F1, F0.5u, and Brier score**. Adopt these as your scorecard headline numbers and use PAN corpora as a calibration set. *Type: shared task + datasets. Difficulty: intermediate→advanced.*

---

## Section D — The improvement side: using stylometry to make generated text more authentic

This is the half that turns measurement into a feedback loop. The frontier (2020–2026) is **authorship style transfer** and **authorship-representation rewards**.

**⭐ Theory-Grounded Evaluation Exposes the Authorship Gap in LLM Personalization (2026)** — the most directly relevant paper to your exact problem.
https://arxiv.org/abs/2604.26460 (HTML: https://arxiv.org/html/2604.26460)
Formalizes "does this LLM text sound like the target author?" as an **authorship-verification** problem and contrasts three scoring approaches you'll choose among: (1) classical stylometrics (function-word distributions — i.e., `pystylometry`), (2) **LUAR**-style neural authorship representations, (3) LLM-as-judge. Key point: inference-time prompting leaves a measurable "authorship gap," and **per-user LoRA adapters are the suggested fix** — precisely the Together AI Llama-3.2-3B QLoRA path. Treat this as the blueprint for your eval layer. *Type: paper. Difficulty: advanced.*

**LUAR — Learning Universal Authorship Representations (Rivera-Soto et al., EMNLP 2021)** — a neural embedding of "writing style" usable as a *similarity reward*.
Repo: https://github.com/LLNL/LUAR · models on HuggingFace (linked in README) · alt impl: https://github.com/noa/uar
Encode the target's real writing and the avatar's output into LUAR vectors; cosine similarity = a content-robust authenticity signal that complements `pystylometry`'s interpretable features. Trained with contrastive learning across Reddit/Amazon/fanfiction. *Type: model + paper. Difficulty: advanced.*

**Wegmann et al. — Style Embeddings ("Same Author or Just Same Topic?")** — style representations explicitly disentangled from topic.
Repo: https://github.com/nlpsoc/Style-Embeddings
Critical because naive embeddings cheat by matching *topic*; this gives a style vector that won't be fooled when your avatar writes about new subjects. *Type: model + paper. Difficulty: advanced.*

**LISA — Interpretable Style Embeddings via Prompting LLMs (EMNLP Findings 2023)** — bridges neural and human-readable style.
PDF: https://aclanthology.org/2023.findings-emnlp.1020.pdf
Produces 768 named, interpretable style attributes (formality, humor, etc.). Useful for telling a writer/PM *why* the avatar is off-voice, not just *that* it is. *Type: paper. Difficulty: advanced.*

### Style transfer / voice cloning (the generation side)

**⭐ STRAP — "Reformulating Unsupervised Style Transfer as Paraphrase Generation" (Krishna et al., EMNLP 2020)**
Paper: https://arxiv.org/abs/2010.05700 · Repo: https://github.com/martiansideofthemoon/style-transfer-paraphrase
The workhorse method: normalize text via a diverse paraphraser, then an *inverse paraphraser* fine-tuned on the target author rewrites it into their voice. Also defines the standard **automatic style-transfer eval triad** (style-transfer accuracy × semantic similarity × fluency) — adopt that triad so you never improve "voice" at the cost of meaning. *Type: paper + code. Difficulty: advanced.*

**STYLL — "Low-Resource Authorship Style Transfer: Can Non-Famous Authors Be Imitated?" (Patel et al., 2023)**
https://arxiv.org/html/2212.08986v3
Imitates ordinary people from *few* examples using in-context learning + style descriptors — the realistic setting for an avatar built from one user's limited corpus. Read for prompt-based imitation before you have enough data to fine-tune. *Type: paper. Difficulty: intermediate→advanced.*

**⭐ Astrapop — "Authorship Style Transfer with Policy Optimization" (2024)** — uses a **style reward in RL**.
https://arxiv.org/html/2403.08043
Trains generation with policy optimization (PPO-style) against an authorship-style reward — exactly the "stylometric/representation distance as reward" pattern. Template for closing the loop: `pystylometry` distance + LUAR similarity → reward → policy update. *Type: paper. Difficulty: advanced.*

**ParaGuide — Guided Diffusion for Plug-and-Play Style Transfer (AAAI 2024)**
https://arxiv.org/html/2308.15459v3
Steers output toward a target style *at inference time* using a style classifier's gradients — useful to nudge authenticity without retraining. Benchmarks on the Enron email corpus. *Type: paper. Difficulty: advanced.*

### Fine-tuning the avatar's voice (Together AI / QLoRA path)

- Together AI fine-tuning overview (LoRA/QLoRA, data format, jobs): https://docs.together.ai/docs/fine-tuning-overview
- The persona-LoRA recommendation is empirically motivated by the *Authorship Gap* paper above — single-author adapters close the gap that prompting alone cannot.
- Data formatting principle (from STRAP/STYLL): build single-author dialogue/QA pairs where the target's turns are the completion; synthetic-question generation fills monologue gaps (which `pystylometry`'s authorship comparisons then score).

---

## Section E — The "passes as human" framing (AI-text detection signals to minimize)

Knowing what *reveals* machine text tells you what to flatten in the avatar's output.

**DetectGPT — "Zero-Shot Machine-Generated Text Detection using Probability Curvature" (Mitchell et al., 2023)**
https://arxiv.org/abs/2301.11305
Detects LLM text via log-probability curvature. Implication: outputs sitting in high-probability "safe" regions read as machine-generated; authentic human writing is more idiosyncratic. *Type: paper. Difficulty: advanced.*

**"Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense" (Krishna et al., NeurIPS 2023)**
Paper: https://arxiv.org/abs/2303.13408 · Repo: https://github.com/martiansideofthemoon/ai-detection-paraphrases
Shows paraphrasing (the STRAP mechanism) shifts the stylometric fingerprint enough to evade detectors — direct evidence that style transfer measurably moves the same features `pystylometry` reads. *Type: paper + code. Difficulty: advanced.*

Backdrop you can cite: LLMs leave their *own* measurable lexical/morphosyntactic fingerprints (the basis for "LLM fingerprinting"). The Stamatatos taxonomy plus the Authorship-Gap paper give you the vocabulary to characterize and reduce them.

---

## Section F — Datasets & benchmarks for training/eval

| Dataset | Use | Access |
|---|---|---|
| **PAN authorship corpora** | Calibrate verification scores; standard baselines/metrics | https://pan.webis.de/ |
| **Blog Authorship Corpus** (Schler et al. 2006; 681K posts, 19,320 bloggers) | "Ordinary author" imitation eval (used by the Authorship-Gap paper) | Search "Blog Authorship Corpus Schler 2006" / Kaggle mirror |
| **Reddit Million User Dataset** | Training/eval set behind LUAR | Via https://github.com/LLNL/LUAR instructions |
| **Enron Email Corpus** | Style-transfer benchmark (STRAP, ParaGuide) | https://www.cs.cmu.edu/~enron/ |
| **CCAT/Reuters journalists, IMDb1M, Victorian authorship** | Closed-set attribution benchmarks (Stamatatos survey) | Linked from the survey references |
| **Project Gutenberg author sets** | Quick Delta experiments (used by faststylometry & stylo) | Bundled in https://github.com/fastdatascience/faststylometry |

---

## Section G — Durable references & courses

- **Computational Stylistics Group** (Eder et al.) — tutorials, datasets, slideshows: https://computationalstylistics.github.io/
- **Christof Schöch's Stylometry Bibliography** — curated master reading list (linked from the `stylo` README).
- **NLTK Book** (free) — tokenization, POS, frequency distributions: https://www.nltk.org/book/
- **spaCy course** (free) — dependency parsing for the `syntactic`/`prosody` modules: https://course.spacy.io/
- **Stamatatos 2009** and **Koppel/Schler/Argamon 2009** surveys (Section C) — the two papers to keep on hand.

---

## The closed loop: measure → score → improve → re-measure

A concrete workflow tying these resources to your avatar pipeline:

1. **Build the reference profile.** Run *all* `pystylometry` modules on the target's real corpus to get a feature vector. Weight by the evidence: keep MTLD/vocd-D/HD-D over raw TTR (McCarthy & Jarvis), z-score/L2-normalize before any distance (Evert et al.), lean on function words + character n-grams (Stamatatos, Mosteller & Wallace).

2. **Define the authenticity score.** Combine two complementary signals: (a) an *interpretable* distance from `pystylometry` — Cosine Delta + per-module deltas — and (b) a *content-robust* neural similarity from **LUAR**/Wegmann embeddings. Calibrate to a probability with faststylometry's `calibrate()`/`predict_proba()` and report PAN-standard metrics (AUC, c@1). The *Authorship-Gap* paper is your template for combining these honestly.

3. **Generate candidate text** with the avatar (prompted in-context per **STYLL**, or via **STRAP** inverse-paraphrasing for stronger control).

4. **Score the candidate** against the reference profile. Also run the STRAP eval triad (style accuracy × semantic similarity × fluency) so you never trade meaning for voice.

5. **Improve.** Two levers, in order of cost: (a) prompt/in-context refinement (STYLL) for quick gains; (b) **per-user QLoRA fine-tuning** on Together AI when you have enough single-author data — the fix the Authorship-Gap paper shows actually closes the gap; optionally (c) use the combined score as an **RL reward** following **Astrapop**.

6. **Re-measure** with the same scorecard, track the delta across iterations, and visualize drift/convergence with `pystylometry.viz`. Use **DetectGPT**/paraphrase-evasion findings as an adversarial check: if a detector still flags it, your distribution is still too "machine-smooth."

**If you read only five things:** Programming Historian lesson (A) → Evert et al. 2017 (B) → Stamatatos 2009 survey (C) → Authorship-Gap 2026 paper (D) → STRAP (D). Those five take you from "compute the metrics" to "use them as a training signal."
