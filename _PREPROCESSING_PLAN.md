# Golden-format target attribution: diarized audio (Part A) + long-form text with inferred target (Part B)

## Context

Two connected data-quality problems in the media → identity pipeline:

**Part A (diarized audio).** A 34-minute two-speaker interview produced only TWO quote Documents ("you"; "do a startup you have to call it Futs…") while the target's long paragraphs were misrouted into biographical *identity* Documents. Root cause (verified): audio > `whisper_max_bytes` is split into ~20 MiB chunks, each diarized independently (`src/anubis/utils/utility.py:1906-2003`); `_merge_diarized_segments_from_chunks` (utility.py:1404) keeps labels verbatim so `speaker_0` aliases different people across chunks; `is_target` is a substring test against the literal label "avatar" (`nodes.py:2007-2011`), which the diarizer stamps only on segments confidently voice-matched to a ~9-second reference clip. Everything else → `_build_biographical_identity_documents`. Live specimen: `data/dbe60d13-.../transcriptions/...CkUcCcRq_eM...json` (raw: 494 micro-segments, target speech under `A`/`B`, 292 chars under `avatar`) vs the hand-annotated golden `..._golden_dataset_hand_annotated.json` (23 coalesced turns, all target speech under one `avatar` label).

**Part B (long-form text).** The same golden-format turn structure must be produced from text where the target is INFERRED from content (no API parameter, no explicit prompting): movie transcripts with `[Name]` markers and unmarked continuation lines (Gray Man → "Dani Miranda"), scripture-style narrative (`data/bible/king_james/king_james_bible (test).txt` → "Jesus"), and character wiki pages (Fallout Curie → structured bio + quotes). Spec: `_PREPROCESSING_PROCESS.md` (repo root) with a worked Gray Man example. The current `dialogue` text branch returns an error Document (`dialogue_text_requires_diarized_segments`, helper_functions.py:1163).

**Golden dataset format (the contract for both parts):** in-memory envelope `{source, filename, model, duration, text, segments}`; `segments` = coalesced turns `{speaker, start?, end?, text}` — ALL target speech under the single label `"avatar"` merged into long turns; each distinct non-target individual keeps their diarizer/roster label. This structure feeds `process_dialogue_json_to_documents` (helper_functions.py:761-876) unchanged → verbatim quote Documents per target turn (with preceding non-target turn as `adapter_prompt`), one role-converted adapter conversation Document, biographical identity Documents from non-target turns → `process_adapter_documents` (nodes.py:2523+) → the three existing store dataset namespaces ONLY (user-confirmed, no new formats): `q_and_a_adapter` (single-turn TRL prompt-completion), `multi_turn_dataset_adapter` (message-list conversations), `langsmith_factual_q_and_a`.

**User decisions:** ambiguous speaker labels lean NON-target and keep their diarized label; remove dead `reference_audio_diarize_max_seconds`; corrected transcript is an in-memory graph structure (no new persistence requirement — existing DEV dump stays as-is); existing three dataset formats only; target inference is content-driven (avatar name used only as a soft prior when ambient, never a required parameter); scripture emits quote/adapter documents in ADDITION to existing document-namespace chunks.

---

# Part A — Diarization target attribution

## A1. Per-chunk speaker-label namespacing — `src/anubis/utils/utility.py`

- `_merge_diarized_segments_from_chunks` (line 1404): add `known_speaker_label: Optional[str] = None`. Per segment: label equals `known_speaker_label` case-insensitively → keep verbatim; otherwise when `known_speaker_label` provided → rewrite to `f"chunk_{chunk_index}.{raw_label}"`; when None (no reference audio, incl. `create_reference_media_from_playlist`) → verbatim (today's behavior). Keep `chunk_idx` on every merged segment. Rewrite the stale docstring claiming known-speaker references unify labels across chunks.
- Sole call site (~line 1993): pass `known_speaker_label=context.audio_diarization_known_speaker_name if encoded_reference_audio else None`.
- Single-request path (≤ 25 MiB, line 1893) untouched; downstream defaults missing `chunk_idx` to 0.

## A2. Adjudication module — `src/subgraphs/process_media_graph/utils/target_attribution.py` (new)

Schemas: `SpeakerLabelAttribution {speaker_label, belongs_to_target, confidence: Literal["high","medium","low"], evidence_summary}`; `TargetSpeakerAttributionResponse {attributions: List[...]}`.

`adjudicate_target_speaker_labels(turns, *, reference_transcript_text, target_name, target_speaker_label, context) -> Optional[Dict[str, bool]]`:
- Labels equal to `target_speaker_label` are target by definition (listed as voice-matcher prior evidence, not re-adjudicated).
- Transcript rendered as `speaker_label: text` lines (mirror `_format_dialogue_transcript`, helper_functions.py:665).
- `init_model(model_without_tools=False, response_format=TargetSpeakerAttributionResponse)`; lazy imports (pattern: `generate_question_for_message`, formatting.py:107-120).
- Validation: drop unknown labels; missing labels → False; only "high"/"medium" promotes — "low" stays non-target; None on exception/empty.
- Length fallback: transcript > `target_speaker_attribution_transcript_character_limit` (default 100000) → adjudicate per `chunk_idx` group, merge maps (namespaced labels never collide).

Prompt file `src/anubis/utils/prompts/target_speaker_attribution_prompt.py` — GPT-5-guide structure. Signals in priority order: voice-matcher confirmations + role continuity around them; interviewer-vs-interviewee roles; content overlap with reference transcript; cross-chunk first-person consistency. Rules: one attribution per provided label, labels copied verbatim, never invented; zero-confirmation chunks still attributed from text; ambiguous → false/"low". Escape hatch: single person throughout → every label target. No pronoun "it", no acronyms.

## A3. Hook into the audio branch — `src/subgraphs/process_media_graph/utils/nodes.py`

- Reference transcript: extend the stored-reference lookup (lines 1744-1757) to also read `value.document.kwargs.page_content` (stored at 1685-1692). Failure-tolerant, default "".
- `chunk_idx` pass-through in segment normalization (lines 2012-2019) and in `coalesce_segments_by_speaker` (helper_functions.py:408-448; opening segment's `chunk_idx` carried onto each turn).
- Adjudication pass after `turns = coalesce_segments_by_speaker(...)` (line 2022), gated on `encoded_reference_audio is not None`, >1 distinct speaker, `enable_target_speaker_attribution == "TRUE"`. Apply as union with diarizer votes (never subtract): mapped-True turns get `is_target=True`, `speaker=target_speaker_label`; all other turns keep their diarized label. Re-run `coalesce_segments_by_speaker` (idempotent) so adjacent avatar turns merge into the golden-format long turns; recompute `distinct_speakers`. Result: `dialogue_payload["segments"]` (line 2052) IS the golden-format segments list.
- Progress events: `target_attribution` (label counts); `target_attribution_failed` + `logger.warning` on exception/None → continue with diarizer votes only; explicit warning when zero target labels remain.
- Generalize lone-speaker promotion (lines 2034-2045): promote when a reference exists, every `chunk_idx` has exactly one distinct raw (prefix-stripped) label, and no turn is target yet. Fallback after adjudication.
- No changes to `process_dialogue_json_to_documents` / quote / identity builders.

---

# Part B — Text dialogue segmentation with inferred target

## B1. New module `src/subgraphs/process_media_graph/utils/text_dialogue_segmentation.py`

All heavy imports lazy. Schemas:
- `SegmentedSpeakerTurn {speaker, text, is_speech}` (`is_speech=False` for stage directions / sound cues / pure narration → speaker "narrator").
- `SpeakerRosterEntry {name, description}`; `WindowSegmentationResult {reasoning, segments, updated_roster, final_attributed_speaker}`.
- `TargetSpeakerInference {reasoning, has_identifiable_target, target_name, matching_roster_names}` (aliases: "Miranda", "Agent Miranda", "Dani" all map to one target; also reused by the structured-web target inference).
- `QuoteBlockAttribution {block_index, quotes_spoken_by_target: bool, reasoning}` (structured-web ambiguous-block classification only — quote text is parser-extracted, never model-echoed).

Functions:
- `split_text_into_dialogue_windows(text, *, window_characters)` — deterministic blank-line/newline boundary split, never mid-line.
- `segment_dialogue_window(window_text, *, roster, last_attributed_speaker, previous_turn_tail)` — one structured call; human message carries roster JSON, last-attributed speaker, final 2 prior turns (read-only, do-not-re-emit), window text.
- `segment_text_into_speaker_turns(text, *, window_characters, max_characters)` — SEQUENTIAL window loop (roster + last-speaker carryover is a chain); case-insensitive roster merge; stops at `max_characters` with logged warning (prefix still yields documents).
- `fold_narrator_segments(segments)` — deterministic: `is_speech=False` segments relabel to the nearest preceding non-target speaker (next one when leading), matching the spec's multi-turn example (`_PREPROCESSING_PROCESS.md` lines 119-132).
- `infer_target_speaker(*, roster, segments, classification_target_name, filename)` — one structured call over roster with per-speaker turn counts + 2-3 sample turns each, the `ContentSituationClassification` prior, and filename/title. Avatar name (when ambient) may be included as a soft prior only.
- `relabel_target_segments(segments, *, target_roster_names)` — target names (case-insensitive) → `speaker="avatar"`, `is_target=True`; others keep names, `is_target=False`.
- `convert_text_dialogue_to_documents(text_content, *, user_id, assistant_id, media_item, classification_target_name)` — orchestrator: windows → fold narrator → infer target → relabel → `coalesce_segments_by_speaker` → `dialogue_payload = {"segments": coalesced, "target_name": ..., "speakers": roster}` → `process_dialogue_json_to_documents`. When `has_identifiable_target` is False → error Document `dialogue_text_target_not_identifiable` so the media job reports the failure.

Post-check guardrail: log a warning when an avatar turn is not fuzzy-contained in the source window (verbatim-echo fidelity).

## B1b. Structured-web extraction module — `src/subgraphs/process_media_graph/utils/structured_web_extraction.py` (new; user-directed design)

Parser-FIRST, not LLM-echo: BeautifulSoup (already a dependency, used in `_httpx_fallback_text`) parses the HTML structure; the LLM only infers the target and classifies ambiguous blocks — quote/bio text is extracted VERBATIM by the parser. Globally applicable to any person, real or imagined, any source (fandom wiki, personal homepage, future crawled pages); functions take `(html_text, url)` so a future n-tree link crawler (audio/video/images/text — out of scope now) can reuse them.

- `parse_html_into_structured_blocks(html_text, *, url) -> dict` — returns `{page_title, infobox_subject_name, blocks: [{heading_path, kind: "paragraph"|"table"|"list", text, table_rows}]}`. MediaWiki-aware selectors first (`.mw-headline` section headings such as "Personality"/"Background", `aside.portable-infobox` subject name, `table.wikitable` rows), generic fallback (title/h1/h2/h3 + p + table) for pages like lexfridman.com (verified: h1 = "Lex Fridman", bio paragraph beginning "Research interests: …").
- `infer_target_from_structured_page(*, page_title, infobox_subject_name, leading_paragraphs, heading_names)` — one structured call reusing `TargetSpeakerInference`; content-inferred, never an explicit parameter.
- `extract_direct_quotes_from_blocks(blocks, *, target_name) -> List[dict]` — deterministic: table rows whose quote cell holds quotation-marked strings (fandom `Context / Comment(s)` layout — sibling context cell becomes the genuine context prompt), quote-section paragraphs, numbered transcript-style rows (the CompanionCurie.txt shape). Quotation marks stripped, text verbatim. One structured LLM classification call ONLY for ambiguous blocks (spoken BY the target vs ABOUT the target); unambiguous structures never round-trip through the model.
- `extract_biographical_prose_blocks(blocks) -> List[dict]` — prose sections (Personality, Background, Research interests, generic bio paragraphs); each section feeds the existing `_build_biographical_identity_documents` (helper_functions.py:~382) per section, preserving `heading_path` as provenance metadata.
- `convert_structured_web_page_to_documents(html_text, *, url, user_id, assistant_id, media_item)` — orchestrator: parse → infer target → quotes → golden-format segments (optional `{"speaker": "context", "text": context_prompt, "is_target": False}` + `{"speaker": "avatar", "text": quote_text, "is_target": True}` per quote) → `process_dialogue_json_to_documents`; biographical prose → biographical pipeline; returns combined documents. No target inferable → error Document `structured_page_target_not_identifiable`.
- HTML availability: the URL article branch currently flattens to text (`WebBaseLoader` / `_httpx_fallback_text`). Retain the raw HTML on the expanded article media item (new metadata field, e.g. `raw_html`) when the fetch succeeds; the structured path runs when the parse detects a subject page (infobox present, or biographical headings + quote structures), otherwise fall through to the plain-text `process_text_to_document` route unchanged.

## B2. Prompts — `src/anubis/utils/prompts/text_dialogue_segmentation_prompt.py` (new)

Three system prompts, GPT-5-guide sections, fully spelled-out prose:
1. `TEXT_DIALOGUE_SEGMENTATION_SYSTEM_PROMPT` — `[Name]` markers open turns; unmarked lines attributed by DIALOGUE LOGIC (the reply "Speaking." after "[Denny] Agent Miranda?" belongs to Miranda), not defaulted to the previous marked speaker; narration with embedded attribution ("And Jesus said unto them, X") splits quoted speech to the named speaker and remaining narration to narrator/`is_speech=false`; verbatim echo, never paraphrase/merge/drop/invent; reuse roster names exactly; never re-emit previous turns. Worked example: Gray Man excerpt (spec lines 22-46) with correct output.
2. `TARGET_SPEAKER_INFERENCE_SYSTEM_PROMPT` — pick the single individual the content centers on (turn counts, being-addressed/being-described evidence, classification prior); list ALL roster aliases; escape hatch `has_identifiable_target=false`.
3. `STRUCTURED_PAGE_TARGET_INFERENCE_SYSTEM_PROMPT` — infer the page's single subject from page title, infobox subject name, leading paragraphs, heading names; escape hatch `has_identifiable_target=false`.
4. `QUOTE_BLOCK_ATTRIBUTION_SYSTEM_PROMPT` — for ambiguous parsed blocks only: decide whether the quoted strings are spoken BY the target versus ABOUT the target; never rewrite the quoted text (the parser already extracted the text verbatim).

## B3. Routing — `process_text_to_document` (helper_functions.py)

- **Dialogue branch** (replace the error return at ~1152-1168): payload has diarized `segments` list → existing `process_dialogue_json_to_documents` path unchanged; otherwise → `convert_text_dialogue_to_documents(...)` with `classification_target_name` from the situation classifier (`has_identifiable_target` gate). If the result is the target-not-identifiable error Document, return early (skip the acceptable-flags stamping loop).
- **Structured web pages** (Curie wiki, lexfridman.com): routed BEFORE text flattening. In the URL article branch, keep the raw HTML on the expanded media item; when `parse_html_into_structured_blocks` detects a subject page, `process_media_item_task` routes to `convert_structured_web_page_to_documents` instead of the plain-text path. Curie ⇒ identity docs from Personality/Background prose + verbatim quote/Q&A/adapter docs from the quote tables; lexfridman.com ⇒ identity docs from the bio paragraph ("Research interests: …").
- **Biographical-facts branch plain-text fallback** (inside block at ~1033): for TEXT uploads (no HTML) whose prose contains dense quotation-marked lines, run the same `extract_direct_quotes_from_blocks` over parser-detected text blocks so pasted bio+quotes files behave like their web counterparts.
- **Reference gate** (scripture, `is_menu_or_religious_text` block at ~973): keep existing document-namespace return, additionally (gated on `narrative_speech_extraction_enabled == "TRUE"`) run `convert_text_dialogue_to_documents` capped by `text_dialogue_segmentation_max_characters` and append dialogue documents (skip on the error Document — the add-on failing must not fail the item). Required because scripture returns early before `ContentSituationClassification` today.
- `ContentSituationClassification` schema unchanged (5000-char slice stays the prior; roster-aware inference makes the final call).

## B4. Windowing

Window 4000 chars default (output echoes the window inside JSON — output length is the binding constraint, this leaves large completion headroom). Sequential carryover: roster + `final_attributed_speaker` + last 2 turns. Final assembly: concatenate all window segments → `fold_narrator_segments` → `relabel_target_segments` → one `coalesce_segments_by_speaker` pass. Cap `text_dialogue_segmentation_max_characters` default 250000 (full movie transcript fits; 4.3MB full bible requires deliberately raising the env). Structured-web extraction needs no windowing — the parser walks the whole DOM deterministically; only the (rare) ambiguous-block classification calls the model, batched in one call.

---

# Config / env (`src/anubis/utils/context.py`, `.env`, `.env.dev`, `.env.example`)

- REMOVE `reference_audio_diarize_max_seconds` (context.py:257-262) + env entries everywhere (dead, user-confirmed).
- ADD: `enable_target_speaker_attribution` (str, "TRUE"), `target_speaker_attribution_transcript_character_limit` (int, 100000), `text_dialogue_segmentation_window_characters` (int, 4000), `text_dialogue_segmentation_max_characters` (int, 250000), `narrative_speech_extraction_enabled` (str, "TRUE"), `structured_web_extraction_enabled` (str, "TRUE"). Uppercase in `.env`/`.env.dev`, empty in `.env.example`.
- Leave `reference_audio_clip_max_seconds` at 9 (verify OpenAI known_speaker_references limits before raising — optional follow-up).

# Verification

**Unit tests** (monkeypatch `init_model` at module level; patterns: tests/unit_tests/test_analysis_pipeline.py:56, test_diarization_dialogue_pipeline.py):
- Part A — new `tests/unit_tests/test_target_speaker_attribution.py`: chunk namespacing on/off, "avatar" preserved, `chunk_idx` retained; attribution union with diarizer votes; "low" confidence stays non-target; re-coalesce merges cross-chunk avatar turns into long golden-format turns; exception → diarizer-votes-only fallback; per-chunk fallback above char limit; generalized lone-speaker promotion; `coalesce_segments_by_speaker` `chunk_idx` pass-through.
- Part B — new `tests/unit_tests/test_text_dialogue_segmentation.py`: window split determinism (never mid-line); two-window roster carryover with boundary same-speaker merge; `fold_narrator_segments` matches spec example ("[line clicks]" folds into surrounding user turn); Gray Man spec excerpt end-to-end with canned LLM results → assert exact golden structure (avatar turns "Speaking.", "Yeah. Hard to miss.", "I'm supposed to be in Singapore."; Denny keeps name; user/assistant alternation per spec lines 100-132); routing (dialogue text no longer errors; target-inference failure returns the new error Document; diarized-segments path regression-green); alias mapping in target inference.
- Structured web — new `tests/unit_tests/test_structured_web_extraction.py` with CHECKED-IN HTML snapshot fixtures (a trimmed fandom-style page with portable-infobox + `Context / Comment(s)` quote table + Personality section, and a trimmed personal-homepage page with h1 + "Research interests:" bio paragraph — live fandom is Cloudflare-protected, so snapshots are required for offline tests): parser block extraction (headings, infobox subject, table rows), deterministic quote extraction with context prompts and stripped quotation marks, biographical prose block selection, orchestrator producing identity docs + avatar quote segments, subject-page detection true/false (article without infobox falls through to plain text).

**Golden-format conformance check (Part A)**: offline script feeding the checked-in RAW dump (`...CkUcCcRq_eM_1782152050694843865.json`) segments through normalization + adjudication (live LLM) and asserting the output shape approaches the hand-annotated golden: all-avatar speech merged into few long turns, non-target labels preserved.

**End-to-end (real APIs, dev stack) — user-supplied test matrix**:

| Class | Source | Exercises | Expected |
|---|---|---|---|
| Monologue single-speaker | https://www.youtube.com/watch?v=7Sk6lTLSZcA ("1984"), https://www.youtube.com/watch?v=0m3hGZvD-0s ("A day in my life"), https://www.youtube.com/watch?v=bCA54RIkpTo | Part A single-speaker path (adjudication escape hatch + generalized lone-speaker promotion across chunks) | all turns target → quote namespace; no misrouted identity docs |
| Dialogue multi-speaker | https://www.youtube.com/watch?v=IvqRSEP_o-o, https://www.youtube.com/watch?v=mQ7ECcjXazw, interviews https://www.youtube.com/watch?v=tlOyZSAZh2k, https://www.youtube.com/watch?v=smK9dgdTl40 | Part A main path (chunk namespacing + adjudication) | golden-format long avatar turns; many verbatim quote docs with genuine `adapter_prompt`; identity docs only from non-target turns; one adapter conversation doc; `target_attribution` progress event |
| Audience Q&A (challenge) | https://www.youtube.com/watch?v=_ySbzVXiwzQ | Part A with many short non-target speakers | avatar answers as quotes; audience questions as genuine prompts |
| Text dialogue | URL in `data/dani_miranda/script.md` → https://scrapsfromtheloft.com/movies/gray-man-2022-transcript/ | Part B dialogue segmentation via URL article loader → `process_text_to_document` dialogue branch | inferred target Dani Miranda; golden-format turns; quotes/Q&A/adapter docs per `_PREPROCESSING_PROCESS.md` worked example |
| Structured wiki bio + quotes | URL in `data/data_pre-processing-challenges/biographical_text_and_quotes.md` → https://fallout.fandom.com/wiki/Curie | Part B structured-web extraction (BeautifulSoup parser-first) | inferred subject Curie; identity docs from Personality/Background prose; verbatim quotes from `Context / Comment(s)` tables as avatar turns with context prompts |
| Personal homepage bio | https://lexfridman.com/ | Part B structured-web extraction, generic (non-wiki) page shape | inferred subject Lex Fridman; identity docs from the "Research interests: … Podcast interests: …" bio paragraph |
| Scripture | `data/bible/king_james/king_james_bible (test).txt` (file upload) | Part B reference-gate add-on (`narrative_speech_extraction_enabled`) | document-namespace chunks PLUS inferred-target (Jesus) quote/adapter docs |

After each ingestion, verify `process_adapter_documents` populated `q_and_a_adapter`, `multi_turn_dataset_adapter`, `langsmith_factual_q_and_a` (formats verified in the earlier session: prompt-completion dicts, message-list conversations, inputs/outputs examples) via `sql/adapter_dataset_store_search.sql`.

Out of scope for this plan (future ingestion work, listed by the user for later): x.com/lexfridman and the social-platform subscribe/pull list (YouTube channel, Instagram, TikTok, Facebook, Reddit, Telegram, Google Scholar), lexfridman.com crawling, discord bot, realtime audio.

# Sequencing

Prompts → pure functions (windowing, folding, namespacing) → LLM wrappers (adjudication, segmentation, inference, wiki) → orchestrators → routing edits → config → tests. Part A's `target_attribution.py` lands first; Part B imports shared golden-format conventions from the same helpers (`coalesce_segments_by_speaker`).

# Risks

- Fandom (and similar Cloudflare-protected hosts) returns a 403 challenge to plain httpx from datacenter addresses (verified from this environment). The extraction functions accept `(html_text, url)` from whatever fetcher succeeds; fetch hardening (browser impersonation / cookie import / browse daemon) is a separate follow-up. End-to-end Curie verification may require a manually saved HTML file uploaded as a file item.
- Multi-protagonist ambiguity (full Gray Man: "Six" vs "Miranda") — content inference picks the most prominent individual; avatar name is only a soft prior.
- Verbatim-echo fidelity — fuzzy-containment post-check logs mutations; line-span schema is a future hardening.
- Cost: full-bible ingestion ≈ ~1000 sequential calls — cap defaults protect against accidental runs; batched-parallel roster snapshots are a follow-up.
- Classification slice (first 5000 chars) can misroute transcripts whose head is credits — head+middle sampling is a cheap later hardening, out of scope.
