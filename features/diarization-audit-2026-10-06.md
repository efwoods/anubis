# Diarization audit — bugs, features, costs (2026-10-06)

Scope: `gpt-4o-transcribe-diarize` in the media-upload pipeline, the voice routes, and the reference-audio anchor, on branch `test`. `anubis/` and `wt/f-anubis/` are byte-identical for `src/anubis/utils/utility.py`, `src/subgraphs/process_media_graph/utils/nodes.py`, `src/api/webapp.py` and `src/anubis/utils/voice/reference_eligibility.py` (`diff -q`). This is a report for future implementation; no code was changed.

Every status below was checked against today's code or today's store rows, because notes in `data/` and `features/bugs/` describe bugs that may already be fixed. Store text lives at `value->'document'->'kwargs'->>'page_content'` (metadata at `value->'document'->'kwargs'->'metadata'`).

## Why diarization matters

Diarization decides which speaker label is the avatar. Every quote, identity fact, voice clip and adapter row from an interview inherits that decision. The first usable upload is cut into a reference clip (`known_speaker_references`), and every later upload is labelled against that reference clip.

## Measured facts

| Fact | Value | Source |
|---|---|---|
| Price, audio input | $2.50 / 1M tokens | developers.openai.com/api/docs/pricing |
| Price, text output | $10.00 / 1M tokens | same page |
| OpenAI's published estimate | $0.006 / minute | same page |
| Context window / max output | 16,000 / 2,000 tokens | developers.openai.com model page |
| Recorded diarization rows | 560 rows, 68,951 input + 64,464 output tokens, $0.81702, 2026-09-07 → 2026-10-04 | `api_metrics` where `model_name ilike '%diar%'` |
| Rows with `media_job_id` | 0 (every recorded row is chat or dictation) | same query, `count(media_job_id)` |
| Diarized store rows by day | 09-10: 157, 09-11: 46, 09-12: 433, 09-17: 733, 10-06: 1 | store rows with `is_target` and `start` |
| Upload recorder | `record_speech_api_usage` landed in 6e71a0d (2026-09-18); no diarized upload since, so the recorder is unexercised | git log + the row above |
| Output-cap truncation | none: on every one of 80 dev dumps (38 media, 45 runs) the last segment ends within 21 s of the media's end (13,104.11 s of 13,113.19 s on the longest) | `data/*/transcriptions/*.json` |
| 2026-09-17 spend | console diarize $5.79 over at least 23,608.219 s = 393.47 min of main-path audio ⇒ $5.79 / 393.47 = **at most $0.014715 / min** (upper bound: prod runs and isolation passes are not in the dev dumps) | OpenAI console + dev dumps |
| Processing time | 2,087.16 s video processed in 1,436.944 s | `data/shivon_zilis/processing_time.md` |
| Speech retries | correct: httpx 0.28.1 rewinds the file before a resend (`_multipart.py:208-210`) | installed package in the prod container |

## Status of previously reported bugs

| Report | Status today | Evidence |
|---|---|---|
| `data/claire_wineland/_BUG_AUTHENTICITY_IS_LOW_QUALITY.md`: the label "avatar" used as a person's name ("I'm proud that avatar is my mom.") | **Fixed for new data; 3 stale rows remain** | Exactly 3 diarized identity rows use "avatar" as a name, all written 2026-09-01. None after the first-person rewriter's `target_name` triage (8f25ee4, 2026-09-08). `target_name="avatar"` is still passed at `nodes.py:2938, 3003, 3015`, so the leak is latent. |
| Same file: another man's speech stored as Claire's ("I would not be the man I am without a woman…") | **Stale row remains; cause not verified** | 1 identity row dated 2026-08-31; no later occurrence. Check: open that row's `namespace_filename` dump and read the turn's label. |
| `data/lex_fridman/urls.md`: interviewer question stored as the avatar's quote (IvqRSEP_o-o) | **Not a bug for current data** | The avatar `59e29e65-1b93-40b0-b579-d68fcc9c3018` is Lex Fridman; the question is Lex's own speech (stored 2026-09-17, `speaker=avatar`). |
| `data/lex_fridman/urls.md`: overlapping speech scattered into fragments (mQ7ECcjXazw) | **Current** | 2026-09-17 ingest: 33 quote rows, 21 under 40 characters, shortest "i", "Do", "man". Raw dump: 88 segments, 55 under 1 s, 69 speaker switches. |
| `data/lex_fridman/urls.md`: words dropped ("let's get crazy, ___ coming years") | **Model limitation** | The words are missing from both the diarizer output and YouTube's captions. |
| Dev dump of mQ7ECcjXazw labelled A/B/C with no "avatar" | **Not a bug** | The stored 2026-09-17 quotes carry `speaker=avatar`, `is_target=true`; target attribution promoted the label. |
| `features/bugs/fp_facts_from_speaker_diarization.md` (avatar `59a0d69a…`, Kbk9BiPhm7o) | **Stale** | The avatar and all of its rows no longer exist. |
| `features/bugs/reference_audio.md`: calibration-sentence recording; false positive on 3gp video | **Calibration built; 3gp not reproduced** | The calibration-sentence comparison exists at `nodes.py:2434-2457`; the 3gp note has no reproduction steps. |

## Current bugs (code-verified, ranked)

1. **Overlapping-speech fragmentation.** `coalesce_segments_by_speaker` (`helper_functions.py:469`) merges only consecutive same-label segments. Interleaved sub-second A/B/A/B fragments therefore become one-word quote rows (see the status table).
2. **The reference is the dominant speaker by talk time, with no identity check.** `_select_dominant_speaker_segments` (`utility.py:1730`). When the interviewer talks more than the avatar, the interviewer becomes the permanent anchor (`store_reference_audio(replace=False)`), and every later upload is labelled against that interviewer. Target attribution only promotes and is grounded on the same reference transcript, so attribution cannot correct the anchor. Code path verified; no occurrence found in the store yet.
3. **Long recordings alias speakers when the reference is cut.** Isolation calls the diarizer without a reference. A file over 25 MiB (`WHISPER_MAX_BYTES=26214400`) is chunked, and `_merge_diarized_segments_from_chunks(known_speaker_label=None)` keeps raw labels across chunks (`utility.py:1632-1648`). `speaker_0` in chunk 0 and `speaker_0` in chunk 1 can be different people, and their talk time is summed into one "dominant" label.
4. **A promoted reference upload is diarized three times over its full length.** `nodes.py:2345` (isolate for the reference clip), then `nodes.py:174` (isolate again for the voice corpus), then `nodes.py:2533` (main pass). `/avatar_voice/samples` diarizes the same take twice (`webapp.py:14074`, `webapp.py:14116`).
5. **Isolation passes are never recorded.** `isolate_dominant_speaker_audio_b64` discards the diarizer's usage and cost. None of the five call sites (`nodes.py:174, 2345`; `webapp.py:14074, 14116, 14477`) write `api_metrics` or a Stripe meter event, so 2 of the 3 passes in bug 4 are invisible.
6. **Chunks are cut at equal time intervals.** `utility.py:2155-2189` slices with ffmpeg `-c copy`, so a turn that crosses a boundary is split under two chunk-namespaced labels.
7. **The YouTube docstring promises subtitles first, but the code always pays to diarize.** `URLDocumentLoaderClass.py:1-9` and `:596`; `get_transcript` is imported and never called on that path.
8. **Hygiene.** The `reference_audio` argument of `transcribe_audio_diarize` is unused; `# asdf` at `utility.py:2072`; the TODOs at `utility.py:1725-1727` describe work already done; the `AUDIO_DIARIZATION_PRICE_PER_MILLION_TOKENS_*` names hold per-token values (`0.0000025`, `0.00001`; the arithmetic in `_diarize_token_cost` is correct).
9. **A plaintext API key** sits in `data/shivon_zilis/_shivon_zilis_test_data.md` (not tracked by git). Rotate the key or remove the file.

## Missing features

- **Diarize once.** Reuse one diarization result for the reference cut, the voice-corpus track and dialogue routing (removes bug 4; `_select_dominant_speaker_segments` is already a pure function).
- **Cost-efficient reference identity check.** Before `store_reference_audio`, run one text-only structured-output call over the already-paid transcript: "which speaker label is <avatar name>?" (self-introductions, being addressed by name). This adds no audio cost; the cost of the text call has not been measured.
- **Fragment merging** before quote pairing: fold sub-second interleaved segments into the surrounding same-speaker turn (fixes bug 1).
- **The avatar's real name as `target_name`** (`_assistant_name(config)`, `nodes.py:202`), which closes the latent label leak.
- **Delete the 4 stale Claire rows** listed in the status table.
- **Per-upload cost preview** from media duration before the job starts (`get_remote_video_duration_seconds`, `get_audio_duration_seconds` already exist).
- **Media duration on each diarization `api_metrics` row,** so dollars per minute comes straight from SQL.
- **Noise reduction.** The noisereduce block is commented out (`utility.py:1385-1448`), and demucs is absent.

## Cost model

These figures use OpenAI's published estimate of $0.006 per minute. That rate is an assumption; the real per-minute rate is not measured.

| Case | Diarizer passes | 60-minute interview |
|---|---|---|
| Upload with a stored reference | 1 | 1 × 60 × $0.006 = $0.36 |
| First upload promoted to reference (today) | 3 | 3 × 60 × $0.006 = $1.08 |
| Same, after diarize-once | 1 | $0.36 |
| `/avatar_voice/samples` with no anchor (today) | 2 | 2 × minutes × $0.006 |

The measured upper bound is $0.014715 per minute, so 60 minutes cost at most 60 × $0.014715 = $0.88290 per pass.

**To measure the real rate** (after bug 5 is fixed, so every pass is recorded), upload one file of known duration on dev and run:
`select sum(cost_usd) from api_metrics where media_job_id = '<job>' and inference_type = 'diarization';` ÷ (duration_seconds / 60).
