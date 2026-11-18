# Cost-efficient diarization and transcript evaluation (2026-11-17)

Scope: the media-upload diarization path and a local evaluation harness, built on branch `f-anubis` and exercised against the Lex Fridman avatar `59e29e65-1b93-40b0-b579-d68fcc9c3018` on the development stack. Nothing was committed, no avatar data was changed, and no paid vendor call was made.

## Evan's decisions (2026-11-17)

1. **Diarizer:** read a published speaker-labelled transcript first and pay a diarizer only when no published transcript exists; one recorded diarizer pass per upload instead of three; fragment merging; an identity check on the reference; a provider switch; and a measured comparison of AssemblyAI, Deepgram and pyannoteAI against the updated OpenAI path on the same Lex Fridman files.
2. **Evaluation home:** LangSmith datasets and experiments are tabled. No LangSmith trace is consumed. Datasets come from the existing local implementation, and the report states what LangSmith would cost.
3. **Lex Fridman data:** rebuild on the development stack after the fixes.

## What the Lex Fridman avatar holds today

Measured with read-only SQL against the `store` table on 2026-11-17.

| Fact | Value |
|---|---|
| Store rows for the avatar | 2,828 (2,292 `analysis`, 398 `identity`, 89 `quote`, 8 `langsmith_factual_q_and_a`, 8 `q_and_a_adapter`, 4 `multi_turn_dataset_adapter`, 1 `reference_audio`, 28 other) |
| Ingestion date of every row | 2026-09-17 |
| Conversation threads for the avatar | 0 |
| Reference clip | cut from `dialogue_long_clip.mp3`, 4.8 s, transcript "We've been working on a device that can read your mind and we would love to see your thoughts." |
| Quote rows from `mQ7ECcjXazw` | 33, of which 21 are under 40 characters |
| Stored question and answer examples | 81, of which 48 pair a real question with the answer and 33 carry a generated question |
| Examples that are a real question (4 words or more) with a real answer (12 words or more) | 9 |

Two findings decide the rebuild:

- **The reference clip is not Lex Fridman.** The stored clip's own transcript is a guest's line, and on 2026-11-06 the clip's voice matched Brian Johnson's reference recording at cosine 0.987. Every upload after the first was labelled against that clip.
- **The stored question and answer dataset carries the speaker confusion.** The first stored example for `IvqRSEP_o-o` holds, as the avatar's answer, Lex Fridman's question followed by Elon Musk's answer in one string.

An evaluation of today's rows would score the avatar against a guest's words, so the dataset has to be rebuilt before a score means anything.

## What was built

### One recorded diarizer pass per upload

A first upload was diarized three times over the upload's full length (`isolate_dominant_speaker_audio_b64` for the reference clip, the same function again for the voice-clone track, then the main pass) and two of the three calls never reached `api_metrics`.

- The promoted upload is now diarized once (`_diarize_promoted_reference_upload`, `process_media_graph/utils/nodes.py`) and the call is recorded. `isolate_dominant_speaker_audio_b64` accepts that response through a `diarization` argument and cuts the reference clip and the voice-clone track from the response with no diarizer call. The dialogue documents reuse the same response (`_relabel_reference_speaker_segments`).
- A chunked recording diarized with no reference now prefixes every speaker label with the chunk index, so two different people in two chunks never share one label (audit bug 3).
- `isolate_dominant_speaker_audio_b64` returns the diarizer response of the function's own paid call under `diarization_response`. The three voice routes in `webapp.py` record that call (`_record_isolation_diarization_usage`), and `/avatar_voice/samples` cuts the anchor from the take's first diarization instead of paying for a second one.

Diarizer calls per upload, before and after:

| Case | Before | Recorded before | After | Recorded after |
|---|---|---|---|---|
| Upload with a usable stored reference | 1 | 1 | 1 | 1 |
| First upload, promoted to the reference | 3 | 1 | 1 | 1 |
| First upload that is the calibration sentence | 2 | 0 | 1 | 1 |
| `/avatar_voice/samples` with no stored anchor | 2 | 0 | 1 | 1 |
| `/avatar_voice/reference` | 1 | 0 | 1 | 1 |
| YouTube video with a published transcript | 1 or 3 | 1 | 0 | not applicable |

At OpenAI's published estimate of $0.006 per minute, a 60-minute first upload goes from 3 × 60 × $0.006 = $1.08 to 1 × 60 × $0.006 = $0.36. The real per-minute rate is not measured; the only measured figure is the upper bound of $0.014715 per minute from 2026-09-17 (`features/diarization-audit-2026-10-06.md` in the `anubis` checkout).

### Identity check on the reference

The reference used to be whoever talked longest. `identify_avatar_speaker_labels` (`target_attribution.py`) makes one text-only structured-output call over the transcript the upload already paid for and decides which speaker labels are the avatar by name: self-introduction, being addressed by name, introduction by a host, first-person biography. No audio is sent.

- `identified` with high or medium confidence: the reference clip and the avatar's turns come from those labels, even when another speaker talks more.
- `absent` with high or medium confidence: no reference is stored from the recording.
- `undetermined`, a low-confidence verdict, or a failed call: the dominant-speaker rule applies as before.

The cost of the identity call has not been measured; the call is recorded through the existing upload recorder when the run carries a `media_job_id`.

### Fragment merging

`merge_interleaved_fragments` (`helper_functions.py`) regroups runs of alternating sub-second segments per speaker before turns are built. Measured on the stored 2026-09-17 diarization dumps for the avatar (`data/59e29e65-…/transcriptions/`), with the default 1.0-second fragment limit:

| Recording | Segments | Turns before → after | Turns under 40 characters before → after | Words before → after |
|---|---|---|---|---|
| `mQ7ECcjXazw` | 88 | 68 → 31 | 53 → 14 | 720 → 720 |
| `CPvkze1twvY` | 4,208 | 1,683 → 1,226 | 1,008 → 512 | 28,855 → 28,855 |
| `IvqRSEP_o-o` | 101 | 34 → 32 | 11 → 9 | 1,185 → 1,185 |
| `dialogue_long_clip.mp3` | 28 | 13 → 11 | 7 → 5 | 115 → 115 |

No word is dropped or added. The limit is `DIARIZATION_FRAGMENT_MAXIMUM_SECONDS`.

### Published transcript first

`src/anubis/utils/speech/published_transcripts.py` reads a transcript that the recording's source publishes with a speaker name and a timestamp on every turn. When one exists for the exact YouTube video id of an upload, no audio is sent to a diarizer, the speakers carry real names, and the avatar's turns are the turns published under the avatar's name. A transcript is applied only when the transcript's own timestamp links point at the same video id, so an episode's transcript is never applied to a clip cut from the episode.

One source is registered: lexfridman.com. Verified on 2026-11-17 against the live site:

- The index page `https://lexfridman.com/podcast` lists 503 episodes, 124 of them with a transcript link.
- `Kbk9BiPhm7o` (Elon Musk and the Neuralink team) parses to 1,772 segments, 766 of them Lex Fridman's, ending at 31,035.8 s.
- Of the avatar's current sources, none has a transcript on the index: `IvqRSEP_o-o` and `n8TZ7ppfgDE` are Lex Clips cuts, `mQ7ECcjXazw` is a 191-second clip, and `0m3hGZvD-0s`, `bCA54RIkpTo` have no transcript. `7Sk6lTLSZcA` has a transcript page that the index does not list (the page is in the site's sitemap only), so that video still goes to the diarizer.

Consequence for the rebuild: ingesting full episodes that have a published transcript costs $0.00 in diarization and yields named, correct speakers. The switch is `PUBLISHED_TRANSCRIPT_LOOKUP_ENABLED`.

### The avatar's real name instead of "avatar"

Fact extraction, scene summaries, target attribution and document metadata now receive the avatar's real name (`avatar_target_name`). The diarizer's known-speaker label stays `avatar`. This closes the latent leak in which a stored fact used "avatar" as a person's name.

### Provider switch

`AUDIO_DIARIZATION_PROVIDER` selects `OPENAI` (default), `ASSEMBLYAI`, `DEEPGRAM` or `PYANNOTEAI` (`src/anubis/utils/speech/diarization_providers.py`, called from `transcribe_audio_diarize`). The hosted providers take a whole recording in one request, so a long recording is not chunked. The avatar's turns are then identified by the existing transcript-reading attribution pass.

The request code follows each vendor's documentation as read on 2026-11-17. The response-normalizing code is unit-tested. **No live request to AssemblyAI, Deepgram or pyannoteAI has been made, because no key exists.**

## Vendor facts (read 2026-11-17)

| | OpenAI `gpt-4o-transcribe-diarize` | AssemblyAI | Deepgram Nova-3 | pyannoteAI |
|---|---|---|---|---|
| List price with diarization | $0.006 per minute = $0.36 per hour (published estimate; billed per token) | $0.21 + $0.02 = $0.23 per hour | 60 × $0.0043 = $0.258 per hour, diarization included | €0.168 per hour with transcription (Developer plan); €0.112 per hour diarization only (Precision-3) |
| Maximum file | 25 MB | 2.2 GB upload, 10 hours | 2 GB | 1 GiB, 24 hours |
| Known speaker from a reference clip | yes, in the request | no (names inferred from transcript text) | not found in the documentation | yes, a separate voiceprint endpoint that returns no words |
| Shutdown | 2027-02-26 | none listed | none listed | none listed |
| Published diarization error rate | not found | not found | not found (chart only) | not found as text (chart only) |

Sources: developers.openai.com/api/docs/pricing and /deprecations; assemblyai.com/pricing; deepgram.com/pricing; pyannote.ai/pricing; each vendor's documentation. The pyannoteAI table was matched by row order; the page's calculator confirms the Precision-3 and the transcription rows on the Developer plan only. No replacement OpenAI model documents `diarized_json` or known-speaker references, so the OpenAI path ends on 2027-02-26 whatever the comparison shows.

**Accuracy on Evan's media: not measured.** The measurement is `scripts/diarization_bakeoff.py`, which runs the same recordings through every provider that has a key and scores each against the published transcript: word error rate, speaker attribution accuracy, precision and recall for Lex Fridman's turns, cost, and speed. The planned first run, with list-price cost per provider for 1,800 s of `JN3KPFbWCy8` (which has a published transcript):

| Provider | Arithmetic | Cost |
|---|---|---|
| OpenAI | 1,800 s / 3,600 × $0.36 | $0.18000 |
| AssemblyAI | 1,800 s / 3,600 × $0.23 | $0.11500 |
| Deepgram | 1,800 s / 3,600 × $0.258 | $0.12900 |
| pyannoteAI | 1,800 s / 3,600 × €0.168 | €0.08400 |

```
docker exec anubis-dev-langgraph-api-dev-1 python scripts/diarization_bakeoff.py \
  --video https://www.youtube.com/watch?v=JN3KPFbWCy8 \
  --avatar-name "Lex Fridman" --maximum-seconds 1800
```

## Transcript evaluation harness (local, no LangSmith traces)

`scripts/evaluate_avatar_against_transcripts.py` with `src/anubis/utils/evaluation/transcript_evaluation.py`.

- **Dataset:** the stored `langsmith_factual_q_and_a` rows, read in place. An example counts as a real question only when a stored quote row records the same question as what was really said before the answer.
- **Trace:** each question goes to `POST /message/{assistant_id}` on a new thread with `stream=false` and the new `evaluation=true` form field. The full record of each question (question, real reply, generated reply, thread id, usage, style analysis, judge verdicts, scores) is one line of `traces.jsonl`. `stream=false` never registers a LangSmith run, and `evaluation=true` (administrator only) skips the learning observation so a replayed interviewer's words are not learned as the administrator's own.
- **Accuracy with respect to the conversation:** a judge model splits the generated reply into claims and marks each claim supported by, contradicted by, or absent from the real reply. Claim precision = supported / all generated claims. Point recall = supported / (supported + real points the reply missed). The embedding cosine similarity of the two replies is computed locally at no model cost.
- **Verisimilitude of the generated text:** the squared Mahalanobis distance of the reply's 28 style features from the avatar's direct-quote cloud, divided by the avatar's empirical threshold (below 1.0 the reply sits inside the person's own quotes); the isolation-forest verdict; the distance from the unmodified-model baseline; and each style feature's mean over the generated replies against the real replies, in the real replies' own standard deviations. The message path now publishes the measured distances under `response_metadata["authenticity_distances"]` beside the existing yes-or-no verdicts.
- **Tuning:** each run writes `summary.json` and `report.md` under `data/evaluations/<assistant_id>/<run label>/`. A prompt or data change is tuned by comparing two runs' summaries on the same examples.

Dry run on today's rows (no model called): 81 stored examples, 9 selected (5 from `IvqRSEP_o-o`, 3 from `mQ7ECcjXazw`, 1 from `dialogue_long_clip.mp3`).

### Cost of an evaluation run

| Item | Value |
|---|---|
| LangSmith traces sent | 0 |
| LangSmith cost if the same turns were traced | examples × (46,515 traces / 6,704 invocations) × ($182.58 / 46,515 traces); for 9 examples = $0.24511 |
| Avatar reply cost per example | not measured; the run sums `api_metrics.cost_usd` over the run's thread ids |
| Judge cost per example | not measured; the run prices the judge's reported tokens at `CLASSIFICATION_MODEL_PROMPT_COST` and `CLASSIFICATION_MODEL_COMPLETION_COST` |

The LangSmith figure derives from invoice GEHMVWPY0003 (2026-10-03): $182.58 for 46,515 base traces produced by 6,704 graph invocations, which is $182.58 / 6,704 = $0.027234 per avatar turn. A three-example pilot measures the two model costs:

```
docker exec -e EVALUATION_API_KEY=<administrator API-KEY> anubis-dev-langgraph-api-dev-1 \
  python scripts/evaluate_avatar_against_transcripts.py \
  --assistant-id 59e29e65-1b93-40b0-b579-d68fcc9c3018 --limit 3
```

## Status of the known bugs

| Bug | Status |
|---|---|
| Overlapping speech scattered into one-word quote rows | fixed in code; stored rows unchanged until the rebuild |
| Reference is the dominant speaker with no identity check | fixed in code |
| Stored Lex Fridman reference clip is a guest's voice | data; fixed by the rebuild |
| Chunk labels collide when no reference exists | fixed for the promoted upload path |
| Three diarizer passes per promoted upload | fixed |
| Isolation passes never recorded | fixed at all five call sites |
| "avatar" passed as the person's name | fixed |
| `transcribe_audio` unbound in the fallback path (`test_a_failed_diarization_does_not_claim_every_speaker_as_the_avatar`) | fixed |
| Chunks cut at equal time intervals split a turn across two chunks | still present on the OpenAI path; the hosted providers do not chunk |
| The diarizer and `whisper-1` shut down on 2027-02-26 | open; the provider comparison decides the replacement |
| "prompt questions are not actually prompting the response" (`IvqRSEP_o-o`) | cause verified: guest and host turns were fused under one label; fixed by a correct reference or a published transcript |

## What needs Evan

1. **Vendor keys** for the comparison: `ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, `PYANNOTEAI_API_KEY` in `.env.dev`.
2. **Recreate the development API container** so the new settings and code load (`env_file` is read when the container is created).
3. **Approve the Lex Fridman rebuild on the development stack:** delete the avatar's 2,828 store rows, store a verified Lex Fridman reference clip, and re-ingest. Sources to add are interviews in which Lex Fridman is the guest (for example `tlOyZSAZh2k`), because those are the recordings in which Lex Fridman answers questions; in Lex Fridman's own podcast the real exchanges are a guest's statement followed by Lex Fridman's next question.
4. **Approve the paid runs:** the provider comparison and the evaluation pilot above.
