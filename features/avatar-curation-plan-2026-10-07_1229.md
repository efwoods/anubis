**Plan date and time:** 2026-10-07 12:29:00 EDT

# Avatar data curation: accuracy and reliability fixes, plus a metering report

## Context

Evan is about to create several new avatars. The data curation behind avatar creation has to be accurate, reliable and cost effective across all three entry points: manual uploads, deep research, and fact editing.

A read-only audit of `wt/f-anubis` (HEAD 43845a9, merged into `test` as 1b8c1ac) found that all three paths write wrong or unreviewed facts:
- Deep research applies facts that only one web page states.
- Deep research fails open: when a check errors, every fact is applied.
- Accepting a research proposal leaves the contradicted fact in place.
- The upload pipeline calls the target person "avatar" instead of the avatar's name.
- Text, PDF and URL uploads never check that the document is about the avatar.
- Deleting or editing a fact through REST never reaches conversations that already loaded the fact.

A peer session's diarization report, `anubis/features/diarization-audit-2026-10-06.md`, covers the diarization internals. This plan builds on that report instead of repeating it.

### Evan's decisions (2026-10-06)

1. **Research auto-apply:** a researched fact applies immediately only when 2 or more independent domains state it. Single-source facts become review cards next to the existing contradiction proposals.
2. **Subject confirmation:** automatic, with a confidence gate. Research stops and raises a review card only when scoping confidence is low. No mandatory confirm click.
3. **Cost:** "free tier does not create avatars". Report the current state of metering and **do not change metering code yet**.
   - Observed: neither `create_avatar` (`src/api/webapp.py:7751`) nor `CreateAvatar*.jsx` contains a tier check. The report records this discrepancy; no gate is added.
4. **Reference voice:** keep the current auto-promotion of the dominant speaker. Two fixes still apply here:
   - Diarize once and reuse the output.
   - Pass the avatar's real name as `target_name`.

## Before editing

- Two other Claude sessions were writing transcripts at 2026-10-06 14:11–14:12 (`a8522047…`, `ba622063…`).
- Re-check transcript modification times and `git -C wt/f-anubis status` before the first edit, following [[concurrent-claude-sessions]].
- Stage only this plan's own hunks.

---

## Phase A: Deep research accuracy

File: `src/anubis/utils/research/deep_research.py`, unless noted otherwise.

1. **The 2-domain rule.**
   - `verify_cluster` (:1003-1060) counts independent registrable domains, not distinct URLs (:1015). Two copies of a page on one domain count once.
   - `partition_verified_facts` (:1075) applies `consistent` clusters with 2 or more domains.
   - Single-domain (`unverified`) clusters become proposals with a new `review_reason="single_source"`, using the existing `build_proposal_document` (:1168) and the existing proposals route and UI.
   - Add a `DEEP_RESEARCH_MIN_INDEPENDENT_SOURCES` setting (default 2) in `context.py`, `.env`, `.env.dev`, and `.env.example` (empty).
2. **Fail closed.**
   - If embedding fails in `cluster_facts` (:987-989), or the verifier call fails (:1053-1059), set `status="unchecked"`.
   - `partition_verified_facts` sends `unchecked` to review, never to apply.
3. **Subject confidence gate.**
   - The scoping step (prompt at :197-208) returns three new fields:
     - `subject_kind`: real public person, fictional character, private person, or other entity
     - `subject_confidence`: 0.0 to 1.0
     - a short list of disambiguating facts
   - The scoping prompt must handle fictional characters (a wiki or source work) and say that a private person may have no public record.
   - Below `DEEP_RESEARCH_MIN_SUBJECT_CONFIDENCE` (new setting, default 0.7), the run writes zero facts and starts no bootstrap or account discovery. The run stores one proposal with `review_reason="subject_ambiguous"`, listing the candidate identities and their source URLs. Resolving that proposal reruns research with the chosen identity as `research_hint`.
   - `create_avatar` research on a personal avatar gets the same `looks_like_a_real_person_name` guard the rename path has (`webapp.py:7848` vs `:8247`).
4. **Keep namesake pages out of the portrait, voice and account steps.**
   - Store each source's `subject_present` verdict on `SearchResult` (from :591).
   - `run_asset_bootstrap` (:1324) and `candidate_accounts_from_research` (`discovered_accounts.py:128`) receive only sources where `subject_present` is true and that supported at least one applied fact.
   - Replace the hardcoded `"confidence": 1.0` (`discovered_accounts.py:364`) with the real value.
5. **The thumbnail check must be able to reject a video** (`asset_bootstrap.py:875-885`).
   - When a portrait exists and no thumbnail matches, acquire no voice and record the outcome `no_matching_video`.
   - A comparison error counts as a non-match.
6. **Accepting a proposal replaces the old fact.**
   - `build_proposal_document` stores the key of `existing_fact` (`identity_key`).
   - `resolve_proposals` (:1518-1580) deletes that key when the owner accepts or edits.
   - The accepted fact is stored with `verification_status="owner_verified"`, not `"inconsistent"` (:1555).
   - Update the label in `AvatarIdentityFacts.jsx:163-170`.
7. **Deduplicate against all held facts, not 200 arbitrary rows** (:378-420, `_EXISTING_FACT_LIMIT` :69).
   - Read with a per-fact semantic query: `store.asearch(namespace, query=fact, limit=5)`, filtered to fact rows (not transcript chunks).
   - Compare each candidate against the held facts that query returns.
   - Owner-edited (`correction_origin="user"`) and `owner_verified` facts are authoritative: a contradicting web fact becomes a proposal and never an overwrite.
8. **Concurrency and claims.**
   - Add a per-avatar lock in `_start_deep_research_job` (`webapp.py:19017`). When the avatar already has a running job, return that existing job.
   - `claim_bootstrap` (`asset_bootstrap.py:234-250`) becomes a conditional write.
   - A `"running"` claim older than `DEEP_RESEARCH_BOOTSTRAP_MEDIA_WAIT_SECONDS × 2` is treated as expired, so a crash or cancel no longer blocks the avatar forever.
9. **Provenance.**
   - Every identity document and proposal gains `research_job_id`, `retrieved_at`, `source_titles` and a numeric `confidence` (`build_identity_document` :1132-1165).
   - Mark the reference audio from bootstrap with `source="deep_research"` and `reference_source_url` (`webapp.py:16662-16680`, `nodes.py:2399`).
10. **Reliability.**
    - The cancel flag is checked after verification and around `await bootstrap_task` (:1403). A cancelled job ends `cancelled`, and the media hand-off and starter regeneration do not run (`webapp.py:19182-19347`).
    - Wrap `run_deep_research` in an overall timeout: `DEEP_RESEARCH_TIMEOUT_SECONDS`.
    - `_cleanup` (`research_jobs.py:141`) never evicts a running job.
    - These failure paths become a job `warnings` list shown in the progress UI, so they stop finishing silently as "completed" with zero facts:
      - an extraction error (:586)
      - a failed read of existing facts (:395)
      - `subject_present=False` on every source
      - no search provider available
11. **Media hand-off.** Send only URLs that supported an applied fact (exclude single-source pages). Videos longer than `DEEP_RESEARCH_BOOTSTRAP_MAX_VIDEO_SECONDS` are skipped (`webapp.py:19193`).

## Phase B: Manual upload accuracy

Files: `src/subgraphs/process_media_graph/utils/{nodes,helper_functions}.py`, `src/anubis/utils/classes/*`.

1. **Real avatar name.** `target_speaker_label` (`nodes.py:2570`) stays the diarizer's speaker label. Every `target_name=` argument (`nodes.py:2695, 2820, 3003, 3015, 3133`) and target attribution (`target_attribution.py`) receive `_assistant_name(config)`.
2. **Diarize once.**
   - On a promoted reference upload, run the main diarization (`nodes.py:2533`) once.
   - Derive the dominant-speaker clip (`nodes.py:2345`) and the voice-clone sample (`nodes.py:174`) from that diarization's segments, instead of calling `isolate_dominant_speaker_audio_b64` (which diarizes internally at `utility.py:1890`).
   - Auto-promotion behavior stays the same.
   - Cross-check the exact steps against the peer report's "Cost model" and "Current bugs" sections before coding.
3. **Subject check for text, PDF and URL uploads.**
   - In `process_text_to_document` (`helper_functions.py:1282`), compare the classifier's inferred `target_name` with the avatar's name.
   - When the inferred name and the avatar's name differ, store the document's content as third-person reference text (namespace `identity`, no first-person rewrite) and emit a `subject_mismatch` progress event, instead of rewriting another person's biography as "I…".
   - Classify a PDF once from its first pages, not page by page (`nodes.py:2067-2090`).
4. **Context-aware fact extraction.**
   - Pass `scene_summary` (`helper_functions.py:960`) into the per-turn `FactRewriterClass.extract` calls (:1003-1031).
   - Add an instruction to the extraction prompt: drop speculative, hypothetical, sarcastic, or quoted-from-someone-else claims.
   - Lone-speaker promotion (`nodes.py:2883-2898`) applies only when that speaker matches the reference clip, not to any single voice.
5. **Full-text biography.** `_build_biographical_identity_documents` (`helper_functions.py:454`) extracts over every 5,000-character window, not only the first one.
6. **Rewrite failure must not store raw text as identity.**
   - `FirstPersonRewriterClass.rewrite` uses `gather(..., return_exceptions=True)` and drops the failed facts.
   - The `else` branch that writes raw chunks (`helper_functions.py:1338-1344, 1463-1483`) is no longer reached on a rewrite error.
7. **Fix provenance pairing.** In `_build_identity_documents_from_facts` (`helper_functions.py:389-392`), pair each fact with the rewritten statement by fact id, not by list index.
8. **Content-hash duplicate detection.**
   - Hash the media bytes or text, or for a URL its canonical id (normalize `youtu.be` and `watch?v=` to one video id).
   - Skip on hash, not filename (`webapp.py:970, 17342-17365`).
9. **Image uploads.** Do not store image descriptions whose only content is `TARGET_NOT_VISIBLE`.
10. **Pages that need JavaScript.** When `_load_article` (`URLDocumentLoaderClass.py:400-470`) returns text that is only an "enable JavaScript" stub or under a minimum length, fail the item with a clear error instead of indexing the stub.

## Phase C: Fact editing

1. **Edits and deletes reach live conversations.**
   - Keep an `identity_revision` row per avatar in the store, bumped by:
     - REST `PUT` and `DELETE /avatar_identity_facts` (`webapp.py:19847, 19873`)
     - `/delete_avatar_document` (:18655)
     - `resolve_proposals`
     - research apply
   - When `load_consciousness` (`src/anubis/utils/nodes.py:1023-1050`) sees a newer revision than the one held in thread state, it re-reads the thread's identity documents by key with a batched get. Missing documents are dropped and edited documents take the new text. The result is written with the existing `op: replace`.
2. **Deleting a fact also sweeps its sources.**
   - Deleting a fact also deletes `quote` rows that share its `original_statement` or `item_job_id`, using the logic of the chat path's `_correction_namespaces` (`identity_tools.py:2184-2192`).
   - Transcript chunks are left alone. They are excluded from the system prompt's identity section only if Evan approves; that step is optional and listed under open questions.
3. **Owner edits are protected.**
   - The re-upload replacement (`vector_store_graph/utils/helper_functions.py:185-245`) keeps rows with `correction_origin="user"` or `verification_status="owner_verified"`.
   - Deep research treats those rows as authoritative (Phase A item 7).
4. **Edit history.** Append to a `correction_history` list (text, timestamp, editor) instead of overwriting `corrected_from` (`identity_tools.py:2410`, `webapp.py:19936`).
5. **Bulk deletes.** `DELETE /avatar_identity_facts` accepts `research_job_id` or `master_job_id` to remove every fact from one run or upload, using Phase A item 9's provenance.
6. **Frontend** (`wt/f-Neural-Nexus-Frontend/src/components/AvatarIdentityFacts.jsx`, `AvatarSettings.jsx`, `research/ResearchPanel.jsx`):
   - Bump `learnedFactsReloadToken` after a document delete (`AvatarSettings.jsx:723-776`).
   - Catch save errors (:113-121, :413-416).
   - Show a failed duplicate delete instead of hiding it (:362-371, :402-411).
   - Add a review-card variant for the `single_source` and `subject_ambiguous` reasons, with "accept all single-source".
   - Show an owner-verified badge.
   - Add a "remove everything from this research run" action.

## Phase D: Metering report (report only, no code changes)

Write `anubis/features/avatar-curation-metering-2026-10-06.md`. Every row must be read from code or measured with a named query; no estimates.

- **Recorded vs not recorded, per curation call site.**
  - Not recorded:
    - research LLM, vision and Tavily calls (the only research row is `inference_type="deep_research"`, `cost_usd=0.0`, `model_name=None`, at `webapp.py:19236-19251`)
    - diarization calls 1 and 2 of a promoted reference upload
    - research bootstrap media
  - Recorded: upload classification and extraction, via `ApiMetricsCallbackHandler` (`metering.py:744-835`) when `media_job_id` is in the run metadata.
- **Settings.** The research defaults in effect, since neither `.env` nor `.env.dev` sets any `DEEP_RESEARCH_*`: 4 topics, 4 queries, 12 sources, 1 follow-up round, concurrency 4, 14 agent turns, Tavily `search_depth:"advanced"`.
- **The tier discrepancy.** "Free tier does not create avatars" vs no gate in `create_avatar` or the create UI. Research starts on every tier (`webapp.py:7837-7848`).
- **Measured cost of one research run.** Create one test avatar on dev and take the OpenAI and Tavily dashboard difference across the run, because `api_metrics` cannot answer this. The run spends money, so ask Evan before running it.
- **Price-field naming hazard.** The `*_per_million_*` fields hold per-token values.

## Docs

Update the `CLAUDE.md` statements the audit proved stale:
- Identity facts are retrieved top-10 per turn (`_RETRIEVAL_LIMIT`, `nodes.py:678`), not "all loaded without limit".
- The Phase 2/3 classifier stubs are replaced by `text_dialogue_segmentation.py`.

---

## Verification

1. **Unit tests.**
   - Run `cd wt/f-anubis && .venv/bin/python -m pytest tests/unit_tests -q -p no:cacheprovider --ignore=tests/unit_tests/test_think_interrupt_flow.py` and compare against the baseline taken at HEAD before the first edit.
   - Run `.venv/bin/python -m ruff check` on the touched files. Never run `ruff format` on `webapp.py`.
2. **New tests:**
   - `partition_verified_facts`: 1-domain → review; 2-domain → apply; `unchecked` → review.
   - `resolve_proposals` removes the old key.
   - The subject-confidence gate writes zero facts.
   - The thumbnail rejection path.
   - Provenance pairing with an empty statement in the middle.
   - Content-hash dedupe on `youtu.be` vs `watch?v=`.
   - The `identity_revision` refresh drops a deleted fact from thread state.
   - Re-upload keeps owner-edited rows.
3. **Dev API.** Coordinate with the active sessions, then `docker restart anubis-dev-langgraph-api-dev-1` and wait until `curl localhost:9600/openapi.json` responds.
4. **End to end on :9600.** Use the dev MCP credentials (admin), with Evan's approval for the spend.
   - Create a public figure (Paul Graham). Confirm with the queries below that every applied fact has 2 or more `source_urls` domains, a `research_job_id`, and `subject_name="Paul Graham"`. Confirm single-source facts appear as proposals.
   - Create an ambiguous name (for example "Michael Jordan" with no description or hint). Confirm the job ends with one `subject_ambiguous` proposal and zero identity facts.
   - Create a fictional character with a wiki URL hint (Jester, Mighty Nein).
   - Upload a podcast where the guest speaks most. Confirm the facts carry the avatar's real name, and that `api_metrics` (the existing call) plus logs show one diarization call.
   - Upload a biography PDF of a different person. Confirm a `subject_mismatch` event and no first-person facts.
   - Edit and delete a fact through the UI at :5173, then send a message in an existing thread. Confirm the deleted fact is absent from the system prompt (LangSmith trace of `think`).
   - Store queries use `value->'document'->'kwargs'->>'page_content'` and `value->'document'->'kwargs'->'metadata'`. A query on `value->'kwargs'` silently returns NULL.

## Open question for implementation time

Should raw transcript chunks stay in the identity retrieval that feeds the system prompt? They restate deleted facts (audit finding C2). Ask Evan before changing it.
