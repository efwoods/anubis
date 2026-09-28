# Watching a livestream together in real time: cost analysis

Written 2026-09-28, after a live session on prod (:8124). Evan watched the Starship
Flight 14 livestream on X with the Shivon Zilis avatar (`ddc68489-…`) over a screen
share. The avatar missed the satellite deployments and said it could not hear the
stream.

**Question:** what would it cost for the avatar to watch and hear a livestream in real
time, the way a person sitting in the room would?

**Short answer:** somewhere between **$0.10 and $4 per viewing hour**, depending on the
architecture. Only the two most expensive architectures react within one second, which
is what "really in the room" needs. Today's pipeline costs well under $0.30 an hour, but
it is effectively blind and deaf to a livestream, for reasons that have nothing to do
with cost (see [What went wrong today](#1-what-went-wrong-today)).

Decision already made by Evan (2026-09-28): **every screen share behaves this way.**
There is no separate watch mode.

---

## 1. What went wrong today

These findings come from the prod API log and the thread checkpoint for 12:57–13:40 UTC.

| # | Finding | Evidence | Effect |
|---|---|---|---|
| 1 | **The unchanged-frame filter drops a livestream.** The filter compares mean brightness against `DEFAULT_AMBIENT_FRAME_CHANGE_THRESHOLD = 0.035` (`Neural-Nexus-Frontend/src/config/ambientFrameChange.js`). The filter also carries a 600 s quiet heartbeat. A spacecraft over a slowly moving Earth, with small telemetry digits changing, stays under 0.035. | Screen observations reached the thread at 12:59:06, 12:59:48, 13:02:29, 13:03:10, 13:11:20, 13:11:57, 13:22:12 and 13:29:25, leaving gaps of 8–11 minutes. | The avatar saw no frame at all during most of the deployment window. |
| 2 | **The screen share never asks for audio.** `requestDisplayMedia()` calls `getDisplayMedia({ video: true, audio: false })` (`src/services/displayCapture.js:31`). | The avatar answered "I can see the livestream, but I can't hear its audio" at 13:01:47. | The avatar has no commentary, no countdown and no cheering. |
| 3 | **Each look takes 20–25 s.** The description (`gpt-5-nano`, measured 10.7 s on average over 1,013 historic rows, 8–16 s today) runs before triage (`gpt-5.6-luna`, 7–12 s today), strictly in series. | Latency breakdown lines, for example turn `841a6c40…`: `resolve_human_message_images_finished` 15 938 ms, `ambient_triage_finished` 25 332 ms. | Even a frame that is captured describes a scene that ended about 20 s earlier. |
| 4 | **Every observation was judged `ignore`.** The speaks-on-intent rules (salience floors, cooldown, "being visible is never a bid") treat a shared screen like a quiet webcam. | Every `[AMBIENT_OBSERVATION … decision=ignore]` in the thread. | The avatar never reacts unprompted, even to "Starship is orbital". |
| 5 | **Observations yield to the person's turns.** An observation that arrives during a typed or spoken turn is refused with 409 (`refuse_ambient_observation_on_busy_thread`). | Evan was talking constantly during the launch. | The busiest moments, when Evan is excited and talking, are the moments with the fewest looks. |
| 6 | **Metering is off.** `api_metrics` has recorded **nothing since 2026-09-22 13:56 UTC**. | `select max(created_at) from api_metrics` | Today's session cost cannot be read from the database. Numbers below use historic rows plus the vendor's per-token prices. |

Findings 1–3 are the "framerate is too slow" symptom, and finding 2 is "cannot hear".
Fixing findings 1, 2 and 4 costs almost nothing, as tier 1 below shows.

---

## 2. Unit costs (measured where possible)

Prices come from `anubis/.env` (prod), which carries the vendor prices the billing code uses.

| Unit | Model | Cost | Latency | Source |
|---|---|---|---|---|
| Describe one still | `gpt-5-nano` ($0.10/M input, $1.25/M output) | **$0.00226** (1,947 input tokens, 1,500 output tokens, most of the output is reasoning) | 10.7 s | `api_metrics`, 1,013 `image_description` rows |
| Triage one observation | `gpt-5.6-luna` ($0.20/M input, $1.20/M output) | **≈ $0.002** (estimate; not metered) | 7–12 s | Prod log timings; tokens estimated from prompt size |
| One avatar reply turn | `gpt-5.6-luna` | **$0.003–0.004** warm cache; **$0.0095** cold. The prompt is about 40 k tokens, about 32 k of which are cached. | 4–5 s | `token_usage` in the prod log |
| Speak one short reaction (about 120 characters) | Cartesia | **$0.005** ($0.04 per 1,000 characters) | < 1 s to first audio | `.env` |
| Transcribe stream audio | `whisper-1` | **$0.006/min = $0.36/hr** | 1–3 s per 10 s chunk | `.env` |
| Diarized transcription | `gpt-4o-transcribe-diarize` | ≈ $0.006/min estimated; the token price is $2.50/M input, $10/M output | Slower | `.env` |
| Non-speech audio events (cheering, engine roar, applause) | MediaPipe Audio Classifier (YAMNet, AudioSet classes) running in the browser | **$0** | < 100 ms | The frontend already loads MediaPipe WASM for motion wireframes |

**Assumption for every tier below:** the avatar reacts out loud about **20 times per viewing
hour** (one reply turn plus one spoken line each), which costs **≈ $0.18/hr**. Reactions
are the only line that grows with how chatty the avatar is.

---

## 3. The options, per viewing hour

"Really there" means seeing continuously, hearing continuously, and reacting within
about one second of an event.

### Tier 0: today (as measured)

- Looks: filter-limited, observed about 16 per hour today, at most 360 per hour at the
  current 30 s browser interval and 10 s server floor.
- Cost: 16 × (0.00226 + 0.002) ≈ **$0.07/hr** as observed, at most **$1.55/hr**.
- Reaction delay: 20–25 s after the frame, plus up to 11 min before a frame is taken at all.
- Hearing: none.
- **Not "really there".** The avatar is blind to slow video, deaf, and silent.

### Tier 1: fix the current pipeline, one still every 10 s

Changes: bypass or loosen the frame filter for screen shares, request tab audio and
transcribe the audio with whisper in 10 s chunks, run the in-browser audio-event
classifier, and let a shared screen count as a standing request to react to milestones.

| Line | Calculation | $/hr |
|---|---|---|
| Describe | 360 × $0.00226 | 0.81 |
| Triage | 360 × $0.002 | 0.72 |
| Whisper | 60 min × $0.006 | 0.36 |
| Audio events | in browser | 0.00 |
| Reactions | 20 × ($0.004 + $0.005) | 0.18 |
| **Total** | | **≈ $2.07** |

Reaction delay: still **20–25 s**. The avatar sees and hears, but reacts like somebody
watching on a 20-second delay.

### Tier 2: frame bursts plus one merged vision-and-triage call (recommended next step)

Changes: the browser grabs a frame every 2 s and sends a **burst of 5 low-detail frames**
every 10 s, along with the whisper transcript and audio events for the same window. One
`gpt-5-nano` call with **minimal reasoning effort** describes the burst as a short clip
and returns the triage decision in the same structured output, which removes the separate
triage call.

- Low detail is about 85 image tokens per frame, so 5 frames are about 425 tokens.
  Adding the prompt gives about 1.5 k input tokens.
- Minimal reasoning cuts output from about 1,500 tokens to about 300.
- About $0.0005 per burst, and **$0.0023 at worst** if reasoning cannot be turned down.

| Line | Calculation | $/hr |
|---|---|---|
| Burst describe and triage | 360 × $0.0005–0.0023 | 0.18–0.83 |
| Whisper | | 0.36 |
| Audio events | in browser | 0.00 |
| Reactions | | 0.18 |
| **Total** | | **≈ $0.72–1.37** |

Reaction delay: **about 3–8 s** after the end of a 10 s window. Events shorter than 10 s
are still caught, because the burst covers the whole window at 2 s spacing.

This tier fits the existing ambient architecture (same endpoint, same hidden observations,
same triage routes) and is the cheapest tier that fixes all three reported symptoms.

### Tier 3: a native realtime model (the "really in the room" tier)

A realtime model takes continuous audio plus periodic frames over one live session and
answers in under a second. The avatar's identity prompt is sent once as session
instructions and then cached.

> ⚠️ **Verify every vendor price in this tier before building.** Tier 3 prices come from
> public list prices as of mid-2025, not from this repository's `.env`, and realtime
> pricing changes often.

**OpenAI `gpt-realtime`** (list: audio in $32/M, cached $0.40/M, audio out $64/M, image in
$5/M, text in $4/M), with text output routed into Cartesia so the avatar keeps its own
voice:

| Line | Calculation | $/hr |
|---|---|---|
| Stream audio in | about 600 tokens/min × 60 × $32/M | 1.15 |
| Frames in | 1 frame per 2 s × about 250 tokens × $5/M | 2.25 |
| Context re-read per response | 20 × about 40 k tokens × $0.40/M (cached) | 0.32 (**$3.20 if the cache misses**) |
| Reactions (text out → Cartesia) | | 0.10 |
| **Total** | | **≈ $3.80** (up to about $6.70) |

**`gpt-realtime-mini`**: roughly one-third of the above, **≈ $1.30–2.20/hr**.

**Google Gemini Live (Flash)**: video at 1 fps is about 258 tokens per frame, which is
about 0.93 M tokens per hour, and audio is about 32 tokens/s, which is about 0.12 M tokens
per hour. At list input prices of $0.50–3/M that is **≈ $0.50–3/hr** before accumulated
context is billed again on each turn. Gemini Live would add a **new model provider**;
`init_model()` supports only `OPEN_AI`, `TOGETHER` and `META`.

Reaction delay: **under 1 s**. Caveats:
- A realtime session holds one socket per viewer for the whole viewing.
- The context window fills within tens of minutes, so the session needs truncation or the
  existing summarization middleware.
- The session bypasses the LangGraph deep agent, so tools, memory writes and the ambient
  checkpoint path must be bridged back into the thread afterwards.

### Tier 4: self-hosted on the RTX 4090

- Components:
  - `faster-whisper` large-v3-turbo transcribes in real time.
  - A small vision-language model (for example Qwen2.5-VL-7B or Moondream) describes a
    frame in about 0.5–2 s.
  - YAMNet classifies sound events.
  - `gpt-5.6-luna` is called only to phrase the reactions.
- Marginal cost: about 450 W × $0.15/kWh ≈ **$0.07/hr** of electricity, plus
  **$0.18/hr** of reactions, **≈ $0.25/hr**.
- Reaction delay: **about 1–3 s**.
- Limits: one GPU serves about 1–3 simultaneous viewers. Operating cost includes uptime,
  queueing, and a second deployment target. The adapter-training plan already claims the
  same GPU.

### Summary

| Tier | $/viewing hour | Reaction delay | Sees slow video | Hears | Fits current architecture |
|---|---|---|---|---|---|
| 0 today | 0.07 (max 1.55) | 20 s – 11 min | ✗ | ✗ | – |
| 1 fixed pipeline | ≈ 2.07 | 20–25 s | ✓ | ✓ | ✓ small change |
| **2 bursts, merged call** | **≈ 0.72–1.37** | **3–8 s** | ✓ | ✓ | ✓ moderate change |
| 3 realtime (OpenAI) | ≈ 3.80 (mini ≈ 1.30–2.20) | < 1 s | ✓ | ✓ | ✗ new session path |
| 3 realtime (Gemini Live) | ≈ 0.50–3 | < 1 s | ✓ | ✓ | ✗ new provider |
| 4 self-hosted | ≈ 0.25 marginal | 1–3 s | ✓ | ✓ | ✗ new infrastructure |

**Per month:** a user who watches 10 hours a month costs about $7–14 on tier 2, $13–38 on
tier 3, and about $2.50 in marginal cost on tier 4. Tier 3 belongs behind the Premium or
Enterprise gate, the same way emotion media is gated.

---

## 4. Recommendation

1. **Ship tier 2 now** on every screen share, per Evan's decision. Tier 2 fixes "cannot
   see", "cannot hear" and "too slow" for about $1 per viewing hour, inside the existing
   ambient pipeline.
2. **Prototype tier 3 with `gpt-realtime-mini`** behind a flag, with text output routed
   into Cartesia so the avatar keeps its cloned voice. The prototype measures real
   per-hour cost on one launch-length session before any tier gate is chosen.
3. **Keep tier 4 as the margin play** once tier 3 proves demand. Self-hosting is the only
   option whose cost does not grow with viewing hours.

---

## 5. Next steps

Markers: **[LLM]** = a Claude session can do the step end to end. **[HUMAN]** = the step
needs Evan (a decision, a vendor console, or a live test in his own browser).

### Tier 2 build

- [ ] **[LLM] Stop the frame filter from dropping video.** Choose one of three fixes and
  test it against a recorded livestream frame sequence:
  - For a **screen** stream, skip `shouldSendCapturedFrame` entirely, as `look_now`
    already does.
  - Lower `VITE_AMBIENT_FRAME_CHANGE_THRESHOLD` for screens only.
  - Compare a block-wise maximum difference instead of the mean.
  - Files: `src/services/ambientCaptureScheduler.js`, `src/config/ambientFrameChange.js`,
    `src/context/MediaShareContext.jsx` (`captureAmbientStills`).
- [ ] **[LLM] Request tab audio.** Change `requestDisplayMedia()` to
  `audio: { suppressLocalAudioPlayback: false }` plus `systemAudio: 'include'`, where
  supported (`src/services/displayCapture.js`). Keep a fallback path for browsers that
  reject audio constraints: phones and Safari give no tab audio.
- [ ] **[LLM] Record the audio in 10 s windows** with `MediaRecorder` on the display
  stream's audio track. Send each window with the burst as a new `stream_audio` form
  field on `POST /message/{assistant_id}` (`ambient=true`). Server: transcribe with
  `transcribe_audio` (whisper-1, already metered as `transcription`) and fold the
  transcript into the observation text as a `stream audio:` section beside `screen:`.
- [ ] **[LLM] Add in-browser audio events** with the MediaPipe `AudioClassifier` (YAMNet)
  on the same audio track. Send the top labels above a confidence floor as
  `stream_sound_events`, rendered like `scene_sound_of` output
  (`background: cheering, engine`).
- [ ] **[LLM] Send frame bursts.** Capture a frame every 2 s into a ring buffer, and send
  the last 5 frames at low detail per observation. Server: describe the burst as one clip
  in one `gpt-5-nano` call with `reasoning_effort="minimal"`. Add a new prompt
  `DESCRIBE_SCREEN_CLIP_PROMPT` in `schema.py` that names what changed across the frames
  (for example a deployment, a stage separation, or on-screen text changes).
- [ ] **[LLM] Merge triage into the burst call** for screen observations. Return
  `{description, decision, salience, reason}` from one structured call, and route that
  output through `route_after_ambient_triage` exactly as a triage decision routes today.
  The unit tests must keep webcam triage unchanged.
- [ ] **[LLM] Treat a watched screen as a standing request.** A screen share should behave
  like the world-facing camera case: the person put the screen in front of the avatar on
  purpose. Add `<SCREEN_SHARED>` guidance to `AMBIENT_CLASSIFY_SYSTEM_PROMPT` so a
  milestone event in the audio or video is a bid. Keep
  `AMBIENT_RESPOND_COOLDOWN_SECONDS` so the avatar does not narrate every burst. Name
  every subject in the prompt text (explicit-naming rule).
- [ ] **[LLM] Stop dropping observations while the person talks.** A screen observation
  refused with 409 should be **queued behind** the person's turn, not discarded, so the
  burst from the excited moment still lands on the thread afterwards.
- [ ] **[LLM] Add the environment variables** to `.env`, `.env.dev`, `.env.example` (no
  values), and `GlobalContext`, plus the `VITE_*` variables in the frontend
  `.env.example`:
  - `AMBIENT_SCREEN_BURST_FRAMES`
  - `AMBIENT_SCREEN_BURST_INTERVAL_SECONDS`
  - `AMBIENT_SCREEN_AUDIO_ENABLED`
  - `AMBIENT_SCREEN_MERGED_TRIAGE_ENABLED`
  - `VITE_AMBIENT_SCREEN_FRAME_SECONDS`
- [ ] **[LLM] Update the "Ambient vision" section of `f-anubis/CLAUDE.md`** with the screen
  path. Add `stream_audio` to the resume context if `look_now` ever needs the audio.
- [ ] **[HUMAN] Live test.** In Chrome, share a **tab** with "Share tab audio" ticked on a
  replayed launch video, then ask "what did they just say?" and "did you see that?".

### Fix the metering (blocks every cost number above)

- [ ] **[LLM] Find why `api_metrics` stopped on 2026-09-22 13:56 UTC.** Check
  `persist_api_metrics_row` (`src/anubis/utils/billing/metering.py`), the prod container
  log for insert errors, and the commits on `anubis` `test` around that date.
- [ ] **[LLM] Meter triage** (currently unpriced) and the new burst call as their own
  `inference_type`s (`ambient_triage`, `screen_clip_description`).
- [ ] **[HUMAN] Read the true session cost** from the OpenAI usage console for
  2026-09-28 12:30–13:45 UTC, and compare it with the $0.07/hr estimate.

### Tier 3 prototype

- [ ] **[HUMAN] Confirm current realtime prices** (OpenAI `gpt-realtime` /
  `gpt-realtime-mini`, Gemini Live Flash) and choose the vendor. Choosing Gemini adds a
  provider to `init_model()`.
- [ ] **[LLM] Build a flagged realtime session:**
  - A WebRTC session from the browser, with an ephemeral key minted by a new API route.
  - Tab audio plus one frame every 2 s goes into the session.
  - The session returns text only.
  - The text is spoken through Cartesia.
  - The session transcript is written back to the LangGraph thread as hidden
    observations, so memory and summarization still see the viewing.
- [ ] **[LLM] Measure one full launch-length session** (about 60 min) and record the
  per-hour cost in this document.
- [ ] **[HUMAN] Decide the tier gate** (Premium or Enterprise) and whether realtime
  replaces tier 2 for gated users or runs beside it.

### Deploy

- [ ] **[HUMAN] Merge `f-anubis` → `anubis` `test`, and `f-Neural-Nexus-Frontend` → the
  frontend.** Prod (`anubis-langgraph-api-prod-1`) bind-mounts the **main `anubis`
  checkout**, not `wt/f-anubis`, so nothing built on the f line reaches prod until the
  merge. Restart prod only after checking for in-flight media jobs
 .
