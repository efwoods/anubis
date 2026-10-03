# Watching a livestream together: cost per viewing minute

Written 2026-09-28. The source document is
[`realtime_livestream_watching_cost_analysis.md`](realtime_livestream_watching_cost_analysis.md).
The source document reports costs per viewing hour. Every figure below is the source
document's input divided down to one viewing minute, with the arithmetic shown. Where the
source document rounded a per-hour total, the per-minute figure below comes from the
unrounded inputs.

**Short answer:**

- The current screen share costs **$0.001136 per viewing minute** at the observed look rate.
  The current screen share hears nothing and reacts 20–25 s late.
- A native realtime model that sees and hears the stream (OpenAI `gpt-realtime`) costs
  **$0.0637 per viewing minute**, or **$0.1117 per viewing minute** if the prompt cache misses.
- The recommended next step (tier 2: frame bursts, whisper audio, one merged call) costs
  **$0.012–$0.0228 per viewing minute**.

---

## 1. Measurement status

- `api_metrics` has recorded nothing since **2026-09-22 13:56 UTC**, so the true cost of the
  2026-09-28 session is **not measured**. The OpenAI usage console for 2026-09-28
  12:30–13:45 UTC would measure the true cost.
- Triage cost ($0.002 per observation) is an **estimate**. Triage is not metered.
- Tier 3 vendor prices are **public list prices from mid-2025**, not prices from `anubis/.env`.
  Verify the tier 3 prices before building.

---

## 2. Cost per picture and picture rate (today)

**One picture costs $0.00426**, not $0.001136. The $0.001136 figure is the cost per viewing
minute, which spreads the per-picture cost across the minutes between pictures.

| Step per picture | Model | Cost | Status |
|---|---|---|---|
| Describe the picture | `gpt-5-nano` | $0.00226 | Measured: mean of 1,013 `image_description` rows in `api_metrics` |
| Triage the observation (react or ignore) | `gpt-5.6-luna` | $0.002 | Estimated: triage is not metered |
| **Total per picture** | | **$0.00426** | Partly measured |

Per-minute derivation at the source document's rate: 16 pictures per hour × $0.00426 ÷ 60 min
= $0.001136 per minute.

**Picture rate.** The source document states 16 pictures per hour. The thread log for the
2026-09-28 session shows **8 pictures in the 43-minute window 12:57–13:40 UTC**, at 12:59:06,
12:59:48, 13:02:29, 13:03:10, 13:11:20, 13:11:57, 13:22:12 and 13:29:25:

| Rate source | Pictures per hour | Arithmetic | $/hr | $/min |
|---|---|---|---|---|
| Source document | 16 | 16 × $0.00426 | 0.06816 | 0.001136 |
| Thread log | 8 ÷ 43 × 60 = 11.16 (rounded to 2 decimal places) | 8 × $0.00426 ÷ 43 × 60 | 0.04755 (rounded to 5 decimal places) | 0.00079256 (rounded to 8 decimal places) |
| Source document's stated cap | 360 | 360 × $0.00426 | 1.5336 | 0.02556 |

**The picture rate is not fixed.**

- The browser attempts a capture every 30 s, which is 120 attempts per hour.
- The unchanged-frame filter drops any frame whose mean brightness differs from the previous
  sent frame by less than `DEFAULT_AMBIENT_FRAME_CHANGE_THRESHOLD = 0.035`
  (`Neural-Nexus-Frontend/src/config/ambientFrameChange.js`). A slowly changing livestream
  rarely crosses the 0.035 threshold, so the picture count depends on how much the screen
  changes.
- A 600 s quiet heartbeat still forces one picture after 600 s without a sent frame.
- The source document states a cap of 360 pictures per hour from the 10 s server floor. The
  30 s browser interval alone limits the rate to 120 pictures per hour, so the 360 cap only
  applies if the browser interval shortens.

**Measurement gap.** Only the $0.00226 describe cost is measured. Triage is not metered, and
`api_metrics` has recorded nothing since 2026-09-22 13:56 UTC. The OpenAI usage console for
2026-09-28 12:30–13:45 UTC would measure the true session total.

---

## 3. Measured on dev, 2026-09-29 (screen share)

Session: localhost:5173 (frontend) → localhost:9600 (`anubis-dev-langgraph-api-dev-1`, source
`wt/f-anubis`), thread `7c850ca6-f84b-44d2-8d10-0c06f24cb7a1`. Dev metering writes
`api_metrics`, so the image-description figures below are measured, not estimated.

| Measure | Value |
|---|---|
| Pictures described | 7, completed between 18:57:32 and 19:02:59 UTC (327.07 s) |
| Picture rate | 6 gaps × 3,600 ÷ 327.07 s = **66.04 pictures per hour** (one picture every 54.51 s) |
| Description cost per picture | **$0.00251115** mean ($0.01757805 ÷ 7); minimum $0.0016949, maximum $0.0037074 |
| Description tokens per picture | 4,049 input tokens on every picture; 1,032–2,642 output tokens |
| Description latency | 9,093–17,405 ms; mean 12,825 ms |
| Description cost per hour | 66.04 × $0.00251115 = **$0.16584** |
| Description cost per minute | $0.16584 ÷ 60 = **$0.0027640** |
| Triage cost | **Not measured.** Triage metering (`inference_type = 'ambient_triage'`) was added on 2026-09-29 and is live only after the dev container restarts. |

Source query:

```sql
select count(*), sum(cost_usd), sum(cost_usd) / count(*), min(cost_usd), max(cost_usd),
       extract(epoch from max(created_at) - min(created_at))
from api_metrics
where thread_id = '7c850ca6-f84b-44d2-8d10-0c06f24cb7a1'
  and inference_type = 'image_description';
```

Triage query, once triage metering is live:

```sql
select inference_type, count(*), sum(cost_usd), sum(cost_usd) / count(*)
from api_metrics
where thread_id = '<thread id>'
  and inference_type in ('image_description', 'ambient_triage')
group by inference_type;
```

**Compared with section 2 (2026-09-28 prod figures):**

- Picture rate: 66.04 pictures per hour measured, against 16 pictures per hour in the source
  document (4.13 times as many pictures).
- Input tokens per description: 4,049 measured, against 1,947 in the historic `api_metrics`
  mean (2.08 times as many input tokens).
- Cost per description: $0.00251115 measured, against $0.00226 historic.

**Excluded from the screen-share figures:**

- One description started at 19:00:19 UTC (dev API log) and wrote no `api_metrics` row, so
  the cost of that description is not recorded.
- The thread's 11 `message` rows ($0.08240522 in total) are the conversation partner's spoken
  turns and the avatar's replies, not screen-share cost.

**Per-reply cost reporting.** Every avatar reply's `response_metadata.turn_cost` carries the
reply's own cost plus every image description and triage call made since the previous reply,
including pictures judged `ignore`, with a per-call breakdown and a total
(`turn_cost.total_cost_usd`). The frontend shows `turn_cost.total_cost_usd` under the reply.

---

## 4. Unit costs

Prices come from `anubis/.env` (prod) unless the source column says otherwise.

| Unit | Model | Cost | Latency | Source |
|---|---|---|---|---|
| Describe one still | `gpt-5-nano` ($0.10/M input, $1.25/M output) | $0.00226 (1,947 input tokens, 1,500 output tokens) | 10.7 s mean | `api_metrics`, 1,013 `image_description` rows |
| Triage one observation | `gpt-5.6-luna` ($0.20/M input, $1.20/M output) | $0.002 (estimate, not metered) | 7–12 s | Prod log timings |
| One avatar reply turn | `gpt-5.6-luna` | $0.004 warm cache (upper end of the $0.003–$0.004 observed range); $0.0095 cold | 4–5 s | `token_usage` in the prod log |
| Speak one 120-character reaction | Cartesia ($0.04 per 1,000 characters) | $0.005 | < 1 s to first audio | `.env` |
| Transcribe stream audio | `whisper-1` | **$0.006 per minute** | 1–3 s per 10 s chunk | `.env` |
| Non-speech audio events | MediaPipe Audio Classifier (YAMNet), in the browser | $0 | < 100 ms | Frontend already loads MediaPipe WASM |

**Assumption for every tier:** the avatar reacts out loud 20 times per viewing hour, which is
20 ÷ 60 = **1/3 reaction per viewing minute**. One reaction is one reply turn plus one spoken
line: $0.004 + $0.005 = $0.009. Reaction cost per viewing minute = $0.009 ÷ 3 = **$0.003**.

---

## 5. The tiers, per viewing minute

### Tier 0: today (as measured)

One look = describe + triage = $0.00226 + $0.002 = **$0.00426**.

| Look rate | Looks per minute | Arithmetic | $/min |
|---|---|---|---|
| Source document's observed rate (16 looks per hour) | 16 ÷ 60 = 4/15 | 4/15 × $0.00426 | **$0.001136** |
| Thread log count (8 observations in the 43-minute window 12:57–13:40 UTC) | 8 ÷ 43 | 8 × $0.00426 ÷ 43 = $0.03408 ÷ 43 | **$0.00079256** (rounded to 8 decimal places) |
| Cap (30 s browser interval, 10 s server floor) | 6 | 6 × $0.00426 | **$0.02556** |

- Reaction cost: $0. Every observation was judged `ignore`.
- Reaction delay: 20–25 s after the frame, plus up to 11 min before a frame is taken at all.
- Hearing: none (`getDisplayMedia({ video: true, audio: false })`,
  `src/services/displayCapture.js:31`).

### Tier 1: fix the current pipeline, one still every 10 s (6 stills per minute)

| Line | Arithmetic | $/min |
|---|---|---|
| Describe | 6 × $0.00226 | 0.01356 |
| Triage | 6 × $0.002 | 0.012 |
| Whisper | 1 min × $0.006 | 0.006 |
| Audio events | in the browser | 0 |
| Reactions | 1/3 × $0.009 | 0.003 |
| **Total** | 0.01356 + 0.012 + 0.006 + 0.003 | **0.03456** |

Reaction delay: 20–25 s.

### Tier 2: frame bursts plus one merged vision-and-triage call (recommended next step)

One burst of 5 low-detail frames every 10 s = 6 bursts per minute. One `gpt-5-nano` call per
burst with minimal reasoning describes the burst and returns the triage decision.
Burst cost is an **assumption** from the source document: $0.0005 with minimal reasoning,
$0.0023 if reasoning cannot be turned down.

| Line | Arithmetic | $/min |
|---|---|---|
| Burst describe and triage | 6 × $0.0005 to 6 × $0.0023 | 0.003–0.0138 |
| Whisper | 1 min × $0.006 | 0.006 |
| Audio events | in the browser | 0 |
| Reactions | 1/3 × $0.009 | 0.003 |
| **Total** | 0.003 + 0.006 + 0.003 to 0.0138 + 0.006 + 0.003 | **0.012–0.0228** |

Reaction delay: 3–8 s after the end of each 10 s window.

### Tier 3: native realtime model (reacts in under 1 s)

**OpenAI `gpt-realtime`** (mid-2025 list: audio in $32/M, cached text $0.40/M, text in $4/M,
image in $5/M), with text output spoken through Cartesia.

Assumed inputs, from the source document: 600 audio tokens per minute, 1 frame every 2 s
(30 frames per minute) at 250 tokens per frame, 40,000 context tokens re-read per response.

| Line | Arithmetic | $/min |
|---|---|---|
| Stream audio in | 600 × $32/M | 0.0192 |
| Frames in | 30 × 250 = 7,500 tokens × $5/M | 0.0375 |
| Context re-read, cached | 1/3 × 40,000 × $0.40/M = $0.016 ÷ 3 | 0.005333… |
| Cartesia speech | 1/3 × $0.005 = $0.005 ÷ 3 | 0.001666… |
| **Total** | $3.822 per hour ÷ 60 | **0.0637** |
| **Total if the cache misses** | context at $4/M: 1/3 × 40,000 × $4/M = $0.16 ÷ 3 = $0.053333…; $6.702 per hour ÷ 60 | **0.1117** |

The table leaves out the realtime model's own text-output tokens (not estimated in the source
document).

**`gpt-realtime-mini`:** the source document's $1.30–$2.20 per hour becomes
$1.30 ÷ 60 = **$0.021666…** to $2.20 ÷ 60 = **$0.036666…** per minute. The source document
derives the mini range from a one-third ratio, not from mini's own prices, so the mini range
is an assumption.

**Google Gemini Live (Flash):** 1 frame per second at 258 tokens = 60 × 258 = 15,480 tokens
per minute; audio at 32 tokens per second = 60 × 32 = 1,920 tokens per minute; total 17,400
tokens per minute.

| List input price | Arithmetic | $/min |
|---|---|---|
| $0.50/M | 17,400 × $0.50/M | **0.0087** |
| $3/M | 17,400 × $3/M | **0.0522** |

The Gemini figures exclude accumulated context billed again on each turn. Gemini Live also
needs a new provider in `init_model()` (supports only `OPEN_AI`, `TOGETHER`, `META`).

### Tier 4: self-hosted on the RTX 4090

| Line | Arithmetic | $/min |
|---|---|---|
| Electricity (assumed 450 W at $0.15/kWh) | 450 W × 1/60 h = 0.0075 kWh × $0.15 | 0.001125 |
| Reactions | 1/3 × $0.009 | 0.003 |
| **Total marginal** | 0.001125 + 0.003 | **0.004125** |

The source document rounds electricity to $0.07 per hour. The unrounded figure is
$0.001125 × 60 = $0.0675 per hour, so the unrounded tier 4 total is $0.2475 per hour.
Reaction delay: 1–3 s. One GPU serves 1–3 simultaneous viewers (source document estimate).

---

## 6. Summary

| Tier | $/viewing minute | Reaction delay | Sees slow video | Hears | Fits current architecture |
|---|---|---|---|---|---|
| 0 today | 0.001136 observed (cap 0.02556) | 20 s – 11 min | ✗ | ✗ | – |
| 1 fixed pipeline | 0.03456 | 20–25 s | ✓ | ✓ | ✓ small change |
| **2 bursts, merged call** | **0.012–0.0228** | **3–8 s** | ✓ | ✓ | ✓ moderate change |
| 3 realtime (OpenAI) | 0.0637 (cache miss 0.1117) | < 1 s | ✓ | ✓ | ✗ new session path |
| 3 realtime (OpenAI mini) | 0.021666…–0.036666… | < 1 s | ✓ | ✓ | ✗ new session path |
| 3 realtime (Gemini Live) | 0.0087–0.0522 | < 1 s | ✓ | ✓ | ✗ new provider |
| 4 self-hosted | 0.004125 marginal | 1–3 s | ✓ | ✓ | ✗ new infrastructure |

**Per month, for a user who watches 600 minutes (10 hours):**

| Tier | Arithmetic | $/month |
|---|---|---|
| 0 today (observed rate) | 600 × $0.001136 | 0.6816 |
| 1 fixed pipeline | 600 × $0.03456 | 20.736 |
| 2 bursts, merged call | 600 × $0.012 to 600 × $0.0228 | 7.20–13.68 |
| 3 realtime (OpenAI) | 600 × $0.0637 (cache miss 600 × $0.1117) | 38.22 (67.02) |
| 3 realtime (Gemini Live) | 600 × $0.0087 to 600 × $0.0522 | 5.22–31.32 |
| 4 self-hosted | 600 × $0.004125 | 2.475 |
