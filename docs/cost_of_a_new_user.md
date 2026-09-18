# What a new user costs

Measured on 2026-09-17 against a brand-new account, then verified line by line
against the vendor consoles. Every figure below comes from either the OpenAI
usage console, the ElevenLabs usage API, the xAI console, or the `api_metrics`
table; nothing here is an estimate unless it says so.

## The headline

**A new user costs about $9.05.** The same account's spend as `api_metrics`
recorded it was **$0.38** — 4% of the truth. That gap is the reason for the
billing changes that accompany this document.

| Source | Figure |
|---|---|
| `api_metrics`, as the product recorded it | $0.38 |
| Reconstructed from LangSmith traces | ~$3.00 |
| **Vendor consoles (the truth)** | **~$9.05** |

## What the account did

One account (`6aac32f516eadce708d1707b`), created 18:35 UTC:

- 3 avatars — a personal avatar, plus avatars of Lex Fridman and Paul Graham
- 2 connected accounts — Gmail and GitHub
- 22 uploads, including **113 minutes of audio and video**
- 47 chat messages across 8 avatars (its own 3, plus 5 public avatars it talked to)
- 5 spoken replies

## Where the money went

| Line | Cost | Share |
|---|---|---|
| `gpt-4o-transcribe-diarize` — audio input $2.99, text input $0.09, text output $2.71 | **$5.79** | 64% |
| `gpt-5.4-nano` — classification, extraction, psycho-analysis, moderation | $2.68 | 30% |
| `gpt-5.6-luna` — 47 chat replies | $0.56 | 6% |
| `gpt-5-nano` — image descriptions | $0.05 | |
| ElevenLabs — 376 characters of speech | $0.06 | |
| xAI — no portrait or emotion video was generated | $0.00 | |
| Tavily — about 20 web searches (estimated; no console reading) | ~$0.16 | |

## Cost per avatar

Vendor cost attributed per avatar, from `api_metrics` rows and LangSmith run
metadata, with diarization apportioned by each avatar's share of media minutes:

| Avatar | What it received | Cost |
|---|---|---|
| Lex Fridman | 13 uploads, **108.4 media minutes**, no chat | **$7.22** |
| Personal avatar | 8 uploads, 4.8 media minutes, 14 messages, 5 spoken replies | **$0.95** |
| Paul Graham | created 21:11, 1 upload plus research | **~$0.43** |
| 5 public avatars it chatted with | 33 messages, no uploads | $0.42 |
| Pipeline work not attributable to one avatar | classification, moderation, helpers | $0.54 |

One avatar is 76% of the bill, and the reason is 108 minutes of audio — not that
the avatar exists, and not that anyone talked to it.

## Cost to create a user

**About $0.03.** Signing up makes no paid call at all. Creating the personal
avatar cost $0.033 in the ten minutes after it appeared, most of which was the
first few chat turns. No deep research, portrait or video fired on this
account's tier.

**Creating avatars is nearly free. Feeding them is what costs money.**

## Expected cost of use

| Action | Cost |
|---|---|
| Chat message | $0.012 |
| Spoken reply | $0.012 |
| **Uploaded media, per minute** | **$0.051 diarization + analysis** |
| Uploaded media, per hour | **~$3.10 and up** |
| Text or PDF upload | $0.01–0.05 by size |
| Creating an avatar | $0.03 |
| Emotion video, when generated | $0.48 each (none fired here) |

A user who sends 100 messages and uploads 30 minutes of media in a month costs
roughly **$3**. One who uploads a two-hour podcast costs **$6 or more in that
single action**.

**Media minutes set the cost of a user; messages barely register.** A cap on
uploaded minutes controls unit economics. A cap on messages does not.

## What was wrong with the accounting, and what changed

### Diarization was computed correctly and then discarded

`gpt-4o-transcribe-diarize` bills **$2.50 per million audio input tokens and
$10.00 per million output tokens**, which is what
`AUDIO_DIARIZATION_PRICE_PER_MILLION_TOKENS_INPUT` and `..._OUTPUT` have always
held, and `_diarize_token_cost` has always applied them correctly. The cost was
simply thrown away before it reached the database — $5.79 of diarization
recorded as $0. It is recorded now.

The per-minute figure (`AUDIO_DIARIZATION_ESTIMATED_PRICE_PER_MINUTE`, $0.006)
is only a fallback for a response that reports no token usage at all, and it is
far too low for one: the real rate works out near **$0.051 per media minute**.
Anything gated on the per-minute estimate — quotas, tier limits, projected burn
— is wrong by roughly 8x.

### Two `/60` bugs

`_transcribe_one_segment_path` and the diarization fallback both multiplied a
duration in **seconds** by a price per **minute**, overstating by 60×. Both
divide by 60 now. Neither had ever shown up, because the cost they computed was
being discarded before it reached the database.

### Uploads recorded nothing

The `document_upload` row is written before the media graph runs, from a
pre-upload token estimate, with no cost. Everything the upload pays for happens
afterwards. Now:

- Every model call made while processing an upload writes its own `api_metrics`
  row, priced from the configured rates and stamped with the media job that made
  it. One handler attached in `init_model` covers every call in the application,
  so classification, dialogue segmentation, first-person rewriting,
  characteristic extraction, the twelve psycho-analysis dimensions and the
  moderation judgement are all recorded without touching their call sites.
- The speech calls report themselves, because they use the OpenAI client
  directly and no LangChain callback ever sees them.
- A chat reply is deliberately left alone: the message endpoint already writes
  that row with its Stripe meter event, and recording it twice would overstate
  every per-message figure the billing portal shows.

### The billing portal billed an estimate

The document-upload meter event was reported **before** processing, carrying a
guess. It is now reported **after** the batch finishes, carrying the tokens the
upload really consumed, summed from the rows its own calls wrote. The estimate
still gates the request — an over-budget upload is still refused before anything
is spent — but it is no longer what the customer pays for. A cancelled or
already-indexed upload now bills nothing.

### Prices that were wrong in the environment files

- `IMAGE_MODEL_COMPLETION_COST=0000004` — a missing decimal point, live in
  production's `.env`. Parsed as a float that is **$4.00 per token**. Corrected
  to `0.00000125`, the measured rate.
- `IMAGE_MODEL_PROMPT_COST` was $0.05 per million against a measured $0.10.
- The production and dev `CLASSIFICATION_MODEL_*` rates disagreed with each
  other; both now carry the measured rates.
- Prompt caching was not modelled at all. Cache writes were the single largest
  line of the inference model's invoice that day. Added:
  `MODEL_CACHED_PROMPT_COST`, `MODEL_CACHE_WRITE_COST`,
  `CLASSIFICATION_MODEL_CACHED_PROMPT_COST`,
  `CLASSIFICATION_MODEL_CACHE_WRITE_COST`, `IMAGE_MODEL_CACHED_PROMPT_COST`.
- Four vendor knobs were documented but set nowhere, so the code used in-code
  defaults: `XAI_IMAGE_COST_PER_IMAGE_USD`, `XAI_VIDEO_COST_PER_SECOND_USD`,
  `ELEVENLABS_TEXT_TO_SPEECH_COST_PER_1000_CHARACTERS_USD`,
  `ELEVENLABS_LIP_SYNC_COST_PER_SECOND_USD`. All four are now set explicitly.

The chat and classification token rates that were already configured turned out
to be **correct** — $0.20 per million input and $1.25 per million output both
match the console to the cent.

## How to re-measure

1. Read the OpenAI console for the day, project **NeuralNexus**, the **Spend
   categories** tab. It gives cost per model split into input, cached input,
   output and, for the diarizer, audio input.
2. Read LangSmith's token counts for the same day and models. Dividing console
   dollars by LangSmith tokens gives each rate; that is how every price in the
   environment files was derived.
3. `api_metrics` should now agree with step 1 for everything except Tavily and
   the speech-character discrepancy below.

## Still open

- **Tavily search is priced nowhere.** About 20 searches went unrecorded; the
  $0.16 above is an estimate at $0.008 per search.
- **Our speech character count disagrees with ElevenLabs.** `api_metrics`
  recorded 1,241 characters for the window; ElevenLabs reported 376 for the same
  voices. The `speech_characters` meter bills customers on our count, so the
  difference matters and should be reconciled against an invoice.
- **Deep research, voice cloning and lip-sync text-to-speech** still record no
  cost outside an upload.
- **`AUDIO_DIARIZATION_PRICE_PER_MILLION_TOKENS_*` are named per million but
  hold per-token values.** The code is consistent with the values; only the
  names mislead. Renaming them means a production container recreate, so the
  names were left alone deliberately.
- **The `gpt-5.4-nano` classification model is no longer used.** Development
  now classifies with `gpt-5-nano`, as production already did, and three classes
  that hard-coded both the model name and its token prices
  (`ImageDescriptionClass`, `FactRewriterClass`, `FirstPersonRewriterClass`) now
  read both from `GlobalContext`. Those hard-coded prices were stale by up to
  4x, so every row they wrote carried a wrong model name and a wrong cost.
