# YC interview questions: Neural Nexus answers

Written 2026-11-14. The fourteen questions come from the screenshot Evan shared in Slack
(`/home/user/Desktop/File.png`). Each answer uses only values read from a named source:

- **Postgres:** the `postgres16` container, database `postgres`. Dev and prod share this database, so
  every count below includes dev and test traffic. The administrator account (`ADMIN_USER_ID` in `.env`)
  is excluded from every user count.
- **Stripe:** the live key (`STRIPE_SECRET_KEY` in `.env`, `sk_live_`), read 2026-11-14.
- **git:** `git shortlog -sn --all` in each repo under `/home/user/gh/anubis-project/`.
- **docs:** `docs/cost_of_a_new_user.md`, `features/milestones-roadmap.md`.

When a question needs a fact the code and data cannot provide, the answer says **"not measured"** and
names what Evan has to fill in. Those gaps are the real homework from the post: *"if you can't answer
some of them because you don't have customers yet, you know what to do."*

---

## 1. What are you building? Be specific.

Neural Nexus (neuralnexus.site) builds an AI avatar of a specific person from that person's own media.
The user uploads text, audio, video, images, PDFs, URLs, or CSV files, or connects accounts such as Gmail
and GitHub. The pipeline:

1. Transcribes the audio and video.
2. Diarizes the speakers so only the target person's words count as the target's identity.
3. Classifies each document.
4. Extracts facts, verbatim quotes, and personality traits into a vector store.

The avatar then answers in that person's voice. Each turn grounds the avatar in the stored facts and
quotes. The avatar can also speak through a cloned voice, show emotion portraits and video, and run
ambient tasks such as inbox triage and screen-share observation.

Stack, from the repo: a LangGraph agent (`src/anubis/graph.py`) behind a FastAPI API with 136 routes
(`grep -cE '^@app\.(get|post|put|delete|patch)' src/api/webapp.py`), Postgres with pgvector, a React
frontend, a Stripe-metered three-tier subscription, and MCP daemons for desktop and mobile.

## 2. Pick one customer. Walk us through how the customer uses the product.

The heaviest non-administrator account by metered events is account `6ab6daff…` (`api_metrics`):

- **2026-10-02:** signed up and created a personal avatar named "Marshall" (`assistant` table,
  `is_personal_avatar_of_creator = true`).
- **2026-10-02 to 2026-10-04:** used the product on 3 distinct days and opened 9 conversation threads with
  2 avatars (`thread` table).
- **Usage:** 72 chat messages, 69 spoken replies (`speech_synthesis`), 32 diarization calls, 5
  transcriptions, 1 image description, and 1 ambient-triage call (`api_metrics` grouped by
  `inference_type`). The 69 spoken replies against 72 messages show the account used voice mode almost
  every turn.
- **Last activity:** 2026-10-04. This customer did not come back after day 3.

**Not measured:** who the person behind `6ab6daff…` is, why the person signed up, and why the person stopped
after 2026-10-04. Evan has to ask the person directly.

## 3. What does the product actually do for them?

From the code, the product gives a person an avatar that:

- Answers questions in the person's voice and phrasing, grounded in uploaded quotes and facts (`quote` and
  `identity` store namespaces).
- Speaks aloud with a cloned voice (`avatar_voice` and `avatar_voice_clips` tables).
- Can be shared publicly. Six avatars are public today: Jarvis, Uncle Jeff, Claire Wineland, Evan Woods,
  Grant Imahara, and Jane Goodall (`assistant` where `metadata.is_public`).

Production-ready use cases listed in `CLAUDE.md`: memorial or legacy avatars, a KJV-grounded pastor
avatar, restaurant menu and ordering, custom git commit messages, and persistent multi-turn threads.

**Not measured:** which of these outcomes a paying customer values. The data shows what people clicked,
not what the product changed for them.

## 4. Are people using the product yet?

Yes, at a small scale. From `api_metrics`, excluding the administrator, 2026-07-25 to 2026-11-13:

| Month | Distinct non-admin users | Chat messages |
|---|---|---|
| 2026-07 | 5 | 57 |
| 2026-08 | 14 | 359 |
| 2026-09 | 12 | 174 |
| 2026-10 | 13 | 89 |
| 2026-11 (through 11-13) | 3 | 3 |

Totals:

- **Users:** 37 distinct non-administrator user ids with 912 metered events. 21 of the 37 users have
  exactly 1 or 2 events.
- **Avatars:** 45 created by 14 distinct owners (`assistant` table), 12 of them personal avatars.
- **Threads:** 895 conversation threads across all accounts, administrator included (`thread` table).

Messages fell from 359 in August to 89 in October, so usage is declining. Some of the 37 users are Evan's
own test accounts. For example, `6aac32f5…` is the 2026-09-17 cost-measurement account from
`docs/cost_of_a_new_user.md`. Which of the other ids are real outside users is not measured. Matching each
id against Auth0 would measure that.

## 5. How much are they paying?

**$0.00 collected.** The Stripe live account has 0 paid charges (`GET /v1/charges`, 2026-11-14).

- **Customers:** 12 Stripe customers.
- **Subscriptions:** 9 in total. 7 are canceled. 2 are on a Pro trial that started 2026-10-16 and ends
  **2026-11-15**, tomorrow. Neither trial account is the administrator.
- **Pricing:** Pro costs $20.00 a month and Premium costs $50.00 a month (active Stripe prices). Each tier
  adds metered overage on messaging tokens, document-upload tokens, and speech characters.

If both trials convert tomorrow, monthly revenue is 2 × $20.00 = $40.00 before overage.

Cost side, for comparison: one new user who uploaded 113 media minutes cost $9.05 in vendor spend, and
$5.79 of that was diarization (`docs/cost_of_a_new_user.md`). A Pro subscriber who uploads that much media
costs $9.05 of a $20.00 month.

## 6. We see this idea a lot. Why will you be the one to crack it?

**Not measured.** Evan has to write this answer. Evidence from the repos that Evan can cite:

- **Speed:** 1 builder shipped 2,174 commits across 8 repos (git counts in question 9).
- **Data quality:** diarization with a reference clip of the target speaker, so a podcast host's avatar
  does not absorb the guests' words.
- **Background:** Evan's prior work on brain-signal classifiers (`Application5.md`: an fMRI Parkinson's
  study, and a Kernel-device laughter classifier with 78% accuracy) points toward the neural-data
  direction in `_NEURAL_DATA.md`.

The repo also tracks the competitors (`.competitors/`).

## 7. Aren't you just building custom software for every customer?

No. Every avatar runs on one graph, the exported `graph` named `"Anubis"` in `langgraph.json`, with one
media pipeline (`process_media_graph`). A new avatar is new data in the store, not new code. The
restaurant, pastor, and memorial avatars all differ only in uploaded media and the identity documents
built from that media.

The custom-work risk is elsewhere: per-site account connectors (Gmail, GitHub, social crawls) and
per-customer adapter training (`features/adapters.md`). Adapters are planned for Premium only and run on a
manual trigger.

## 8. What will you get done by demo day?

**Not measured.** The demo-day date and a committed target are not in the repo. The open roadmap items in
`features/milestones-roadmap.md` that move the questions above:

- **Turn the 2 trials into paying customers.** The trials end 2026-11-15.
- **Phase 10, latency:** the roadmap target is under 1 s against 2 to 20 s today, as written in the
  roadmap. The current latency is not re-measured.
- **Phase 11, Stripe metering:** live mode still needs the live billing objects provisioned and the live
  webhook registered (`_CONTEXT.md`).
- **Diarization replacement:** OpenAI shuts down `gpt-4o-transcribe-diarize` and `whisper-1` on 2027-02-26
  (memory `diarization-replacement-research`). The current pipeline stops working on that date unless a
  replacement ships first.

Evan has to pick one number, either paying customers or weekly active users, and commit to a target.

## 9. Who does what on the team?

From git authorship (`git shortlog -sn --all`), Evan Woods is the only author in every repo:

| Repo | Commits | First commit |
|---|---|---|
| `anubis` (agent + API) | 1,535 | 2026-01-26 |
| `Neural-Nexus-Frontend` | 556 | 2025-06-09 |
| `anubis-customer-portal` | 34 | 2026-07-13 |
| `anubis-mcp-server-ubuntu-desktop` | 15 | 2026-07-06 |
| `anubis-adapter` | 12 | 2026-07-05 |
| `anubis-mcp-server-mobile` | 11 | 2026-07-27 |
| `anubis-discord` | 8 | 2026-07-08 |
| `anubis-mcp-server-mac-os` | 3 | 2026-08-10 |

Total: 2,174 commits.

**Not measured:** non-engineering roles. The Slack workspace is `neuralnexusteam`, but git shows no second
contributor. If co-founders exist, Evan has to list each person and each person's role. If no co-founder
exists, YC will ask about that next.

## 10. How many customers have you spoken to?

**Not measured.** The repo has no record of customer conversations. Evan has to count customer
conversations from Evan's own records. The 2 trial accounts and the 37 user ids from question 4 are the
first list to call.

## 11. How are you selling?

What the code shows is self-serve: sign up on neuralnexus.site, a free tier, and Stripe Checkout for Pro
with a 30-day trial (`_CONTEXT.md`). The repo lists funding and outreach channels (Kickstarter, Patreon,
StartEngine, the LangChain Slack) in `features/milestones-roadmap.md`, but no sales motion with results.

**Not measured:** where the 37 users came from. No referral or source field is recorded. Adding a signup
source field would measure acquisition.

## 12. What's actually built? Show us.

Built and running on prod (`anubis-langgraph-api-prod-1`):

- **Media ingestion:** files and URLs, transcription, diarization with a reference speaker, and
  classification (roadmap phases 1 and 2).
- **Chat:** streaming SSE chat with persistent threads (phase 9).
- **Avatars:** full create, edit, delete, and share; public avatars; anonymous visitors.
- **Voice:** voice cloning with a standard-voice fallback and a provider switch between ElevenLabs and
  Cartesia.
- **Media generation:** emotion portraits and video.
- **Account connections:** Gmail, GitHub, and social crawls, plus inbox triage.
- **Billing:** three tiers with metering, and an administrator spend panel.
- **Desktop and mobile:** MCP daemons for screen-share and camera observation.
- **Tests:** 198 files in `tests/unit_tests/`.

Specified but not built yet (`features/milestones-roadmap.md`): dialogue formatting (phase 3), target
relabeling (phase 4), most of the 17 identity dimensions (phase 5), the adapter training loop (phase 7),
the eval suite (phase 8), and the latency target (phase 10).

For the "show us" part, Evan should demo one public avatar live, for example Jane Goodall, rather than
walk through a feature list.

## 13. Will your first customers actually use the product?

The data so far says only a few do, and not for long:

- **Retention:** 21 of the 37 non-administrator users have 1 or 2 metered events.
- **Longest use:** only 3 users were active on 10 or more distinct days. `245c0ffc…` was active on 18
  days from 2026-07-25 to 2026-11-02, `ce3420d1…` on 14 days from 2026-07-30 to 2026-08-20, and
  `961c67b7…` on 10 days from 2026-08-10 to 2026-09-02. Whether any of the 3 users is Evan's own account
  is not measured.
- **Monthly messages:** fell from 359 (August) to 89 (October).

Tomorrow's Pro trial outcome (2026-11-15) is the first real paid-usage signal.

## 14. Where do they need the product most?

**Not measured.** No customer has said so on record. Usage points to voice: the heaviest account
(question 2) took a spoken reply on 69 of 72 messages. Across all accounts, `speech_synthesis` is the
second-largest metered category with 1,549 events, after `message` with 3,423. That figure includes the
administrator.

Evan has to confirm the need by asking the 2 trial users and the top accounts from question 13 directly.

---

### Queries used

```sql
-- non-admin users and messages per month (ADMIN = ADMIN_USER_ID from .env)
select date_trunc('month', created_at)::date,
       count(distinct regexp_replace(user_id, '^auth0\|', '')) filter (where regexp_replace(user_id, '^auth0\|', '') <> :ADMIN),
       count(*) filter (where regexp_replace(user_id, '^auth0\|', '') <> :ADMIN and inference_type = 'message')
from api_metrics group by 1 order by 1;
```

Stripe: `GET /v1/charges?limit=100`, `GET /v1/subscriptions?status=all&limit=100`,
`GET /v1/customers?limit=100`, `GET /v1/prices?active=true` with the live key.
