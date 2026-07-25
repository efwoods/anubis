


# BUG LISTING:
## BUG 1: anonymous usage does not report accurate usage in the customer-portal; 

# src/security/auth.py get_anonymous_user_with_anonymous_api_key:
I need to use the hashed_ip for the simulation of different anonymous users (they are the result of using a VPN then using the _hash_key function to create the hashed_ip) of each vpn; I will manually alter these values during dev testing:
    if request.app.state.context.dev == "TRUE":
        hashed_ip = _hash_key("172.18.0.1")
        # hashed_ip = '2a1201bb6c0061be63fc4ce58a048136fa91d3afea9e21f62ae7988a20cc09f1' # VPN_SIMULATED
        # hashed_ip = '72aefc13eebd36bf5ec1cbfa1f2e930117a62e07f600dc618c18725f3d52be15' # NO_VPN_SIMULATED


In production, i am identifying the following bug (Messaging tokens342,864 / 200,000 tokens (+142,864 over)):
Neural Nexus

Customer portal
Browsing as anonymous

You are currently an anonymous user. Usage below is read-only until you sign up. To start the free Pro trial (to create avatars, upload documents, and more!) use Sign up for free Pro trial in the plan section.
Current subscription
active

Neural Nexus Free Tier — Base Subscription

Price varies with usage
Upgrade from anonymous free tier

Anonymous usage is tracked by a hash of your network address (the same scheme Neural Nexus uses). Create an account with email and password to start the free Pro trial, then open Stripe Checkout.
Neural Nexus Free Tier — Base Subscription

$0/month

    Messaging tokens: 200,000 tokens/month · $2.00 per 1M over

Neural Nexus Pro Tier — Base Subscription

$20.00/month

30-day free trial

    Document upload tokens: 10,000,000 tokens/month · $3.00 per 1M over
    Messaging tokens: 5,000,000 tokens/month · $1.50 per 1M over

Neural Nexus Premium Tier — Base Subscription

$50.00/month

    Adapter training: 5 units/month · $5.00 per unit over
    Adapter inference tokens: 10,000,000 tokens/month · $4.00 per 1M over
    Document upload tokens: 40,000,000 tokens/month · $2.50 per 1M over
    Messaging tokens: 20,000,000 tokens/month · $1.25 per 1M over

Usage this period
Jul 15, 2026 – Aug 15, 2026
Messaging tokens342,864 / 200,000 tokens (+142,864 over)
0 tokens remaining$2.00 per 1,000,000 tokens over allotment
-----
## api response:
data: {"type": "usage_estimate", "input_tokens": 22274, "usage": {"meter": "messaging_tokens", "tier": "free", "monthly_allotment": 200000, "used_to_date": 0, "remaining": 200000, "pay_per_use_enabled": false, "usage_period_start": "2026-07-01T00:00:00+00:00", "usage_period_end": "2026-08-01T00:00:00+00:00"}, "thread_id": "cb958bba-0fc3-4264-8b1c-0fd7c3e29e0e", "request_id": "fce6c3ab-c06b-4e7a-b6bc-b189cc80ba72"}

data: {"type": "assistant_token", "text": "Hey"}

data: {"type": "assistant_token", "text": "\u2014"}

data: {"type": "assistant_token", "text": "what"}

data: {"type": "assistant_token", "text": "\u2019s"}

data: {"type": "assistant_token", "text": " up"}

data: {"type": "assistant_token", "text": "?"}

: keepalive

data: {"type": "done", "content": "Hey\u2014what\u2019s up?", "thread_id": "cb958bba-0fc3-4264-8b1c-0fd7c3e29e0e", "request_id": "fce6c3ab-c06b-4e7a-b6bc-b189cc80ba72", "total_response_time_ms": 17895, "response_metadata": {"finish_reason": "stop", "model_name": "gpt-5.4-nano-2026-03-17", "service_tier": "default", "model_provider": "openai", "sentiment": {"base_emotion": "neutral", "emotion": "neutral", "score": 0.5597519278526306}, "token_usage": {"prompt_tokens": 19836, "completion_tokens": 9, "total_tokens": 19845}, "features": {"moving_average_ttr": 1.0, "mtld_lexical_diversity": 3.0, "hdd_lexical_diversity": 1.0, "lexical_density_content_word_ratio": 0.5, "noun_density": 0.16666666666666666, "verb_density": 0.3333333333333333, "adjective_density": 0.0, "adverb_density": 0.0, "pronoun_density": 0.16666666666666666, "preposition_density": 0.0, "noun_to_verb_ratio": 0.6666666666666666, "mean_sentence_length_words": 3.0, "stdev_sentence_length_words": 0.0, "interrogative_sentence_ratio": 1.0, "exclamatory_sentence_ratio": 0.0, "comma_rate_per_word": 0.0, "semicolon_rate_per_word": 0.0, "colon_rate_per_word": 0.0, "dash_rate_per_word": 0.3333333333333333, "ellipsis_rate_per_word": 0.0, "exclamation_rate_per_word": 0.0, "question_mark_rate_per_word": 0.3333333333333333, "all_caps_word_ratio": 0.0, "words_per_paragraph": 3.0, "transition_word_rate_per_word": 0.0, "lexical_entropy_bits": 1.584962500721156, "average_word_length_characters": 3.6666666666666665, "key_phrase_rate": 0.0, "key_phrase_rate_description": "The key_phrase_rate is the rate of detected avatar signature key phrases per total word when compared against the ground truth dataset (direct quotes of the avatar), and is the rate of baseline ChatGPT signature key phrases per total word when compared against the baseline ChatGPT dataset."}, "comparison_to_unmodified_llm_response_analysis": {"no_statistically_significantly_difference_from_unmodified_llm_response_using_squared_mahalanobis_distance": false, "unmodified_llm_comparison_isolation_forest_shap_values": {"mtld_lexical_diversity": -0.14960766849237384, "noun_density": -0.14131669250864398, "verb_density": -0.19548582656292893, "adjective_density": -0.1441411335687152, "pronoun_density": -0.17351241449700464, "preposition_density": -0.1314117884261159, "interrogative_sentence_ratio": -0.29745612659298876, "dash_rate_per_word": -0.15881606491014613, "question_mark_rate_per_word": -0.2810603747075978, "lexical_entropy_bits": -0.24023538799435445}, "unmodified_llm_comparison_isolation_forest_shap_values_description": "Negative values indicate dissimilarity from unmodified llm dataset. Positive values indicate similarity to unmodified llm responses. Scale is -1 to 1.", "no_statistically_significant_difference_between_sample_and_unmodified_llm_according_to_isolation_forest": false}}, "usage": {"prompt_tokens": 19836, "completion_tokens": 9, "total_tokens": 19845, "meter": "messaging_tokens", "tier": "free", "monthly_allotment": 200000, "used_to_date": 19845, "remaining": 180155, "pay_per_use_enabled": false, "usage_period_start": "2026-07-01T00:00:00+00:00", "usage_period_end": "2026-08-01T00:00:00+00:00"}}


The anonymous usage of messaging api and the stripe customer portal are not currently synchronized. the stripe customer portal needs to be the source of truth.

## BUG 2: verify_subscription_status does not return anonymous usage statistics for the anonymous user

## Test Assistant ID:
`cd8ddcc4-6051-4adb-8876-231e0f3a7105`

## LOCAL ports:
### Customer portal:
http://localhost:5171/

### API:
http://localhost:8123/docs#GET/verify_subscription_status
http://localhost:8123/docs#POST/message/{assistant_id}

### UI:
http://localhost:8501/?assistant_id=cd8ddcc4-6051-4adb-8876-231e0f3a7105&thread_id=1ef9048a-c547-472f-bd37-7091e805def3


## api response:

data: {"type": "usage_estimate", "input_tokens": 22852, "usage": {"meter": "messaging_tokens", "tier": "free", "monthly_allotment": 200000, "used_to_date": 20648, "remaining": 179352, "pay_per_use_enabled": false, "usage_period_start": "2026-07-01T00:00:00+00:00", "usage_period_end": "2026-08-01T00:00:00+00:00"}, "thread_id": "19cf7d24-6d89-4e8d-834e-e211799305d7", "request_id": "d3520ba4-1f0a-4e51-af97-5422ced6d5b7"}

data: {"type": "assistant_token", "text": "Hey"}

data: {"type": "assistant_token", "text": "."}

data: {"type": "assistant_token", "text": " What"}

data: {"type": "assistant_token", "text": "\u2019s"}

data: {"type": "assistant_token", "text": " up"}

data: {"type": "assistant_token", "text": "?"}

: keepalive

data: {"type": "done", "content": "Hey. What\u2019s up?", "thread_id": "19cf7d24-6d89-4e8d-834e-e211799305d7", "request_id": "d3520ba4-1f0a-4e51-af97-5422ced6d5b7", "total_response_time_ms": 10996, "response_metadata": {"finish_reason": "stop", "model_name": "gpt-5.4-nano-2026-03-17", "service_tier": "default", "model_provider": "openai", "sentiment": {"base_emotion": "surprise", "emotion": "curiosity", "score": 0.5777334570884705}, "token_usage": {"prompt_tokens": 19836, "completion_tokens": 9, "total_tokens": 19845}, "features": {"moving_average_ttr": 1.0, "mtld_lexical_diversity": 4.0, "hdd_lexical_diversity": 1.0, "lexical_density_content_word_ratio": 0.3333333333333333, "noun_density": 0.16666666666666666, "verb_density": 0.16666666666666666, "adjective_density": 0.0, "adverb_density": 0.0, "pronoun_density": 0.16666666666666666, "preposition_density": 0.0, "noun_to_verb_ratio": 1.0, "mean_sentence_length_words": 1.5, "stdev_sentence_length_words": 0.5, "interrogative_sentence_ratio": 0.5, "exclamatory_sentence_ratio": 0.0, "comma_rate_per_word": 0.0, "semicolon_rate_per_word": 0.0, "colon_rate_per_word": 0.0, "dash_rate_per_word": 0.0, "ellipsis_rate_per_word": 0.0, "exclamation_rate_per_word": 0.0, "question_mark_rate_per_word": 0.3333333333333333, "all_caps_word_ratio": 0.0, "words_per_paragraph": 3.0, "transition_word_rate_per_word": 0.0, "lexical_entropy_bits": 1.584962500721156, "average_word_length_characters": 3.6666666666666665, "key_phrase_rate": 0.0, "key_phrase_rate_description": "The key_phrase_rate is the rate of detected avatar signature key phrases per total word when compared against the ground truth dataset (direct quotes of the avatar), and is the rate of baseline ChatGPT signature key phrases per total word when compared against the baseline ChatGPT dataset."}, "comparison_to_unmodified_llm_response_analysis": {"no_statistically_significantly_difference_from_unmodified_llm_response_using_squared_mahalanobis_distance": false, "unmodified_llm_comparison_isolation_forest_shap_values": {"moving_average_ttr": -0.14289254891262756, "noun_density": -0.19333661647153233, "adjective_density": -0.16366041944067852, "adverb_density": -0.15924921847533952, "pronoun_density": -0.14366704673555225, "preposition_density": -0.16416515323919134, "dash_rate_per_word": -0.13435983702040796, "question_mark_rate_per_word": -0.39268620347508976, "lexical_entropy_bits": -0.2924854193982227, "average_word_length_characters": -0.1265410150922277}, "unmodified_llm_comparison_isolation_forest_shap_values_description": "Negative values indicate dissimilarity from unmodified llm dataset. Positive values indicate similarity to unmodified llm responses. Scale is -1 to 1.", "no_statistically_significant_difference_between_sample_and_unmodified_llm_according_to_isolation_forest": false}}, "usage": {"prompt_tokens": 19836, "completion_tokens": 9, "total_tokens": 19845, "meter": "messaging_tokens", "tier": "free", "monthly_allotment": 200000, "used_to_date": 40493, "remaining": 159507, "pay_per_use_enabled": false, "usage_period_start": "2026-07-01T00:00:00+00:00", "usage_period_end": "2026-08-01T00:00:00+00:00"}}

## customer portal usage (reports 282,448 tokens with overage and above reports 19845 total tokens and 40493 used_to_date with 159507 remaining):

Neural Nexus

Customer portal
Browsing as anonymous

You are currently an anonymous user. Usage below is read-only until you sign up. To start the free Pro trial (to create avatars, upload documents, and more!) use Sign up for free Pro trial in the plan section.
Current subscription
active

Neural Nexus Free Tier — Base Subscription

Price varies with usage

Your free trial has not been used yet — it is included with the pro tier below.
Upgrade from anonymous free tier

Anonymous usage is tracked by a hash of your network address (the same scheme Neural Nexus uses). Create an account with email and password to start the free Pro trial, then open Stripe Checkout.
Neural Nexus Free Tier — Base Subscription

$0/month

    Messaging tokens: 200,000 tokens/month · $2.00 per 1M over

Neural Nexus Pro Tier — Base Subscription

$20.00/month

30-day free trial available

    Document upload tokens: 10,000,000 tokens/month · $3.00 per 1M over
    Messaging tokens: 5,000,000 tokens/month · $1.50 per 1M over

Neural Nexus Premium Tier — Base Subscription

$50.00/month

    Adapter training: 5 units/month · $5.00 per unit over
    Adapter inference tokens: 10,000,000 tokens/month · $4.00 per 1M over
    Document upload tokens: 40,000,000 tokens/month · $2.50 per 1M over
    Messaging tokens: 20,000,000 tokens/month · $1.25 per 1M over

Usage this period
Jul 15, 2026 – Aug 15, 2026
Messaging tokens282,448 / 200,000 tokens (+82,448 over)
Allotment exhausted — enable pay-per-use or upgrade to continue$2.00 per 1,000,000 tokens over allotment
Powered by Stripe
Terms
Privacy
Support


---------------------

# RESOLUTION EVIDENCE (2026-07-25)

## BUG 1 + BUG 2 — both closed; all three surfaces now report one number

`curl /verify_subscription_status --header 'Accept: application/json' --header 'API-KEY: YOUR_SECRET_TOKEN'`
(anonymous requester, no bearer token — BUG 2 was that this endpoint refused
anonymous callers and returned no usage at all):

```json
{
  "status": "active",
  "tier": "free",
  "subscription_id": "sub_1Ttg5RLimk9GVblrWUpiDWPx",
  "customer_id": "cus_UtT1Pxg26KRKUX",
  "email": null,
  "anonymous": true,
  "pay_per_use_enabled": false,
  "cancel_at_period_end": false,
  "usage_period_start": "2026-07-16T03:35:49+00:00",
  "usage_period_end": "2026-08-16T03:35:49+00:00",
  "meters": {
    "messaging_tokens": {
      "monthly_allotment": 200000,
      "used_to_date": 342864,
      "remaining": 0,
      "over_allotment": 142864,
      "overage_price_per_million": 2.0,
      "overage_price_per_unit_usd": null
    }
  }
}
```

Cross-check against the other two surfaces for the same visitor
(`cus_UtT1Pxg26KRKUX`, hashed IP of `172.18.0.1`):

| surface | used_to_date | period |
|---|---|---|
| Stripe customer portal | 342,864 | Jul 15 – Aug 15 (rendered in the browser's local time zone) |
| `/verify_subscription_status` | 342,864 | 2026-07-16T03:35:49Z – 2026-08-16T03:35:49Z |
| `POST /message/{assistant_id}` SSE `usage_estimate` + `done` | 342,864 | 2026-07-16T03:35:49Z – 2026-08-16T03:35:49Z |

The remaining apparent date gap is a rendering difference only: the portal shows
the same instants in the viewer's local time zone (Jul 16 03:35 UTC is Jul 15
20:35 US/Pacific), not a different window. `max(local, stripe)` reconciliation
plus the anonymous Stripe billing-cycle window closed the numeric gap; the
calendar-month window (Jul 1 – Aug 1) that the earlier captures above show is
gone.

## Follow-on: usage stopped ADVANCING while messaging continued

After the reconciliation landed, the dev anonymous identity read 342,864 on
every surface but never grew, and messaging kept answering despite
`"remaining": 0`. Cause: `sha256("172.18.0.1")` =
`245c0ffc0f6a0215471542b9add1fa5331647f4af18c431f039c66dbee92732e` was listed in
`ADMIN_METERING_BYPASS_IDENTIFIERS`, which suppresses BOTH enforcement (402/429)
AND the metering writes (Stripe meter event + `api_metrics` row). A frozen number
on every surface is not a synchronization failure — it is the bypass working as
specified — but it is indistinguishable from one while testing.

Fix: split the bypass in two (`MeteringBypass` in
`src/anubis/utils/billing/gating.py`). The three anonymous dev identities moved
to the new `DEV_METERED_ENFORCEMENT_BYPASS_IDENTIFIERS`, which skips enforcement
only and is honored only when `DEV=TRUE`; their usage is still metered, so the
number keeps advancing in step across all three surfaces. Responses now carry
`admin_enforcement_bypass: true` for that mode, keeping
`admin_metering_bypass: true` to mean "never recorded anywhere".