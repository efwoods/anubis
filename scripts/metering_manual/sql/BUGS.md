# created account: immediately premium and past due (created a new account; did not recreate bug)

{
  "status": "past_due",
  "tier": "premium",
  "subscription_id": "sub_1Tw1YhLimk9GVblrYh79XBvv",
  "customer_id": "cus_UvtKi5sL4rTPMc",
  "email": "yirogo5370@diarshop.com",
  "pay_per_use_enabled": false,
  "cancel_at_period_end": false,
  "usage_period_start": "2026-07-22T18:09:18+00:00",
  "usage_period_end": "2026-08-22T18:09:18+00:00",
  "meters": {
    "messaging_tokens": {
      "monthly_allotment": 20000000,
      "used_to_date": 0,
      "remaining": 20000000,
      "overage_price_per_million": 1.25,
      "overage_price_per_unit_usd": null
    },
    "document_upload_tokens": {
      "monthly_allotment": 40000000,
      "used_to_date": 0,
      "remaining": 40000000,
      "overage_price_per_million": 2.5,
      "overage_price_per_unit_usd": null
    },
    "adapter_inference_tokens": {
      "monthly_allotment": 10000000,
      "used_to_date": 0,
      "remaining": 10000000,
      "overage_price_per_million": 4.0,
      "overage_price_per_unit_usd": null
    },
    "adapter_training_units": {
      "monthly_allotment": 5,
      "used_to_date": 0,
      "remaining": 5,
      "overage_price_per_million": null,
      "overage_price_per_unit_usd": 5.0
    }
  }
}


## appropriate behavior on new account (did not recreate bug; original bug did not re-create)
{
  "status": "trialing",
  "tier": "pro",
  "subscription_id": "sub_1TwC6LLimk9GVblrRFB4RjNe",
  "customer_id": "cus_Uw4DtZm5rA9xTg",
  "email": "nemalal833@besteya.com",
  "pay_per_use_enabled": false,
  "cancel_at_period_end": false,
  "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
  "usage_period_end": "2026-08-23T02:11:11.551562+00:00",
  "meters": {
    "messaging_tokens": {
      "monthly_allotment": 5000000,
      "used_to_date": 0,
      "remaining": 5000000,
      "overage_price_per_million": 1.5,
      "overage_price_per_unit_usd": null
    },
    "document_upload_tokens": {
      "monthly_allotment": 10000000,
      "used_to_date": 0,
      "remaining": 10000000,
      "overage_price_per_million": 3.0,
      "overage_price_per_unit_usd": null
    }
  }
}

# pay_per_use_enables but does not disable on click in UI (needs to be instant [clear cache])
22:19 -> 22:24
directory: /home/user/gh/anubis-project/anubis-customer-portal

Pay-per-use past allotment
Requests stop once a meter's monthly allotment is exhausted (HTTP 402).

Enable pay-per-use
Saved. Changes can take up to 5 minutes to apply to the Neural Nexus API (it caches account lookups).

{
  "status": "trialing",
  "tier": "pro",
  "subscription_id": "sub_1TwC6LLimk9GVblrRFB4RjNe",
  "customer_id": "cus_Uw4DtZm5rA9xTg",
  "email": "nemalal833@besteya.com",
  "pay_per_use_enabled": true,
  "cancel_at_period_end": false,
  "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
  "usage_period_end": "2026-08-23T02:11:11.551562+00:00",
  "meters": {
    "messaging_tokens": {
      "monthly_allotment": 5000000,
      "used_to_date": 20959,
      "remaining": 4979041,
      "overage_price_per_million": 1.5,
      "overage_price_per_unit_usd": null
    },
    "document_upload_tokens": {
      "monthly_allotment": 10000000,
      "used_to_date": 0,
      "remaining": 10000000,
      "overage_price_per_million": 3.0,
      "overage_price_per_unit_usd": null
    }
  }
}


# Document upload tokens:

This is the monthly allotment and used amount (the media processing started):
  "playlists_expanding": 1,
  "estimated_tokens_total": 143762,
  "usage": {
    "meter": "document_upload_tokens",
    "tier": "pro",
    "monthly_allotment": 10000000,
    "used_to_date": 40143762,
    "remaining": 0,
    "pay_per_use_enabled": true,
    "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
    "usage_period_end": "2026-08-23T02:11:11.551562+00:00"
  },
  "message": "Media processing started; enumerating 1 playlist(s) in the background"

this is the upload file:
/home/user/gh/anubis-project/anubis/data/shivon_zilis/_shivon_zilis_test_data.md

curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-tDFV1JPbM-cTP7vWAYtzNdOP9DqSMglRTd3nLOkZWPk' \
  --form 'files=@_shivon_zilis_test_data.md' \
  --form 'url=[""]' \
  --form 'assistant_id=70fc7621-f590-4d71-8081-8786b8e7a810' \
  --form 'reference_audio=false' \
  --form 'reference_image=false' \
  --form 'create_reference_media_from_playlist=false'


{
  "job_id": "c48a8874-d881-456f-8711-68a378247b49",
  "status": "queued",
  "status_url": "/media_job/c48a8874-d881-456f-8711-68a378247b49",
  "progress_url": "/media_job/c48a8874-d881-456f-8711-68a378247b49/progress",
  "cancel_url": "/media_job/c48a8874-d881-456f-8711-68a378247b49/cancel",
  "items_accepted": 2,
  "filenames": [
    "https://www.youtube.com/watch?v=-tQwzhHjAVI",
    "https://www.youtube.com/watch?v=CkUcCcRq_eM&list=PL9rU625vkl4UlyAT5THtDV3cOB2KOkASX"
  ],
  "items": [
    {
      "job_id": "d74e2e75-6da8-4e8c-9826-077d862cbba8",
      "filename": "https://www.youtube.com/watch?v=-tQwzhHjAVI",
      "status": "queued",
      "estimated_tokens": 26100,
      "status_url": "/media_job/d74e2e75-6da8-4e8c-9826-077d862cbba8",
      "progress_url": "/media_job/d74e2e75-6da8-4e8c-9826-077d862cbba8/progress",
      "cancel_url": "/media_job/d74e2e75-6da8-4e8c-9826-077d862cbba8/cancel"
    },
    {
      "job_id": "499e69cc-9ce5-4bd6-909a-bd1cc3c76d82",
      "filename": "https://www.youtube.com/watch?v=CkUcCcRq_eM&list=PL9rU625vkl4UlyAT5THtDV3cOB2KOkASX",
      "status": "queued",
      "estimated_tokens": 18000,
      "status_url": "/media_job/499e69cc-9ce5-4bd6-909a-bd1cc3c76d82",
      "progress_url": "/media_job/499e69cc-9ce5-4bd6-909a-bd1cc3c76d82/progress",
      "cancel_url": "/media_job/499e69cc-9ce5-4bd6-909a-bd1cc3c76d82/cancel"
    }
  ],
  "playlists_expanding": 1,
  "estimated_tokens_total": 143762,
  "usage": {
    "meter": "document_upload_tokens",
    "tier": "pro",
    "monthly_allotment": 10000000,
    "used_to_date": 40143762,
    "remaining": 0,
    "pay_per_use_enabled": true,
    "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
    "usage_period_end": "2026-08-23T02:11:11.551562+00:00"
  },
  "message": "Media processing started; enumerating 1 playlist(s) in the background"
}

job_id:  c48a8874-d881-456f-8711-68a378247b49



## Document upload tokens (these media jobs need to be declined):

request:
curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-tDFV1JPbM-cTP7vWAYtzNdOP9DqSMglRTd3nLOkZWPk' \
  --form 'files=@test_data.md' \
  --form 'url=[""]' \
  --form 'assistant_id=70fc7621-f590-4d71-8081-8786b8e7a810' \
  --form 'reference_audio=false' \
  --form 'reference_image=false' \
  --form 'create_reference_media_from_playlist=false'

test data file: 
/home/user/gh/anubis-project/anubis/data/shivon_zilis/test_data.md

{
  "job_id": "1ba561e5-de65-47b0-93e7-769c1b403af8",
  "status": "queued",
  "status_url": "/media_job/1ba561e5-de65-47b0-93e7-769c1b403af8",
  "progress_url": "/media_job/1ba561e5-de65-47b0-93e7-769c1b403af8/progress",
  "cancel_url": "/media_job/1ba561e5-de65-47b0-93e7-769c1b403af8/cancel",
  "items_accepted": 1,
  "filenames": [
    "test_data.md"
  ],
  "items": [
    {
      "job_id": "20409587-b2a7-48ba-a947-96e52a4acd80",
      "filename": "test_data.md",
      "status": "queued",
      "estimated_tokens": 2,
      "status_url": "/media_job/20409587-b2a7-48ba-a947-96e52a4acd80",
      "progress_url": "/media_job/20409587-b2a7-48ba-a947-96e52a4acd80/progress",
      "cancel_url": "/media_job/20409587-b2a7-48ba-a947-96e52a4acd80/cancel"
    }
  ],
  "playlists_expanding": 0,
  "estimated_tokens_total": 2,
  "usage": {
    "meter": "document_upload_tokens",
    "tier": "pro",
    "monthly_allotment": 10000000,
    "used_to_date": 40143764,
    "remaining": 0,
    "pay_per_use_enabled": true,
    "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
    "usage_period_end": "2026-08-23T02:11:11.551562+00:00"
  },
  "message": "Media processing started"
}


## Document processing job results:
curl '/media_jobs?include_finished=true&assistant_id=70fc7621-f590-4d71-8081-8786b8e7a810' \
  --header 'Accept: application/json' \
  --header 'API-KEY: sk-tDFV1JPbM-cTP7vWAYtzNdOP9DqSMglRTd3nLOkZWPk'


  {
  "count": 2,
  "jobs": [
    {
      "job_id": "1ba561e5-de65-47b0-93e7-769c1b403af8",
      "assistant_id": "70fc7621-f590-4d71-8081-8786b8e7a810",
      "status": "completed",
      "created_at": 1784774406.236248,
      "started_at": 1784774406.2398767,
      "finished_at": 1784774406.2982645,
      "duration_seconds": 0.058,
      "children_total": 1,
      "children_completed": 1,
      "children_error": 0,
      "children_cancelled": 0,
      "children_running": 0,
      "children_queued": 0,
      "status_url": "/media_job/1ba561e5-de65-47b0-93e7-769c1b403af8",
      "progress_url": "/media_job/1ba561e5-de65-47b0-93e7-769c1b403af8/progress",
      "cancel_url": "/media_job/1ba561e5-de65-47b0-93e7-769c1b403af8/cancel"
    },
    {
      "job_id": "c48a8874-d881-456f-8711-68a378247b49",
      "assistant_id": "70fc7621-f590-4d71-8081-8786b8e7a810",
      "status": "completed",
      "created_at": 1784773902.4919572,
      "started_at": 1784773902.4952786,
      "finished_at": 1784773984.6264706,
      "duration_seconds": 82.131,
      "children_total": 6,
      "children_completed": 0,
      "children_error": 6,
      "children_cancelled": 0,
      "children_running": 0,
      "children_queued": 0,
      "status_url": "/media_job/c48a8874-d881-456f-8711-68a378247b49",
      "progress_url": "/media_job/c48a8874-d881-456f-8711-68a378247b49/progress",
      "cancel_url": "/media_job/c48a8874-d881-456f-8711-68a378247b49/cancel"
    }
  ]
}


### document upload tokens setup and response

### Script set:
-- mutate_document_upload_token_usage.sql
-- Premium allotment = 40_000_000 | Pro = 10_000_000 | Free = N/A (403 capability)
INSERT INTO api_metrics (
  id, created_at, user_id, stripe_customer_id, inference_type,
  prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name
)
VALUES (
  gen_random_uuid(), now(),
  '6a6177ccfd677c585b38d703',   -- $NN_USER_ID
  'cus_Uw4DtZm5rA9xTg',         -- $NN_CUSTOMER
  'test_inject',
  0, 0,
  40000000,                     -- set to tier allotment (or higher)
  0, 0,
  'document_upload_tokens'
);


###  after processing (used_to_date increased:       "used_to_date": 40143764,):
{
  "status": "trialing",
  "tier": "pro",
  "subscription_id": "sub_1TwC6LLimk9GVblrRFB4RjNe",
  "customer_id": "cus_Uw4DtZm5rA9xTg",
  "email": "nemalal833@besteya.com",
  "pay_per_use_enabled": true,
  "cancel_at_period_end": false,
  "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
  "usage_period_end": "2026-08-23T02:11:11.551562+00:00",
  "meters": {
    "messaging_tokens": {
      "monthly_allotment": 5000000,
      "used_to_date": 20020959,
      "remaining": 0,
      "overage_price_per_million": 1.5,
      "overage_price_per_unit_usd": null
    },
    "document_upload_tokens": {
      "monthly_allotment": 10000000,
      "used_to_date": 40143764,
      "remaining": 0,
      "overage_price_per_million": 3.0,
      "overage_price_per_unit_usd": null
    }
  }
}

<!-- 
# Free trial canceled  free-tier; free-tier tokens used
curl /verify_subscription_status \
  --header 'Accept: application/json' \
  --header 'API-KEY: sk-tDFV1JPbM-cTP7vWAYtzNdOP9DqSMglRTd3nLOkZWPk'



{
  "status": "canceled",
  "tier": "free",
  "subscription_id": "sub_1TwC6LLimk9GVblrRFB4RjNe",
  "customer_id": "cus_Uw4DtZm5rA9xTg",
  "email": "nemalal833@besteya.com",
  "pay_per_use_enabled": true,
  "cancel_at_period_end": false,
  "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
  "usage_period_end": "2026-08-23T02:11:11.551562+00:00",
  "meters": {
    "messaging_tokens": {
      "monthly_allotment": 5000000,
      "used_to_date": 20020959,
      "remaining": 0,
      "overage_price_per_million": 1.5,
      "overage_price_per_unit_usd": null
    },
    "document_upload_tokens": {
      "monthly_allotment": 10000000,
      "used_to_date": 40143764,
      "remaining": 0,
      "overage_price_per_million": 3.0,
      "overage_price_per_unit_usd": null
    }
  }
}



curl /message/70fc7621-f590-4d71-8081-8786b8e7a810 \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-tDFV1JPbM-cTP7vWAYtzNdOP9DqSMglRTd3nLOkZWPk' \
  --form 'message=test' \
  --form 'your_name=' \
  --form 'your_description=' \
  --form 'conversation_title=' \
  --form 'files=[""]' \
  --form 'thread_id=' \
  --form 'stream=true' \
  --form 'feedback=false' \
  --form 'like=false' \
  --form 'dislike=false' \
  --form 'user_timezone=' \
  --form 'include_quality_metrics=true' \
  --form 'include_usage_metrics=true' \
  --form 'adapter=false'



  data: {"type": "usage_estimate", "input_tokens": 20834, "usage": {"meter": "messaging_tokens", "tier": "pro", "monthly_allotment": 5000000, "used_to_date": 20020959, "remaining": 0, "pay_per_use_enabled": true, "usage_period_start": "2026-07-23T02:11:11.551562+00:00", "usage_period_end": "2026-08-23T02:11:11.551562+00:00"}, "thread_id": "f06128c7-8806-4052-b285-f51a6285b80c", "request_id": "a38f4274-7ad7-45bc-9000-385e1494b186"}

data: {"type": "assistant_token", "text": "Hey"}

data: {"type": "assistant_token", "text": "."}

data: {"type": "assistant_token", "text": " What"}

data: {"type": "assistant_token", "text": " do"}

data: {"type": "assistant_token", "text": " you"}

data: {"type": "assistant_token", "text": " want"}

data: {"type": "assistant_token", "text": " to"}

data: {"type": "assistant_token", "text": " test"}

data: {"type": "assistant_token", "text": "?"}

: keepalive

data: {"type": "done", "content": "Hey. What do you want to test?", "thread_id": "f06128c7-8806-4052-b285-f51a6285b80c", "request_id": "a38f4274-7ad7-45bc-9000-385e1494b186", "total_response_time_ms": 12286, "response_metadata": {"finish_reason": "stop", "model_name": "gpt-5.4-nano-2026-03-17", "service_tier": "default", "model_provider": "openai", "sentiment": {"base_emotion": "surprise", "emotion": "curiosity", "score": 0.5681958198547363}, "token_usage": {"prompt_tokens": 20953, "completion_tokens": 12, "total_tokens": 20965}, "features": {"moving_average_ttr": 1.0, "mtld_lexical_diversity": 7.0, "hdd_lexical_diversity": 0.9999999999999998, "lexical_density_content_word_ratio": 0.4444444444444444, "noun_density": 0.1111111111111111, "verb_density": 0.3333333333333333, "adjective_density": 0.0, "adverb_density": 0.0, "pronoun_density": 0.2222222222222222, "preposition_density": 0.1111111111111111, "noun_to_verb_ratio": 0.5, "mean_sentence_length_words": 3.5, "stdev_sentence_length_words": 2.5, "interrogative_sentence_ratio": 0.5, "exclamatory_sentence_ratio": 0.0, "comma_rate_per_word": 0.0, "semicolon_rate_per_word": 0.0, "colon_rate_per_word": 0.0, "dash_rate_per_word": 0.0, "ellipsis_rate_per_word": 0.0, "exclamation_rate_per_word": 0.0, "question_mark_rate_per_word": 0.14285714285714285, "all_caps_word_ratio": 0.0, "words_per_paragraph": 7.0, "transition_word_rate_per_word": 0.0, "lexical_entropy_bits": 2.807354922057604, "average_word_length_characters": 3.142857142857143, "key_phrase_rate": 0.0, "key_phrase_rate_description": "The key_phrase_rate is the rate of detected avatar signature key phrases per total word when compared against the ground truth dataset (direct quotes of the avatar), and is the rate of baseline ChatGPT signature key phrases per total word when compared against the baseline ChatGPT dataset."}, "comparison_to_unmodified_llm_response_analysis": {"no_statistically_significantly_difference_from_unmodified_llm_response_using_squared_mahalanobis_distance": false, "unmodified_llm_comparison_isolation_forest_shap_values": {"moving_average_ttr": -0.1238767028870047, "lexical_density_content_word_ratio": -0.15644158228040025, "noun_density": -0.15306600238435789, "verb_density": -0.20483511165671842, "adjective_density": -0.1959971934936796, "adverb_density": -0.15331724370194405, "pronoun_density": -0.17464149170906998, "question_mark_rate_per_word": -0.3226150560438386, "lexical_entropy_bits": -0.2863873615591808, "average_word_length_characters": -0.14186573254467527}, "unmodified_llm_comparison_isolation_forest_shap_values_description": "Negative values indicate dissimilarity from unmodified llm dataset. Positive values indicate similarity to unmodified llm responses. Scale is -1 to 1.", "no_statistically_significant_difference_between_sample_and_unmodified_llm_according_to_isolation_forest": false}}, "usage": {"prompt_tokens": 20953, "completion_tokens": 12, "total_tokens": 20965, "meter": "messaging_tokens", "tier": "pro", "monthly_allotment": 5000000, "used_to_date": 20041924, "remaining": 0, "pay_per_use_enabled": true, "usage_period_start": "2026-07-23T02:11:11.551562+00:00", "usage_period_end": "2026-08-23T02:11:11.551562+00:00"}}


## Free-tier is reporting the following in the UI:
Usage this period
Jun 30, 2026 – Jul 31, 2026
Messaging tokens
41,924 / 200,000 tokens
158,076 tokens remaining
$2.00 per 1,000,000 tokens over allotment


TEST MODE — this portal is connected to the Stripe test environment. No real charges occur.
Neural Nexus
Customer portal

Light mode
nemalal833
Sign out
Current subscription
anonymous free tier
Neural Nexus Free Tier — Base Subscription

Price varies with usage

Switch plan
Upgrades apply immediately (with proration). Downgrades apply at the end of the current billing period — unused allotment continues until then.

Neural Nexus Free Tier — Base Subscription
✓ Active plan

$0/month

Messaging tokens: 200,000 tokens/month · $2.00 per 1M over
Current plan
Neural Nexus Pro Tier — Base Subscription
$20.00/month

30-day free trial

Document upload tokens: 10,000,000 tokens/month · $3.00 per 1M over
Messaging tokens: 5,000,000 tokens/month · $1.50 per 1M over
Switch to this plan
Neural Nexus Premium Tier — Base Subscription
$50.00/month

Adapter training: 5 units/month · $5.00 per unit over
Adapter inference tokens: 10,000,000 tokens/month · $4.00 per 1M over
Document upload tokens: 40,000,000 tokens/month · $2.50 per 1M over
Messaging tokens: 20,000,000 tokens/month · $1.25 per 1M over
Switch to this plan
Usage this period
Jun 30, 2026 – Jul 31, 2026
Messaging tokens
41,924 / 200,000 tokens
158,076 tokens remaining
$2.00 per 1,000,000 tokens over allotment
Pay-per-use past allotment
Usage past your monthly allotment continues and bills at your tier's overage rate.

Disable pay-per-use
Payment method
VISA •••• 4242
Expires 04/2027
Make default
Remove
Add payment method
Billing information
Name
nemalal833
Email
nemalal833@besteya.com
Billing address
—
Phone number
—
Update information
Invoice history
Jul 22, 2026
$0.00
paid
Free trial for 1 × Neural Nexus Pro Tier — Base Subscription
View
PDF
Powered by Stripe
Terms
Privacy
Support
 -->

# UI and /verify_subscription_status disagree after subscription:

curl /verify_subscription_status \
  --header 'Accept: application/json' \
  --header 'API-KEY: sk-tDFV1JPbM-cTP7vWAYtzNdOP9DqSMglRTd3nLOkZWPk'


{
  "status": "canceled",
  "tier": "free",
  "subscription_id": "sub_1TwC6LLimk9GVblrRFB4RjNe",
  "customer_id": "cus_Uw4DtZm5rA9xTg",
  "email": "nemalal833@besteya.com",
  "pay_per_use_enabled": true,
  "cancel_at_period_end": false,
  "usage_period_start": "2026-07-23T02:11:11.551562+00:00",
  "usage_period_end": "2026-08-23T02:11:11.551562+00:00",
  "meters": {
    "messaging_tokens": {
      "monthly_allotment": 5000000,
      "used_to_date": 20041924,
      "remaining": 0,
      "overage_price_per_million": 1.5,
      "overage_price_per_unit_usd": null
    },
    "document_upload_tokens": {
      "monthly_allotment": 10000000,
      "used_to_date": 40143764,
      "remaining": 0,
      "overage_price_per_million": 3.0,
      "overage_price_per_unit_usd": null
    }
  }
}

# /list_avatar_documents not uploading document although the document has been processed:

curl /list_avatar_documents \
  --header 'Accept: application/json' \
  --header 'API-KEY: sk-ubr9cE1qg6sCpY2w455Fks0M27iEZD_T1QU02s7u9h4'


## /media_job/{job_id}/progress

curl /media_job/236a356a-1b3e-46c9-8111-0c8b8373bfb7/progress \
  --header 'Accept: application/json' \
  --header 'API-KEY: sk-ubr9cE1qg6sCpY2w455Fks0M27iEZD_T1QU02s7u9h4'

data: {"type": "status", "status": "completed", "started_at": 1784779721.1266122, "elapsed_seconds": 199.298}

data: {"type": "media_progress", "stage": "labeling", "total": 1, "item_job_id": "611032fe-9f91-437c-b939-0306f7bc8361", "item_filename": "test_data.md", "started_at": 1784779721.1266122, "elapsed_seconds": 199.298}

data: {"type": "media_progress", "stage": "converting_started", "total": 1, "skipped": 0, "item_job_id": "611032fe-9f91-437c-b939-0306f7bc8361", "item_filename": "test_data.md", "started_at": 1784779721.1266122, "elapsed_seconds": 199.298}

data: {"type": "media_progress", "stage": "converting", "current": 1, "total": 1, "filename": "test_data.md", "item_job_id": "611032fe-9f91-437c-b939-0306f7bc8361", "item_filename": "test_data.md", "started_at": 1784779721.1266122, "elapsed_seconds": 199.299}

data: {"type": "media_progress", "stage": "converting_complete", "total": 1, "skipped": 0, "errors": 0, "indexed": 1, "item_job_id": "611032fe-9f91-437c-b939-0306f7bc8361", "item_filename": "test_data.md", "started_at": 1784779721.1266122, "elapsed_seconds": 199.299}

data: {"type": "done", "status": "completed", "result": {"items_total": 1, "items_completed": 1, "items_error": 0, "items_cancelled": 0, "items": [{"job_id": "611032fe-9f91-437c-b939-0306f7bc8361", "filename": "test_data.md", "namespace_filename": "f296af30-1b1b-5a5e-990e-e92fc5368b77", "status": "completed", "estimated_tokens": 2, "error": null}], "message": "Batch processing finished"}, "error": null, "finished_at": 1784779721.1863663, "duration_seconds": 0.06, "started_at": 1784779721.1266122, "elapsed_seconds": 199.299}

## initial document_upload request: test_data.md contains only the word `please` (one token)


curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-ubr9cE1qg6sCpY2w455Fks0M27iEZD_T1QU02s7u9h4' \
  --form 'files=@test_data.md' \
  --form 'url=[""]' \
  --form 'assistant_id=8a4bf526-d42b-4571-92e5-9f8a0c831810' \
  --form 'reference_audio=false' \
  --form 'reference_image=false' \
  --form 'create_reference_media_from_playlist=false'

# Free tier (no trial)
## messaging works (blocks as intended)
## Document Upload functions (blocks as intended)

# Pro tier (no trial)
## messaging (blocks as intended)
## Document upload functions below and above

# Premium (no trial)
## messaging blocks as intended past allotment
## document upload blocks as intended pas allotment

