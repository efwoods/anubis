-- mutate_anonymous_messaging_over_allotment.sql
-- Force anonymous free-tier messaging past the monthly allotment (200_000).
--
-- Anonymous users are always free: messaging_tokens only (no document_upload).
-- Enforcement blocks when used + this_request_estimate >= allotment, so inject
-- exactly 200000 (or higher). Include stripe_customer_id — fetch_usage_since
-- sums on customer id when present (see anonymous_billing_customers).
--
-- hashed_ip: 245c0ffc0f6a0215471542b9add1fa5331647f4af18c431f039c66dbee92732e
-- customer:  cus_UtT1Pxg26KRKUX
--
-- Run clear_anonymous_usage.sql first if you need a clean zero before inject.

INSERT INTO api_metrics (
  id, created_at, user_id, stripe_customer_id, inference_type,
  prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name
)
VALUES (
  gen_random_uuid(), now(),
  '245c0ffc0f6a0215471542b9add1fa5331647f4af18c431f039c66dbee92732e',
  'cus_UtT1Pxg26KRKUX',
  'test_inject',
  0, 0,
  200000,              -- free-tier messaging allotment; raise to go further over
  0, 0,
  'messaging_tokens'
);
