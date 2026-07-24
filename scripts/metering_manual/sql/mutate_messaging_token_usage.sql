-- mutate_messaging_token_usage.sql
-- Premium allotment = 20_000_000 | Pro = 5_000_000 | Free = 200_000
--
-- total_tokens carries the injected amount. It must be the tier's allotment (or
-- higher) to force the over-allotment path: enforcement blocks when
-- period usage + this request's estimate >= allotment. A value of 0 inserts a
-- row that adds nothing to SUM(total_tokens) and therefore changes no gate.
--
-- user_id uses the SAME 'auth0|' prefixed spelling the API writes, so the
-- cleanup path (clear_testing.sql / nn_clear) matches these rows. Enforcement
-- keys on stripe_customer_id when present, so a mismatched user_id would still
-- count against the allotment while escaping deletion.
INSERT INTO api_metrics (
  id, created_at, user_id, stripe_customer_id, inference_type,
  prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name
)
VALUES (
  gen_random_uuid(), now(),
  'auth0|6a624194d35cf2fb8de3e51a',   -- $NN_USER_ID
  'cus_UwI5WZe7H8TOnx',               -- $NN_CUSTOMER
  'test_inject',
  0, 0,
  5000000000,                            -- pro allotment; raise to 20000000 for premium
  0, 0,
  'messaging_tokens'
);
