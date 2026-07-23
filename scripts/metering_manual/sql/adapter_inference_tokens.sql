-- mutate_adapter_inference_token_usage.sql
-- Premium only: allotment = 10_000_000
INSERT INTO api_metrics (
  id, created_at, user_id, stripe_customer_id, inference_type,
  prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name
)
VALUES (
  gen_random_uuid(), now(),
  'auth0|6a6177ccfd677c585b38d703',
  'cus_Uw4DtZm5rA9xTg',
  'test_inject',
  0, 0,
  10000000,
  0, 0,
  'adapter_inference_tokens'
);
