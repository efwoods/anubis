-- clear_anonymous_usage.sql
-- Zero month-to-date usage for one anonymous visitor (hashed IP).
--
-- Anonymous allotment gating prefers stripe_customer_id when the visitor has
-- an anonymous billing customer (anonymous_billing_customers). Older rows may
-- only have user_id = hashed_ip and a NULL customer — delete BOTH so
-- used_to_date actually drops to 0.
--
-- hashed_ip: 245c0ffc0f6a0215471542b9add1fa5331647f4af18c431f039c66dbee92732e
-- mapped customer (from anonymous_billing_customers): cus_UtT1Pxg26KRKUX

DELETE FROM api_metrics
WHERE user_id = '245c0ffc0f6a0215471542b9add1fa5331647f4af18c431f039c66dbee92732e'
   OR stripe_customer_id = 'cus_UtT1Pxg26KRKUX';
