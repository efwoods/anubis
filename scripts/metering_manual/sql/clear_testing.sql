-- Remove injected test rows for the manual-validation account.
--
-- Deleting rows is the ONLY way to make used_to_date go down inside a period:
-- used_to_date is SUM(total_tokens) over a time window, not a stored counter,
-- so inserting a row with total_tokens = 0 adds zero and leaves the total
-- exactly where the earlier injects put it. (The other way down is waiting for
-- the period to advance past those rows' created_at.)
--
-- The match is on the billing identity, not on user_id alone: the inject .sql
-- files in this directory write the bare Auth0 subject while the API writes the
-- 'auth0|' prefixed form, and enforcement sums on stripe_customer_id whenever
-- one is present — so a user_id-only DELETE can remove nothing while those rows
-- keep counting against the allotment.
DELETE FROM api_metrics
WHERE inference_type = 'test_inject'
  AND COALESCE(stripe_customer_id, user_id) IN (
      'cus_Uw4DtZm5rA9xTg',                 -- $NN_CUSTOMER
      'auth0|6a6177ccfd677c585b38d703',     -- $NN_USER_ID
      '6a6177ccfd677c585b38d703'            -- unprefixed spelling used by the inject files
  );
