-- usage_report.sql — current-period usage AND lifetime usage, per user, per meter.
--
-- Why one table serves both needs
-- ------------------------------
-- api_metrics is append-only: nothing is ever zeroed, decremented, or rolled
-- over at a period boundary. Both numbers are derived from the same rows by
-- changing only the WHERE clause:
--
--   billing  -> SUM(total_tokens) WHERE created_at >= period_start   (a window)
--   metrics  -> SUM(total_tokens) with no time predicate             (all history)
--
-- That is why the monthly "reset" needs no reset job: when the period rolls
-- over, period_start advances and the previous period's rows simply fall out
-- of the window, while remaining on disk for lifetime reporting forever.
--
-- Billing identity
-- ----------------
-- fetch_usage_since (src/anubis/utils/billing/metering.py) sums on
-- stripe_customer_id when the account has one and falls back to user_id, so
-- this report groups by COALESCE(stripe_customer_id, user_id) to match the
-- enforcement path exactly. Grouping that way also folds together rows written
-- under different spellings of the same Auth0 subject (for example
-- 'auth0|6a6177...' from the API and '6a6177...' from a hand-run inject
-- script), which the API already treats as one payer via the customer id.
--
-- Period start
-- ------------
-- Pass the value /verify_subscription_status reports as usage_period_start.
-- That is the authoritative per-user window: resolve_usage_period_start_for_user
-- takes the LATEST of the cached Stripe current_period_start, the user's
-- usage_period_anchor expanded to a monthly boundary, and the environment
-- default. For anyone who subscribed or upgraded mid-month it is NOT the first
-- of the calendar month, so a report that assumes date_trunc('month', now())
-- will disagree with the API.
--
--   docker exec -i postgres16 psql -U postgres -d postgres \
--     -v period_start='2026-07-23T02:11:11.551562+00:00' \
--     -f scripts/metering_manual/sql/usage_report.sql
--
-- With no -v the report falls back to the start of the current UTC calendar
-- month, which is what USAGE_PERIOD_DAYS=0 yields for a user with no anchor.

\if :{?period_start}
\else
  \set period_start ''
\endif

\pset null '-'

-- ---------------------------------------------------------------------------
-- Per billing identity, per meter: current period vs lifetime.
-- ---------------------------------------------------------------------------
WITH period AS (
    SELECT COALESCE(
               NULLIF(:'period_start', '')::timestamptz,
               date_trunc('month', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
           ) AS period_start
)
SELECT
    COALESCE(metric.stripe_customer_id, metric.user_id) AS billing_identity,
    string_agg(DISTINCT metric.user_id, ', ')           AS user_ids,
    metric.meter_event_name                             AS meter,
    -- Billing number: only what falls inside the current usage period.
    COALESCE(
        SUM(metric.total_tokens)
            FILTER (WHERE metric.created_at >= period.period_start),
        0
    )                                                   AS current_period_used,
    -- Metrics number: every row ever written for this payer and meter.
    SUM(metric.total_tokens)                            AS lifetime_used,
    COUNT(*) FILTER (WHERE metric.created_at >= period.period_start)
                                                        AS current_period_rows,
    COUNT(*)                                            AS lifetime_rows,
    MIN(metric.created_at)                              AS first_usage_at,
    MAX(metric.created_at)                              AS last_usage_at
FROM api_metrics AS metric
CROSS JOIN period
WHERE metric.meter_event_name IN (
    'messaging_tokens',
    'document_upload_tokens',
    'adapter_training_units',      -- units, not tokens, stored in total_tokens
    'adapter_inference_tokens'
)
GROUP BY 1, 3, period.period_start
ORDER BY 1, 3;

-- ---------------------------------------------------------------------------
-- Platform totals per meter: the same two numbers across every user, plus the
-- distinct payer count, for capacity and revenue reporting.
-- ---------------------------------------------------------------------------
WITH period AS (
    SELECT COALESCE(
               NULLIF(:'period_start', '')::timestamptz,
               date_trunc('month', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
           ) AS period_start
)
SELECT
    metric.meter_event_name                             AS meter,
    COALESCE(
        SUM(metric.total_tokens)
            FILTER (WHERE metric.created_at >= period.period_start),
        0
    )                                                   AS current_period_used,
    SUM(metric.total_tokens)                            AS lifetime_used,
    COUNT(DISTINCT COALESCE(metric.stripe_customer_id, metric.user_id))
        FILTER (WHERE metric.created_at >= period.period_start)
                                                        AS payers_active_this_period,
    COUNT(DISTINCT COALESCE(metric.stripe_customer_id, metric.user_id))
                                                        AS payers_lifetime
FROM api_metrics AS metric
CROSS JOIN period
WHERE metric.meter_event_name IN (
    'messaging_tokens',
    'document_upload_tokens',
    'adapter_training_units',
    'adapter_inference_tokens'
)
GROUP BY 1, period.period_start
ORDER BY 1;

-- ---------------------------------------------------------------------------
-- Month-by-month history per meter — proves the window advances and shows what
-- each closed period billed. Calendar months here, so this is an approximation
-- for users whose anchor is not the first of the month; the per-user query
-- above with the API's usage_period_start is the billing-accurate view.
-- ---------------------------------------------------------------------------
SELECT
    date_trunc('month', metric.created_at AT TIME ZONE 'UTC') AS calendar_month_utc,
    metric.meter_event_name                                   AS meter,
    SUM(metric.total_tokens)                                  AS used_in_month,
    COUNT(DISTINCT COALESCE(metric.stripe_customer_id, metric.user_id)) AS payers
FROM api_metrics AS metric
WHERE metric.meter_event_name IN (
    'messaging_tokens',
    'document_upload_tokens',
    'adapter_training_units',
    'adapter_inference_tokens'
)
GROUP BY 1, 2
ORDER BY 1 DESC, 2;
