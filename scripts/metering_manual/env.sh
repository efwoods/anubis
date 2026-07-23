# scripts/metering_manual/env.sh
#
# Sourceable curl + SQL helper library for the metering manual-validation
# walkthrough. Source it, fill in the four account variables (printed for you by
# provision_account.py), then call the helper functions per situation.
#
#   source scripts/metering_manual/env.sh
#
# Every helper prints the HTTP status so status IS the pass/fail signal, exactly
# as the playbook prescribes: 200/202 success, 402 allotment, 403 capability,
# 429 rate limit.

# ---------------------------------------------------------------------------
# Config — EDIT after creating the fresh account (provision_account.py prints
# these four export lines ready to paste).
# ---------------------------------------------------------------------------
export NN_API="http://localhost:8123"       # dev API (NOT 8900 on this branch)
export NN_API_KEY="sk-ubr9cE1qg6sCpY2w455Fks0M27iEZD_T1QU02s7u9h4"  # raw API key for the test account
export NN_AVATAR_ID="8a4bf526-d42b-4571-92e5-9f8a0c831810" # assistant_id from /create_avatar
export NN_USER_ID="auth0|6a61928d122ee3c9a8f2734c"         # Auth0 user_id — for SQL meter injection
export NN_CUSTOMER="cus_Uw63jpTP11EDCv"         # Stripe customer id — for SQL meter injection

# Fixed infra (do not edit unless the container/DB names changed).
PG=(docker exec -i postgres16 psql -U postgres -d postgres -v ON_ERROR_STOP=1)
_HDR=(-H "API-KEY: ${NN_API_KEY}")
# ~6-estimated-token upload fixture, created on demand (the playbook's
# data/shivon_zilis/test_tokens_1_tokens.md does not exist on this branch).
NN_FIXTURE="${NN_FIXTURE:-/tmp/nn_upload_fixture.md}"

nn_fixture() { printf 'one two three four five six\n' > "$NN_FIXTURE"; echo "wrote $NN_FIXTURE"; }

# --- Read-only inspection --------------------------------------------------

# Full subscription + per-meter allotment/usage/remaining snapshot.
nn_verify() { curl -s "${_HDR[@]}" "$NN_API/verify_subscription_status" | jq .; }

# Compact one-line tier/status/ppu view.
nn_tier() {
  curl -s "${_HDR[@]}" "$NN_API/verify_subscription_status" \
    | jq -c '{tier,status,ppu:.pay_per_use_enabled,cancel_at_period_end,meters:(.meters|keys)}'
}

# Raw usage per meter straight from api_metrics, keyed and windowed exactly the
# way enforcement is. Two details make the difference between agreeing with
# /verify_subscription_status and quietly disagreeing with it:
#
#   1. Identity. fetch_usage_since sums on stripe_customer_id whenever the
#      account has one and only falls back to user_id otherwise, so a query that
#      filters on user_id alone misses rows written under a different spelling
#      of the same subject (the inject .sql files use the bare Auth0 subject,
#      the API writes the 'auth0|' prefixed form) while enforcement counts both.
#   2. Window. The period start is the account's usage_period_anchor / Stripe
#      billing period, NOT date_trunc('month'). Anyone who subscribed mid-month
#      has a window starting on their signup day-of-month, so the calendar month
#      over-counts by everything spent before the anchor.
#
# So the period start is read from the API rather than assumed. Includes
# injected test rows; use to sanity-check what the API reports.
nn_usage() {
  local period_start
  period_start=$(curl -s "${_HDR[@]}" "$NN_API/verify_subscription_status" \
    | jq -r '.usage_period_start // empty')
  [[ -n "$period_start" ]] || {
    echo "could not read usage_period_start from the API; is it running?" >&2
    return 1
  }
  echo "usage period starts $period_start"
  "${PG[@]}" -tA -c "
    SELECT meter_event_name,
           sum(total_tokens) FILTER (WHERE created_at >= '$period_start'::timestamptz) AS used_this_period,
           sum(total_tokens)                                                           AS used_lifetime,
           count(*)          FILTER (WHERE created_at >= '$period_start'::timestamptz) AS rows_this_period,
           count(*)                                                                    AS rows_lifetime
    FROM api_metrics
    WHERE COALESCE(stripe_customer_id, user_id) IN ('$NN_CUSTOMER', '$NN_USER_ID')
    GROUP BY 1 ORDER BY 1;"
}

# Full platform report: current-period and lifetime usage for every user and
# every meter. Applies THIS account's period start to all rows, so the lifetime
# columns are always right but the current-period columns are only billing-exact
# for accounts sharing this anchor — see the header of usage_report.sql.
nn_usage_report() {
  local period_start
  period_start=$(curl -s "${_HDR[@]}" "$NN_API/verify_subscription_status" \
    | jq -r '.usage_period_start // empty')
  "${PG[@]}" -v period_start="${period_start}" \
    -f - < "$(dirname "${BASH_SOURCE[0]}")/sql/usage_report.sql"
}

# --- Metered actions (print HTTP status + relevant payload) ----------------

# Send one message. Arg1: adapter flag (default false). Prints HTTP status and
# the terminal done.usage frame (meter/tier/allotment/used/remaining/ppu).
nn_msg() {
  local adapter="${1:-false}"
  local out http body
  out=$(curl -s -N -w $'\n__HTTP__%{http_code}' "${_HDR[@]}" \
    --data-urlencode "message=Reply with exactly: ok" \
    --data "stream=true" \
    --data "include_quality_metrics=false" \
    --data "include_usage_metrics=true" \
    --data "adapter=${adapter}" \
    "$NN_API/message/$NN_AVATAR_ID")
  http="${out##*__HTTP__}"; body="${out%$'\n'__HTTP__*}"
  echo "HTTP $http   (adapter=$adapter)"
  echo "$body" | sed -n 's/^data: //p' | jq -Rc 'fromjson? | select(.type=="done") | .usage'
  [[ "$http" =~ ^2 ]] || { echo "-- error body --"; echo "$body" | tail -c 600; echo; }
}

# Upload the fixture (document_upload_tokens meter). Prints HTTP status + body.
nn_upload() {
  [[ -f "$NN_FIXTURE" ]] || nn_fixture >/dev/null
  local out http body
  out=$(curl -s -w $'\n__HTTP__%{http_code}' "${_HDR[@]}" \
    -F "assistant_id=$NN_AVATAR_ID" \
    -F "files=@$NN_FIXTURE;type=text/markdown" \
    "$NN_API/update_avatar_identity_with_media")
  http="${out##*__HTTP__}"; body="${out%$'\n'__HTTP__*}"
  echo "HTTP $http"; echo "$body" | head -c 700; echo
}

# --- Billing state toggles -------------------------------------------------

# Enable/disable pay-per-use. Arg1: true|false. On an ACTIVE paid tier PPU is
# INFERRED true, so you MUST call `nn_ppu false` before every "over, PPU off"
# case. Enabling with no card on file returns 402 (expected).
nn_ppu()       { echo -n "set_pay_per_use=$1 -> "; curl -s -o /dev/null -w '%{http_code}\n' "${_HDR[@]}" -X POST "$NN_API/set_pay_per_use?enabled=$1"; }
nn_ppu_v()     { curl -s -w $'\nHTTP %{http_code}\n' "${_HDR[@]}" -X POST "$NN_API/set_pay_per_use?enabled=$1"; }   # verbose (shows body)

# POST /subscribe?tier=free|pro|premium. Prints body + HTTP status.
nn_subscribe() { curl -s -w $'\nHTTP %{http_code}\n' "${_HDR[@]}" -X POST "$NN_API/subscribe?tier=$1"; }

# --- Meter injection (the "manually mutate state" primitive) ---------------
# Injects a synthetic api_metrics row so month-to-date usage for <meter> equals
# <tokens>. inference_type='test_inject' so nn_clear only removes YOUR rows.
# Enforcement blocks when used + this-request-estimate >= allotment, so passing
# the exact allotment forces the over-allotment path.
nn_inject() {  # nn_inject <meter> <tokens>
  local meter="$1" tokens="$2"
  nn_clear
  "${PG[@]}" -c "
    INSERT INTO api_metrics
      (id, created_at, user_id, stripe_customer_id, inference_type,
       prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name)
    VALUES
      (gen_random_uuid(), now(), '$NN_USER_ID', '$NN_CUSTOMER', 'test_inject',
       0, 0, $tokens, 0, 0, '$meter');"
  echo "injected $tokens into $meter"
}
# Remove injected rows. Matches on the billing identity rather than on
# user_id alone: the standalone .sql inject files write the bare Auth0 subject
# while NN_USER_ID carries the 'auth0|' prefix, so a user_id-only DELETE removes
# nothing yet still reports success — and because enforcement keys on
# stripe_customer_id, those orphaned rows keep counting against the allotment.
# The DELETE count is printed so "cleared" is never claimed without evidence.
nn_clear() {
  "${PG[@]}" -c "
    DELETE FROM api_metrics
    WHERE inference_type = 'test_inject'
      AND COALESCE(stripe_customer_id, user_id) IN ('$NN_CUSTOMER', '$NN_USER_ID');"
}

# Reset used_to_date to ZERO by deleting EVERY api_metrics row for this account,
# genuine usage included. nn_clear deliberately removes only rows it injected
# (inference_type = 'test_inject'), so it reports "DELETE 0" and leaves the
# counters untouched whenever the usage came from real /message and
# /update_avatar_identity_with_media calls — which is the usual reason a period
# looks impossible to reset. Deleting is the only in-period way down: used_to_date
# is SUM(total_tokens) over the period window, not a stored counter, so no
# compensating row can lower it. Test accounts only.
nn_clear_all() {
  "${PG[@]}" -c "
    DELETE FROM api_metrics
    WHERE COALESCE(stripe_customer_id, user_id) IN ('$NN_CUSTOMER', '$NN_USER_ID');"
}

# --- Whole billing states --------------------------------------------------
# Status and tier come from the Stripe subscription; the ALLOTMENTS come from
# app_metadata.trial_context, which no product code clears — so canceling alone
# leaves the trial's allotments in place. nn_scenario sets both halves and polls
# the API until it agrees. Names: free | free-expired-trial | canceled |
# canceled-in-trial | premium-trial | premium-active.
nn_scenario() { python "$(dirname "${BASH_SOURCE[0]}")/stripe_setup.py" scenario "$1"; }

# Stripe subscriptions + the Auth0 billing metadata (trial_context and whether
# its window is still open) in one view.
nn_state() { python "$(dirname "${BASH_SOURCE[0]}")/stripe_setup.py" show; }

# Convenience: inject exactly the current tier's allotment for a meter (reads
# the live allotment from /verify_subscription_status).
nn_inject_full() {  # nn_inject_full <meter>
  local meter="$1" allot
  allot=$(curl -s "${_HDR[@]}" "$NN_API/verify_subscription_status" | jq -r ".meters[\"$meter\"].monthly_allotment // empty")
  [[ -n "$allot" ]] || { echo "tier has no '$meter' meter"; return 1; }
  nn_inject "$meter" "$allot"
}

echo "metering helpers loaded. API=$NN_API  user=$NN_USER_ID"
echo "  inspect : nn_verify | nn_tier | nn_usage | nn_state"
echo "  act     : nn_msg [adapter] | nn_upload"
echo "  billing : nn_ppu true|false | nn_subscribe <tier> | nn_scenario <name>"
echo "  inject  : nn_inject <meter> <tokens> | nn_inject_full <meter> | nn_clear | nn_clear_all"
echo "  meters  : messaging_tokens document_upload_tokens adapter_inference_tokens"
