---
name: sleep-analysis
description: Sleep quality scoring and "how is the user sleeping right now" SOP. Use whenever the prompt mentions sleep, rest, recovery, HRV, fatigue, energy, readiness, or "am I rested".
license: MIT
metadata:
  author: anubis
  version: "0.1"
allowed-tools: read_file, write_file, execute, ingest_sleep
---

# Sleep Analysis SOP

## When to use

Prompts that include: sleep, rested, fatigue, energy, recovery, readiness, HRV, deep, REM, "how am I doing".

## What to produce

A `SleepOverview` block usable inside an AAR, or — if the user asked a stand-alone sleep question — a single Slack message.

## Inputs

`/data/sleep/clean/<run>.parquet` with columns: `date`, `total_sleep_min`, `efficiency_pct`, `rem_min`, `deep_min`, `awake_min`, `hrv_ms`, `score`.

## Steps

### 1. Load and validate

Call `ingest_sleep(provider="oura", days=14)` (or apple/whoop/fitbit per `SLEEP_PROVIDER`). The 14-day window is intentional: it gives a 7-day rolling average and a 7-day baseline.

### 2. Compute composite

Use the provider score when available. When not, compute:

```
composite = 0.40 * normalize(efficiency_pct, 70, 95)
          + 0.20 * normalize(deep_min, 45, 110)
          + 0.20 * normalize(rem_min, 60, 130)
          + 0.20 * normalize(hrv_ms, 25, 90)
```

`normalize(x, lo, hi)` clips to `[0, 100]`.

### 3. Trend

Report `composite_today`, `7d_avg`, and `hrv_trend_pct = (hrv_today - hrv_7d_avg) / hrv_7d_avg * 100`.

### 4. Recommendation

Compose one sentence specific to tomorrow's calendar load. If meeting density tomorrow > 4 hrs of recorded calls, recommend a 21:30 wind-down with no screens after 22:00. Otherwise recommend 22:30 wind-down. Always pull the calendar load from `/data/meet/clean/<run>.parquet` if present; if not, fall back to "wind-down by 22:30".

### 5. Output

Return a `SleepOverview` block:

```python
SleepOverview(
    last_night_score=composite_today,
    seven_day_avg_score=composite_7d_avg,
    hrv_trend_pct=hrv_trend_pct,
    recommendation=recommendation_sentence,
)
```

## Notes

- HRV trend is the single most predictive signal for "how today will go". A drop of >= 15% vs 7d avg means flag this in the AAR's `what_did_not_work` regardless of composite score.
- Time of measurement matters: if the most recent record's date is older than today, mark sleep ingest stale in `what_did_not_work`.
