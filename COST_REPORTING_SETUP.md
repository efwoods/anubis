# Cost & Usage Reporting Setup Guide

This guide walks through setting up the cost/usage reporting feature for personal avatars.

## Overview

The cost/usage reporting system enables personal avatars to:
- Track and report platform costs via Google Sheets
- Compute metrics: cost per avatar, per message, per conversation, per new user
- Track feature development costs with git correlation
- Monitor spend thresholds and detect anomalies
- Send alerts to agent inbox

## Quick Setup

### 1. Google Sheets API Credentials

**Option A: Service Account (Recommended)**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable Google Sheets API
4. Create a service account:
   - Go to IAM & Admin → Service Accounts
   - Create service account
   - Download JSON key
5. Share your spreadsheet with the service account email
6. Add to `.env`:
```bash
GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON='{"type": "service_account", "project_id": "...", ...}'
```

**Option B: OAuth**

Use existing OAuth credentials (requires user interaction):
```bash
GOOGLE_OAUTH_CLIENT_ID=your_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret
```

### 2. Configure Target Spreadsheet

```bash
# The target spreadsheet URL
COST_REPORTING_SPREADSHEET_URL=https://docs.google.com/spreadsheets/d/1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA/edit

# Sheet range for metrics (optional, default: Metrics!A:G)
COST_REPORTING_RANGE=Metrics!A:G
```

### 3. Set Alert Thresholds (Optional)

```bash
# Daily spend threshold in USD
COST_ALERT_DAILY_THRESHOLD_USD=100.0

# Monthly spend threshold in USD
COST_ALERT_MONTHLY_THRESHOLD_USD=3000.0
```

## Spreadsheet Format

The reporting system appends rows to the configured sheet with this format:

| Column | Description |
|--------|-------------|
| A | Timestamp (ISO 8601) |
| B | Cost per avatar (USD) |
| C | Avg cost per message (USD) |
| D | Avg cost per conversation (USD) |
| E | Cost per new user (USD) |
| F | Total spend (USD) |
| G | Total tokens |

### Example Sheet Structure

Create a sheet named "Metrics" with headers:

```
| Timestamp | Cost/Avatar | Cost/Message | Cost/Conversation | Cost/User | Total Cost | Total Tokens |
|-----------|-------------|--------------|-------------------|-----------|------------|--------------|
| 2026-09-13T12:30:00Z | 0.0234 | 0.0012 | 0.0456 | 1.23 | 234.56 | 1000000 |
```

## Using the Tools

### As a Personal Avatar

```python
# Get current metrics
metrics = await get_cost_metrics(period_days=30)

# Update the Google Sheet
result = await update_cost_report(force_update=True)

# Track a feature's cost
tracking = await track_feature_cost(
    feature_name="ambient vision feature",
    period_days=7
)

# Predict cost of planned work
prediction = await predict_feature_cost(
    estimated_hours=20.0,
    similar_features=["voice mode", "emotion media"]
)

# Check for cost alerts
alerts = await check_cost_alerts(send_to_inbox=True)
```

### Programmatic Usage

```python
from src.anubis.utils.reporting import (
    cost_metrics,
    google_sheets,
    feature_tracking,
    alerts,
)

# Compute all metrics
metrics = await cost_metrics.compute_all_cost_metrics(pool)

# Update sheet
await google_sheets.update_cost_report(context, metrics)

# Run cost monitoring
monitoring = await alerts.run_cost_monitoring(
    context, pool, store, user_id, assistant_id
)
```

## Provider Adapters

Integration points for external usage APIs:

- **OpenAI**: Organization usage endpoint
- **ElevenLabs**: Character usage tracking
- **xAI**: Grok billing API
- **Cursor**: Development tool usage
- **Claude**: Anthropic API usage

**Note**: Adapter interfaces are complete; full implementations require API keys and endpoint access.

## Cost Metrics Explained

### Cost per Avatar
Total spend divided by number of active avatars in the period.

### Average Cost per Message
Total spend divided by number of message inferences (excludes uploads, analysis).

### Average Cost per Conversation
Total spend divided by number of distinct conversation threads.

### Cost per New User
Total spend by users who first appeared in the period, divided by count of new users.

## Alerting

Alerts are sent to the agent inbox when:

1. **Threshold Breach**: Daily or monthly spend exceeds configured limit
2. **Anomaly Detection**: Today's spend is >2 standard deviations above average

Alerts appear in the inbox with:
- Subject line describing the alert
- Detailed metrics
- Severity level (high/medium)

## Feature Cost Tracking

Track development costs by correlating:
- Git commits in time window
- API usage from `api_metrics`
- Duration and tokens consumed

Predict future costs based on:
- Historical cost per hour
- Estimated development time
- Similar feature patterns

## Troubleshooting

### "No credentials configured"
- Verify `GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON` or OAuth credentials are set
- Check JSON format is valid
- Ensure service account has Sheets API access

### "Failed to write sheet"
- Confirm spreadsheet is shared with service account email
- Verify spreadsheet URL/ID is correct
- Check sheet name matches `COST_REPORTING_RANGE`

### "No data returned"
- Ensure `api_metrics` table has data
- Check time period includes actual usage
- Verify database pool is connected

### Alerts not appearing
- Check thresholds are configured and reasonable
- Verify spend actually exceeds threshold
- Confirm inbox delivery is working

## Architecture

```
src/anubis/utils/reporting/
├── google_sheets.py      # Sheets API integration
├── cost_metrics.py       # Metrics from api_metrics
├── feature_tracking.py   # Git + cost correlation
├── alerts.py             # Threshold & anomaly detection
└── providers/            # External usage adapters
    ├── openai_adapter.py
    ├── elevenlabs_adapter.py
    ├── xai_adapter.py
    ├── cursor_adapter.py
    └── claude_adapter.py
```

## Data Sources

- **Primary**: `api_metrics` table (tokens, cost, model, inference type per request)
- **Secondary**: Git log (commits, authors, timestamps)
- **Future**: External provider APIs (OpenAI, ElevenLabs, etc.)

## Security

- Service account JSON stored as environment variable
- No credentials hardcoded in code
- Sheets access scoped to single spreadsheet
- All operations logged for audit

## Next Steps

1. ✅ Set up Google Sheets credentials
2. ✅ Configure target spreadsheet URL
3. ✅ Test with personal avatar tools
4. ⬜ Set up automated reporting schedule
5. ⬜ Complete provider API integrations
6. ⬜ Build Grafana dashboard

## Support

For issues or questions:
1. Check logs for error details
2. Verify environment variables are set
3. Test Google Sheets access manually
4. Review `api_metrics` table for data

## Reference

- Target Spreadsheet: https://docs.google.com/spreadsheets/d/1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA/edit
- Google Sheets API: https://developers.google.com/sheets/api
- Service Account Setup: https://cloud.google.com/iam/docs/service-accounts
