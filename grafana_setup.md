# Grafana + Postgres Setup for Metrics Dashboard

## 1. Install Grafana

```bash
# Using Docker
docker run -d -p 3000:3000 --name grafana \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana:latest
```

## 2. Add Postgres Data Source

1. Open Grafana at http://localhost:3000 (admin/admin)
2. Go to Configuration > Data Sources > Add data source
3. Select PostgreSQL
4. Configure:
   - Host: your_postgres_host:5432
   - Database: your_database_name
   - User: your_db_user
   - Password: your_db_password
   - SSL Mode: require/disable based on your setup

## 3. Create Dashboard Panels

### Active Users (Daily/Monthly)
```sql
SELECT
  DATE(timestamp) as date,
  COUNT(DISTINCT user_id) as daily_active_users
FROM api_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

### Response Latency
```sql
SELECT
  DATE(timestamp) as date,
  AVG(request_latency_ms) as avg_latency,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY request_latency_ms) as p95_latency,
  percentile_cont(0.99) WITHIN GROUP (ORDER BY request_latency_ms) as p99_latency
FROM api_metrics
WHERE endpoint IN ('/message', '/message/{assistant_id}')
  AND timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

### Token Costs
```sql
SELECT
  DATE(timestamp) as date,
  SUM(cost_usd) as daily_cost,
  SUM(total_tokens) as daily_tokens
FROM api_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

### Conversations per User
```sql
SELECT
  user_id,
  COUNT(DISTINCT conversation_id) as conversation_count
FROM api_metrics
WHERE conversation_id IS NOT NULL
  AND timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id
ORDER BY conversation_count DESC
LIMIT 100;
```

### Feedback Scores
```sql
SELECT
  DATE(am.timestamp) as date,
  AVG(uf.rating) as avg_rating,
  COUNT(uf.id) as feedback_count
FROM api_metrics am
LEFT JOIN user_feedback uf ON am.conversation_id = uf.conversation_id
WHERE am.timestamp >= NOW() - INTERVAL '30 days'
  AND uf.rating IS NOT NULL
GROUP BY DATE(am.timestamp)
ORDER BY date DESC;
```

## 4. Optional: Prometheus for Alerting

If you want alerting on high latency/costs:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'anubis-api'
    static_configs:
      - targets: ['your-api-host:8000']
    metrics_path: '/metrics'  # You'll need to add a /metrics endpoint
```

## 5. Sample Dashboard JSON

Import this JSON into Grafana to get a pre-built dashboard:

```json
{
  "dashboard": {
    "title": "Anubis API Metrics",
    "panels": [
      {
        "title": "Daily Active Users",
        "type": "graph",
        "targets": [{
          "rawSql": "SELECT DATE(timestamp) as time, COUNT(DISTINCT user_id) as value FROM api_metrics WHERE $__timeFilter(timestamp) GROUP BY DATE(timestamp) ORDER BY time",
          "format": "time_series"
        }]
      },
      {
        "title": "Response Latency",
        "type": "graph",
        "targets": [{
          "rawSql": "SELECT DATE(timestamp) as time, AVG(request_latency_ms) as value FROM api_metrics WHERE endpoint LIKE '/message%' AND $__timeFilter(timestamp) GROUP BY DATE(timestamp) ORDER BY time",
          "format": "time_series"
        }]
      },
      {
        "title": "Daily Token Costs",
        "type": "graph",
        "targets": [{
          "rawSql": "SELECT DATE(timestamp) as time, SUM(cost_usd) as value FROM api_metrics WHERE $__timeFilter(timestamp) GROUP BY DATE(timestamp) ORDER BY time",
          "format": "time_series"
        }]
      }
    ]
  }
}
```

## 6. Retention Analysis

For user retention, you can create cohort analysis:

```sql
WITH user_first_seen AS (
  SELECT
    user_id,
    DATE(MIN(timestamp)) as first_date
  FROM api_metrics
  GROUP BY user_id
),
daily_active AS (
  SELECT
    user_id,
    DATE(timestamp) as activity_date
  FROM api_metrics
  GROUP BY user_id, DATE(timestamp)
)
SELECT
  first_date,
  COUNT(DISTINCT CASE WHEN activity_date = first_date THEN user_id END) as day_0,
  COUNT(DISTINCT CASE WHEN activity_date = first_date + INTERVAL '1 day' THEN user_id END) as day_1,
  COUNT(DISTINCT CASE WHEN activity_date = first_date + INTERVAL '7 days' THEN user_id END) as day_7,
  COUNT(DISTINCT CASE WHEN activity_date = first_date + INTERVAL '30 days' THEN user_id END) as day_30
FROM user_first_seen ufs
LEFT JOIN daily_active da ON ufs.user_id = da.user_id
WHERE first_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY first_date
ORDER BY first_date DESC;
```