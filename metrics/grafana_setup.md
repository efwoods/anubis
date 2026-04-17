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

#### Daily Active Users
```sql
SELECT
  DATE(timestamp) as date,
  COUNT(DISTINCT user_id) as daily_active_users
FROM api_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

#### Monthly Active Users
```sql
SELECT
  DATE_TRUNC('month', timestamp) as month,
  COUNT(DISTINCT user_id) as monthly_active_users
FROM api_metrics
WHERE timestamp >= NOW() - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY month DESC;
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
  COUNT(DISTINCT thread_id) as conversation_count
FROM api_metrics
WHERE thread_id IS NOT NULL
  AND timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id
ORDER BY conversation_count DESC
LIMIT 100;
```

### Average Daily Messages
```sql
SELECT
  DATE(timestamp) as date,
  COUNT(*) as total_messages,
  COUNT(DISTINCT user_id) as active_users,
  CASE 
    WHEN COUNT(DISTINCT user_id) > 0 
    THEN ROUND(COUNT(*)::numeric / COUNT(DISTINCT user_id), 2) 
    ELSE 0 
  END as avg_messages_per_user
FROM api_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

### Average Daily Messages per User
```sql
SELECT
  user_id,
  COUNT(*) as total_messages,
  COUNT(DISTINCT DATE(timestamp)) as active_days,
  CASE 
    WHEN COUNT(DISTINCT DATE(timestamp)) > 0 
    THEN ROUND(COUNT(*)::numeric / COUNT(DISTINCT DATE(timestamp)), 2) 
    ELSE 0 
  END as avg_daily_messages
FROM api_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id
HAVING COUNT(DISTINCT DATE(timestamp)) > 0
ORDER BY avg_daily_messages DESC
LIMIT 100;
```

### Average Feedback Rating  Scores
```sql
SELECT
  DATE(uf.timestamp) as date,
  AVG(uf.rating) as avg_rating,
  COUNT(uf.id) as feedback_count
FROM user_feedback uf
WHERE uf.timestamp >= NOW() - INTERVAL '30 days'
  AND uf.feedback_type = 'rating'
  AND uf.rating IS NOT NULL
GROUP BY DATE(uf.timestamp)
ORDER BY date DESC;
```

### Number of Unique Users and Count of Total Feedback per Feedback Type
```sql
SELECT
  feedback_type,
  COUNT(*) as feedback_count,
  COUNT(DISTINCT user_id) as users_providing_feedback
FROM user_feedback
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY feedback_type
ORDER BY feedback_count DESC;
```

### Daily Feedback Activity for the Last Month
```sql
SELECT
  DATE(timestamp) as date,
  feedback_type,
  COUNT(*) as feedback_count
FROM user_feedback
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp), feedback_type
ORDER BY date DESC, feedback_count DESC;
```

### Total Feedback Engagement
```sql
SELECT
  COUNT(*) as total_feedback_events,
  COUNT(DISTINCT user_id) as users_providing_feedback,
  COUNT(DISTINCT thread_id) as conversations_with_feedback,
  SUM(CASE WHEN feedback_type = 'like' THEN 1 ELSE 0 END) as likes,
  SUM(CASE WHEN feedback_type = 'dislike' THEN 1 ELSE 0 END) as dislikes,
  SUM(CASE WHEN feedback_type = 'edit' THEN 1 ELSE 0 END) as edits,
  SUM(CASE WHEN feedback_type = 'rating' THEN 1 ELSE 0 END) as ratings
FROM user_feedback
WHERE timestamp >= NOW() - INTERVAL '30 days';
```


## Retention Analysis

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
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [{
          "rawSql": "SELECT DATE(timestamp) as time, COUNT(DISTINCT user_id) as \"Daily Active Users\" FROM api_metrics WHERE $__timeFilter(timestamp) GROUP BY DATE(timestamp) ORDER BY time",
          "format": "time_series"
        }]
      },
      {
        "title": "Monthly Active Users",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [{
          "rawSql": "SELECT DATE_TRUNC('month', timestamp) as time, COUNT(DISTINCT user_id) as \"Monthly Active Users\" FROM api_metrics WHERE $__timeFilter(timestamp) GROUP BY DATE_TRUNC('month', timestamp) ORDER BY time",
          "format": "time_series"
        }]
      },
      {
        "title": "Response Latency",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [{
          "rawSql": "SELECT DATE(timestamp) as time, AVG(request_latency_ms) as \"Avg Latency\", percentile_cont(0.95) WITHIN GROUP (ORDER BY request_latency_ms) as \"P95 Latency\" FROM api_metrics WHERE endpoint LIKE '/message%' AND $__timeFilter(timestamp) GROUP BY DATE(timestamp) ORDER BY time",
          "format": "time_series"
        }]
      },
      {
        "title": "Daily Token Costs",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [{
          "rawSql": "SELECT DATE(timestamp) as time, SUM(cost_usd) as \"Daily Cost ($)\" FROM api_metrics WHERE $__timeFilter(timestamp) GROUP BY DATE(timestamp) ORDER BY time",
          "format": "time_series"
        }]
      },
      {
        "title": "Average Daily Messages per User",
        "type": "table",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
        "targets": [{
          "rawSql": "SELECT user_id as \"User ID\", COUNT(*) as \"Total Messages\", COUNT(DISTINCT DATE(timestamp)) as \"Active Days\", CASE WHEN COUNT(DISTINCT DATE(timestamp)) > 0 THEN ROUND(COUNT(*)::numeric / COUNT(DISTINCT DATE(timestamp)), 2) ELSE 0 END as \"Avg Daily Messages\" FROM api_metrics WHERE $__timeFilter(timestamp) GROUP BY user_id HAVING COUNT(DISTINCT DATE(timestamp)) > 0 ORDER BY \"Avg Daily Messages\" DESC LIMIT 20",
          "format": "table"
        }]
      },
      {
        "title": "Conversations per User",
        "type": "table",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
        "targets": [{
          "rawSql": "SELECT user_id as \"User ID\", COUNT(DISTINCT thread_id) as \"Conversations\" FROM api_metrics WHERE thread_id IS NOT NULL AND $__timeFilter(timestamp) GROUP BY user_id ORDER BY \"Conversations\" DESC LIMIT 20",
          "format": "table"
        }]
      },
      {
        "title": "Feedback Count by Type",
        "type": "barchart",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
        "targets": [{
          "rawSql": "SELECT feedback_type as \"Type\", COUNT(*) as \"Count\", COUNT(DISTINCT user_id) as \"Users\" FROM user_feedback WHERE $__timeFilter(timestamp) GROUP BY feedback_type ORDER BY \"Count\" DESC",
          "format": "table"
        }]
      },
      {
        "title": "Daily Feedback Activity",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
        "targets": [{
          "rawSql": "SELECT DATE(timestamp) as time, feedback_type, COUNT(*) as feedback_count FROM user_feedback WHERE $__timeFilter(timestamp) GROUP BY DATE(timestamp), feedback_type ORDER BY time, feedback_type",
          "format": "time_series"
        }]
      },
      {
        "title": "Total Feedback Engagement",
        "type": "stat",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 32},
        "targets": [{
          "rawSql": "SELECT COUNT(*) as \"Total Feedback\", COUNT(DISTINCT user_id) as \"Users\", COUNT(DISTINCT thread_id) as \"Conversations\", SUM(CASE WHEN feedback_type = 'like' THEN 1 ELSE 0 END) as \"Likes\", SUM(CASE WHEN feedback_type = 'dislike' THEN 1 ELSE 0 END) as \"Dislikes\", SUM(CASE WHEN feedback_type = 'edit' THEN 1 ELSE 0 END) as \"Edits\", SUM(CASE WHEN feedback_type = 'rating' THEN 1 ELSE 0 END) as \"Ratings\" FROM user_feedback WHERE $__timeFilter(timestamp)",
          "format": "table"
        }]
      }
    ]
  }
}
```
