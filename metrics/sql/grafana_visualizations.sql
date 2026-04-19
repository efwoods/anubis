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


# Model Usage Analysis
## Per model usage

```sql
SELECT
  timestamp,
  request_id,
  elem->>'model' AS model,
  (elem->>'cost')::numeric AS cost,
  (elem->>'latency')::int AS latency_ms,
  (elem->>'prompt_tokens')::int AS prompt_tokens,
  (elem->>'completion_tokens')::int AS completion_tokens,
  (elem->>'total_tokens')::int AS total_tokens,
  request_latency_ms AS request_latency_ms
FROM api_metrics,
     jsonb_array_elements(COALESCE(model_type_response_metrics, '[]'::jsonb)) AS elem
WHERE endpoint LIKE '/message%';
```

`latency_ms` in each row is the full `/message` request wall-clock (milliseconds), duplicated per model until per-model timings are instrumented. Use `request_latency_ms` on the parent row for a single per-request value.

## Per inference type usage
```sql
SELECT
  timestamp,
  request_id,
  elem->>'inference_type' AS inference_type,
  (elem->>'cost')::numeric AS cost,
  (elem->>'latency')::int AS latency_ms,
  (elem->>'prompt_tokens')::int AS prompt_tokens,
  (elem->>'completion_tokens')::int AS completion_tokens,
  (elem->>'total_tokens')::int AS total_tokens,
  request_latency_ms AS request_latency_ms
FROM api_metrics,
     jsonb_array_elements(COALESCE(inference_type_response_metrics, '[]'::jsonb)) AS elem
WHERE endpoint LIKE '/message%';
```