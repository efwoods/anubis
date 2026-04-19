-- Create metrics table
CREATE TABLE IF NOT EXISTS api_metrics (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    user_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    endpoint VARCHAR(255),
    method VARCHAR(10),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    request_latency_ms INTEGER,
    response_status INTEGER,
    cost_usd DECIMAL(10,6),
    thread_id VARCHAR(255),
    feedback_type VARCHAR(50),
    rating DECIMAL(3,2),
    inference_type_response_metrics JSONB,
    model_type_response_metrics JSONB,
    langsmith_trace_id VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_metrics_user_id ON api_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON api_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_metrics_thread_id ON api_metrics(thread_id);

-- Optional: GIN indexes for JSONB filtering in Grafana / SQL
-- CREATE INDEX IF NOT EXISTS idx_api_metrics_inference_json ON api_metrics USING GIN (inference_type_response_metrics);
-- CREATE INDEX IF NOT EXISTS idx_api_metrics_model_json ON api_metrics USING GIN (model_type_response_metrics);

-- Migration from older schema (if `model` column still exists):
-- ALTER TABLE api_metrics DROP COLUMN IF EXISTS model;
-- ALTER TABLE api_metrics ADD COLUMN IF NOT EXISTS inference_type_response_metrics JSONB;
-- ALTER TABLE api_metrics ADD COLUMN IF NOT EXISTS model_type_response_metrics JSONB;
