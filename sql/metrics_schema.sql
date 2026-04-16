-- Create metrics table
CREATE TABLE IF NOT EXISTS api_metrics (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    user_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    endpoint VARCHAR(255),
    method VARCHAR(10),
    model VARCHAR(255),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    request_latency_ms INTEGER,
    response_status INTEGER,
    cost_usd DECIMAL(10,6),
    thread_id VARCHAR(255),
    feedback_rating DECIMAL(3,2),
    langsmith_trace_id VARCHAR(255),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_metrics_user_id ON api_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON api_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_metrics_thread_id ON api_metrics(thread_id);

-- Optional: user_feedback table for detailed feedback
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    user_id VARCHAR(255),
    thread_id VARCHAR(255),
    feedback_type VARCHAR(50), -- 'like', 'dislike', 'rating', 'edit'
    rating DECIMAL(3,2), -- 1-5 scale
    comment TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);