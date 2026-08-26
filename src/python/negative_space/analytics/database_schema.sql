-- Phase 6: Analytics Engine Database Schema
-- PostgreSQL 15+ compatible
-- Created: November 8, 2025
-- Purpose: Time-series data storage, metrics, events, and analytics

-- ============================================================================
-- 1. ANALYTICS_EVENTS - Core event storage for real-time analytics
-- ============================================================================
-- Stores all incoming events with automatic timestamp and metadata
-- Partitioned by time for efficient querying

CREATE TABLE IF NOT EXISTS analytics_events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL,  -- Client-provided event ID for deduplication
    event_type VARCHAR(255) NOT NULL,  -- e.g., 'metric_collected', 'anomaly_detected'
    source VARCHAR(255) NOT NULL,  -- e.g., 'imaging_service', 'user_action'
    payload JSONB NOT NULL,  -- Flexible event data structure
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE,  -- When event was processed
    metadata JSONB,  -- Additional context (user_id, session_id, etc.)
    INDEX idx_analytics_events_type_time (event_type, created_at DESC),
    INDEX idx_analytics_events_source (source),
    INDEX idx_analytics_events_event_id (event_id),
    INDEX idx_analytics_events_created (created_at DESC)
) PARTITION BY RANGE (created_at);

-- Create weekly partitions for last 12 weeks + 4 weeks ahead
-- This query creates partitions dynamically in init script

-- ============================================================================
-- 2. METRICS_TIMESERIES - High-performance time-series metrics storage
-- ============================================================================
-- Stores pre-aggregated metrics with precise timestamp tracking
-- Optimized for range queries and downsampling

CREATE TABLE IF NOT EXISTS metrics_timeseries (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,  -- e.g., 'event_rate', 'error_rate', 'latency_p99'
    metric_type VARCHAR(50) NOT NULL,  -- 'gauge', 'counter', 'histogram'
    value DOUBLE PRECISION NOT NULL,  -- Metric value
    tags JSONB,  -- Key-value pairs for filtering (e.g., service, region, environment)
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,  -- Metric timestamp (may differ from created_at)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    INDEX idx_metrics_name_time (metric_name, timestamp DESC),
    INDEX idx_metrics_tags (tags) USING GIN,  -- JSONB index for tag filtering
    INDEX idx_metrics_timestamp (timestamp DESC),
    INDEX idx_metrics_metric_type (metric_type)
) PARTITION BY RANGE (timestamp);

-- Create daily partitions for efficient storage and querying
-- Automatic partition management via init script

-- ============================================================================
-- 3. AGGREGATED_METRICS - Pre-computed aggregations for dashboard
-- ============================================================================
-- Stores minute/hour/day level aggregations to speed up dashboard queries
-- Updated by batch processing jobs

CREATE TABLE IF NOT EXISTS aggregated_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    aggregation_level VARCHAR(50) NOT NULL,  -- 'minute', 'hour', 'day'
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    count BIGINT NOT NULL DEFAULT 0,  -- Number of raw metric points
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    avg_value DOUBLE PRECISION,
    median_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION,
    p95_value DOUBLE PRECISION,
    p99_value DOUBLE PRECISION,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    INDEX idx_agg_metrics_name_level (metric_name, aggregation_level, period_start DESC),
    INDEX idx_agg_metrics_period (period_start DESC, period_end DESC),
    INDEX idx_agg_metrics_tags (tags) USING GIN
) PARTITION BY RANGE (period_start);

-- ============================================================================
-- 4. ANOMALY_DETECTIONS - Detected anomalies and thresholds
-- ============================================================================
-- Records anomalies detected by statistical or ML algorithms
-- Supports investigation and alerting

CREATE TABLE IF NOT EXISTS anomaly_detections (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    detection_method VARCHAR(100) NOT NULL,  -- 'isolation_forest', 'zscore', 'iqr', 'autoencoder'
    anomaly_type VARCHAR(50) NOT NULL,  -- 'outlier', 'trend_change', 'seasonality_break'
    severity_score DOUBLE PRECISION NOT NULL,  -- 0.0-1.0, higher = more severe
    confidence DOUBLE PRECISION NOT NULL,  -- 0.0-1.0, confidence in detection
    expected_value DOUBLE PRECISION,  -- Expected value if applicable
    observed_value DOUBLE PRECISION,  -- Actual observed value
    threshold_lower DOUBLE PRECISION,
    threshold_upper DOUBLE PRECISION,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,  -- When anomaly started
    resolved_at TIMESTAMP WITH TIME ZONE,  -- When anomaly resolved
    alert_triggered BOOLEAN DEFAULT FALSE,
    context JSONB,  -- Additional context for investigation
    tags JSONB,
    INDEX idx_anomaly_metric_time (metric_name, detected_at DESC),
    INDEX idx_anomaly_severity (severity_score DESC),
    INDEX idx_anomaly_resolved (resolved_at),
    INDEX idx_anomaly_method (detection_method)
) PARTITION BY RANGE (detected_at);

-- ============================================================================
-- 5. STATISTICAL_SUMMARIES - Pre-computed statistical summaries
-- ============================================================================
-- Stores statistical summaries for quick analysis
-- Updated periodically by batch processes

CREATE TABLE IF NOT EXISTS statistical_summaries (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    window_size INTERVAL NOT NULL,  -- e.g., '1 hour', '1 day'
    summary_start TIMESTAMP WITH TIME ZONE NOT NULL,
    summary_end TIMESTAMP WITH TIME ZONE NOT NULL,
    data_points BIGINT NOT NULL,
    mean DOUBLE PRECISION,
    median DOUBLE PRECISION,
    mode DOUBLE PRECISION,
    std_dev DOUBLE PRECISION,
    variance DOUBLE PRECISION,
    skewness DOUBLE PRECISION,
    kurtosis DOUBLE PRECISION,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    range_value DOUBLE PRECISION,
    iqr_value DOUBLE PRECISION,
    coeff_of_variation DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    INDEX idx_stats_metric_time (metric_name, summary_start DESC),
    INDEX idx_stats_window (window_size)
) PARTITION BY RANGE (summary_start);

-- ============================================================================
-- 6. CORRELATION_ANALYSIS - Correlation between metrics
-- ============================================================================
-- Stores computed correlations for relationship analysis

CREATE TABLE IF NOT EXISTS correlation_analysis (
    id BIGSERIAL PRIMARY KEY,
    metric_a VARCHAR(255) NOT NULL,
    metric_b VARCHAR(255) NOT NULL,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    pearson_correlation DOUBLE PRECISION,  -- Pearson correlation coefficient
    spearman_correlation DOUBLE PRECISION,  -- Spearman rank correlation
    p_value DOUBLE PRECISION,  -- Statistical significance
    data_points BIGINT NOT NULL,
    significance_level VARCHAR(10),  -- 'significant', 'marginal', 'insignificant'
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    INDEX idx_corr_metrics (metric_a, metric_b, window_start DESC),
    INDEX idx_corr_significance (significance_level),
    CONSTRAINT unique_correlation UNIQUE (metric_a, metric_b, window_start, window_end)
);

-- ============================================================================
-- 7. ALERTS - Alert definitions and firing history
-- ============================================================================
-- Manages alert thresholds and records alert events

CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    metric_name VARCHAR(255) NOT NULL,
    condition_type VARCHAR(50) NOT NULL,  -- 'threshold', 'change', 'anomaly', 'composite'
    threshold_value DOUBLE PRECISION,
    threshold_operator VARCHAR(10),  -- '>', '<', '>=', '<=', '==', '!='
    enabled BOOLEAN DEFAULT TRUE,
    severity VARCHAR(50) NOT NULL,  -- 'info', 'warning', 'critical'
    notification_channels JSONB,  -- e.g., ['email', 'slack', 'pagerduty']
    cooldown_minutes INTEGER DEFAULT 15,  -- Min time between successive alerts
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    INDEX idx_alert_metric (metric_name),
    INDEX idx_alert_enabled (enabled),
    INDEX idx_alert_severity (severity)
);

CREATE TABLE IF NOT EXISTS alert_events (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION,
    message TEXT,
    status VARCHAR(50),  -- 'triggered', 'acknowledged', 'resolved'
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    INDEX idx_alert_events_alert (alert_id, triggered_at DESC),
    INDEX idx_alert_events_status (status),
    INDEX idx_alert_events_time (triggered_at DESC)
) PARTITION BY RANGE (triggered_at);

-- ============================================================================
-- 8. ANALYSIS_SESSIONS - Track analysis sessions and queries
-- ============================================================================
-- Records user queries and analysis activities for auditing

CREATE TABLE IF NOT EXISTS analysis_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    user_id VARCHAR(255),
    query_type VARCHAR(100),  -- 'timeseries_query', 'anomaly_detection', 'correlation'
    metrics_queried JSONB,  -- List of metrics in query
    filter_criteria JSONB,  -- Tags, time ranges, etc.
    execution_time_ms BIGINT,  -- Query execution time
    result_count BIGINT,  -- Number of results returned
    query_text TEXT,  -- Actual query for debugging
    status VARCHAR(50),  -- 'success', 'error', 'timeout'
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    INDEX idx_session_user (user_id),
    INDEX idx_session_type (query_type),
    INDEX idx_session_time (created_at DESC)
) PARTITION BY RANGE (created_at);

-- ============================================================================
-- 9. CACHE_ENTRIES - Redis-backed cache metadata
-- ============================================================================
-- Tracks what's cached for cache management

CREATE TABLE IF NOT EXISTS cache_entries (
    id BIGSERIAL PRIMARY KEY,
    cache_key VARCHAR(512) NOT NULL UNIQUE,
    cache_type VARCHAR(100),  -- 'aggregated_metrics', 'statistical_summary', 'query_result'
    value_type VARCHAR(100),
    size_bytes BIGINT,
    ttl_seconds INTEGER,
    hit_count BIGINT DEFAULT 0,
    miss_count BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    INDEX idx_cache_type (cache_type),
    INDEX idx_cache_expires (expires_at),
    INDEX idx_cache_accessed (last_accessed DESC)
);

-- ============================================================================
-- INDEXES - Additional composite indexes for common queries
-- ============================================================================

-- Analytics events by source and time
CREATE INDEX IF NOT EXISTS idx_events_source_time
    ON analytics_events(source, created_at DESC);

-- Metrics by multiple dimensions
CREATE INDEX IF NOT EXISTS idx_metrics_composite
    ON metrics_timeseries(metric_name, metric_type, timestamp DESC);

-- Anomalies by metric and time
CREATE INDEX IF NOT EXISTS idx_anomaly_composite
    ON anomaly_detections(metric_name, severity_score DESC, detected_at DESC);

-- ============================================================================
-- VIEWS - Useful pre-defined views for analytics
-- ============================================================================

-- View: Recent anomalies (unresolved)
CREATE OR REPLACE VIEW v_recent_unresolved_anomalies AS
SELECT
    ad.id,
    ad.metric_name,
    ad.anomaly_type,
    ad.severity_score,
    ad.confidence,
    ad.observed_value,
    ad.expected_value,
    ad.detected_at,
    ad.detection_method,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ad.detected_at))/3600 as hours_since_detection
FROM anomaly_detections ad
WHERE ad.resolved_at IS NULL
ORDER BY ad.detected_at DESC;

-- View: Top anomalies by severity
CREATE OR REPLACE VIEW v_top_anomalies_by_severity AS
SELECT
    metric_name,
    anomaly_type,
    COUNT(*) as count,
    AVG(severity_score) as avg_severity,
    MAX(severity_score) as max_severity,
    COUNT(CASE WHEN resolved_at IS NULL THEN 1 END) as unresolved_count
FROM anomaly_detections
WHERE detected_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY metric_name, anomaly_type
ORDER BY max_severity DESC;

-- View: Metrics with recent anomalies
CREATE OR REPLACE VIEW v_metrics_with_anomalies AS
SELECT DISTINCT
    m.metric_name,
    COUNT(DISTINCT ad.id) as anomaly_count,
    MAX(ad.severity_score) as max_severity,
    MAX(ad.detected_at) as last_anomaly
FROM metrics_timeseries m
LEFT JOIN anomaly_detections ad ON m.metric_name = ad.metric_name
WHERE ad.detected_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY m.metric_name
ORDER BY anomaly_count DESC;

-- ============================================================================
-- COMMENTS - Documentation for schema
-- ============================================================================

COMMENT ON TABLE analytics_events IS 'Stores all incoming events with flexible JSONB payload for extensibility';
COMMENT ON TABLE metrics_timeseries IS 'High-performance time-series storage for metrics with partitioning';
COMMENT ON TABLE aggregated_metrics IS 'Pre-aggregated metrics for dashboard performance';
COMMENT ON TABLE anomaly_detections IS 'Detected anomalies with severity, confidence, and resolution tracking';
COMMENT ON TABLE statistical_summaries IS 'Computed statistical summaries for analysis';
COMMENT ON TABLE correlation_analysis IS 'Correlation metrics between different measurements';
COMMENT ON TABLE alerts IS 'Alert definitions and configuration';
COMMENT ON TABLE alert_events IS 'Alert firing history and status tracking';
COMMENT ON TABLE analysis_sessions IS 'Audit trail of user queries and analysis';
COMMENT ON TABLE cache_entries IS 'Metadata for cache management';

COMMENT ON COLUMN metrics_timeseries.tags IS 'JSONB tags for filtering, e.g., {"service":"imaging","region":"us-east"}';
COMMENT ON COLUMN anomaly_detections.severity_score IS 'Severity of anomaly from 0.0 (none) to 1.0 (critical)';
COMMENT ON COLUMN anomaly_detections.confidence IS 'Confidence in detection from 0.0 (random) to 1.0 (certain)';

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
