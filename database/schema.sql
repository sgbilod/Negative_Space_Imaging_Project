-- ============================================================================
-- NEGATIVE SPACE IMAGING PROJECT - POSTGRESQL SCHEMA
--
-- Database: negative_space
-- Version: 1.0.0
-- PostgreSQL: 14+
--
-- Purpose:
--   Complete schema for imaging analysis, security, and processing management
--
-- Features:
--   - JSONB for flexible metadata storage
--   - Comprehensive audit trail
--   - Performance-optimized indexes
--   - Foreign key constraints with CASCADE
--   - Timezone-aware timestamps (UTC)
--   - Enum types for status tracking
--
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- User roles for authorization
CREATE TYPE user_role AS ENUM (
    'admin',           -- Full system access
    'analyst',         -- Can analyze images and view results
    'viewer',          -- Read-only access to results
    'api_user'         -- API-based access
);

-- Image processing status
CREATE TYPE analysis_status AS ENUM (
    'pending',         -- Waiting to be processed
    'processing',      -- Currently being analyzed
    'completed',       -- Successfully processed
    'failed',          -- Processing failed
    'archived'         -- Archived/completed
);

-- Processing queue status
CREATE TYPE queue_status AS ENUM (
    'queued',          -- In queue, not started
    'processing',      -- Currently processing
    'completed',       -- Successfully completed
    'failed',          -- Failed to process
    'retrying',        -- Retrying after failure
    'cancelled'        -- Explicitly cancelled
);

-- Security action types
CREATE TYPE audit_action AS ENUM (
    'login',           -- User login
    'logout',          -- User logout
    'create_image',    -- Image uploaded
    'delete_image',    -- Image deleted
    'run_analysis',    -- Analysis started
    'view_results',    -- Results accessed
    'export_data',     -- Data exported
    'api_call',        -- API endpoint called
    'permission_change', -- Permissions modified
    'config_change'    -- Configuration changed
);

-- Audit status
CREATE TYPE audit_status AS ENUM (
    'success',         -- Action succeeded
    'failure',         -- Action failed
    'partial'          -- Partially successful
);

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',

    -- Constraints
    CONSTRAINT valid_email CHECK (email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

-- Images table
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_name VARCHAR(512) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    original_filename VARCHAR(512),
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    image_dimensions JSONB DEFAULT '{"width": 0, "height": 0, "channels": 0}',
    analysis_status analysis_status NOT NULL DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    uploaded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT positive_file_size CHECK (file_size > 0),
    CONSTRAINT valid_mime_type CHECK (mime_type ~ '^[a-z]+/[a-z0-9\-\.]+$')
);

-- Analysis Results table
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL UNIQUE REFERENCES images(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    algorithm_version VARCHAR(50) NOT NULL,
    algorithm_parameters JSONB DEFAULT '{}',
    negative_space_data JSONB NOT NULL,
    anomalies JSONB DEFAULT '[]',
    confidence_score NUMERIC(5, 4) NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    quality_metrics JSONB DEFAULT '{}',
    visualization_path VARCHAR(1024),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT positive_processing_time CHECK (processing_time_ms > 0),
    CONSTRAINT valid_algorithm_version CHECK (algorithm_version ~ '^\d+\.\d+\.\d+$')
);

-- Processing Queue table
CREATE TABLE processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status queue_status NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 5,
    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    last_error_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion_at TIMESTAMP WITH TIME ZONE,
    worker_id VARCHAR(255),

    -- Constraints
    CONSTRAINT valid_priority CHECK (priority >= 1 AND priority <= 10)
);

-- API Keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    scopes JSONB DEFAULT '["read:images", "read:results"]',
    ip_whitelist JSONB DEFAULT 'null',
    rate_limit_per_minute INTEGER DEFAULT 100,

    -- Constraints
    CONSTRAINT valid_scopes CHECK (scopes IS NULL OR jsonb_typeof(scopes) = 'array')
);

-- Security Audit table
CREATE TABLE security_audit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action audit_action NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    status audit_status NOT NULL DEFAULT 'success',
    ip_address INET,
    user_agent TEXT,
    error_message TEXT,
    changes_before JSONB,
    changes_after JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_ip CHECK (ip_address IS NULL OR family(ip_address) = 4 OR family(ip_address) = 6)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Users indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_is_active ON users(is_active);
CREATE INDEX idx_users_created_at ON users(created_at DESC);
CREATE INDEX idx_users_last_login ON users(last_login_at DESC);

-- Images indexes
CREATE INDEX idx_images_user_id ON images(user_id);
CREATE INDEX idx_images_analysis_status ON images(analysis_status);
CREATE INDEX idx_images_created_at ON images(created_at DESC);
CREATE INDEX idx_images_uploaded_at ON images(uploaded_at DESC);
CREATE INDEX idx_images_file_hash ON images(file_hash);
CREATE INDEX idx_images_user_created ON images(user_id, created_at DESC);
CREATE INDEX idx_images_status_created ON images(analysis_status, created_at DESC);
-- JSONB indexes
CREATE INDEX idx_images_metadata ON images USING GIN(metadata);
CREATE INDEX idx_images_dimensions ON images USING GIN(image_dimensions);

-- Analysis Results indexes
CREATE INDEX idx_analysis_results_image_id ON analysis_results(image_id);
CREATE INDEX idx_analysis_results_user_id ON analysis_results(user_id);
CREATE INDEX idx_analysis_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX idx_analysis_results_algorithm_version ON analysis_results(algorithm_version);
CREATE INDEX idx_analysis_results_confidence ON analysis_results(confidence_score DESC);
CREATE INDEX idx_analysis_results_user_created ON analysis_results(user_id, created_at DESC);
-- JSONB indexes
CREATE INDEX idx_analysis_results_negative_space ON analysis_results USING GIN(negative_space_data);
CREATE INDEX idx_analysis_results_anomalies ON analysis_results USING GIN(anomalies);
CREATE INDEX idx_analysis_results_quality_metrics ON analysis_results USING GIN(quality_metrics);

-- Processing Queue indexes
CREATE INDEX idx_queue_image_id ON processing_queue(image_id);
CREATE INDEX idx_queue_user_id ON processing_queue(user_id);
CREATE INDEX idx_queue_status ON processing_queue(status);
CREATE INDEX idx_queue_priority ON processing_queue(priority DESC);
CREATE INDEX idx_queue_created_at ON processing_queue(created_at DESC);
CREATE INDEX idx_queue_status_priority ON processing_queue(status, priority DESC);
CREATE INDEX idx_queue_retry_count ON processing_queue(retry_count);
CREATE INDEX idx_queue_next_retry ON processing_queue(created_at)
    WHERE status IN ('failed', 'retrying');

-- API Keys indexes
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX idx_api_keys_created_at ON api_keys(created_at DESC);
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at);

-- Security Audit indexes
CREATE INDEX idx_audit_user_id ON security_audit(user_id);
CREATE INDEX idx_audit_action ON security_audit(action);
CREATE INDEX idx_audit_timestamp ON security_audit(timestamp DESC);
CREATE INDEX idx_audit_ip_address ON security_audit(ip_address);
CREATE INDEX idx_audit_user_timestamp ON security_audit(user_id, timestamp DESC);
CREATE INDEX idx_audit_status ON security_audit(status);
CREATE INDEX idx_audit_resource ON security_audit(resource_type, resource_id);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- User activity summary
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT
    u.id,
    u.email,
    u.role,
    COUNT(DISTINCT i.id) as total_images,
    COUNT(DISTINCT ar.id) as total_analyses,
    MAX(i.uploaded_at) as last_image_upload,
    MAX(ar.created_at) as last_analysis,
    u.last_login_at,
    u.created_at
FROM users u
LEFT JOIN images i ON u.id = i.user_id
LEFT JOIN analysis_results ar ON u.id = ar.user_id
GROUP BY u.id, u.email, u.role, u.last_login_at, u.created_at;

-- Processing queue status summary
CREATE OR REPLACE VIEW queue_status_summary AS
SELECT
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) as avg_queue_time_seconds,
    MIN(created_at) as oldest_item,
    MAX(created_at) as newest_item
FROM processing_queue
GROUP BY status;

-- Image analysis status
CREATE OR REPLACE VIEW image_analysis_status AS
SELECT
    i.id,
    i.file_name,
    i.user_id,
    i.analysis_status,
    ar.confidence_score,
    ar.processing_time_ms,
    ar.algorithm_version,
    ar.created_at,
    CASE
        WHEN i.analysis_status = 'completed' THEN TRUE
        ELSE FALSE
    END as is_analyzed
FROM images i
LEFT JOIN analysis_results ar ON i.id = ar.image_id
ORDER BY i.created_at DESC;

-- Recent audit events
CREATE OR REPLACE VIEW recent_audit_events AS
SELECT
    id,
    user_id,
    action,
    resource_type,
    resource_id,
    status,
    ip_address,
    timestamp,
    error_message
FROM security_audit
ORDER BY timestamp DESC
LIMIT 1000;

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for users table
CREATE TRIGGER users_update_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for images table
CREATE TRIGGER images_update_updated_at BEFORE UPDATE ON images
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for analysis_results table
CREATE TRIGGER analysis_results_update_updated_at BEFORE UPDATE ON analysis_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to validate image status transition
CREATE OR REPLACE FUNCTION validate_image_status_transition()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.analysis_status = OLD.analysis_status THEN
        RETURN NEW;
    END IF;

    -- Define valid status transitions
    IF OLD.analysis_status = 'pending' AND NEW.analysis_status NOT IN ('processing', 'failed') THEN
        RAISE EXCEPTION 'Invalid status transition from pending to %', NEW.analysis_status;
    END IF;

    IF OLD.analysis_status = 'processing' AND NEW.analysis_status NOT IN ('completed', 'failed') THEN
        RAISE EXCEPTION 'Invalid status transition from processing to %', NEW.analysis_status;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for image status validation
CREATE TRIGGER validate_image_status BEFORE UPDATE ON images
    FOR EACH ROW EXECUTE FUNCTION validate_image_status_transition();

-- Function to create audit log entry
CREATE OR REPLACE FUNCTION log_audit_event()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO security_audit (
        action,
        resource_type,
        resource_id,
        status,
        ip_address
    ) VALUES (
        'create_image'::audit_action,
        'image',
        NEW.id::TEXT,
        'success'::audit_status,
        NULL
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for audit logging on image creation
CREATE TRIGGER audit_image_creation AFTER INSERT ON images
    FOR EACH ROW EXECUTE FUNCTION log_audit_event();

-- Function to validate queue status transitions
CREATE OR REPLACE FUNCTION validate_queue_status_transition()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = OLD.status THEN
        RETURN NEW;
    END IF;

    -- Valid transitions
    IF OLD.status = 'queued' AND NEW.status NOT IN ('processing', 'cancelled') THEN
        RAISE EXCEPTION 'Invalid queue transition from queued to %', NEW.status;
    END IF;

    IF OLD.status = 'processing' AND NEW.status NOT IN ('completed', 'failed') THEN
        RAISE EXCEPTION 'Invalid queue transition from processing to %', NEW.status;
    END IF;

    IF OLD.status = 'failed' AND NEW.status NOT IN ('retrying', 'cancelled') THEN
        RAISE EXCEPTION 'Invalid queue transition from failed to %', NEW.status;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for queue status validation
CREATE TRIGGER validate_queue_status BEFORE UPDATE ON processing_queue
    FOR EACH ROW EXECUTE FUNCTION validate_queue_status_transition();

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Create application role
CREATE ROLE app_user WITH LOGIN PASSWORD 'change_me_in_production';

-- Grant permissions on tables
GRANT SELECT, INSERT, UPDATE ON users TO app_user;
GRANT SELECT, INSERT, UPDATE ON images TO app_user;
GRANT SELECT, INSERT, UPDATE ON analysis_results TO app_user;
GRANT SELECT, INSERT, UPDATE ON processing_queue TO app_user;
GRANT SELECT, INSERT, UPDATE ON api_keys TO app_user;
GRANT INSERT ON security_audit TO app_user;
GRANT SELECT ON security_audit TO app_user;

-- Grant permissions on views
GRANT SELECT ON user_activity_summary TO app_user;
GRANT SELECT ON queue_status_summary TO app_user;
GRANT SELECT ON image_analysis_status TO app_user;
GRANT SELECT ON recent_audit_events TO app_user;

-- Grant permissions on sequences
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE users IS 'User accounts with authentication and role-based access';
COMMENT ON TABLE images IS 'Uploaded images awaiting or completed analysis';
COMMENT ON TABLE analysis_results IS 'Results from negative space analysis on images';
COMMENT ON TABLE processing_queue IS 'Task queue for image processing with retry logic';
COMMENT ON TABLE api_keys IS 'API authentication keys with expiration and rate limiting';
COMMENT ON TABLE security_audit IS 'Complete audit trail of all user actions and system events';

COMMENT ON COLUMN users.password_hash IS 'Bcrypt hash of user password';
COMMENT ON COLUMN users.metadata IS 'Additional user data (preferences, settings)';
COMMENT ON COLUMN images.file_hash IS 'SHA-256 hash of file content for deduplication';
COMMENT ON COLUMN images.metadata IS 'Image metadata (EXIF, color space, etc.)';
COMMENT ON COLUMN analysis_results.negative_space_data IS 'Complex analysis results in JSON format';
COMMENT ON COLUMN analysis_results.anomalies IS 'Array of detected anomalies with details';
COMMENT ON COLUMN processing_queue.worker_id IS 'Identifies which worker processes this task';

-- ============================================================================
-- DATABASE STATISTICS AND ANALYSIS
-- ============================================================================

-- Analyze tables for query planner
ANALYZE users;
ANALYZE images;
ANALYZE analysis_results;
ANALYZE processing_queue;
ANALYZE api_keys;
ANALYZE security_audit;
