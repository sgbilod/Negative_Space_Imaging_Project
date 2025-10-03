-- Negative Space Imaging Project Migration
-- Add image verification and API access tables
-- Version: 1.0.0

-- Add image verification table
CREATE TABLE IF NOT EXISTS image_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    verification_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    verified_by UUID REFERENCES users(id) ON DELETE SET NULL,
    verification_date TIMESTAMP WITH TIME ZONE,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create index on image_verifications
CREATE INDEX idx_image_verifications_image_id ON image_verifications(image_id);
CREATE INDEX idx_image_verifications_status ON image_verifications(status);

-- Add API keys table for programmatic access
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    api_key VARCHAR(64) NOT NULL UNIQUE,
    permissions JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes on api_keys
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);

-- Add trigger for updated_at
CREATE TRIGGER update_api_keys_updated_at
    BEFORE UPDATE ON api_keys
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();

-- Update system version in settings
UPDATE settings SET value = '"1.1.0"'::jsonb WHERE key = 'system.version';

-- Insert system event for migration
INSERT INTO system_events (
    event_type,
    severity,
    message,
    details
) VALUES (
    'system.migration',
    'info',
    'Database migrated to version 1.1.0',
    '{"version": "1.1.0", "migration_id": "V1.0.0__add_verification_and_api_tables", "migrated_at": "' || NOW()::TEXT || '"}'::jsonb
);
