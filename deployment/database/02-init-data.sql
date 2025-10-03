-- Add initial data for Negative Space Imaging Project
-- This script inserts initial data required for the application

-- Add admin user
-- Password is 'admin123' (hashed)
INSERT INTO users (username, email, password_hash, first_name, last_name, role)
VALUES
    ('admin', 'admin@negativespaceimagingproject.com',
     '$2a$12$1lG7BQO.o4arRPsqMQFQU.SBQXkA3IKWD1FkPS9vw1SQfVT4ynk9m',
     'System', 'Administrator', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Add test user
-- Password is 'test123' (hashed)
INSERT INTO users (username, email, password_hash, first_name, last_name, role)
VALUES
    ('test', 'test@negativespaceimagingproject.com',
     '$2a$12$yl4vJFN6BXkWmQI/Yw1ZJu4RfJCtfzQ6/0c/FP7fHRTtZ.j26fmw2',
     'Test', 'User', 'user')
ON CONFLICT (username) DO NOTHING;

-- Add demo project for admin
INSERT INTO projects (name, description, owner_id, is_public)
VALUES
    ('Demo Project', 'A demonstration project with sample data',
     (SELECT id FROM users WHERE username = 'admin'), TRUE)
ON CONFLICT DO NOTHING;

-- Add default image sources
INSERT INTO image_sources (name, description, source_type, location, metadata)
VALUES
    ('Local Storage', 'Local file system storage', 'local', '/app/data/images',
     '{"retention_policy": "permanent", "default": true}'),
    ('Demo Images', 'Sample astronomical images for demonstration', 'local', '/app/data/demo',
     '{"retention_policy": "permanent", "read_only": true}'),
    ('Hubble Archive', 'Hubble Space Telescope Public Archive', 'remote', 'https://hla.stsci.edu/hlaview.html',
     '{"api_required": true, "retention_policy": "30 days", "format": "fits"}'
    )
ON CONFLICT DO NOTHING;

-- Add system settings
INSERT INTO system_settings (key, value, description)
VALUES
    ('processing.default_parameters',
     '{"quality": "high", "detect_threshold": 0.75, "max_batch_size": 4, "use_gpu": true}',
     'Default parameters for image processing'),
    ('security.password_policy',
     '{"min_length": 8, "require_uppercase": true, "require_numbers": true, "require_special": true, "max_age_days": 90}',
     'Password policy settings'),
    ('system.maintenance',
     '{"cleanup_interval_days": 7, "backup_interval_hours": 24, "max_log_age_days": 30}',
     'System maintenance settings'),
    ('ui.defaults',
     '{"theme": "dark", "items_per_page": 20, "default_view": "grid"}',
     'Default UI settings'),
    ('api.rate_limits',
     '{"authenticated": {"requests_per_minute": 60}, "unauthenticated": {"requests_per_minute": 10}}',
     'API rate limiting settings')
ON CONFLICT DO NOTHING;
