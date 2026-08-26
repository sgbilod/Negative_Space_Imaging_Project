# PostgreSQL Database System - Negative Space Imaging Project

## Overview

This directory contains the complete PostgreSQL database schema, migrations, and management tools for the Negative Space Imaging Project. The system is designed for production-grade imaging analysis with comprehensive audit trails, performance optimization, and data integrity.

## Architecture

### Database Structure

```
negative_space/
├── schema.sql                    # Complete schema definition
├── migrationRunner.js            # Migration management framework
├── migrations/
│   ├── 001_initial_schema.js    # Initial schema setup
│   └── 002_add_audit_tables.js  # Enhanced audit trail
└── README.md                     # This file
```

### Core Tables

#### **Users** (`users`)

User accounts with authentication and role-based access control.

```sql
Fields:
- id (UUID, PK)
- email (VARCHAR 255, UNIQUE)
- password_hash (VARCHAR 255)
- full_name (VARCHAR 255)
- role (user_role: admin, analyst, viewer, api_user)
- is_active (BOOLEAN)
- is_email_verified (BOOLEAN)
- last_login_at (TIMESTAMP)
- created_at, updated_at (TIMESTAMP)
- metadata (JSONB)

Indexes:
- email, role, is_active, created_at, last_login
- Composite: (user_id, created_at)
```

#### **Images** (`images`)

Uploaded images awaiting or completed analysis.

```sql
Fields:
- id (UUID, PK)
- user_id (UUID, FK → users)
- file_name (VARCHAR 512)
- file_path (VARCHAR 1024)
- file_size (BIGINT)
- mime_type (VARCHAR 100)
- file_hash (VARCHAR 64, UNIQUE)
- image_dimensions (JSONB)
- analysis_status (analysis_status: pending, processing, completed, failed, archived)
- metadata (JSONB)
- uploaded_at, processed_at, created_at, updated_at (TIMESTAMP)

Indexes:
- user_id, file_hash, analysis_status, created_at
- JSONB: metadata, image_dimensions
- Composite: (user_id, created_at), (analysis_status, created_at)
```

#### **Analysis Results** (`analysis_results`)

Results from negative space analysis on images.

```sql
Fields:
- id (UUID, PK)
- image_id (UUID, UNIQUE FK → images)
- user_id (UUID, FK → users)
- algorithm_version (VARCHAR 50)
- algorithm_parameters (JSONB)
- negative_space_data (JSONB)
- anomalies (JSONB[])
- confidence_score (DECIMAL 5,4: 0.0-1.0)
- processing_time_ms (INTEGER)
- quality_metrics (JSONB)
- visualization_path (VARCHAR 1024)
- notes (TEXT)
- created_at, updated_at (TIMESTAMP)

Indexes:
- image_id, user_id, algorithm_version, created_at
- confidence_score (DESC)
- JSONB: negative_space_data, anomalies, quality_metrics
- Composite: (user_id, created_at)
```

#### **Processing Queue** (`processing_queue`)

Task queue for image processing with automatic retry logic.

```sql
Fields:
- id (UUID, PK)
- image_id (UUID, FK → images)
- user_id (UUID, FK → users)
- status (queue_status: queued, processing, completed, failed, retrying, cancelled)
- priority (INTEGER: 1-10)
- retry_count (INTEGER)
- max_retries (INTEGER)
- error_message (TEXT)
- last_error_at (TIMESTAMP)
- created_at, started_at, completed_at, estimated_completion_at (TIMESTAMP)
- worker_id (VARCHAR 255)

Indexes:
- image_id, user_id, status, created_at
- Composite: (status, priority), (created_at, status)
- Filtered: status IN ('failed', 'retrying')
```

#### **API Keys** (`api_keys`)

API authentication keys with expiration and rate limiting.

```sql
Fields:
- id (UUID, PK)
- user_id (UUID, FK → users)
- key_hash (VARCHAR 255, UNIQUE)
- name (VARCHAR 255)
- description (TEXT)
- is_active (BOOLEAN)
- last_used_at (TIMESTAMP)
- created_at, expires_at (TIMESTAMP)
- scopes (JSONB[])
- ip_whitelist (JSONB)
- rate_limit_per_minute (INTEGER)

Indexes:
- user_id, key_hash, is_active, created_at, expires_at
```

#### **Security Audit** (`security_audit`)

Complete audit trail of all user actions and system events.

```sql
Fields:
- id (UUID, PK)
- user_id (UUID, FK → users, nullable)
- action (audit_action: login, logout, create_image, delete_image, run_analysis, view_results, export_data, api_call, permission_change, config_change)
- resource_type (VARCHAR 100)
- resource_id (VARCHAR 255)
- status (audit_status: success, failure, partial)
- ip_address (INET)
- user_agent (TEXT)
- error_message (TEXT)
- changes_before, changes_after (JSONB)
- timestamp (TIMESTAMP)

Indexes:
- user_id, action, status, ip_address, created_at
- Composite: (user_id, timestamp), (resource_type, resource_id)
```

### Enhancement Tables (Migration 002)

#### **Audit Log Details** (`audit_log_details`)

Enhanced audit with detailed event tracking.

```sql
Fields:
- id (UUID, PK)
- audit_id (UUID, FK → security_audit)
- event_name (VARCHAR 255)
- event_description (TEXT)
- event_data (JSONB)
- system_state (JSONB)
- duration_ms (INTEGER)
- success_count, failure_count (INTEGER)
- created_at (TIMESTAMP)
```

#### **Analysis Cache** (`analysis_cache`)

Caches frequently accessed analysis results.

```sql
Fields:
- id (UUID, PK)
- cache_key (VARCHAR 255, UNIQUE)
- image_id (UUID, FK → images)
- analysis_id (UUID, FK → analysis_results)
- algorithm_fingerprint (TEXT)
- cached_result (JSONB)
- hit_count (INTEGER)
- last_accessed_at (TIMESTAMP)
- created_at, expires_at (TIMESTAMP)
- size_bytes (INTEGER)
```

#### **Performance Metrics** (`performance_metrics`)

Query performance and system metrics tracking.

```sql
Fields:
- id (UUID, PK)
- metric_name (VARCHAR 255)
- metric_type (VARCHAR 50)
- value (DECIMAL 15,4)
- unit (VARCHAR 50)
- labels (JSONB)
- context (JSONB)
- recorded_at (TIMESTAMP)
- sample_count (INTEGER)
- percentile_p50, percentile_p95, percentile_p99 (DECIMAL)
```

## Views

### `user_activity_summary`

Summary of user activity including image uploads and analyses.

### `queue_status_summary`

Processing queue statistics by status with timing metrics.

### `image_analysis_status`

Analysis status for all images with performance metrics.

### `recent_audit_events`

Most recent 1000 audit events with full details.

### `analysis_performance_summary` (Enhancement)

Analysis performance categorization (cached/fast/normal/slow).

### `audit_summary_by_user` (Enhancement)

User-level audit summary with event counts and durations.

### `duplicate_images_report` (Enhancement)

Report of duplicate images with detection metadata.

## Functions and Triggers

### Automatic Timestamp Management

- `update_updated_at_column()` - Automatically updates `updated_at` on record changes
- Applied to: `users`, `images`, `analysis_results`

### Data Validation

- `validate_image_status_transition()` - Enforces valid status transitions
- `validate_queue_status_transition()` - Enforces valid queue state transitions

### Audit Logging

- `log_audit_event()` - Automatically logs image creation events
- `cleanup_expired_cache()` - Removes expired cache entries
- `aggregate_performance_metrics()` - Aggregates metrics by time window
- `mark_duplicate_images()` - Marks images as duplicates

## Indexes Strategy

### Performance Optimization

**Foreign Key Indexes:**

- Every foreign key is indexed by default
- Enables efficient joins and cascading deletes

**Time-Based Queries:**

- `created_at`, `updated_at`, `timestamp` indexed DESC
- Composite indexes on user_id + created_at for user timelines

**Status/Filter Indexes:**

- `analysis_status`, `queue_status`, `is_active` indexed separately
- Composite indexes for common filter combinations: (status, priority), (status, created_at)

**JSONB Indexes:**

- GIN indexes on `metadata`, `image_dimensions`, `negative_space_data`, `anomalies`
- Enable efficient searching within JSON fields

**Covered Queries:**

- Composite indexes designed to cover common query patterns
- Example: (user_id, created_at) for "show latest images for user"

## Installation & Setup

### Prerequisites

- PostgreSQL 14 or higher
- Node.js 14+ (for migration runner)
- npm or yarn
- `knex` CLI (optional, for manual testing)

### Installation Steps

1. **Create Database**

```bash
createdb negative_space
```

2. **Install Dependencies**

```bash
npm install knex pg
```

3. **Configure Environment**

```bash
cat > .env << EOF
DB_HOST=localhost
DB_PORT=5432
DB_NAME=negative_space
DB_USER=app_user
DB_PASSWORD=your_secure_password
DB_CLIENT=pg
DB_SSL=false
EOF
```

4. **Run Migrations**

```bash
# Run all pending migrations
node database/migrationRunner.js up

# Or test first with dry-run
node database/migrationRunner.js up --dry-run
```

5. **Verify Installation**

```bash
node database/migrationRunner.js status
```

## Migration Management

### Commands

```bash
# Run all pending migrations
node migrationRunner.js up

# Rollback last migration
node migrationRunner.js down

# Show current migration status
node migrationRunner.js status

# Validate migration files
node migrationRunner.js validate

# Show migration history
node migrationRunner.js history [limit]

# Dry-run (test without applying)
node migrationRunner.js [command] --dry-run
```

### Migration Files

#### 001_initial_schema.js

- Creates all core tables (users, images, analysis_results, processing_queue, api_keys, security_audit)
- Establishes all primary indexes and relationships
- Sets up triggers for timestamp management
- Creates views for common queries
- Configures role permissions

**Status Transitions:**

```
Images:  pending → processing → completed
         pending → failed
         pending → archived
         processing → completed
         processing → failed

Queue:   queued → processing → completed
         queued → cancelled
         processing → completed
         processing → failed
         failed → retrying
         failed → cancelled
```

#### 002_add_audit_tables.js

- Creates enhancement tables (audit_log_details, analysis_cache, performance_metrics)
- Adds columns to existing tables (audit tracking, duplicate detection, performance stats)
- Creates advanced functions and views
- Implements cache management
- Adds performance monitoring capabilities

### Transaction Safety

All migrations run within database transactions. If any step fails:

1. Entire migration is rolled back
2. Database state remains unchanged
3. Migration marked as failed with error details
4. Can be retried after fixing the issue

### Migration History

Migration history is tracked in the `migrations_history` table:

- `name`: Migration identifier
- `status`: pending, running, success, failed, rolled_back
- `batch`: Batch number for coordinated migrations
- `duration_ms`: Execution time
- `error_message`: Error details if failed
- `started_at`, `completed_at`: Timestamps

## Backup & Recovery

### Backup Strategy

```bash
# Full backup
pg_dump negative_space > backup_$(date +%Y%m%d_%H%M%S).sql

# Compressed backup
pg_dump -Fc negative_space > backup_$(date +%Y%m%d_%H%M%S).dump

# Backup with custom format (faster restore)
pg_dump -Fc -j 4 negative_space > backup_parallel.dump
```

### Recovery

```bash
# From SQL backup
psql negative_space < backup_20250120_120000.sql

# From compressed backup
pg_restore -d negative_space backup_20250120_120000.dump

# Parallel restore (faster)
pg_restore -j 4 -d negative_space backup_20250120_120000.dump
```

## Performance Tuning

### Connection Pooling

```javascript
// In application code
const pool = {
  min: 2,
  max: 10,
  acquireTimeoutMillis: 30000,
  idleTimeoutMillis: 30000,
};
```

### Query Optimization

```sql
-- Analyze and get query plan
EXPLAIN ANALYZE
SELECT * FROM images WHERE user_id = ? AND created_at > ? ORDER BY created_at DESC;

-- Update statistics for better planning
ANALYZE images;
ANALYZE analysis_results;

-- Vacuum to reclaim space
VACUUM ANALYZE;
```

### Common Queries

```sql
-- User's latest images with analysis status
SELECT i.*, ar.confidence_score
FROM images i
LEFT JOIN analysis_results ar ON i.id = ar.image_id
WHERE i.user_id = $1
ORDER BY i.created_at DESC
LIMIT 50;

-- Pending analysis queue
SELECT * FROM processing_queue
WHERE status IN ('queued', 'retrying')
ORDER BY priority DESC, created_at ASC
LIMIT 10;

-- Analysis performance summary
SELECT
  a.algorithm_version,
  COUNT(*) as count,
  AVG(a.processing_time_ms) as avg_time,
  AVG(a.confidence_score) as avg_confidence
FROM analysis_results a
WHERE a.created_at > NOW() - INTERVAL '24 hours'
GROUP BY a.algorithm_version;
```

## Security Considerations

### Role-Based Access

```sql
-- Admin user (full access)
INSERT INTO users (email, password_hash, role)
VALUES ('admin@example.com', '$2b$12$...', 'admin');

-- Analyst user (can run analysis)
INSERT INTO users (email, password_hash, role)
VALUES ('analyst@example.com', '$2b$12$...', 'analyst');

-- Viewer user (read-only)
INSERT INTO users (email, password_hash, role)
VALUES ('viewer@example.com', '$2b$12$...', 'viewer');

-- API user (programmatic access)
INSERT INTO users (email, password_hash, role)
VALUES ('api@example.com', '$2b$12$...', 'api_user');
```

### Password Security

- Use bcrypt for hashing (`password_hash` field)
- Never store plaintext passwords
- Minimum 12-character passwords recommended
- Enforce password changes periodically

### API Keys

```sql
-- Create API key for application
INSERT INTO api_keys (user_id, key_hash, name, scopes)
VALUES (
  $1,
  'sha256_hash_of_key',
  'Production API Key',
  '["read:images", "read:results", "write:results"]'
);

-- Set expiration
UPDATE api_keys SET expires_at = NOW() + INTERVAL '1 year'
WHERE id = $1;

-- Rate limiting
UPDATE api_keys SET rate_limit_per_minute = 100
WHERE id = $1;
```

### Audit Trail

Complete audit trail in `security_audit` table:

- All user actions logged automatically
- IP addresses and user agents captured
- Before/after state of changes tracked
- Queries in `audit_summary_by_user` and `recent_audit_events` views

## Troubleshooting

### Migration Failures

```bash
# Check migration status
node migrationRunner.js status

# View history with errors
node migrationRunner.js history 20

# Validate migration files
node migrationRunner.js validate

# Rollback if needed
node migrationRunner.js down
```

### Connection Issues

```bash
# Test connection
psql -h localhost -U app_user -d negative_space -c "SELECT 1;"

# Check running connections
SELECT * FROM pg_stat_activity WHERE datname = 'negative_space';

# Terminate stale connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'negative_space' AND state = 'inactive';
```

### Performance Issues

```bash
# Check slow queries
SELECT * FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;

-- Rebuild index
REINDEX INDEX idx_images_created_at;

-- Vacuum and analyze
VACUUM ANALYZE images;
```

## Development Workflow

### Adding a New Migration

1. Create migration file in `database/migrations/` directory:

```javascript
// database/migrations/003_your_migration_name.js
module.exports = {
  async up(knex) {
    // Create tables, add columns, etc.
  },

  async down(knex) {
    // Reverse changes
  },
};
```

2. Test with dry-run:

```bash
node migrationRunner.js up --dry-run
```

3. Apply migration:

```bash
node migrationRunner.js up
```

4. Verify success:

```bash
node migrationRunner.js status
```

### Schema Changes

Always follow this pattern:

1. Make change in `schema.sql` (for reference)
2. Create migration file
3. Test in development
4. Review changes
5. Apply to staging
6. Deploy to production

## Support & Documentation

- PostgreSQL Documentation: https://www.postgresql.org/docs/
- Knex.js: http://knexjs.org/
- UUID Extension: https://www.postgresql.org/docs/current/uuid-ossp.html
- JSONB Guide: https://www.postgresql.org/docs/current/datatype-json.html

## License

This database schema is part of the Negative Space Imaging Project and follows the same license as the main project.

---

**Last Updated:** 2025-01-20
**Schema Version:** 1.0.0
**PostgreSQL Version:** 14+
