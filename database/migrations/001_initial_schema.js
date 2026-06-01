/**
 * NEGATIVE SPACE IMAGING PROJECT - DATABASE MIGRATION
 *
 * Migration: 001_initial_schema
 * Purpose: Create initial database schema with all core tables
 * Version: 1.0.0
 * Date: 2025
 *
 * Tables Created:
 *   - users
 *   - images
 *   - analysis_results
 *   - processing_queue
 *   - api_keys
 *   - security_audit
 */

'use strict';

/**
 * Upgrade function - Executes migration
 * @param {Object} knex - Knex database instance
 * @returns {Promise<void>}
 */
async function up(knex) {
  console.log('🔄 Running migration 001_initial_schema (UP)...');

  try {
    // Enable required PostgreSQL extensions
    await knex.raw('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"');
    console.log('✓ uuid-ossp extension enabled');

    await knex.raw('CREATE EXTENSION IF NOT EXISTS "pgcrypto"');
    console.log('✓ pgcrypto extension enabled');

    await knex.raw('CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"');
    console.log('✓ pg_stat_statements extension enabled');

    // Create enum types
    await knex.raw(`
      CREATE TYPE user_role AS ENUM (
        'admin', 'analyst', 'viewer', 'api_user'
      )
    `);
    console.log('✓ user_role enum created');

    await knex.raw(`
      CREATE TYPE analysis_status AS ENUM (
        'pending', 'processing', 'completed', 'failed', 'archived'
      )
    `);
    console.log('✓ analysis_status enum created');

    await knex.raw(`
      CREATE TYPE queue_status AS ENUM (
        'queued', 'processing', 'completed', 'failed', 'retrying', 'cancelled'
      )
    `);
    console.log('✓ queue_status enum created');

    await knex.raw(`
      CREATE TYPE audit_action AS ENUM (
        'login', 'logout', 'create_image', 'delete_image', 'run_analysis',
        'view_results', 'export_data', 'api_call', 'permission_change', 'config_change'
      )
    `);
    console.log('✓ audit_action enum created');

    await knex.raw(`
      CREATE TYPE audit_status AS ENUM (
        'success', 'failure', 'partial'
      )
    `);
    console.log('✓ audit_status enum created');

    // Create users table
    await knex.schema.createTable('users', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table.string('email', 255).notNullable().unique().index();
      table.string('password_hash', 255).notNullable();
      table.string('full_name', 255);
      table
        .enum('role', ['admin', 'analyst', 'viewer', 'api_user'])
        .notNullable()
        .defaultTo('viewer')
        .index();
      table.boolean('is_active').notNullable().defaultTo(true).index();
      table.boolean('is_email_verified').notNullable().defaultTo(false);
      table.dateTime('last_login_at').nullable().index();
      table.dateTime('created_at').notNullable().defaultTo(knex.fn.now()).index();
      table.dateTime('updated_at').notNullable().defaultTo(knex.fn.now());
      table.jsonb('metadata').defaultTo('{}');
    });
    console.log('✓ users table created');

    // Create images table
    await knex.schema.createTable('images', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table
        .uuid('user_id')
        .notNullable()
        .index()
        .references('id')
        .inTable('users')
        .onDelete('CASCADE');
      table.string('file_name', 512).notNullable();
      table.string('file_path', 1024).notNullable();
      table.bigInteger('file_size').notNullable();
      table.string('mime_type', 100).notNullable();
      table.string('original_filename', 512);
      table.string('file_hash', 64).notNullable().unique().index();
      table.jsonb('image_dimensions').defaultTo('{"width": 0, "height": 0, "channels": 0}');
      table
        .enum('analysis_status', ['pending', 'processing', 'completed', 'failed', 'archived'])
        .notNullable()
        .defaultTo('pending')
        .index();
      table.jsonb('metadata').defaultTo('{}').index();
      table.dateTime('uploaded_at').notNullable().defaultTo(knex.fn.now()).index();
      table.dateTime('processed_at').nullable();
      table.dateTime('created_at').notNullable().defaultTo(knex.fn.now()).index();
      table.dateTime('updated_at').notNullable().defaultTo(knex.fn.now());

      // Composite indexes
      table.index(['user_id', 'created_at']);
      table.index(['analysis_status', 'created_at']);
    });
    console.log('✓ images table created');

    // Create analysis_results table
    await knex.schema.createTable('analysis_results', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table
        .uuid('image_id')
        .notNullable()
        .unique()
        .index()
        .references('id')
        .inTable('images')
        .onDelete('CASCADE');
      table
        .uuid('user_id')
        .notNullable()
        .index()
        .references('id')
        .inTable('users')
        .onDelete('CASCADE');
      table.string('algorithm_version', 50).notNullable().index();
      table.jsonb('algorithm_parameters').defaultTo('{}');
      table.jsonb('negative_space_data').notNullable().index();
      table.jsonb('anomalies').defaultTo('[]').index();
      table.decimal('confidence_score', 5, 4).notNullable();
      table.integer('processing_time_ms').notNullable();
      table.jsonb('quality_metrics').defaultTo('{}').index();
      table.string('visualization_path', 1024);
      table.text('notes');
      table.dateTime('created_at').notNullable().defaultTo(knex.fn.now()).index();
      table.dateTime('updated_at').notNullable().defaultTo(knex.fn.now());

      // Composite index
      table.index(['user_id', 'created_at']);
    });
    console.log('✓ analysis_results table created');

    // Create processing_queue table
    await knex.schema.createTable('processing_queue', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table
        .uuid('image_id')
        .notNullable()
        .index()
        .references('id')
        .inTable('images')
        .onDelete('CASCADE');
      table
        .uuid('user_id')
        .notNullable()
        .index()
        .references('id')
        .inTable('users')
        .onDelete('CASCADE');
      table
        .enum('status', ['queued', 'processing', 'completed', 'failed', 'retrying', 'cancelled'])
        .notNullable()
        .defaultTo('queued')
        .index();
      table.integer('priority').notNullable().defaultTo(5);
      table.integer('retry_count').notNullable().defaultTo(0);
      table.integer('max_retries').notNullable().defaultTo(3);
      table.text('error_message');
      table.dateTime('last_error_at').nullable();
      table.dateTime('created_at').notNullable().defaultTo(knex.fn.now()).index();
      table.dateTime('started_at').nullable();
      table.dateTime('completed_at').nullable();
      table.dateTime('estimated_completion_at').nullable();
      table.string('worker_id', 255);

      // Composite indexes
      table.index(['status', 'priority']);
      table.index(['created_at', 'status']);
    });
    console.log('✓ processing_queue table created');

    // Create api_keys table
    await knex.schema.createTable('api_keys', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table
        .uuid('user_id')
        .notNullable()
        .index()
        .references('id')
        .inTable('users')
        .onDelete('CASCADE');
      table.string('key_hash', 255).notNullable().unique().index();
      table.string('name', 255).notNullable();
      table.text('description');
      table.boolean('is_active').notNullable().defaultTo(true).index();
      table.dateTime('last_used_at').nullable();
      table.dateTime('created_at').notNullable().defaultTo(knex.fn.now()).index();
      table.dateTime('expires_at').nullable().index();
      table.jsonb('scopes').defaultTo('["read:images", "read:results"]');
      table.jsonb('ip_whitelist').defaultTo('null');
      table.integer('rate_limit_per_minute').defaultTo(100);
    });
    console.log('✓ api_keys table created');

    // Create security_audit table
    await knex.schema.createTable('security_audit', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table
        .uuid('user_id')
        .nullable()
        .index()
        .references('id')
        .inTable('users')
        .onDelete('SET NULL');
      table
        .enum('action', [
          'login',
          'logout',
          'create_image',
          'delete_image',
          'run_analysis',
          'view_results',
          'export_data',
          'api_call',
          'permission_change',
          'config_change',
        ])
        .notNullable()
        .index();
      table.string('resource_type', 100);
      table.string('resource_id', 255);
      table
        .enum('status', ['success', 'failure', 'partial'])
        .notNullable()
        .defaultTo('success')
        .index();
      table.specificType('ip_address', 'INET');
      table.text('user_agent');
      table.text('error_message');
      table.jsonb('changes_before');
      table.jsonb('changes_after');
      table.dateTime('timestamp').notNullable().defaultTo(knex.fn.now()).index();

      // Composite indexes
      table.index(['user_id', 'timestamp']);
      table.index(['resource_type', 'resource_id']);
    });
    console.log('✓ security_audit table created');

    // Create update_updated_at_column function
    await knex.raw(`
      CREATE OR REPLACE FUNCTION update_updated_at_column()
      RETURNS TRIGGER AS $$
      BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);
    console.log('✓ update_updated_at_column function created');

    // Create triggers for updated_at
    await knex.raw(`
      CREATE TRIGGER users_update_updated_at BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    `);
    await knex.raw(`
      CREATE TRIGGER images_update_updated_at BEFORE UPDATE ON images
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    `);
    await knex.raw(`
      CREATE TRIGGER analysis_results_update_updated_at BEFORE UPDATE ON analysis_results
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    `);
    console.log('✓ updated_at triggers created');

    // Create validation functions
    await knex.raw(`
      CREATE OR REPLACE FUNCTION validate_image_status_transition()
      RETURNS TRIGGER AS $$
      BEGIN
        IF NEW.analysis_status = OLD.analysis_status THEN
          RETURN NEW;
        END IF;

        IF OLD.analysis_status = 'pending' AND NEW.analysis_status NOT IN ('processing', 'failed') THEN
          RAISE EXCEPTION 'Invalid status transition from pending to %', NEW.analysis_status;
        END IF;

        IF OLD.analysis_status = 'processing' AND NEW.analysis_status NOT IN ('completed', 'failed') THEN
          RAISE EXCEPTION 'Invalid status transition from processing to %', NEW.analysis_status;
        END IF;

        RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);
    console.log('✓ validate_image_status_transition function created');

    await knex.raw(`
      CREATE TRIGGER validate_image_status BEFORE UPDATE ON images
        FOR EACH ROW EXECUTE FUNCTION validate_image_status_transition();
    `);
    console.log('✓ validate_image_status trigger created');

    // Create audit logging function
    await knex.raw(`
      CREATE OR REPLACE FUNCTION log_audit_event()
      RETURNS TRIGGER AS $$
      BEGIN
        INSERT INTO security_audit (
          action, resource_type, resource_id, status, ip_address
        ) VALUES (
          'create_image'::audit_action, 'image', NEW.id::TEXT, 'success'::audit_status, NULL
        );
        RETURN NEW;
      END;
      $$ LANGUAGE plpgsql;
    `);
    await knex.raw(`
      CREATE TRIGGER audit_image_creation AFTER INSERT ON images
        FOR EACH ROW EXECUTE FUNCTION log_audit_event();
    `);
    console.log('✓ audit logging functions and triggers created');

    // Create queue status validation
    await knex.raw(`
      CREATE OR REPLACE FUNCTION validate_queue_status_transition()
      RETURNS TRIGGER AS $$
      BEGIN
        IF NEW.status = OLD.status THEN
          RETURN NEW;
        END IF;

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
    `);
    await knex.raw(`
      CREATE TRIGGER validate_queue_status BEFORE UPDATE ON processing_queue
        FOR EACH ROW EXECUTE FUNCTION validate_queue_status_transition();
    `);
    console.log('✓ queue status validation created');

    // Create views
    await knex.raw(`
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
    `);
    console.log('✓ user_activity_summary view created');

    await knex.raw(`
      CREATE OR REPLACE VIEW queue_status_summary AS
      SELECT
        status,
        COUNT(*) as count,
        AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) as avg_queue_time_seconds,
        MIN(created_at) as oldest_item,
        MAX(created_at) as newest_item
      FROM processing_queue
      GROUP BY status;
    `);
    console.log('✓ queue_status_summary view created');

    await knex.raw(`
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
        CASE WHEN i.analysis_status = 'completed' THEN TRUE ELSE FALSE END as is_analyzed
      FROM images i
      LEFT JOIN analysis_results ar ON i.id = ar.image_id
      ORDER BY i.created_at DESC;
    `);
    console.log('✓ image_analysis_status view created');

    await knex.raw(`
      CREATE OR REPLACE VIEW recent_audit_events AS
      SELECT
        id, user_id, action, resource_type, resource_id, status, ip_address, timestamp, error_message
      FROM security_audit
      ORDER BY timestamp DESC
      LIMIT 1000;
    `);
    console.log('✓ recent_audit_events view created');

    // Create app_user role and grant permissions
    await knex.raw(`
      DO $$
      BEGIN
        CREATE ROLE app_user WITH LOGIN PASSWORD 'change_me_in_production';
      EXCEPTION WHEN duplicate_object THEN
        NULL;
      END
      $$;
    `);
    console.log('✓ app_user role created');

    await knex.raw(`
      GRANT SELECT, INSERT, UPDATE ON users TO app_user;
      GRANT SELECT, INSERT, UPDATE ON images TO app_user;
      GRANT SELECT, INSERT, UPDATE ON analysis_results TO app_user;
      GRANT SELECT, INSERT, UPDATE ON processing_queue TO app_user;
      GRANT SELECT, INSERT, UPDATE ON api_keys TO app_user;
      GRANT INSERT ON security_audit TO app_user;
      GRANT SELECT ON security_audit TO app_user;
      GRANT SELECT ON user_activity_summary TO app_user;
      GRANT SELECT ON queue_status_summary TO app_user;
      GRANT SELECT ON image_analysis_status TO app_user;
      GRANT SELECT ON recent_audit_events TO app_user;
      GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
    `);
    console.log('✓ permissions granted to app_user');

    // Analyze tables
    await knex.raw('ANALYZE users');
    await knex.raw('ANALYZE images');
    await knex.raw('ANALYZE analysis_results');
    await knex.raw('ANALYZE processing_queue');
    await knex.raw('ANALYZE api_keys');
    await knex.raw('ANALYZE security_audit');
    console.log('✓ database statistics analyzed');

    console.log('✅ Migration 001_initial_schema (UP) completed successfully');
  } catch (error) {
    console.error('❌ Migration 001_initial_schema (UP) failed:', error.message);
    throw error;
  }
}

/**
 * Downgrade function - Rollback migration
 * @param {Object} knex - Knex database instance
 * @returns {Promise<void>}
 */
async function down(knex) {
  console.log('🔄 Running migration 001_initial_schema (DOWN)...');

  try {
    // Drop views
    await knex.raw('DROP VIEW IF EXISTS recent_audit_events');
    await knex.raw('DROP VIEW IF EXISTS image_analysis_status');
    await knex.raw('DROP VIEW IF EXISTS queue_status_summary');
    await knex.raw('DROP VIEW IF EXISTS user_activity_summary');
    console.log('✓ views dropped');

    // Drop triggers
    await knex.raw('DROP TRIGGER IF EXISTS validate_queue_status ON processing_queue');
    await knex.raw('DROP TRIGGER IF EXISTS audit_image_creation ON images');
    await knex.raw('DROP TRIGGER IF EXISTS validate_image_status ON images');
    await knex.raw('DROP TRIGGER IF EXISTS analysis_results_update_updated_at ON analysis_results');
    await knex.raw('DROP TRIGGER IF EXISTS images_update_updated_at ON images');
    await knex.raw('DROP TRIGGER IF EXISTS users_update_updated_at ON users');
    console.log('✓ triggers dropped');

    // Drop functions
    await knex.raw('DROP FUNCTION IF EXISTS validate_queue_status_transition()');
    await knex.raw('DROP FUNCTION IF EXISTS log_audit_event()');
    await knex.raw('DROP FUNCTION IF EXISTS validate_image_status_transition()');
    await knex.raw('DROP FUNCTION IF EXISTS update_updated_at_column()');
    console.log('✓ functions dropped');

    // Drop tables
    await knex.schema.dropTableIfExists('security_audit');
    await knex.schema.dropTableIfExists('api_keys');
    await knex.schema.dropTableIfExists('processing_queue');
    await knex.schema.dropTableIfExists('analysis_results');
    await knex.schema.dropTableIfExists('images');
    await knex.schema.dropTableIfExists('users');
    console.log('✓ tables dropped');

    // Drop enum types
    await knex.raw('DROP TYPE IF EXISTS audit_status');
    await knex.raw('DROP TYPE IF EXISTS audit_action');
    await knex.raw('DROP TYPE IF EXISTS queue_status');
    await knex.raw('DROP TYPE IF EXISTS analysis_status');
    await knex.raw('DROP TYPE IF EXISTS user_role');
    console.log('✓ enum types dropped');

    // Drop role (optional - commented to avoid permission errors)
    // await knex.raw('DROP ROLE IF EXISTS app_user');

    console.log('✅ Migration 001_initial_schema (DOWN) completed successfully');
  } catch (error) {
    console.error('❌ Migration 001_initial_schema (DOWN) failed:', error.message);
    throw error;
  }
}

module.exports = { up, down };
