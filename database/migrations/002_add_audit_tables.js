/**
 * NEGATIVE SPACE IMAGING PROJECT - DATABASE MIGRATION
 *
 * Migration: 002_add_audit_tables
 * Purpose: Add enhanced audit trail and analytics tables
 * Version: 1.0.0
 * Date: 2025
 *
 * Tables Created:
 *   - audit_log_details (Enhanced audit with detailed tracking)
 *   - analysis_cache (Cache for frequently accessed results)
 *   - performance_metrics (Query performance and system metrics)
 *
 * Changes to Existing Tables:
 *   - users: Add audit_event_count and api_usage tracking
 *   - images: Add analysis_retry_count and failure_reason
 *   - analysis_results: Add cache_hit and cached_at fields
 *   - processing_queue: Add estimated_priority and complexity_score
 *   - api_keys: Add last_rate_limit_reset and request_count
 */

'use strict';

/**
 * Upgrade function - Executes migration
 * @param {Object} knex - Knex database instance
 * @returns {Promise<void>}
 */
async function up(knex) {
  console.log('🔄 Running migration 002_add_audit_tables (UP)...');

  try {
    // Create audit_log_details table
    await knex.schema.createTable('audit_log_details', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table
        .uuid('audit_id')
        .notNullable()
        .references('id')
        .inTable('security_audit')
        .onDelete('CASCADE')
        .index();
      table.string('event_name', 255).notNullable().index();
      table.text('event_description');
      table.jsonb('event_data').defaultTo('{}').index();
      table.jsonb('system_state').defaultTo('{}');
      table.integer('duration_ms');
      table.integer('success_count').defaultTo(0);
      table.integer('failure_count').defaultTo(0);
      table.dateTime('created_at').notNullable().defaultTo(knex.fn.now()).index();

      // Composite indexes
      table.index(['audit_id', 'created_at']);
      table.index(['event_name', 'created_at']);
    });
    console.log('✓ audit_log_details table created');

    // Create analysis_cache table
    await knex.schema.createTable('analysis_cache', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table.string('cache_key', 255).notNullable().unique().index();
      table.uuid('image_id').references('id').inTable('images').onDelete('CASCADE').index();
      table
        .uuid('analysis_id')
        .references('id')
        .inTable('analysis_results')
        .onDelete('CASCADE')
        .index();
      table.text('algorithm_fingerprint').notNullable();
      table.jsonb('cached_result').notNullable();
      table.integer('hit_count').defaultTo(0);
      table.dateTime('last_accessed_at').nullable();
      table.dateTime('created_at').notNullable().defaultTo(knex.fn.now()).index();
      table.dateTime('expires_at').nullable().index();
      table.integer('size_bytes').notNullable();

      // Composite indexes
      table.index(['image_id', 'created_at']);
      table.index(['expires_at', 'created_at']);
    });
    console.log('✓ analysis_cache table created');

    // Create performance_metrics table
    await knex.schema.createTable('performance_metrics', (table) => {
      table.uuid('id').primary().defaultTo(knex.raw('uuid_generate_v4()'));
      table.string('metric_name', 255).notNullable().index();
      table.string('metric_type', 50).notNullable(); // histogram, gauge, counter
      table.decimal('value', 15, 4).notNullable();
      table.string('unit', 50).notNullable();
      table.jsonb('labels').defaultTo('{}');
      table.jsonb('context').defaultTo('{}');
      table.dateTime('recorded_at').notNullable().defaultTo(knex.fn.now()).index();
      table.integer('sample_count').defaultTo(1);
      table.decimal('percentile_p50', 15, 4);
      table.decimal('percentile_p95', 15, 4);
      table.decimal('percentile_p99', 15, 4);

      // Composite indexes
      table.index(['metric_name', 'recorded_at']);
      table.index(['recorded_at', 'metric_type']);
    });
    console.log('✓ performance_metrics table created');

    // Add columns to users table
    await knex.schema.table('users', (table) => {
      table.integer('audit_event_count').defaultTo(0).after('metadata');
      table.integer('api_request_count').defaultTo(0).after('audit_event_count');
      table.integer('failed_login_attempts').defaultTo(0).after('api_request_count');
      table.dateTime('account_locked_until').nullable().after('failed_login_attempts');
      table.jsonb('security_settings').defaultTo('{}').after('account_locked_until');
    });
    console.log('✓ users table enhanced');

    // Add columns to images table
    await knex.schema.table('images', (table) => {
      table.integer('analysis_retry_count').defaultTo(0).after('updated_at');
      table.text('failure_reason').nullable().after('analysis_retry_count');
      table.dateTime('last_retry_at').nullable().after('failure_reason');
      table.jsonb('processing_history').defaultTo('[]').after('last_retry_at');
      table.boolean('is_duplicate').defaultTo(false).after('processing_history');
      table
        .uuid('duplicate_of')
        .nullable()
        .references('id')
        .inTable('images')
        .onDelete('SET NULL')
        .after('is_duplicate');
    });
    console.log('✓ images table enhanced');

    // Add columns to analysis_results table
    await knex.schema.table('analysis_results', (table) => {
      table.boolean('cache_hit').defaultTo(false).after('updated_at');
      table.dateTime('cached_at').nullable().after('cache_hit');
      table.integer('cache_ttl_seconds').defaultTo(86400).after('cached_at');
      table.jsonb('performance_stats').defaultTo('{}').after('cache_ttl_seconds');
      table.text('optimization_suggestions').nullable().after('performance_stats');
    });
    console.log('✓ analysis_results table enhanced');

    // Add columns to processing_queue table
    await knex.schema.table('processing_queue', (table) => {
      table.integer('estimated_priority').defaultTo(5).after('priority');
      table.decimal('complexity_score', 5, 4).defaultTo(0.5).after('estimated_priority');
      table.jsonb('processing_metadata').defaultTo('{}').after('complexity_score');
      table.text('processing_notes').nullable().after('processing_metadata');
    });
    console.log('✓ processing_queue table enhanced');

    // Add columns to api_keys table
    await knex.schema.table('api_keys', (table) => {
      table.dateTime('last_rate_limit_reset').nullable().after('rate_limit_per_minute');
      table.integer('request_count_current_window').defaultTo(0).after('last_rate_limit_reset');
      table.integer('total_requests').defaultTo(0).after('request_count_current_window');
      table.jsonb('usage_stats').defaultTo('{}').after('total_requests');
    });
    console.log('✓ api_keys table enhanced');

    // Create indexes on new columns
    await knex.raw('CREATE INDEX idx_audit_log_details_audit_id ON audit_log_details(audit_id)');
    await knex.raw(
      'CREATE INDEX idx_audit_log_details_event_name ON audit_log_details(event_name)',
    );
    await knex.raw('CREATE INDEX idx_analysis_cache_image_id ON analysis_cache(image_id)');
    await knex.raw('CREATE INDEX idx_analysis_cache_analysis_id ON analysis_cache(analysis_id)');
    await knex.raw('CREATE INDEX idx_analysis_cache_expires_at ON analysis_cache(expires_at)');
    await knex.raw(
      'CREATE INDEX idx_performance_metrics_metric_name ON performance_metrics(metric_name)',
    );
    await knex.raw(
      'CREATE INDEX idx_performance_metrics_recorded_at ON performance_metrics(recorded_at DESC)',
    );
    await knex.raw('CREATE INDEX idx_users_audit_event_count ON users(audit_event_count DESC)');
    await knex.raw('CREATE INDEX idx_images_is_duplicate ON images(is_duplicate)');
    await knex.raw('CREATE INDEX idx_images_duplicate_of ON images(duplicate_of)');
    await knex.raw('CREATE INDEX idx_analysis_results_cache_hit ON analysis_results(cache_hit)');
    console.log('✓ indexes created on enhanced tables');

    // Create function for cache management
    await knex.raw(`
      CREATE OR REPLACE FUNCTION cleanup_expired_cache()
      RETURNS void AS $$
      BEGIN
        DELETE FROM analysis_cache
        WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
      END;
      $$ LANGUAGE plpgsql;
    `);
    console.log('✓ cleanup_expired_cache function created');

    // Create function for performance metrics aggregation
    await knex.raw(`
      CREATE OR REPLACE FUNCTION aggregate_performance_metrics(
        p_metric_name VARCHAR,
        p_window_minutes INT DEFAULT 60
      )
      RETURNS TABLE(
        avg_value DECIMAL,
        min_value DECIMAL,
        max_value DECIMAL,
        sample_count BIGINT,
        window_start TIMESTAMP WITH TIME ZONE,
        window_end TIMESTAMP WITH TIME ZONE
      ) AS $$
      BEGIN
        RETURN QUERY
        SELECT
          AVG(value)::DECIMAL,
          MIN(value)::DECIMAL,
          MAX(value)::DECIMAL,
          COUNT(*)::BIGINT,
          date_trunc('minute', recorded_at)::TIMESTAMP WITH TIME ZONE,
          (date_trunc('minute', recorded_at) + (p_window_minutes || ' minutes')::INTERVAL)::TIMESTAMP WITH TIME ZONE
        FROM performance_metrics
        WHERE metric_name = p_metric_name
          AND recorded_at > CURRENT_TIMESTAMP - (p_window_minutes || ' minutes')::INTERVAL
        GROUP BY date_trunc('minute', recorded_at)
        ORDER BY date_trunc('minute', recorded_at) DESC;
      END;
      $$ LANGUAGE plpgsql;
    `);
    console.log('✓ aggregate_performance_metrics function created');

    // Create function for duplicate detection
    await knex.raw(`
      CREATE OR REPLACE FUNCTION mark_duplicate_images(
        p_original_id UUID,
        p_duplicate_id UUID
      )
      RETURNS void AS $$
      BEGIN
        UPDATE images
        SET is_duplicate = TRUE,
            duplicate_of = p_original_id,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_duplicate_id;

        INSERT INTO security_audit (action, resource_type, resource_id, status)
        VALUES ('create_image'::audit_action, 'image_duplicate', p_duplicate_id::TEXT, 'success'::audit_status);
      END;
      $$ LANGUAGE plpgsql;
    `);
    console.log('✓ mark_duplicate_images function created');

    // Create enhanced views
    await knex.raw(`
      CREATE OR REPLACE VIEW analysis_performance_summary AS
      SELECT
        ar.id,
        ar.image_id,
        ar.confidence_score,
        ar.processing_time_ms,
        ar.cache_hit,
        ar.cached_at,
        i.file_size,
        CASE
          WHEN ar.cache_hit THEN 'cached'
          WHEN ar.processing_time_ms < 1000 THEN 'fast'
          WHEN ar.processing_time_ms < 5000 THEN 'normal'
          ELSE 'slow'
        END as performance_category,
        ar.created_at
      FROM analysis_results ar
      JOIN images i ON ar.image_id = i.id
      ORDER BY ar.created_at DESC;
    `);
    console.log('✓ analysis_performance_summary view created');

    await knex.raw(`
      CREATE OR REPLACE VIEW audit_summary_by_user AS
      SELECT
        u.id,
        u.email,
        COUNT(DISTINCT ad.id) as total_events,
        COUNT(DISTINCT CASE WHEN ad.success_count > 0 THEN ad.id END) as successful_events,
        COUNT(DISTINCT CASE WHEN ad.failure_count > 0 THEN ad.id END) as failed_events,
        MAX(ad.created_at) as last_event,
        AVG(ad.duration_ms) as avg_event_duration_ms,
        u.audit_event_count
      FROM users u
      LEFT JOIN security_audit sa ON u.id = sa.user_id
      LEFT JOIN audit_log_details ad ON sa.id = ad.audit_id
      GROUP BY u.id, u.email, u.audit_event_count;
    `);
    console.log('✓ audit_summary_by_user view created');

    await knex.raw(`
      CREATE OR REPLACE VIEW duplicate_images_report AS
      SELECT
        original.id as original_id,
        original.file_name as original_name,
        original.file_size,
        COUNT(duplicates.id) as duplicate_count,
        STRING_AGG(DISTINCT duplicates.file_name, ', ') as duplicate_names,
        original.created_at as first_uploaded,
        MAX(duplicates.created_at) as last_duplicate_found
      FROM images original
      LEFT JOIN images duplicates ON duplicates.duplicate_of = original.id
      WHERE original.is_duplicate = FALSE
      GROUP BY original.id, original.file_name, original.file_size, original.created_at
      HAVING COUNT(duplicates.id) > 0
      ORDER BY duplicate_count DESC;
    `);
    console.log('✓ duplicate_images_report view created');

    // Grant permissions on new tables and functions
    await knex.raw(`
      GRANT SELECT, INSERT, UPDATE ON audit_log_details TO app_user;
      GRANT SELECT, INSERT, UPDATE ON analysis_cache TO app_user;
      GRANT SELECT, INSERT ON performance_metrics TO app_user;
      GRANT SELECT ON analysis_performance_summary TO app_user;
      GRANT SELECT ON audit_summary_by_user TO app_user;
      GRANT SELECT ON duplicate_images_report TO app_user;
    `);
    console.log('✓ permissions granted on new tables');

    console.log('✅ Migration 002_add_audit_tables (UP) completed successfully');
  } catch (error) {
    console.error('❌ Migration 002_add_audit_tables (UP) failed:', error.message);
    throw error;
  }
}

/**
 * Downgrade function - Rollback migration
 * @param {Object} knex - Knex database instance
 * @returns {Promise<void>}
 */
async function down(knex) {
  console.log('🔄 Running migration 002_add_audit_tables (DOWN)...');

  try {
    // Drop views
    await knex.raw('DROP VIEW IF EXISTS duplicate_images_report');
    await knex.raw('DROP VIEW IF EXISTS audit_summary_by_user');
    await knex.raw('DROP VIEW IF EXISTS analysis_performance_summary');
    console.log('✓ views dropped');

    // Drop functions
    await knex.raw('DROP FUNCTION IF EXISTS mark_duplicate_images(UUID, UUID)');
    await knex.raw('DROP FUNCTION IF EXISTS aggregate_performance_metrics(VARCHAR, INT)');
    await knex.raw('DROP FUNCTION IF EXISTS cleanup_expired_cache()');
    console.log('✓ functions dropped');

    // Drop indexes
    await knex.raw('DROP INDEX IF EXISTS idx_analysis_results_cache_hit');
    await knex.raw('DROP INDEX IF EXISTS idx_images_duplicate_of');
    await knex.raw('DROP INDEX IF EXISTS idx_images_is_duplicate');
    await knex.raw('DROP INDEX IF EXISTS idx_users_audit_event_count');
    await knex.raw('DROP INDEX IF EXISTS idx_performance_metrics_recorded_at');
    await knex.raw('DROP INDEX IF EXISTS idx_performance_metrics_metric_name');
    await knex.raw('DROP INDEX IF EXISTS idx_analysis_cache_expires_at');
    await knex.raw('DROP INDEX IF EXISTS idx_analysis_cache_analysis_id');
    await knex.raw('DROP INDEX IF EXISTS idx_analysis_cache_image_id');
    await knex.raw('DROP INDEX IF EXISTS idx_audit_log_details_event_name');
    await knex.raw('DROP INDEX IF EXISTS idx_audit_log_details_audit_id');
    console.log('✓ indexes dropped');

    // Drop tables
    await knex.schema.dropTableIfExists('performance_metrics');
    await knex.schema.dropTableIfExists('analysis_cache');
    await knex.schema.dropTableIfExists('audit_log_details');
    console.log('✓ tables dropped');

    // Remove columns from existing tables
    await knex.schema.table('api_keys', (table) => {
      table.dropColumn('usage_stats');
      table.dropColumn('total_requests');
      table.dropColumn('request_count_current_window');
      table.dropColumn('last_rate_limit_reset');
    });
    console.log('✓ api_keys columns removed');

    await knex.schema.table('processing_queue', (table) => {
      table.dropColumn('processing_notes');
      table.dropColumn('processing_metadata');
      table.dropColumn('complexity_score');
      table.dropColumn('estimated_priority');
    });
    console.log('✓ processing_queue columns removed');

    await knex.schema.table('analysis_results', (table) => {
      table.dropColumn('optimization_suggestions');
      table.dropColumn('performance_stats');
      table.dropColumn('cache_ttl_seconds');
      table.dropColumn('cached_at');
      table.dropColumn('cache_hit');
    });
    console.log('✓ analysis_results columns removed');

    await knex.schema.table('images', (table) => {
      table.dropColumn('duplicate_of');
      table.dropColumn('is_duplicate');
      table.dropColumn('processing_history');
      table.dropColumn('last_retry_at');
      table.dropColumn('failure_reason');
      table.dropColumn('analysis_retry_count');
    });
    console.log('✓ images columns removed');

    await knex.schema.table('users', (table) => {
      table.dropColumn('security_settings');
      table.dropColumn('account_locked_until');
      table.dropColumn('failed_login_attempts');
      table.dropColumn('api_request_count');
      table.dropColumn('audit_event_count');
    });
    console.log('✓ users columns removed');

    console.log('✅ Migration 002_add_audit_tables (DOWN) completed successfully');
  } catch (error) {
    console.error('❌ Migration 002_add_audit_tables (DOWN) failed:', error.message);
    throw error;
  }
}

module.exports = { up, down };
