/**
 * NEGATIVE SPACE IMAGING PROJECT - MIGRATION RUNNER
 *
 * Purpose:
 *   Manages database migrations with transaction safety, rollback capability,
 *   and comprehensive logging
 *
 * Features:
 *   - Version control for migrations
 *   - Transaction safety with automatic rollback
 *   - Migration history tracking
 *   - Forward and backward migration support
 *   - Dry-run capability for testing
 *   - Detailed logging and error reporting
 *
 * Usage:
 *   node migrationRunner.js up              # Run all pending migrations
 *   node migrationRunner.js down            # Rollback last migration
 *   node migrationRunner.js status          # Show migration status
 *   node migrationRunner.js --dry-run up    # Test migrations without applying
 */

'use strict';

const fs = require('fs');
const path = require('path');
const knex = require('knex');

/**
 * Migration Runner Configuration
 */
const config = {
  database: {
    client: process.env.DB_CLIENT || 'pg',
    connection: {
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'negative_space',
      user: process.env.DB_USER || 'app_user',
      password: process.env.DB_PASSWORD || 'change_me_in_production',
      ssl: process.env.DB_SSL === 'true' ? { rejectUnauthorized: false } : false,
    },
    migrations: {
      directory: path.join(__dirname, 'migrations'),
      tableName: 'migrations_history',
      loadExtensions: ['.js'],
    },
    pool: {
      min: 2,
      max: 10,
      acquireTimeoutMillis: 30000,
      idleTimeoutMillis: 30000,
    },
    // Don't use Knex's migration system - manage ourselves
    disableTransactions: false,
  },
  migrationTimeout: 300000, // 5 minutes
  dryRun: false,
};

/**
 * Migration History Tracking
 */
class MigrationTracker {
  constructor(kx) {
    this.knex = kx;
  }

  /**
   * Initialize migration tracking table
   */
  async initialize() {
    const tableName = config.migrations.tableName;
    const exists = await this.knex.schema.hasTable(tableName);

    if (!exists) {
      await this.knex.schema.createTable(tableName, (table) => {
        table.uuid('id').primary().defaultTo(this.knex.raw('uuid_generate_v4()'));
        table.integer('batch').notNullable();
        table.string('name', 255).notNullable().unique();
        table.string('status', 50).notNullable().defaultTo('pending'); // pending, running, success, failed, rolled_back
        table.text('error_message').nullable();
        table.integer('duration_ms').nullable();
        table.dateTime('started_at').nullable();
        table.dateTime('completed_at').nullable();
        table.dateTime('rolled_back_at').nullable();
        table.json('metadata').defaultTo('{}');
        table.dateTime('created_at').notNullable().defaultTo(this.knex.fn.now());
        table.dateTime('updated_at').notNullable().defaultTo(this.knex.fn.now());

        table.index('batch');
        table.index('status');
        table.index('created_at');
      });
      console.log('✓ Migration history table created');
    }
  }

  /**
   * Get all completed migrations
   */
  async getCompleted() {
    return this.knex(config.migrations.tableName)
      .where('status', '=', 'success')
      .orderBy('batch', 'asc')
      .orderBy('created_at', 'asc');
  }

  /**
   * Get migration status
   */
  async getStatus(name) {
    return this.knex(config.migrations.tableName).where('name', '=', name).first();
  }

  /**
   * Record migration start
   */
  async recordStart(name, batch) {
    return this.knex(config.migrations.tableName)
      .insert({
        name,
        batch,
        status: 'running',
        started_at: new Date(),
      })
      .onConflict('name')
      .merge();
  }

  /**
   * Record migration success
   */
  async recordSuccess(name, durationMs) {
    return this.knex(config.migrations.tableName).where('name', '=', name).update({
      status: 'success',
      duration_ms: durationMs,
      completed_at: new Date(),
      updated_at: new Date(),
    });
  }

  /**
   * Record migration failure
   */
  async recordFailure(name, errorMessage) {
    return this.knex(config.migrations.tableName).where('name', '=', name).update({
      status: 'failed',
      error_message: errorMessage,
      completed_at: new Date(),
      updated_at: new Date(),
    });
  }

  /**
   * Record migration rollback
   */
  async recordRollback(name) {
    return this.knex(config.migrations.tableName).where('name', '=', name).update({
      status: 'rolled_back',
      rolled_back_at: new Date(),
      updated_at: new Date(),
    });
  }

  /**
   * Get next batch number
   */
  async getNextBatch() {
    const result = await this.knex(config.migrations.tableName)
      .where('status', '=', 'success')
      .max('batch as max_batch')
      .first();

    return (result?.max_batch || 0) + 1;
  }

  /**
   * Get migration history
   */
  async getHistory(limit = 50) {
    return this.knex(config.migrations.tableName).orderBy('created_at', 'desc').limit(limit);
  }
}

/**
 * Migration Runner
 */
class MigrationRunner {
  constructor(kx, tracker) {
    this.knex = kx;
    this.tracker = tracker;
    this.migrationsDir = config.migrations.directory;
  }

  /**
   * Load migration files
   */
  async loadMigrations() {
    const files = fs
      .readdirSync(this.migrationsDir)
      .filter((f) => f.endsWith('.js'))
      .sort();

    return files.map((file) => ({
      name: path.basename(file, '.js'),
      path: path.join(this.migrationsDir, file),
      module: require(path.join(this.migrationsDir, file)),
    }));
  }

  /**
   * Get pending migrations
   */
  async getPending() {
    const allMigrations = await this.loadMigrations();
    const completed = await this.tracker.getCompleted();
    const completedNames = completed.map((m) => m.name);

    return allMigrations.filter((m) => !completedNames.includes(m.name));
  }

  /**
   * Run migrations up
   */
  async up() {
    const pending = await this.getPending();

    if (pending.length === 0) {
      console.log('✓ No pending migrations');
      return { status: 'success', message: 'No migrations to run', count: 0 };
    }

    const batch = await this.tracker.getNextBatch();
    const results = [];
    let successCount = 0;
    let failureCount = 0;

    console.log(`\n📊 Found ${pending.length} pending migration(s), batch: ${batch}\n`);

    for (const migration of pending) {
      const result = await this.runMigration(migration, 'up', batch);
      results.push(result);

      if (result.status === 'success') {
        successCount++;
      } else {
        failureCount++;
      }
    }

    console.log(`\n✅ Migration run complete: ${successCount} succeeded, ${failureCount} failed\n`);

    return {
      status: failureCount === 0 ? 'success' : 'partial',
      migrations: results,
      count: successCount,
      failed: failureCount,
    };
  }

  /**
   * Run single migration
   */
  async runMigration(migration, direction, batch) {
    const startTime = Date.now();
    const action = direction === 'up' ? '▶️ Running' : '⏮️ Rolling back';

    console.log(`${action}: ${migration.name}`);

    try {
      if (!config.dryRun) {
        await this.tracker.recordStart(migration.name, batch);
      }

      // Run migration function
      const migrationFn = migration.module[direction];
      if (!migrationFn) {
        throw new Error(`Migration function '${direction}' not found`);
      }

      if (config.dryRun) {
        console.log(`   [DRY-RUN] Would execute ${direction} migration for ${migration.name}`);
      } else {
        // Wrap in transaction
        await this.knex.transaction(async (trx) => {
          // Use transaction-aware knex instance
          await migrationFn(trx);
        });
      }

      const durationMs = Date.now() - startTime;

      if (!config.dryRun) {
        if (direction === 'up') {
          await this.tracker.recordSuccess(migration.name, durationMs);
        } else {
          await this.tracker.recordRollback(migration.name);
        }
      }

      console.log(`   ✓ ${migration.name} completed in ${durationMs}ms`);

      return {
        name: migration.name,
        status: 'success',
        duration: durationMs,
        direction,
      };
    } catch (error) {
      const durationMs = Date.now() - startTime;

      console.error(`   ✗ ${migration.name} failed: ${error.message}`);

      if (!config.dryRun) {
        await this.tracker.recordFailure(migration.name, error.message);
      }

      return {
        name: migration.name,
        status: 'failed',
        error: error.message,
        duration: durationMs,
        direction,
      };
    }
  }

  /**
   * Rollback last migration
   */
  async down() {
    const completed = await this.tracker.getCompleted();

    if (completed.length === 0) {
      console.log('✓ No migrations to rollback');
      return { status: 'success', message: 'No migrations to rollback', count: 0 };
    }

    // Get last migration from completed list
    const lastMigration = completed[completed.length - 1];
    const allMigrations = await this.loadMigrations();
    const migration = allMigrations.find((m) => m.name === lastMigration.name);

    if (!migration) {
      throw new Error(`Migration file not found for: ${lastMigration.name}`);
    }

    console.log(`\n⏮️ Rolling back migration: ${migration.name}\n`);

    const result = await this.runMigration(migration, 'down', null);

    console.log(`\n✅ Rollback complete\n`);

    return {
      status: result.status,
      migration: result,
      count: result.status === 'success' ? 1 : 0,
    };
  }

  /**
   * Show migration status
   */
  async status() {
    const completed = await this.tracker.getCompleted();
    const pending = await this.getPending();
    const history = await this.tracker.getHistory(10);

    console.log('\n📊 MIGRATION STATUS\n');
    console.log(`✓ Completed: ${completed.length}`);
    console.log(`⏳ Pending: ${pending.length}`);

    if (pending.length > 0) {
      console.log('\nPending Migrations:');
      pending.forEach((m) => console.log(`  - ${m.name}`));
    }

    console.log('\n📋 Recent Migrations:');
    history.forEach((m) => {
      const statusIcon =
        {
          success: '✓',
          failed: '✗',
          rolled_back: '↩️',
          running: '⏳',
          pending: '⏹️',
        }[m.status] || '?';

      console.log(
        `  ${statusIcon} ${m.name.padEnd(40)} [${m.status}] ` +
          `(${m.duration_ms || 'N/A'}ms) - ${new Date(m.created_at).toLocaleString()}`,
      );
    });

    console.log();
  }

  /**
   * Validate migration files
   */
  async validate() {
    const migrations = await this.loadMigrations();
    const issues = [];

    console.log('\n🔍 Validating migrations...\n');

    for (const migration of migrations) {
      if (!migration.module.up) {
        issues.push(`${migration.name}: Missing 'up' function`);
      }
      if (!migration.module.down) {
        issues.push(`${migration.name}: Missing 'down' function`);
      }
    }

    if (issues.length === 0) {
      console.log(`✓ All ${migrations.length} migrations are valid\n`);
      return { status: 'valid', count: migrations.length };
    } else {
      console.log('✗ Validation issues found:');
      issues.forEach((issue) => console.log(`  - ${issue}`));
      console.log();
      return { status: 'invalid', issues };
    }
  }
}

/**
 * CLI Command Handler
 */
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'status';
  const options = args.slice(1);

  // Check for dry-run flag
  if (options.includes('--dry-run')) {
    config.dryRun = true;
    console.log('🧪 DRY-RUN MODE: No changes will be applied\n');
  }

  try {
    // Initialize database connection
    const db = knex(config.database);
    const tracker = new MigrationTracker(db);
    const runner = new MigrationRunner(db, tracker);

    // Initialize migration tracking
    await tracker.initialize();

    // Handle commands
    switch (command) {
      case 'up':
        console.log('🚀 Running migrations...');
        await runner.up();
        break;

      case 'down':
        console.log('🔄 Rolling back migrations...');
        await runner.down();
        break;

      case 'status':
        await runner.status();
        break;

      case 'validate':
        await runner.validate();
        break;

      case 'history':
        const limit = parseInt(options[0]) || 50;
        const history = await tracker.getHistory(limit);
        console.log(`\n📋 Migration History (last ${limit}):\n`);
        history.forEach((m) => {
          console.log(`  ${m.name.padEnd(40)} [${m.status}] (${m.duration_ms || 'N/A'}ms)`);
        });
        console.log();
        break;

      default:
        console.log(`
Unknown command: ${command}

Usage:
  node migrationRunner.js up               # Run pending migrations
  node migrationRunner.js down             # Rollback last migration
  node migrationRunner.js status           # Show migration status
  node migrationRunner.js validate         # Validate migration files
  node migrationRunner.js history [limit]  # Show migration history
  node migrationRunner.js [cmd] --dry-run  # Test without applying changes

Examples:
  node migrationRunner.js up --dry-run     # Test pending migrations
  node migrationRunner.js history 20       # Show last 20 migrations
        `);
        process.exit(1);
    }

    await db.destroy();
    console.log('✅ Done\n');
  } catch (error) {
    console.error('❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

module.exports = {
  MigrationRunner,
  MigrationTracker,
  config,
};
