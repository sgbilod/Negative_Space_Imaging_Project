#!/usr/bin/env node

/**
 * DOCKER INITIALIZATION SCRIPT FOR NEGATIVE SPACE IMAGING PROJECT
 *
 * Purpose:
 *   Comprehensive initialization and health verification for all Docker services
 *   Validates database connectivity, Redis connection, Express server, and environment
 *
 * Usage:
 *   node docker-init.js [OPTIONS]
 *
 * Options:
 *   --verbose              Enable verbose output with detailed debug info
 *   --log-file FILE        Custom log file path
 *   --test-mode            Run in test mode without modifying services
 *   --timeout SECONDS      Set connection timeout in seconds (default: 30)
 *   --retry-attempts NUM   Number of retry attempts (default: 3)
 *   --help                 Display help message
 *
 * Exit Codes:
 *   0 - All services initialized successfully
 *   1 - One or more services failed initialization
 *   2 - Environment validation failed
 *   3 - Script execution error
 *
 * Author: DevOps Team
 * Date: 2025-10-17
 * Version: 1.0.0
 *
 */

'use strict';

// ============================================================================
// IMPORTS AND DEPENDENCIES
// ============================================================================

const fs = require('fs');
const path = require('path');
const os = require('os');
const net = require('net');
const { promisify } = require('util');
const { exec } = require('child_process');

const execAsync = promisify(exec);

// PostgreSQL client
let pg = null;
try {
  pg = require('pg');
} catch (e) {
  // PostgreSQL will be optional
}

// Redis client
let redis = null;
try {
  redis = require('redis');
} catch (e) {
  // Redis will be optional
}

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

const COLORS = {
  RESET: '\x1b[0m',
  GREEN: '\x1b[32m',
  RED: '\x1b[31m',
  YELLOW: '\x1b[33m',
  BLUE: '\x1b[34m',
  CYAN: '\x1b[36m',
  BOLD: '\x1b[1m',
};

const LOG_LEVELS = {
  ERROR: 'ERROR',
  WARN: 'WARN',
  INFO: 'INFO',
  OK: 'OK',
  DEBUG: 'DEBUG',
};

// Service configuration with defaults
const SERVICE_CONFIG = {
  postgres: {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432', 10),
    database: process.env.DB_NAME || 'negative_space',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    connectionTimeoutMillis: 5000,
    idleTimeoutMillis: 30000,
  },
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    password: process.env.REDIS_PASSWORD || undefined,
    retryStrategy: (options) => {
      if (options.total_retry_time > 1000 * 60 * 60) {
        return new Error('Retry time exhausted');
      }
      return Math.min(options.attempt * 100, 3000);
    },
  },
  express: {
    host: process.env.EXPRESS_HOST || 'localhost',
    port: parseInt(process.env.EXPRESS_PORT || '3000', 10),
    healthEndpoint: '/health',
  },
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const STATE = {
  verbose: false,
  testMode: false,
  logFile: null,
  timeout: 30000,
  retryAttempts: 3,
  exitCode: 0,
  results: {
    environment: { passed: false, errors: [] },
    postgres: { passed: false, errors: [] },
    redis: { passed: false, errors: [] },
    express: { passed: false, errors: [] },
    initialization: { passed: false, errors: [] },
  },
};

// ============================================================================
// LOGGING FUNCTIONS
// ============================================================================

/**
 * Log message with timestamp and level
 * @param {string} level - Log level
 * @param {string} message - Log message
 * @param {Error} [error] - Optional error object
 */
function log(level, message, error = null) {
  const timestamp = new Date().toISOString();
  const colorCode = getColorCode(level);
  const consoleMsg = `${colorCode}[${timestamp}] [${level}]${COLORS.RESET} ${message}`;
  const fileMsg = `[${timestamp}] [${level}] ${message}`;

  // Log to console based on verbosity
  if (STATE.verbose || level === LOG_LEVELS.ERROR || level === LOG_LEVELS.WARN) {
    console.log(consoleMsg);
  }

  // Log error details if provided
  if (error && STATE.verbose) {
    console.error(`${colorCode}[${timestamp}] [${level}] Error Details:${COLORS.RESET}`);
    console.error(error.stack || error.message);
  }

  // Log to file if configured
  if (STATE.logFile) {
    try {
      fs.appendFileSync(STATE.logFile, `${fileMsg}\n`);
      if (error) {
        fs.appendFileSync(STATE.logFile, `${error.stack || error.message}\n`);
      }
    } catch (writeError) {
      console.error('Failed to write to log file:', writeError.message);
    }
  }
}

/**
 * Get color code for log level
 * @param {string} level - Log level
 * @returns {string} Color code
 */
function getColorCode(level) {
  const colorMap = {
    [LOG_LEVELS.ERROR]: COLORS.RED,
    [LOG_LEVELS.WARN]: COLORS.YELLOW,
    [LOG_LEVELS.INFO]: COLORS.BLUE,
    [LOG_LEVELS.OK]: COLORS.GREEN,
    [LOG_LEVELS.DEBUG]: COLORS.CYAN,
  };
  return colorMap[level] || COLORS.RESET;
}

// ============================================================================
// COMMAND-LINE ARGUMENT PARSING
// ============================================================================

/**
 * Parse command-line arguments
 */
function parseArguments() {
  const args = process.argv.slice(2);

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case '--verbose':
        STATE.verbose = true;
        log(LOG_LEVELS.DEBUG, 'Verbose mode enabled');
        break;

      case '--test-mode':
        STATE.testMode = true;
        log(LOG_LEVELS.INFO, 'Running in test mode');
        break;

      case '--log-file':
        STATE.logFile = args[++i];
        if (!STATE.logFile) {
          throw new Error('--log-file requires a file path');
        }
        initializeLogFile();
        break;

      case '--timeout':
        STATE.timeout = parseInt(args[++i], 10) * 1000;
        if (isNaN(STATE.timeout) || STATE.timeout <= 0) {
          throw new Error('--timeout must be a positive number');
        }
        log(LOG_LEVELS.DEBUG, `Timeout set to ${STATE.timeout}ms`);
        break;

      case '--retry-attempts':
        STATE.retryAttempts = parseInt(args[++i], 10);
        if (isNaN(STATE.retryAttempts) || STATE.retryAttempts < 0) {
          throw new Error('--retry-attempts must be a non-negative number');
        }
        log(LOG_LEVELS.DEBUG, `Retry attempts set to ${STATE.retryAttempts}`);
        break;

      case '--help':
        showUsage();
        process.exit(0);
        break;

      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }
}

/**
 * Initialize log file
 */
function initializeLogFile() {
  try {
    const logDir = path.dirname(STATE.logFile);
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    // Write header
    const header = `
================================================================================
DOCKER INITIALIZATION LOG
================================================================================
Timestamp: ${new Date().toISOString()}
Hostname: ${os.hostname()}
Platform: ${os.platform()}
User: ${process.env.USER || process.env.USERNAME || 'unknown'}
Node Version: ${process.version}
================================================================================
`;

    fs.writeFileSync(STATE.logFile, header);
    log(LOG_LEVELS.INFO, `Log file initialized: ${STATE.logFile}`);
  } catch (error) {
    console.error('Failed to initialize log file:', error.message);
    STATE.logFile = null;
  }
}

/**
 * Display usage information
 */
function showUsage() {
  console.log(`
${COLORS.BOLD}Docker Initialization Script${COLORS.RESET}

${COLORS.BOLD}Usage:${COLORS.RESET}
  node docker-init.js [OPTIONS]

${COLORS.BOLD}Options:${COLORS.RESET}
  --verbose              Enable verbose output with detailed debug information
  --log-file FILE        Specify custom log file path
  --test-mode            Run in test mode without modifying services
  --timeout SECONDS      Set connection timeout (default: 30 seconds)
  --retry-attempts NUM   Number of retry attempts (default: 3)
  --help                 Display this help message

${COLORS.BOLD}Examples:${COLORS.RESET}
  # Standard initialization
  node docker-init.js

  # Verbose mode with logging
  node docker-init.js --verbose --log-file docker-init.log

  # Test mode with custom timeout
  node docker-init.js --test-mode --timeout 60

${COLORS.BOLD}Exit Codes:${COLORS.RESET}
  0 - All services initialized successfully
  1 - One or more services failed initialization
  2 - Environment validation failed
  3 - Script execution error
  `);
}

// ============================================================================
// ENVIRONMENT VALIDATION
// ============================================================================

/**
 * Validate environment variables
 * @returns {Promise<boolean>} True if all required variables are present
 */
async function validateEnvironment() {
  log(LOG_LEVELS.INFO, 'Validating environment variables...');

  const requiredEnvVars = ['NODE_ENV', 'DATABASE_URL', 'REDIS_URL', 'JWT_SECRET'];

  const warnings = [];
  const errors = [];

  requiredEnvVars.forEach((envVar) => {
    if (!process.env[envVar]) {
      errors.push(`Missing required environment variable: ${envVar}`);
    }
  });

  // Check for recommended variables
  const recommendedEnvVars = ['LOG_LEVEL', 'CORS_ALLOWED_ORIGINS', 'DB_POOL_SIZE'];

  recommendedEnvVars.forEach((envVar) => {
    if (!process.env[envVar]) {
      warnings.push(`Recommended environment variable not set: ${envVar}`);
    }
  });

  // Validate specific values
  if (
    process.env.NODE_ENV &&
    !['development', 'testing', 'production'].includes(process.env.NODE_ENV)
  ) {
    errors.push(
      `Invalid NODE_ENV: ${process.env.NODE_ENV}. Must be: development, testing, or production`,
    );
  }

  if (
    process.env.LOG_LEVEL &&
    !['debug', 'info', 'warn', 'error'].includes(process.env.LOG_LEVEL)
  ) {
    warnings.push(`Invalid LOG_LEVEL: ${process.env.LOG_LEVEL}`);
  }

  // Report findings
  if (warnings.length > 0) {
    warnings.forEach((warning) => {
      log(LOG_LEVELS.WARN, warning);
      STATE.results.environment.errors.push(warning);
    });
  }

  if (errors.length > 0) {
    errors.forEach((error) => {
      log(LOG_LEVELS.ERROR, error);
      STATE.results.environment.errors.push(error);
    });
    return false;
  }

  log(LOG_LEVELS.OK, 'Environment validation passed');
  STATE.results.environment.passed = true;
  return true;
}

// ============================================================================
// CONNECTION UTILITIES
// ============================================================================

/**
 * Check if a port is accessible
 * @param {string} host - Hostname
 * @param {number} port - Port number
 * @param {number} timeoutMs - Timeout in milliseconds
 * @returns {Promise<boolean>} True if port is accessible
 */
function checkPort(host, port, timeoutMs = 5000) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    const onError = () => resolve(false);

    socket.setTimeout(timeoutMs);
    socket.once('error', onError);
    socket.once('timeout', onError);

    socket.connect(port, host, () => {
      socket.destroy();
      resolve(true);
    });
  });
}

/**
 * Retry a function with exponential backoff
 * @param {Function} fn - Function to retry
 * @param {string} name - Function name for logging
 * @param {number} maxAttempts - Maximum retry attempts
 * @returns {Promise<*>} Result of function
 */
async function retryWithBackoff(fn, name, maxAttempts = STATE.retryAttempts) {
  let lastError;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      log(LOG_LEVELS.DEBUG, `${name} - Attempt ${attempt}/${maxAttempts}`);
      return await fn();
    } catch (error) {
      lastError = error;
      log(LOG_LEVELS.DEBUG, `${name} failed: ${error.message}`);

      if (attempt < maxAttempts) {
        const delayMs = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
        log(LOG_LEVELS.DEBUG, `Retrying ${name} in ${delayMs}ms...`);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }

  throw lastError;
}

// ============================================================================
// POSTGRESQL HEALTH CHECK
// ============================================================================

/**
 * Test PostgreSQL connection
 * @returns {Promise<boolean>} True if connection is successful
 */
async function checkPostgresHealth() {
  log(LOG_LEVELS.INFO, 'Testing PostgreSQL connectivity...');

  try {
    // First check if port is accessible
    const portAccessible = await checkPort(
      SERVICE_CONFIG.postgres.host,
      SERVICE_CONFIG.postgres.port,
      5000,
    );

    if (!portAccessible) {
      throw new Error(
        `PostgreSQL port ${SERVICE_CONFIG.postgres.port} is not accessible on ${SERVICE_CONFIG.postgres.host}`,
      );
    }

    log(LOG_LEVELS.OK, `PostgreSQL port ${SERVICE_CONFIG.postgres.port} is accessible`);

    // If postgres package is available, test actual connection
    if (pg && !STATE.testMode) {
      return await retryWithBackoff(async () => {
        const client = new pg.Client(SERVICE_CONFIG.postgres);
        try {
          await client.connect();
          const result = await client.query('SELECT NOW() as current_time');
          log(
            LOG_LEVELS.OK,
            `PostgreSQL query successful - Current time: ${result.rows[0].current_time}`,
          );
          STATE.results.postgres.passed = true;
          await client.end();
          return true;
        } catch (error) {
          throw error;
        }
      }, 'PostgreSQL Connection');
    }

    STATE.results.postgres.passed = true;
    return true;
  } catch (error) {
    const errorMsg = `PostgreSQL health check failed: ${error.message}`;
    log(LOG_LEVELS.ERROR, errorMsg, error);
    STATE.results.postgres.errors.push(errorMsg);
    STATE.exitCode = 1;
    return false;
  }
}

// ============================================================================
// REDIS HEALTH CHECK
// ============================================================================

/**
 * Test Redis connection
 * @returns {Promise<boolean>} True if connection is successful
 */
async function checkRedisHealth() {
  log(LOG_LEVELS.INFO, 'Testing Redis connectivity...');

  try {
    // First check if port is accessible
    const portAccessible = await checkPort(
      SERVICE_CONFIG.redis.host,
      SERVICE_CONFIG.redis.port,
      5000,
    );

    if (!portAccessible) {
      throw new Error(
        `Redis port ${SERVICE_CONFIG.redis.port} is not accessible on ${SERVICE_CONFIG.redis.host}`,
      );
    }

    log(LOG_LEVELS.OK, `Redis port ${SERVICE_CONFIG.redis.port} is accessible`);

    // If redis package is available, test actual connection
    if (redis && !STATE.testMode) {
      return await retryWithBackoff(async () => {
        const client = redis.createClient({
          host: SERVICE_CONFIG.redis.host,
          port: SERVICE_CONFIG.redis.port,
          password: SERVICE_CONFIG.redis.password,
          connectTimeout: 5000,
        });

        return new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            client.quit();
            reject(new Error('Redis connection timeout'));
          }, STATE.timeout);

          client.on('ready', () => {
            clearTimeout(timeout);
            log(LOG_LEVELS.OK, 'Redis connected successfully');

            client.ping((err, reply) => {
              client.quit();

              if (err) {
                reject(err);
              } else {
                log(LOG_LEVELS.OK, `Redis ping response: ${reply}`);
                STATE.results.redis.passed = true;
                resolve(true);
              }
            });
          });

          client.on('error', (err) => {
            clearTimeout(timeout);
            reject(err);
          });
        });
      }, 'Redis Connection');
    }

    STATE.results.redis.passed = true;
    return true;
  } catch (error) {
    const errorMsg = `Redis health check failed: ${error.message}`;
    log(LOG_LEVELS.ERROR, errorMsg, error);
    STATE.results.redis.errors.push(errorMsg);
    STATE.exitCode = 1;
    return false;
  }
}

// ============================================================================
// EXPRESS SERVER HEALTH CHECK
// ============================================================================

/**
 * Test Express server health endpoint
 * @returns {Promise<boolean>} True if server is healthy
 */
async function checkExpressHealth() {
  log(LOG_LEVELS.INFO, 'Testing Express server connectivity...');

  try {
    const portAccessible = await checkPort(
      SERVICE_CONFIG.express.host,
      SERVICE_CONFIG.express.port,
      5000,
    );

    if (!portAccessible) {
      throw new Error(
        `Express server port ${SERVICE_CONFIG.express.port} is not accessible on ${SERVICE_CONFIG.express.host}`,
      );
    }

    log(LOG_LEVELS.OK, `Express server port ${SERVICE_CONFIG.express.port} is accessible`);

    if (!STATE.testMode) {
      return await retryWithBackoff(async () => {
        const healthUrl = `http://${SERVICE_CONFIG.express.host}:${SERVICE_CONFIG.express.port}${SERVICE_CONFIG.express.healthEndpoint}`;

        const response = await new Promise((resolve, reject) => {
          const request = require('http').get(healthUrl, { timeout: 5000 }, (res) => {
            let data = '';
            res.on('data', (chunk) => {
              data += chunk;
            });
            res.on('end', () => {
              resolve({ status: res.statusCode, body: data });
            });
          });

          request.on('error', reject);
          request.on('timeout', () => {
            request.abort();
            reject(new Error('Health check request timeout'));
          });
        });

        if (response.status !== 200) {
          throw new Error(`Health check returned status ${response.status}`);
        }

        log(LOG_LEVELS.OK, 'Express server health check passed');
        STATE.results.express.passed = true;
        return true;
      }, 'Express Health Check');
    }

    STATE.results.express.passed = true;
    return true;
  } catch (error) {
    const errorMsg = `Express health check failed: ${error.message}`;
    log(LOG_LEVELS.ERROR, errorMsg, error);
    STATE.results.express.errors.push(errorMsg);
    STATE.exitCode = 1;
    return false;
  }
}

// ============================================================================
// INITIALIZATION PHASE
// ============================================================================

/**
 * Run initialization tasks
 * @returns {Promise<boolean>} True if initialization successful
 */
async function runInitialization() {
  log(LOG_LEVELS.INFO, 'Running initialization tasks...');

  try {
    if (STATE.testMode) {
      log(LOG_LEVELS.INFO, 'Running in test mode - skipping actual modifications');
    }

    // Initialize cache connections (if needed)
    log(LOG_LEVELS.DEBUG, 'Initialization tasks completed successfully');
    STATE.results.initialization.passed = true;
    return true;
  } catch (error) {
    const errorMsg = `Initialization failed: ${error.message}`;
    log(LOG_LEVELS.ERROR, errorMsg, error);
    STATE.results.initialization.errors.push(errorMsg);
    STATE.exitCode = 1;
    return false;
  }
}

// ============================================================================
// REPORTING AND OUTPUT
// ============================================================================

/**
 * Generate and output health report
 */
function generateReport() {
  const reportTimestamp = new Date().toISOString();
  const reportData = {
    timestamp: reportTimestamp,
    hostname: os.hostname(),
    platform: os.platform(),
    nodeVersion: process.version,
    environment: process.env.NODE_ENV || 'unknown',
    exitCode: STATE.exitCode,
    results: STATE.results,
    summary: {
      total: Object.keys(STATE.results).length,
      passed: Object.values(STATE.results).filter((r) => r.passed).length,
      failed: Object.values(STATE.results).filter((r) => !r.passed && r.errors.length > 0).length,
    },
  };

  console.log('\n');
  console.log(`${COLORS.BOLD}${'='.repeat(72)}`);
  console.log('DOCKER INITIALIZATION REPORT');
  console.log(`${'='.repeat(72)}${COLORS.RESET}`);
  console.log('');

  // Summary section
  console.log(`${COLORS.BOLD}SUMMARY:${COLORS.RESET}`);
  console.log(`  Timestamp: ${reportTimestamp}`);
  console.log(`  Hostname: ${os.hostname()}`);
  console.log(`  Platform: ${os.platform()}`);
  console.log(
    `  Overall Status: ${STATE.exitCode === 0 ? `${COLORS.GREEN}HEALTHY${COLORS.RESET}` : `${COLORS.RED}DEGRADED${COLORS.RESET}`}`,
  );
  console.log('');

  // Service status
  console.log(`${COLORS.BOLD}SERVICE STATUS:${COLORS.RESET}`);
  Object.entries(STATE.results).forEach(([service, result]) => {
    const status = result.passed
      ? `${COLORS.GREEN}✓${COLORS.RESET}`
      : `${COLORS.RED}✗${COLORS.RESET}`;
    console.log(`  ${status} ${service.padEnd(20)} ${result.passed ? 'PASSED' : 'FAILED'}`);

    if (result.errors.length > 0) {
      result.errors.forEach((error) => {
        console.log(`     ${COLORS.RED}→${COLORS.RESET} ${error}`);
      });
    }
  });

  console.log('');
  console.log(`${COLORS.BOLD}RESULTS:${COLORS.RESET}`);
  console.log(`  Total Checks: ${reportData.summary.total}`);
  console.log(`  Passed: ${reportData.summary.passed}`);
  console.log(`  Failed: ${reportData.summary.failed}`);

  if (STATE.logFile) {
    console.log('');
    console.log(`${COLORS.BOLD}LOG FILE:${COLORS.RESET}`);
    console.log(`  ${STATE.logFile}`);
  }

  console.log('');
  console.log(`${COLORS.BOLD}${'='.repeat(72)}${COLORS.RESET}`);

  // Save JSON report
  try {
    const reportFile = path.join('logs', `docker-init-${Date.now()}.json`);
    fs.mkdirSync('logs', { recursive: true });
    fs.writeFileSync(reportFile, JSON.stringify(reportData, null, 2));
    log(LOG_LEVELS.INFO, `Report saved to: ${reportFile}`);
  } catch (error) {
    log(LOG_LEVELS.WARN, `Failed to save report file: ${error.message}`);
  }
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

/**
 * Main entry point
 */
async function main() {
  try {
    log(LOG_LEVELS.INFO, 'Starting Docker initialization script...');

    // Parse arguments
    parseArguments();

    // Validate environment
    const envValid = await validateEnvironment();
    if (!envValid && !STATE.testMode) {
      log(LOG_LEVELS.ERROR, 'Environment validation failed');
      STATE.exitCode = 2;
    }

    // Run health checks
    log(LOG_LEVELS.INFO, 'Phase 1: Running health checks...');
    await checkPostgresHealth();
    await checkRedisHealth();
    await checkExpressHealth();

    // Run initialization
    log(LOG_LEVELS.INFO, 'Phase 2: Running initialization...');
    await runInitialization();

    // Generate report
    generateReport();

    // Exit with appropriate code
    process.exit(STATE.exitCode);
  } catch (error) {
    log(LOG_LEVELS.ERROR, `Unexpected error: ${error.message}`, error);
    console.error(error);
    process.exit(3);
  }
}

// Execute main function
if (require.main === module) {
  main();
}

// Export for testing
module.exports = {
  checkPort,
  validateEnvironment,
  checkPostgresHealth,
  checkRedisHealth,
  checkExpressHealth,
  retryWithBackoff,
};
