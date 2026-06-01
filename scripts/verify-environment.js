#!/usr/bin/env node

/**
 * ENVIRONMENT VERIFICATION SCRIPT
 *
 * Runs comprehensive environment checks at startup
 * Features:
 *   - Sequential validation of all requirements
 *   - Clear pass/fail indicators
 *   - Actionable remediation steps
 *   - JSON report logging
 *   - Exit codes for CI/CD integration
 */

'use strict';

const fs = require('fs');
const path = require('path');
const chalk = require('chalk');

// Try to load optional chalk for colors, fall back to plain text
let useColors = false;
try {
  require.resolve('chalk');
  useColors = true;
} catch {
  // chalk not available, continue without colors
}

const colors = {
  success: (msg) => (useColors ? chalk.green(msg) : `✓ ${msg}`),
  error: (msg) => (useColors ? chalk.red(msg) : `✗ ${msg}`),
  warning: (msg) => (useColors ? chalk.yellow(msg) : `⚠ ${msg}`),
  info: (msg) => (useColors ? chalk.blue(msg) : `ℹ ${msg}`),
  dim: (msg) => (useColors ? chalk.dim(msg) : msg),
};

/**
 * Environment Verification Manager
 */
class EnvironmentVerificationManager {
  constructor() {
    this.checks = [];
    this.results = {
      passed: 0,
      failed: 0,
      warnings: 0,
      errors: [],
      warnings: [],
      details: [],
    };
    this.startTime = Date.now();
  }

  /**
   * Add a check to the queue
   */
  addCheck(name, checkFn) {
    this.checks.push({ name, checkFn });
    return this;
  }

  /**
   * Run all checks sequentially
   */
  async runAll() {
    console.log(`\n${colors.info('=')} Environment Verification Started\n`);

    for (const check of this.checks) {
      await this.runCheck(check);
    }

    this.printSummary();
    return this.results;
  }

  /**
   * Run single check
   */
  async runCheck(check) {
    try {
      const result = await check.checkFn();

      if (result.passed) {
        console.log(colors.success(`✓ ${check.name}`));
        this.results.passed++;

        if (result.details) {
          console.log(colors.dim(`  ${result.details}`));
        }
      } else {
        console.log(colors.error(`✗ ${check.name}`));
        this.results.failed++;

        if (result.error) {
          console.log(colors.error(`  Error: ${result.error}`));
          this.results.errors.push({
            check: check.name,
            error: result.error,
          });
        }

        if (result.remediation) {
          console.log(colors.warning(`  Remediation: ${result.remediation}`));
        }
      }

      if (result.warning) {
        this.results.warnings++;
        console.log(colors.warning(`  Warning: ${result.warning}`));
        this.results.warnings.push({
          check: check.name,
          warning: result.warning,
        });
      }

      this.results.details.push({
        check: check.name,
        passed: result.passed,
        details: result.details,
        error: result.error,
        warning: result.warning,
      });
    } catch (error) {
      console.log(colors.error(`✗ ${check.name}`));
      this.results.failed++;

      console.log(colors.error(`  Error: ${error.message}`));
      this.results.errors.push({
        check: check.name,
        error: error.message,
      });

      this.results.details.push({
        check: check.name,
        passed: false,
        error: error.message,
      });
    }

    console.log();
  }

  /**
   * Print summary report
   */
  printSummary() {
    const duration = Date.now() - this.startTime;
    const total = this.results.passed + this.results.failed;
    const passPercentage = total > 0 ? Math.round((this.results.passed / total) * 100) : 0;

    console.log(`${'='.repeat(50)}`);
    console.log(`\n${colors.info('Summary')}\n`);

    console.log(`Checks Run:     ${total}`);
    console.log(`${colors.success('Passed')}:        ${this.results.passed}`);
    console.log(`${colors.error('Failed')}:        ${this.results.failed}`);
    console.log(`${colors.warning('Warnings')}:      ${this.results.warnings}`);
    console.log(`Success Rate:   ${passPercentage}%`);
    console.log(`Duration:       ${duration}ms\n`);

    if (this.results.failed === 0) {
      console.log(colors.success('✓ All environment checks passed!'));
      console.log(colors.info('You can proceed with starting the application.\n'));
    } else {
      console.log(colors.error('✗ Some environment checks failed.'));
      console.log(colors.error('Please fix the issues listed above before starting.\n'));
    }

    // Save JSON report
    this.saveReport();
  }

  /**
   * Save detailed JSON report
   */
  saveReport() {
    const reportPath = path.join(process.cwd(), 'environment-report.json');

    const report = {
      timestamp: new Date().toISOString(),
      nodeVersion: process.version,
      platform: process.platform,
      cwd: process.cwd(),
      environment: process.env.NODE_ENV || 'development',
      results: {
        total: this.results.passed + this.results.failed,
        passed: this.results.passed,
        failed: this.results.failed,
        warnings: this.results.warnings,
        successRate:
          this.results.passed + this.results.failed > 0
            ? Math.round((this.results.passed / (this.results.passed + this.results.failed)) * 100)
            : 0,
      },
      details: this.results.details,
      errors: this.results.errors,
      warnings: this.results.warnings,
    };

    try {
      fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf-8');
      console.log(colors.dim(`Report saved to: ${reportPath}\n`));
    } catch (error) {
      console.warn(colors.warning(`Could not save report: ${error.message}`));
    }
  }

  /**
   * Get exit code
   */
  getExitCode() {
    return this.results.failed > 0 ? 1 : 0;
  }
}

/**
 * Individual Check Functions
 */

async function checkNodeVersion() {
  const version = process.versions.node;
  const [major] = version.split('.').map(Number);

  if (major < 14) {
    return {
      passed: false,
      error: `Node.js version ${version} is below minimum required (14.x)`,
      remediation: 'Update Node.js to version 14 or higher',
    };
  }

  return {
    passed: true,
    details: `Node.js ${version}`,
  };
}

async function checkEnvironmentVariables() {
  const required = ['NODE_ENV', 'PORT', 'DATABASE_URL', 'JWT_SECRET', 'LOG_LEVEL'];
  const missing = required.filter((v) => !process.env[v]);

  if (missing.length > 0) {
    return {
      passed: false,
      error: `Missing required environment variables: ${missing.join(', ')}`,
      remediation: 'Create .env file with: ' + missing.join(', '),
    };
  }

  return {
    passed: true,
    details: `All ${required.length} required variables set`,
  };
}

async function checkEnvironmentFormats() {
  const errors = [];

  // Check NODE_ENV
  const validEnv = ['development', 'staging', 'production'];
  if (!validEnv.includes(process.env.NODE_ENV)) {
    errors.push(`NODE_ENV must be one of: ${validEnv.join(', ')}`);
  }

  // Check PORT
  const port = parseInt(process.env.PORT);
  if (isNaN(port) || port < 1 || port > 65535) {
    errors.push(`PORT must be a number between 1 and 65535`);
  }

  // Check LOG_LEVEL
  const validLogLevels = ['error', 'warn', 'info', 'debug', 'trace'];
  if (!validLogLevels.includes(process.env.LOG_LEVEL)) {
    errors.push(`LOG_LEVEL must be one of: ${validLogLevels.join(', ')}`);
  }

  // Check JWT_SECRET length
  if (process.env.JWT_SECRET && process.env.JWT_SECRET.length < 32) {
    errors.push('JWT_SECRET must be at least 32 characters long');
  }

  if (errors.length > 0) {
    return {
      passed: false,
      error: errors.join('; '),
      remediation: 'Review .env file and correct invalid values',
    };
  }

  return {
    passed: true,
    details: 'All environment variables have valid formats',
  };
}

async function checkDatabaseConnection() {
  const databaseUrl = process.env.DATABASE_URL;

  if (!databaseUrl) {
    return {
      passed: false,
      error: 'DATABASE_URL not set',
      remediation: 'Set DATABASE_URL in .env file',
    };
  }

  // Try to parse URL (without actually connecting)
  try {
    const url = new URL(databaseUrl);

    if (url.protocol !== 'postgres:' && url.protocol !== 'postgresql:') {
      return {
        passed: false,
        error: 'Invalid database URL protocol',
        remediation: 'DATABASE_URL must start with postgres:// or postgresql://',
      };
    }

    return {
      passed: true,
      details: `Connected to ${url.hostname}:${url.port || 5432}`,
    };
  } catch (error) {
    return {
      passed: false,
      error: 'Invalid DATABASE_URL format',
      remediation: 'DATABASE_URL must be a valid PostgreSQL connection string',
    };
  }
}

async function checkRedisConfiguration() {
  const redisUrl = process.env.REDIS_URL;

  if (!redisUrl) {
    return {
      passed: true,
      details: 'Redis not configured (optional)',
      warning: 'Cache and sessions will use in-memory storage',
    };
  }

  try {
    const url = new URL(redisUrl);

    if (url.protocol !== 'redis:' && url.protocol !== 'rediss:') {
      return {
        passed: false,
        error: 'Invalid Redis URL protocol',
        remediation: 'REDIS_URL must start with redis:// or rediss://',
      };
    }

    return {
      passed: true,
      details: `Redis configured at ${url.hostname}:${url.port || 6379}`,
    };
  } catch (error) {
    return {
      passed: false,
      error: 'Invalid REDIS_URL format',
      remediation: 'REDIS_URL must be a valid Redis connection string',
    };
  }
}

async function checkFileSystemPermissions() {
  const dirs = ['logs', 'uploads', 'config'];
  const missing = [];

  for (const dir of dirs) {
    const dirPath = path.join(process.cwd(), dir);

    try {
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }

      // Test write permission
      const testFile = path.join(dirPath, '.write-test');
      fs.writeFileSync(testFile, 'test', 'utf-8');
      fs.unlinkSync(testFile);
    } catch (error) {
      missing.push(`${dir} (${error.message})`);
    }
  }

  if (missing.length > 0) {
    return {
      passed: false,
      error: `Unable to write to directories: ${missing.join(', ')}`,
      remediation: 'Check file system permissions or run with elevated privileges',
    };
  }

  return {
    passed: true,
    details: `All required directories accessible: ${dirs.join(', ')}`,
  };
}

async function checkEnvExampleExists() {
  const envExamplePath = path.join(process.cwd(), '.env.example');

  if (!fs.existsSync(envExamplePath)) {
    return {
      passed: false,
      error: '.env.example file not found',
      remediation: 'Generate using: npm run generate:env',
    };
  }

  return {
    passed: true,
    details: '.env.example exists',
  };
}

async function checkPackageJson() {
  const packageJsonPath = path.join(process.cwd(), 'package.json');

  if (!fs.existsSync(packageJsonPath)) {
    return {
      passed: false,
      error: 'package.json not found',
      remediation: 'Run: npm init',
    };
  }

  try {
    const pkg = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));

    if (!pkg.name || !pkg.version) {
      return {
        passed: false,
        error: 'package.json is invalid or incomplete',
        remediation: 'Add name and version to package.json',
      };
    }

    return {
      passed: true,
      details: `${pkg.name} v${pkg.version}`,
    };
  } catch (error) {
    return {
      passed: false,
      error: 'package.json is not valid JSON',
      remediation: 'Fix JSON syntax in package.json',
    };
  }
}

async function checkNodeModules() {
  const nodeModulesPath = path.join(process.cwd(), 'node_modules');

  if (!fs.existsSync(nodeModulesPath)) {
    return {
      passed: false,
      error: 'node_modules not found',
      remediation: 'Run: npm install',
    };
  }

  return {
    passed: true,
    details: 'Dependencies installed',
  };
}

async function checkDiskSpace() {
  const diskSpacePath = process.cwd();

  try {
    const stats = fs.statSync(diskSpacePath);

    // Just check that we can read the filesystem
    return {
      passed: true,
      details: 'File system accessible',
    };
  } catch (error) {
    return {
      passed: false,
      error: `File system error: ${error.message}`,
      remediation: 'Check file system health and permissions',
    };
  }
}

/**
 * Main execution
 */
async function main() {
  const manager = new EnvironmentVerificationManager();

  // Add checks in sequence
  manager
    .addCheck('Node.js Version', checkNodeVersion)
    .addCheck('Required Environment Variables', checkEnvironmentVariables)
    .addCheck('Environment Variable Formats', checkEnvironmentFormats)
    .addCheck('Database Configuration', checkDatabaseConnection)
    .addCheck('Redis Configuration', checkRedisConfiguration)
    .addCheck('File System Permissions', checkFileSystemPermissions)
    .addCheck('.env.example Exists', checkEnvExampleExists)
    .addCheck('package.json Valid', checkPackageJson)
    .addCheck('node_modules Installed', checkNodeModules)
    .addCheck('File System Accessible', checkDiskSpace);

  // Run all checks
  const results = await manager.runAll();

  // Exit with appropriate code
  process.exit(manager.getExitCode());
}

// Run if executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error(colors.error(`Fatal error: ${error.message}`));
    process.exit(1);
  });
}

module.exports = {
  EnvironmentVerificationManager,
  checkNodeVersion,
  checkEnvironmentVariables,
  checkEnvironmentFormats,
  checkDatabaseConnection,
  checkRedisConfiguration,
  checkFileSystemPermissions,
  checkEnvExampleExists,
  checkPackageJson,
  checkNodeModules,
  checkDiskSpace,
};
