/**
 * ENVIRONMENT VALIDATOR
 *
 * Comprehensive validation system for environment variables
 * Features:
 *   - Format validation (URLs, ports, etc.)
 *   - Service connectivity testing
 *   - File system permissions verification
 *   - Port availability checking
 *   - Clear error reporting with remediation steps
 */

'use strict';

const fs = require('fs');
const path = require('path');
const net = require('net');
const dns = require('dns').promises;

/**
 * Validation Result
 */
class ValidationResult {
  constructor() {
    this.errors = [];
    this.warnings = [];
    this.success = [];
  }

  addError(variable, message, remediation) {
    this.errors.push({
      type: 'ERROR',
      variable,
      message,
      remediation,
      timestamp: new Date().toISOString(),
    });
  }

  addWarning(variable, message, remediation) {
    this.warnings.push({
      type: 'WARNING',
      variable,
      message,
      remediation,
      timestamp: new Date().toISOString(),
    });
  }

  addSuccess(variable, message) {
    this.success.push({
      type: 'SUCCESS',
      variable,
      message,
      timestamp: new Date().toISOString(),
    });
  }

  isValid() {
    return this.errors.length === 0;
  }

  hasIssues() {
    return this.errors.length > 0 || this.warnings.length > 0;
  }

  toJSON() {
    return {
      valid: this.isValid(),
      hasIssues: this.hasIssues(),
      errors: this.errors,
      warnings: this.warnings,
      success: this.success,
      summary: {
        totalErrors: this.errors.length,
        totalWarnings: this.warnings.length,
        totalSuccess: this.success.length,
      },
    };
  }

  toString() {
    let output = '\n📋 ENVIRONMENT VALIDATION REPORT\n';
    output += '═'.repeat(50) + '\n\n';

    if (this.errors.length > 0) {
      output += `❌ ERRORS (${this.errors.length}):\n`;
      this.errors.forEach((err, i) => {
        output += `\n  ${i + 1}. ${err.variable}\n`;
        output += `     Error: ${err.message}\n`;
        output += `     Fix: ${err.remediation}\n`;
      });
      output += '\n';
    }

    if (this.warnings.length > 0) {
      output += `⚠️  WARNINGS (${this.warnings.length}):\n`;
      this.warnings.forEach((warn, i) => {
        output += `\n  ${i + 1}. ${warn.variable}\n`;
        output += `     Warning: ${warn.message}\n`;
        output += `     Suggestion: ${warn.remediation}\n`;
      });
      output += '\n';
    }

    if (this.success.length > 0) {
      output += `✅ VALIDATED (${this.success.length}):\n`;
      this.success.forEach((succ, i) => {
        output += `  ${i + 1}. ${succ.variable}: ${succ.message}\n`;
      });
      output += '\n';
    }

    output += '═'.repeat(50) + '\n';
    output += `Status: ${this.isValid() ? '✅ VALID' : '❌ INVALID'}\n`;
    output += '═'.repeat(50) + '\n';

    return output;
  }
}

/**
 * Environment Validator
 */
class EnvironmentValidator {
  constructor(env = process.env) {
    this.env = env;
    this.result = new ValidationResult();
  }

  /**
   * Validate all environment variables
   */
  async validateAll() {
    console.log('🔍 Starting environment validation...\n');

    await this.validateRequired();
    await this.validateFormats();
    await this.validateConnectivity();
    await this.validateFileSystemPermissions();
    await this.validatePortAvailability();
    await this.validateCredentials();

    return this.result;
  }

  /**
   * Validate required variables exist
   */
  async validateRequired() {
    console.log('Checking required variables...');

    const required = ['NODE_ENV', 'PORT', 'DATABASE_URL', 'JWT_SECRET', 'LOG_LEVEL'];

    required.forEach((variable) => {
      if (!this.env[variable]) {
        this.result.addError(
          variable,
          `Missing required environment variable`,
          `Set ${variable} in .env file or system environment`,
        );
      } else {
        this.result.addSuccess(variable, 'Variable defined');
      }
    });
  }

  /**
   * Validate variable formats
   */
  async validateFormats() {
    console.log('Validating variable formats...');

    // NODE_ENV validation
    const validNodeEnv = ['development', 'staging', 'production'];
    if (this.env.NODE_ENV && !validNodeEnv.includes(this.env.NODE_ENV)) {
      this.result.addError(
        'NODE_ENV',
        `Invalid value: ${this.env.NODE_ENV}`,
        `Set NODE_ENV to one of: ${validNodeEnv.join(', ')}`,
      );
    } else if (this.env.NODE_ENV) {
      this.result.addSuccess('NODE_ENV', `Valid: ${this.env.NODE_ENV}`);
    }

    // PORT validation
    if (this.env.PORT) {
      const port = parseInt(this.env.PORT);
      if (isNaN(port) || port < 1 || port > 65535) {
        this.result.addError(
          'PORT',
          `Invalid port number: ${this.env.PORT}`,
          `Set PORT to a number between 1 and 65535`,
        );
      } else {
        this.result.addSuccess('PORT', `Valid: ${port}`);
      }
    }

    // DATABASE_URL validation
    if (this.env.DATABASE_URL) {
      if (!this.isValidDatabaseUrl(this.env.DATABASE_URL)) {
        this.result.addError(
          'DATABASE_URL',
          `Invalid database URL format`,
          `Set DATABASE_URL to valid format: postgres://user:password@host:port/database`,
        );
      } else {
        this.result.addSuccess('DATABASE_URL', 'Valid database URL format');
      }
    }

    // REDIS_URL validation (if present)
    if (this.env.REDIS_URL) {
      if (!this.isValidRedisUrl(this.env.REDIS_URL)) {
        this.result.addWarning(
          'REDIS_URL',
          `Invalid Redis URL format`,
          `Set REDIS_URL to valid format: redis://user:password@host:port`,
        );
      } else {
        this.result.addSuccess('REDIS_URL', 'Valid Redis URL format');
      }
    }

    // JWT_SECRET validation
    if (this.env.JWT_SECRET) {
      if (this.env.JWT_SECRET.length < 32) {
        this.result.addWarning(
          'JWT_SECRET',
          `JWT_SECRET is weak (${this.env.JWT_SECRET.length} chars)`,
          `Use a JWT_SECRET with at least 32 characters for better security`,
        );
      } else {
        this.result.addSuccess('JWT_SECRET', `Valid (${this.env.JWT_SECRET.length} chars)`);
      }
    }

    // LOG_LEVEL validation
    const validLogLevels = ['debug', 'info', 'warn', 'error', 'fatal'];
    if (this.env.LOG_LEVEL && !validLogLevels.includes(this.env.LOG_LEVEL)) {
      this.result.addWarning(
        'LOG_LEVEL',
        `Invalid log level: ${this.env.LOG_LEVEL}`,
        `Set LOG_LEVEL to one of: ${validLogLevels.join(', ')}`,
      );
    } else if (this.env.LOG_LEVEL) {
      this.result.addSuccess('LOG_LEVEL', `Valid: ${this.env.LOG_LEVEL}`);
    }

    // MAX_FILE_SIZE validation
    if (this.env.MAX_FILE_SIZE) {
      const size = parseInt(this.env.MAX_FILE_SIZE);
      if (isNaN(size) || size <= 0) {
        this.result.addError(
          'MAX_FILE_SIZE',
          `Invalid file size: ${this.env.MAX_FILE_SIZE}`,
          `Set MAX_FILE_SIZE to a positive number (bytes)`,
        );
      } else {
        this.result.addSuccess('MAX_FILE_SIZE', `Valid: ${size} bytes`);
      }
    }

    // PROCESSING_TIMEOUT validation
    if (this.env.PROCESSING_TIMEOUT) {
      const timeout = parseInt(this.env.PROCESSING_TIMEOUT);
      if (isNaN(timeout) || timeout < 1000) {
        this.result.addWarning(
          'PROCESSING_TIMEOUT',
          `Processing timeout too low: ${timeout}ms`,
          `Set PROCESSING_TIMEOUT to at least 1000ms (1 second)`,
        );
      } else {
        this.result.addSuccess('PROCESSING_TIMEOUT', `Valid: ${timeout}ms`);
      }
    }

    // AWS credentials validation (if S3 is used)
    if (this.env.AWS_ACCESS_KEY_ID) {
      if (!this.env.AWS_SECRET_ACCESS_KEY) {
        this.result.addError(
          'AWS_SECRET_ACCESS_KEY',
          `AWS_ACCESS_KEY_ID set but AWS_SECRET_ACCESS_KEY missing`,
          `Set AWS_SECRET_ACCESS_KEY in environment for S3 access`,
        );
      } else {
        this.result.addSuccess('AWS Credentials', 'AWS credentials configured');
      }
    }
  }

  /**
   * Validate service connectivity
   */
  async validateConnectivity() {
    console.log('Validating service connectivity...');

    // PostgreSQL connectivity
    if (this.env.DATABASE_URL) {
      try {
        const dbUrl = new URL(this.env.DATABASE_URL);
        const isReachable = await this.checkPortAvailability(dbUrl.hostname, dbUrl.port || 5432);

        if (isReachable) {
          this.result.addSuccess(
            'PostgreSQL',
            `Reachable at ${dbUrl.hostname}:${dbUrl.port || 5432}`,
          );
        } else {
          this.result.addWarning(
            'PostgreSQL',
            `Cannot connect to database server`,
            `Verify PostgreSQL is running on ${dbUrl.hostname}:${dbUrl.port || 5432}`,
          );
        }
      } catch (error) {
        this.result.addWarning(
          'PostgreSQL',
          `Failed to validate connectivity: ${error.message}`,
          `Check DATABASE_URL format and ensure PostgreSQL is accessible`,
        );
      }
    }

    // Redis connectivity
    if (this.env.REDIS_URL) {
      try {
        const redisUrl = new URL(this.env.REDIS_URL);
        const isReachable = await this.checkPortAvailability(
          redisUrl.hostname,
          redisUrl.port || 6379,
        );

        if (isReachable) {
          this.result.addSuccess(
            'Redis',
            `Reachable at ${redisUrl.hostname}:${redisUrl.port || 6379}`,
          );
        } else {
          this.result.addWarning(
            'Redis',
            `Cannot connect to Redis server`,
            `Verify Redis is running on ${redisUrl.hostname}:${redisUrl.port || 6379}`,
          );
        }
      } catch (error) {
        this.result.addWarning(
          'Redis',
          `Failed to validate connectivity: ${error.message}`,
          `Check REDIS_URL format and ensure Redis is accessible`,
        );
      }
    }

    // API service check (if API_URL is set)
    if (this.env.API_URL) {
      try {
        const apiUrl = new URL(this.env.API_URL);
        const isReachable = await this.checkPortAvailability(apiUrl.hostname, apiUrl.port || 80);

        if (isReachable) {
          this.result.addSuccess(
            'API Service',
            `Reachable at ${apiUrl.hostname}:${apiUrl.port || 80}`,
          );
        } else {
          this.result.addWarning(
            'API Service',
            `Cannot connect to API service`,
            `Verify API service is running on ${apiUrl.hostname}:${apiUrl.port || 80}`,
          );
        }
      } catch (error) {
        this.result.addWarning(
          'API Service',
          `Failed to validate connectivity: ${error.message}`,
          `Check API_URL format`,
        );
      }
    }
  }

  /**
   * Validate file system permissions
   */
  async validateFileSystemPermissions() {
    console.log('Validating file system permissions...');

    const directoriesToCheck = [
      this.env.LOG_DIR || 'logs',
      this.env.UPLOAD_DIR || 'uploads',
      'config',
    ];

    for (const dir of directoriesToCheck) {
      try {
        const fullPath = path.resolve(dir);

        if (!fs.existsSync(fullPath)) {
          try {
            fs.mkdirSync(fullPath, { recursive: true });
            this.result.addSuccess(dir, `Directory created: ${fullPath}`);
          } catch (mkdirError) {
            this.result.addError(
              dir,
              `Cannot create directory: ${mkdirError.message}`,
              `Ensure parent directory exists and you have write permissions`,
            );
          }
        } else {
          // Check read/write permissions
          fs.accessSync(fullPath, fs.constants.R_OK | fs.constants.W_OK);
          this.result.addSuccess(dir, `Directory writable: ${fullPath}`);
        }
      } catch (error) {
        this.result.addError(
          dir,
          `Permission denied for ${dir}: ${error.message}`,
          `Check directory permissions or create ${dir} with write access`,
        );
      }
    }
  }

  /**
   * Validate port availability
   */
  async validatePortAvailability() {
    console.log('Validating port availability...');

    if (this.env.PORT) {
      try {
        const port = parseInt(this.env.PORT);
        const isAvailable = await this.isPortAvailable(port);

        if (isAvailable) {
          this.result.addSuccess('PORT', `Port ${port} is available`);
        } else {
          this.result.addError(
            'PORT',
            `Port ${port} is already in use`,
            `Change PORT to an available port or stop the service using port ${port}`,
          );
        }
      } catch (error) {
        this.result.addWarning(
          'PORT',
          `Could not verify port availability: ${error.message}`,
          `Manually verify port ${this.env.PORT} is available`,
        );
      }
    }
  }

  /**
   * Validate credentials format
   */
  async validateCredentials() {
    console.log('Validating credentials...');

    // Database credentials from URL
    if (this.env.DATABASE_URL) {
      try {
        const url = new URL(this.env.DATABASE_URL);
        if (!url.username) {
          this.result.addWarning(
            'DATABASE_URL',
            `Database URL has no username`,
            `Add credentials to DATABASE_URL: postgres://user:password@host/db`,
          );
        } else if (!url.password) {
          this.result.addWarning(
            'DATABASE_URL',
            `Database URL has no password`,
            `Add password to DATABASE_URL: postgres://user:password@host/db`,
          );
        } else {
          this.result.addSuccess('Database Credentials', 'Credentials present in DATABASE_URL');
        }
      } catch (error) {
        // Already handled in format validation
      }
    }

    // COOKIE_SECRET validation
    if (this.env.COOKIE_SECRET) {
      if (this.env.COOKIE_SECRET.length < 32) {
        this.result.addWarning(
          'COOKIE_SECRET',
          `COOKIE_SECRET is weak (${this.env.COOKIE_SECRET.length} chars)`,
          `Use COOKIE_SECRET with at least 32 characters`,
        );
      } else {
        this.result.addSuccess('COOKIE_SECRET', `Valid (${this.env.COOKIE_SECRET.length} chars)`);
      }
    }

    // BCRYPT_SALT_ROUNDS validation
    if (this.env.BCRYPT_SALT_ROUNDS) {
      const rounds = parseInt(this.env.BCRYPT_SALT_ROUNDS);
      if (isNaN(rounds) || rounds < 10) {
        this.result.addWarning(
          'BCRYPT_SALT_ROUNDS',
          `Bcrypt salt rounds too low: ${rounds}`,
          `Set BCRYPT_SALT_ROUNDS to at least 10 (recommended: 12+)`,
        );
      } else {
        this.result.addSuccess('BCRYPT_SALT_ROUNDS', `Valid: ${rounds} rounds`);
      }
    }
  }

  /**
   * Check if URL is valid PostgreSQL connection string
   */
  isValidDatabaseUrl(url) {
    try {
      const parsed = new URL(url);
      return (
        (parsed.protocol === 'postgres:' || parsed.protocol === 'postgresql:') && parsed.hostname
      );
    } catch {
      return false;
    }
  }

  /**
   * Check if URL is valid Redis connection string
   */
  isValidRedisUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === 'redis:' && parsed.hostname;
    } catch {
      return false;
    }
  }

  /**
   * Check if port is available
   */
  isPortAvailable(port) {
    return new Promise((resolve) => {
      const server = net.createServer();

      server.once('error', (err) => {
        if (err.code === 'EADDRINUSE') {
          resolve(false);
        } else {
          resolve(false);
        }
      });

      server.once('listening', () => {
        server.close();
        resolve(true);
      });

      server.listen(port);
    });
  }

  /**
   * Check if host:port is reachable
   */
  checkPortAvailability(host, port) {
    return new Promise((resolve) => {
      const socket = net.createConnection({ host, port, timeout: 2000 });

      socket.on('connect', () => {
        socket.destroy();
        resolve(true);
      });

      socket.on('error', () => {
        resolve(false);
      });

      socket.on('timeout', () => {
        socket.destroy();
        resolve(false);
      });
    });
  }
}

module.exports = {
  EnvironmentValidator,
  ValidationResult,
};
