/**
 * CONFIGURATION SERVICE
 *
 * Type-safe configuration management with environment-specific overrides
 * Features:
 *   - Centralized config loading
 *   - Environment-specific settings (dev, staging, prod)
 *   - Type validation
 *   - Default values
 *   - Clear error messages
 */

'use strict';

const path = require('path');

/**
 * Default Configuration
 */
const defaultConfig = {
  // Application
  node_env: process.env.NODE_ENV || 'development',
  port: parseInt(process.env.PORT || '3000'),
  host: process.env.HOST || 'localhost',

  // Database
  database: {
    url: process.env.DATABASE_URL || 'postgres://postgres:postgres@localhost:5432/negative_space',
    ssl: process.env.DB_SSL === 'true',
    logging: process.env.DB_LOGGING === 'true',
    sync: process.env.DB_SYNC === 'true',
    pool: {
      min: parseInt(process.env.DB_POOL_MIN || '2'),
      max: parseInt(process.env.DB_POOL_MAX || '10'),
    },
  },

  // Redis (optional)
  redis: {
    url: process.env.REDIS_URL || null,
    enabled: !!process.env.REDIS_URL,
  },

  // JWT
  jwt: {
    secret: process.env.JWT_SECRET,
    expiresIn: process.env.JWT_EXPIRES_IN || '24h',
    refreshSecret: process.env.JWT_REFRESH_SECRET,
    refreshExpiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d',
  },

  // Security
  security: {
    bcryptSaltRounds: parseInt(process.env.BCRYPT_SALT_ROUNDS || '12'),
    cookieSecret: process.env.COOKIE_SECRET,
    csrfEnabled: process.env.CSRF_ENABLED !== 'false',
    corsAllowedOrigins: process.env.CORS_ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    sessionTimeout: parseInt(process.env.SESSION_TIMEOUT || '3600000'), // 1 hour
  },

  // Logging
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    dir: process.env.LOG_DIR || 'logs',
    filename: process.env.LOG_FILENAME || 'app.log',
    maxSize: process.env.LOG_MAX_SIZE || '10m',
    maxFiles: parseInt(process.env.LOG_MAX_FILES || '7'),
  },

  // File Storage
  storage: {
    uploadDir: process.env.UPLOAD_DIR || 'uploads',
    maxFileSize: parseInt(process.env.MAX_FILE_SIZE || '10485760'), // 10MB
    allowedFileTypes: process.env.ALLOWED_FILE_TYPES?.split(',') || [
      'image/jpeg',
      'image/png',
      'image/tiff',
      'application/dicom',
    ],
  },

  // Imaging Configuration
  imaging: {
    defaultAlgorithm: process.env.DEFAULT_ALGORITHM || 'advanced',
    maxImageDimension: parseInt(process.env.MAX_IMAGE_DIMENSION || '4096'),
    processingTimeout: parseInt(process.env.PROCESSING_TIMEOUT || '30000'), // 30 seconds
    enableGpu: process.env.ENABLE_GPU === 'true',
    cacheTtl: parseInt(process.env.CACHE_TTL || '86400'), // 1 day
  },

  // HIPAA Compliance
  compliance: {
    hipaaEnabled: process.env.HIPAA_ENABLED === 'true',
    dataEncryption: process.env.DATA_ENCRYPTION === 'true',
    auditLogEnabled: process.env.AUDIT_LOG_ENABLED === 'true',
    autoLogout: parseInt(process.env.AUTO_LOGOUT || '900000'), // 15 minutes
  },

  // AWS (optional)
  aws: {
    enabled: !!process.env.AWS_ACCESS_KEY_ID,
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION || 'us-east-1',
    s3Bucket: process.env.AWS_S3_BUCKET,
  },

  // API Keys (optional)
  api: {
    apiKey: process.env.API_KEY,
    apiSecret: process.env.API_SECRET,
  },
};

/**
 * Environment-specific overrides
 */
const environmentOverrides = {
  production: {
    database: {
      ssl: true,
      logging: false,
      sync: false,
    },
    logging: {
      level: 'warn',
    },
    compliance: {
      hipaaEnabled: true,
      dataEncryption: true,
      auditLogEnabled: true,
    },
    security: {
      csrfEnabled: true,
    },
  },

  staging: {
    logging: {
      level: 'debug',
    },
    database: {
      logging: true,
    },
  },

  development: {
    logging: {
      level: 'debug',
    },
    database: {
      logging: true,
      sync: true,
    },
  },
};

/**
 * Configuration Service
 */
class ConfigService {
  constructor(env = process.env, customConfig = {}) {
    this.env = env;
    this.config = this.loadConfig(customConfig);
  }

  /**
   * Load and merge configuration
   */
  loadConfig(customConfig = {}) {
    const nodeEnv = process.env.NODE_ENV || 'development';

    // Start with defaults
    let config = JSON.parse(JSON.stringify(defaultConfig));

    // Apply environment-specific overrides
    if (environmentOverrides[nodeEnv]) {
      config = this.deepMerge(config, environmentOverrides[nodeEnv]);
    }

    // Apply custom config
    if (customConfig && Object.keys(customConfig).length > 0) {
      config = this.deepMerge(config, customConfig);
    }

    return config;
  }

  /**
   * Deep merge objects
   */
  deepMerge(target, source) {
    const result = JSON.parse(JSON.stringify(target));

    Object.keys(source).forEach((key) => {
      if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
        result[key] = this.deepMerge(result[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    });

    return result;
  }

  /**
   * Get configuration value by path
   * @param {string} path - Dot notation path (e.g., 'database.url')
   * @param {*} defaultValue - Default value if not found
   */
  get(path, defaultValue = undefined) {
    return this.getNestedValue(this.config, path, defaultValue);
  }

  /**
   * Get nested value from object
   */
  getNestedValue(obj, path, defaultValue) {
    const keys = path.split('.');
    let value = obj;

    for (const key of keys) {
      if (value === null || value === undefined) {
        return defaultValue;
      }
      value = value[key];
    }

    return value !== undefined ? value : defaultValue;
  }

  /**
   * Get entire configuration object
   */
  getAll() {
    return JSON.parse(JSON.stringify(this.config));
  }

  /**
   * Get specific section
   */
  getSection(section) {
    if (!this.config[section]) {
      throw new Error(`Configuration section not found: ${section}`);
    }
    return JSON.parse(JSON.stringify(this.config[section]));
  }

  /**
   * Validate configuration
   */
  validate() {
    const errors = [];

    // Required configurations
    if (!this.config.jwt.secret) {
      errors.push('JWT_SECRET is required');
    }

    if (!this.config.database.url) {
      errors.push('DATABASE_URL is required');
    }

    if (this.config.jwt.secret && this.config.jwt.secret.length < 32) {
      errors.push('JWT_SECRET must be at least 32 characters');
    }

    if (this.config.port < 1 || this.config.port > 65535) {
      errors.push(`PORT must be between 1 and 65535, got ${this.config.port}`);
    }

    if (this.config.node_env === 'production') {
      if (!this.config.database.ssl) {
        errors.push('DATABASE_SSL should be true in production');
      }
      if (this.config.logging.level !== 'warn' && this.config.logging.level !== 'error') {
        errors.push('LOG_LEVEL should be warn or error in production');
      }
    }

    if (errors.length > 0) {
      const message = `Configuration validation failed:\n  - ${errors.join('\n  - ')}`;
      throw new Error(message);
    }

    return true;
  }

  /**
   * Export config as formatted string
   */
  toString() {
    return JSON.stringify(this.config, null, 2);
  }

  /**
   * Check if running in specific environment
   */
  isProduction() {
    return this.config.node_env === 'production';
  }

  isStaging() {
    return this.config.node_env === 'staging';
  }

  isDevelopment() {
    return this.config.node_env === 'development';
  }

  /**
   * Get database connection info
   */
  getDatabaseUrl() {
    return this.config.database.url;
  }

  /**
   * Get Redis URL if configured
   */
  getRedisUrl() {
    return this.config.redis.url;
  }

  /**
   * Check if Redis is enabled
   */
  isRedisEnabled() {
    return this.config.redis.enabled;
  }

  /**
   * Get storage paths
   */
  getStoragePaths() {
    return {
      uploads: path.resolve(this.config.storage.uploadDir),
      logs: path.resolve(this.config.logging.dir),
    };
  }

  /**
   * Get security configuration
   */
  getSecurityConfig() {
    return {
      bcryptSaltRounds: this.config.security.bcryptSaltRounds,
      jwtSecret: this.config.jwt.secret,
      cookieSecret: this.config.security.cookieSecret,
      corsOrigins: this.config.security.corsAllowedOrigins,
    };
  }

  /**
   * Get imaging configuration
   */
  getImagingConfig() {
    return {
      defaultAlgorithm: this.config.imaging.defaultAlgorithm,
      maxImageDimension: this.config.imaging.maxImageDimension,
      processingTimeout: this.config.imaging.processingTimeout,
      enableGpu: this.config.imaging.enableGpu,
      cacheTtl: this.config.imaging.cacheTtl,
    };
  }

  /**
   * Get compliance settings
   */
  getComplianceSettings() {
    return {
      hipaaEnabled: this.config.compliance.hipaaEnabled,
      dataEncryption: this.config.compliance.dataEncryption,
      auditLogEnabled: this.config.compliance.auditLogEnabled,
      autoLogout: this.config.compliance.autoLogout,
    };
  }

  /**
   * Create masked config for logging (removes sensitive data)
   */
  getMaskedConfig() {
    const masked = JSON.parse(JSON.stringify(this.config));

    // Mask sensitive fields
    const maskField = (obj, fieldName) => {
      if (obj[fieldName]) {
        obj[fieldName] = '*'.repeat(Math.min(obj[fieldName].length, 20));
      }
    };

    maskField(masked.jwt, 'secret');
    maskField(masked.jwt, 'refreshSecret');
    maskField(masked.security, 'cookieSecret');
    maskField(masked.database.url, 'password');
    maskField(masked.aws, 'accessKeyId');
    maskField(masked.aws, 'secretAccessKey');
    maskField(masked.api, 'apiKey');
    maskField(masked.api, 'apiSecret');

    return masked;
  }
}

module.exports = {
  ConfigService,
  defaultConfig,
  environmentOverrides,
};
