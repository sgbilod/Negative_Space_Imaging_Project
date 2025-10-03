import dotenv from 'dotenv';
import path from 'path';

// Load environment variables from .env file
dotenv.config();

const environment = process.env.NODE_ENV || 'development';

/**
 * Application configuration based on environment
 */
export const config = {
  // Server configuration
  server: {
    port: parseInt(process.env.PORT || '3000', 10),
    env: environment,
    isDevelopment: environment === 'development',
    isProduction: environment === 'production',
    isTest: environment === 'test',
  },

  // Database configuration
  database: {
    url: process.env.DATABASE_URL || 'postgres://postgres:postgres@localhost:5432/negative_space',
    ssl: process.env.DB_SSL === 'true',
    logging: process.env.DB_LOGGING === 'true',
    synchronize: process.env.DB_SYNC === 'true' || environment === 'development',
  },

  // Security configuration
  security: {
    jwtSecret: process.env.JWT_SECRET || 'super-secret-key-that-should-be-changed-in-production',
    jwtExpiresIn: process.env.JWT_EXPIRES_IN || '1d',
    bcryptSaltRounds: parseInt(process.env.BCRYPT_SALT_ROUNDS || '12', 10),
    cookieSecret: process.env.COOKIE_SECRET || 'cookie-secret-key',
    csrfEnabled: process.env.CSRF_ENABLED !== 'false',
  },

  // CORS configuration
  cors: {
    allowedOrigins: process.env.CORS_ALLOWED_ORIGINS
      ? process.env.CORS_ALLOWED_ORIGINS.split(',')
      : ['http://localhost:3000', 'http://localhost:5173'],
  },

  // Logging configuration
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    directory: process.env.LOG_DIR || path.join(process.cwd(), 'logs'),
    filename: process.env.LOG_FILENAME || 'app.log',
    maxSize: process.env.LOG_MAX_SIZE || '10m',
    maxFiles: parseInt(process.env.LOG_MAX_FILES || '7', 10),
  },

  // File storage configuration
  storage: {
    uploadDir: process.env.UPLOAD_DIR || path.join(process.cwd(), 'uploads'),
    maxFileSize: parseInt(process.env.MAX_FILE_SIZE || '10485760', 10), // 10MB
    allowedTypes: process.env.ALLOWED_FILE_TYPES
      ? process.env.ALLOWED_FILE_TYPES.split(',')
      : ['image/jpeg', 'image/png', 'image/tiff', 'application/dicom'],
  },

  // Negative space imaging processing configuration
  imaging: {
    defaultAlgorithm: process.env.DEFAULT_ALGORITHM || 'advanced',
    maxImageDimension: parseInt(process.env.MAX_IMAGE_DIMENSION || '4096', 10),
    processingTimeout: parseInt(process.env.PROCESSING_TIMEOUT || '30000', 10), // 30 seconds
    enableGPU: process.env.ENABLE_GPU === 'true',
    cacheTTL: parseInt(process.env.CACHE_TTL || '86400', 10), // 24 hours
  },

  // HIPAA compliance configuration
  hipaa: {
    enabled: process.env.HIPAA_ENABLED === 'true' || environment === 'production',
    dataEncryption: process.env.DATA_ENCRYPTION === 'true' || environment === 'production',
    auditLogEnabled: process.env.AUDIT_LOG_ENABLED === 'true' || environment === 'production',
    autoLogout: parseInt(process.env.AUTO_LOGOUT || '900000', 10), // 15 minutes
  },
};
