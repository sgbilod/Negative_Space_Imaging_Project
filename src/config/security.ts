/**
 * Security Configuration for Negative Space Imaging System
 * 
 * This file contains security-related configurations for the application,
 * including rate limiting, CORS, content security policy, and encryption settings.
 * 
 * Note: For production deployment, sensitive values should be set via environment
 * variables rather than hardcoded in this file.
 */

import { SecurityConfig } from '../types/security';

const securityConfig: SecurityConfig = {
  /**
   * Rate limiting configuration to prevent abuse
   */
  rateLimit: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again after 15 minutes',
  },

  /**
   * CORS (Cross-Origin Resource Sharing) configuration
   */
  cors: {
    origin: process.env.CORS_ORIGIN || ['http://localhost:3000', 'https://negative-space-imaging.com'],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: [
      'Content-Type', 
      'Authorization', 
      'X-Requested-With', 
      'X-Signature', 
      'X-Timestamp',
      'X-Request-ID'
    ],
    exposedHeaders: ['Content-Range', 'X-Content-Range'],
    credentials: true,
    maxAge: 86400, // 24 hours
  },

  /**
   * Content Security Policy settings
   * Restricts the sources from which resources can be loaded
   */
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", 'https://cdn.jsdelivr.net'],
      styleSrc: ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com', 'https://cdn.jsdelivr.net'],
      imgSrc: ["'self'", 'data:', 'blob:'],
      fontSrc: ["'self'", 'https://fonts.gstatic.com', 'data:'],
      connectSrc: ["'self'", 'https://api.negative-space-imaging.com'],
      frameSrc: ["'none'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      workerSrc: ["'self'", 'blob:'],
      frameAncestors: ["'none'"],
      formAction: ["'self'"],
      baseUri: ["'self'"],
      upgradeInsecureRequests: [],
    },
  },

  /**
   * Cross-Origin Embedder Policy settings
   */
  crossOriginEmbedderPolicy: true,

  /**
   * Cross-Origin Opener Policy settings
   */
  crossOriginOpenerPolicy: true,

  /**
   * Cross-Origin Resource Policy settings
   */
  crossOriginResourcePolicy: { policy: 'same-origin' },

  /**
   * DNS Prefetch Control settings
   */
  dnsPrefetchControl: { allow: false },

  /**
   * Expect-CT settings
   */
  expectCt: {
    maxAge: 86400,
    enforce: true,
  },

  /**
   * Frameguard settings to prevent clickjacking
   */
  frameguard: { action: 'deny' },

  /**
   * Hide X-Powered-By header to prevent information disclosure
   */
  hidePoweredBy: true,

  /**
   * HTTP Strict Transport Security settings
   */
  hsts: {
    maxAge: 31536000, // 1 year
    includeSubDomains: true,
    preload: true,
  },

  /**
   * IE No Open settings
   */
  ieNoOpen: true,

  /**
   * MIME sniffing prevention
   */
  noSniff: true,

  /**
   * Origin Agent Cluster header
   */
  originAgentCluster: true,

  /**
   * Cross-domain policy settings
   */
  permittedCrossDomainPolicies: { permittedPolicies: 'none' },

  /**
   * Referrer Policy settings
   */
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },

  /**
   * XSS protection settings
   */
  xssFilter: true,

  /**
   * Routes that require request signatures for enhanced security
   */
  signedRoutes: [
    '/api/images/upload',
    '/api/images/analyze',
    '/api/users/update-password',
    '/api/auth/reset-password',
    '/api/admin/*',
  ],

  /**
   * Signature timeout in milliseconds
   * Requests with signatures older than this will be rejected
   */
  signatureTimeout: 5 * 60 * 1000, // 5 minutes

  /**
   * Encryption settings
   * CRITICAL: These values MUST be set via environment variables in production
   */
  encryption: {
    // Encryption key must be provided via ENCRYPTION_KEY environment variable
    // Generate with: openssl rand -hex 32
    getKey: (): string => {
      const key = process.env.ENCRYPTION_KEY;
      if (!key && process.env.NODE_ENV === 'production') {
        throw new Error('ENCRYPTION_KEY environment variable must be set in production');
      }
      return key || '5d9a3f4c8b7e6d2a1c8b7e6d2a1c8b7e6d2a1c8b7e6d2a1c8b7e6d2a1c8b7e';
    },
    
    // Initialization vector must be provided via ENCRYPTION_IV environment variable
    // Generate with: openssl rand -hex 8
    getIv: (): string => {
      const iv = process.env.ENCRYPTION_IV;
      if (!iv && process.env.NODE_ENV === 'production') {
        throw new Error('ENCRYPTION_IV environment variable must be set in production');
      }
      return iv || '9f8e7d6c5b4a3f2e';
    },
  },

  /**
   * JWT (JSON Web Token) settings
   * CRITICAL: JWT_SECRET MUST be set via environment variable
   */
  jwt: {
    // JWT secret must be provided via JWT_SECRET environment variable
    // Generate with: openssl rand -hex 32
    getSecret: (): string => {
      const secret = process.env.JWT_SECRET;
      if (!secret) {
        throw new Error('JWT_SECRET environment variable must be set');
      }
      return secret;
    },
    
    // Token expiration times
    expiresIn: {
      access: '1h', // 1 hour
      refresh: '7d', // 7 days
    },
  },

  /**
   * Password policy settings
   */
  passwordPolicy: {
    minLength: 12,
    requireUppercase: true,
    requireLowercase: true,
    requireNumbers: true,
    requireSpecialChars: true,
    passwordHistoryLimit: 5, // Remember last 5 passwords
    maxPasswordAge: 90, // 90 days
  },

  /**
   * Session security settings
   */
  session: {
    name: 'nsi.sid',
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    maxAge: 3600000, // 1 hour
    sameSite: 'strict',
  },

  /**
   * Security audit and logging settings
   */
  audit: {
    enabled: true,
    logLevel: process.env.AUDIT_LOG_LEVEL || 'info',
    sensitiveFields: [
      'password',
      'passwordConfirm',
      'currentPassword',
      'newPassword',
      'token',
      'credit_card',
      'ssn',
      'medicalRecord',
    ],
  },

  /**
   * HIPAA compliance settings
   */
  hipaa: {
    enabledFor: {
      audit: true,
      encryption: true,
      authentication: true,
      authorization: true,
    },
    dataRetentionPeriod: 6 * 30 * 24 * 60 * 60 * 1000, // 6 months in milliseconds
    requiredRoles: ['admin', 'medical_staff', 'compliance_officer'],
  }
};

export default securityConfig;
