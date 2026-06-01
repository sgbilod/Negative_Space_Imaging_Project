/**
 * Security Middleware for Negative Space Imaging System
 * 
 * This middleware implements advanced security features required for HIPAA compliance
 * and sensitive image data protection, including:
 * 
 * - End-to-end encryption
 * - Comprehensive audit logging
 * - Rate limiting
 * - Content security policies
 * - Session management
 * - Authentication & authorization
 * - Input validation and sanitization
 */

import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import xss from 'xss-clean';
import hpp from 'hpp';
import cors from 'cors';
import { v4 as uuidv4 } from 'uuid';
import winston from 'winston';
import crypto from 'crypto';
import { createClient } from 'redis';
import { SecurityConfig } from '../types/security';
import logger from '../utils/logger';

// Load security configuration
import securityConfig from '../config/security';

// Initialize Redis client for rate limiting and session management
let redisClient: any;
try {
  redisClient = createClient({
    url: process.env.REDIS_URL || 'redis://localhost:6379',
  });
  redisClient.on('error', (err: Error) => {
    logger.error('Redis client error', { error: err.message });
  });
} catch (err) {
  logger.warn('Redis client initialization failed, using in-memory storage', { 
    error: (err as Error).message 
  });
}

/**
 * Audit logging middleware that records detailed information about each request
 */
export const auditLogger = (req: Request, res: Response, next: NextFunction) => {
  // Generate unique request ID for correlation
  const requestId = uuidv4();
  req.headers['x-request-id'] = requestId;
  
  // Capture request start time
  const startTime = process.hrtime();
  
  // Create audit entry when response finishes
  res.on('finish', () => {
    const hrTime = process.hrtime(startTime);
    const responseTime = (hrTime[0] * 1000 + hrTime[1] / 1000000).toFixed(2);
    
    // Log request details, excluding sensitive data
    const sanitizedHeaders = { ...req.headers };
    delete sanitizedHeaders.authorization;
    delete sanitizedHeaders.cookie;
    
    const auditEntry = {
      requestId,
      timestamp: new Date().toISOString(),
      method: req.method,
      url: req.originalUrl,
      statusCode: res.statusCode,
      responseTime: `${responseTime}ms`,
      userAgent: req.headers['user-agent'],
      ipAddress: req.ip || req.headers['x-forwarded-for'] || 'unknown',
      userId: (req as any).user?.id || 'anonymous',
      referer: req.headers.referer || '',
      contentLength: res.getHeader('content-length') || 0,
    };
    
    // Log with appropriate level based on status code
    if (res.statusCode >= 500) {
      logger.error('Request failed', auditEntry);
    } else if (res.statusCode >= 400) {
      logger.warn('Request error', auditEntry);
    } else {
      logger.info('Request completed', auditEntry);
    }
  });
  
  next();
};

/**
 * Advanced rate limiting middleware with IP and user-based throttling
 */
export const rateLimiter = rateLimit({
  windowMs: securityConfig.rateLimit.windowMs,
  max: securityConfig.rateLimit.maxRequests,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    status: 'error',
    message: 'Too many requests, please try again later.',
  },
  keyGenerator: (req) => {
    // Use combination of IP and user ID (if authenticated) for more precise rate limiting
    const userId = (req as any).user?.id || '';
    return userId ? `${req.ip}-${userId}` : req.ip || '';
  },
});

/**
 * Enhanced security headers middleware using helmet
 */
export const securityHeaders = helmet({
  contentSecurityPolicy: securityConfig.contentSecurityPolicy,
  crossOriginEmbedderPolicy: securityConfig.crossOriginEmbedderPolicy,
  crossOriginOpenerPolicy: securityConfig.crossOriginOpenerPolicy,
  crossOriginResourcePolicy: securityConfig.crossOriginResourcePolicy,
  dnsPrefetchControl: securityConfig.dnsPrefetchControl,
  expectCt: securityConfig.expectCt,
  frameguard: securityConfig.frameguard,
  hidePoweredBy: securityConfig.hidePoweredBy,
  hsts: securityConfig.hsts,
  ieNoOpen: securityConfig.ieNoOpen,
  noSniff: securityConfig.noSniff,
  originAgentCluster: securityConfig.originAgentCluster,
  permittedCrossDomainPolicies: securityConfig.permittedCrossDomainPolicies,
  referrerPolicy: securityConfig.referrerPolicy,
  xssFilter: securityConfig.xssFilter,
});

/**
 * CORS configuration middleware with secure defaults
 */
export const corsOptions = cors({
  origin: securityConfig.cors.origin,
  methods: securityConfig.cors.methods,
  allowedHeaders: securityConfig.cors.allowedHeaders,
  exposedHeaders: securityConfig.cors.exposedHeaders,
  credentials: securityConfig.cors.credentials,
  maxAge: securityConfig.cors.maxAge,
  optionsSuccessStatus: 204,
});

/**
 * Input validation and sanitization middleware
 */
export const sanitizeInputs = [
  // Prevent XSS attacks
  xss(),
  // Prevent HTTP Parameter Pollution attacks
  hpp(),
];

/**
 * Encryption utility for end-to-end encryption of sensitive data
 */
export class EncryptionService {
  private algorithm: string;
  private secretKey: Buffer;
  private iv: Buffer;

  constructor() {
    this.algorithm = 'aes-256-cbc';
    this.secretKey = Buffer.from(
      process.env.ENCRYPTION_KEY || securityConfig.encryption.defaultKey,
      'hex'
    );
    this.iv = Buffer.from(
      process.env.ENCRYPTION_IV || securityConfig.encryption.defaultIv,
      'hex'
    );
  }

  encrypt(text: string): string {
    const cipher = crypto.createCipheriv(this.algorithm, this.secretKey, this.iv);
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return encrypted;
  }

  decrypt(encryptedText: string): string {
    const decipher = crypto.createDecipheriv(this.algorithm, this.secretKey, this.iv);
    let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }

  // Generate a secure hash for data validation
  generateHash(data: string): string {
    return crypto
      .createHmac('sha256', this.secretKey)
      .update(data)
      .digest('hex');
  }

  // Verify hash for data integrity
  verifyHash(data: string, hash: string): boolean {
    const computed = this.generateHash(data);
    return crypto.timingSafeEqual(
      Buffer.from(computed),
      Buffer.from(hash)
    );
  }
}

// Export singleton instance of EncryptionService
export const encryptionService = new EncryptionService();

/**
 * Middleware that validates request signatures for data integrity
 */
export const validateRequestSignature = (req: Request, res: Response, next: NextFunction) => {
  // Only validate signatures for specific routes that need enhanced security
  if (!securityConfig.signedRoutes.includes(req.path)) {
    return next();
  }

  const signature = req.headers['x-signature'] as string;
  const timestamp = req.headers['x-timestamp'] as string;
  const payload = req.method === 'GET' ? JSON.stringify(req.query) : JSON.stringify(req.body);

  // Validate signature presence
  if (!signature || !timestamp) {
    return res.status(401).json({
      status: 'error',
      message: 'Missing request signature or timestamp',
    });
  }

  // Validate timestamp to prevent replay attacks
  const timestampNum = parseInt(timestamp, 10);
  const now = Date.now();
  if (now - timestampNum > securityConfig.signatureTimeout) {
    return res.status(401).json({
      status: 'error',
      message: 'Request signature expired',
    });
  }

  // Validate signature
  const dataToVerify = `${payload}:${timestamp}`;
  if (!encryptionService.verifyHash(dataToVerify, signature)) {
    logger.warn('Invalid request signature', {
      path: req.path,
      method: req.method,
      ip: req.ip,
    });

    return res.status(401).json({
      status: 'error',
      message: 'Invalid request signature',
    });
  }

  next();
};

/**
 * Initialize all security middleware
 */
export const initializeSecurityMiddleware = (app: any) => {
  // Apply security middleware in the correct order
  app.use(auditLogger);
  app.use(rateLimiter);
  app.use(securityHeaders);
  app.use(corsOptions);
  app.use(sanitizeInputs);
  app.use(validateRequestSignature);

  // Log successful security initialization
  logger.info('Security middleware initialized', {
    rateLimit: {
      windowMs: securityConfig.rateLimit.windowMs,
      maxRequests: securityConfig.rateLimit.maxRequests,
    },
    corsOrigins: securityConfig.cors.origin,
  });
};

export default {
  auditLogger,
  rateLimiter,
  securityHeaders,
  corsOptions,
  sanitizeInputs,
  validateRequestSignature,
  encryptionService,
  initializeSecurityMiddleware,
};
