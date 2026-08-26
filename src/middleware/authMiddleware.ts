/**
 * Authentication and Authorization Middleware
 * JWT verification, role-based access control, and request validation
 */

import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { configService } from '../config/serverConfig';
import { log } from '../services/loggingService';
import { v4 as uuidv4 } from 'uuid';

/**
 * Extended Express Request with user context
 */
export interface AuthenticatedRequest extends Request {
  userId?: string;
  user?: {
    id: string;
    email: string;
    roles: string[];
  };
  requestId?: string;
  startTime?: number;
}

/**
 * JWT payload interface
 */
interface JWTPayload {
  id: string;
  email: string;
  roles: string[];
  iat?: number;
  exp?: number;
}

/**
 * Request ID middleware - adds unique ID to each request for tracking
 */
export const requestIdMiddleware = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): void => {
  const requestId = (req.headers['x-request-id'] as string) || uuidv4();
  req.requestId = requestId;
  req.startTime = Date.now();

  // Add request ID to response headers
  res.setHeader('X-Request-ID', requestId);

  next();
};

/**
 * Request logging middleware - logs all incoming requests
 */
export const requestLoggingMiddleware = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): void => {
  const config = configService.getConfig();

  if (!config.logging.requestLogging) {
    next();
    return;
  }

  // Log request
  log.debug(`Incoming ${req.method} ${req.path}`, {
    requestId: req.requestId,
    method: req.method,
    path: req.path,
    ip: req.ip,
  });

  // Log response when it's sent
  const originalSend = res.send;
  res.send = function (data: any): Response {
    const duration = req.startTime ? Date.now() - req.startTime : 0;
    const statusCode = res.statusCode;

    log.request(req.method, req.path, statusCode, duration, req.requestId || 'unknown');

    return originalSend.call(this, data);
  };

  next();
};

/**
 * JWT authentication middleware
 * Verifies JWT token and extracts user information
 */
export const authenticateToken = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): void => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

    if (!token) {
      log.security('Missing JWT token', 'high', {
        requestId: req.requestId,
        path: req.path,
      });

      res.status(401).json({
        error: 'Missing authentication token',
        requestId: req.requestId,
      });
      return;
    }

    const config = configService.getConfig();
    const decoded = jwt.verify(token, config.jwt.secret) as JWTPayload;

    req.user = {
      id: decoded.id,
      email: decoded.email,
      roles: decoded.roles || [],
    };

    req.userId = decoded.id;

    log.auth('Token verified', decoded.id, true, {
      requestId: req.requestId,
    });

    next();
  } catch (error) {
    log.security('JWT verification failed', 'medium', {
      requestId: req.requestId,
      error: (error as Error).message,
    });

    if (error instanceof jwt.TokenExpiredError) {
      res.status(401).json({
        error: 'Token expired',
        requestId: req.requestId,
      });
      return;
    }

    res.status(403).json({
      error: 'Invalid token',
      requestId: req.requestId,
    });
  }
};

/**
 * Authorization middleware - role-based access control
 * @param allowedRoles - Array of roles that are allowed to access this route
 */
export const authorize = (allowedRoles: string[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        error: 'Authentication required',
        requestId: req.requestId,
      });
      return;
    }

    const userRoles = req.user.roles || [];
    const hasRequiredRole = allowedRoles.some((role) => userRoles.includes(role));

    if (!hasRequiredRole) {
      log.security('Unauthorized access attempt', 'medium', {
        requestId: req.requestId,
        userId: req.userId,
        requiredRoles: allowedRoles,
        userRoles,
        path: req.path,
      });

      res.status(403).json({
        error: 'Insufficient permissions',
        requiredRoles: allowedRoles,
        requestId: req.requestId,
      });
      return;
    }

    log.auth(`Access granted for role: ${userRoles.join(', ')}`, req.userId, true, {
      requestId: req.requestId,
      path: req.path,
    });

    next();
  };
};

/**
 * Optional authentication middleware - doesn't fail if token is missing
 */
export const optionalAuth = (
  req: AuthenticatedRequest,
  _res: Response,
  next: NextFunction,
): void => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (token) {
      const config = configService.getConfig();
      const decoded = jwt.verify(token, config.jwt.secret) as JWTPayload;

      req.user = {
        id: decoded.id,
        email: decoded.email,
        roles: decoded.roles || [],
      };

      req.userId = decoded.id;
    }
  } catch (error) {
    log.debug('Optional authentication failed, continuing without user', {
      requestId: req.requestId,
    });
  }

  next();
};

/**
 * Request validation middleware
 * Validates request body against schema
 */
export const validateRequest = (schema: any) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    try {
      const { error, value } = schema.validate(req.body, {
        abortEarly: false,
        stripUnknown: true,
      });

      if (error) {
        const validationErrors = error.details.map((detail: any) => ({
          field: detail.path.join('.'),
          message: detail.message,
        }));

        log.warn('Request validation failed', {
          requestId: req.requestId,
          errors: validationErrors,
        });

        res.status(400).json({
          error: 'Validation failed',
          details: validationErrors,
          requestId: req.requestId,
        });
        return;
      }

      // Replace body with validated data
      req.body = value;
      next();
    } catch (error) {
      log.error('Request validation error', error as Error, {
        requestId: req.requestId,
      });

      res.status(500).json({
        error: 'Internal validation error',
        requestId: req.requestId,
      });
    }
  };
};

/**
 * Combine multiple validation schemas
 */
export const combineSchemas = (schemas: any[]): any => {
  return {
    validate: (data: any, options: any = {}) => {
      let value = data;
      const errors: any[] = [];

      for (const schema of schemas) {
        const result = schema.validate(value, { ...options, abortEarly: false });
        if (result.error) {
          errors.push(...result.error.details);
        }
        value = result.value;
      }

      if (errors.length > 0) {
        return {
          error: {
            details: errors,
            message: `Validation failed with ${errors.length} errors`,
          },
          value,
        };
      }

      return { value };
    },
  };
};

export default {
  requestIdMiddleware,
  requestLoggingMiddleware,
  authenticateToken,
  authorize,
  optionalAuth,
  validateRequest,
  combineSchemas,
};
