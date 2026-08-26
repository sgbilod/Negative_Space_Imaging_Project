/**
 * Request Logging Middleware
 *
 * Logs HTTP requests with method, path, status code, and response time.
 * Provides visibility into API usage and performance.
 *
 * @module middleware/requestLogger
 */

import { Request, Response, NextFunction } from 'express';
import logger from '../utils/logger';

/**
 * Request logging middleware
 * Logs request details and response information
 */
export function requestLogger(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const startTime = Date.now();
  const requestId = generateRequestId();

  // Add request ID to response headers
  res.setHeader('X-Request-ID', requestId);

  // Store original send function
  const originalSend = res.send;

  // Override send to log response
  res.send = function (data: unknown): Response {
    const responseTime = Date.now() - startTime;
    const statusCode = res.statusCode;

    // Log the request
    logRequest({
      requestId,
      method: req.method,
      path: req.path,
      query: req.query,
      statusCode,
      responseTime,
      userAgent: req.get('user-agent'),
      ip: req.ip,
    });

    // Call original send
    return originalSend.call(this, data);
  };

  next();
}

/**
 * Detailed request logger interface
 */
interface RequestLog {
  requestId: string;
  method: string;
  path: string;
  query: Record<string, unknown>;
  statusCode: number;
  responseTime: number;
  userAgent?: string;
  ip?: string;
}

/**
 * Log request details
 */
function logRequest(log: RequestLog): void {
  const logLevel = log.statusCode >= 400 ? 'warn' : 'info';

  logger[logLevel](
    `${log.method} ${log.path} - ${log.statusCode} (${log.responseTime}ms)`,
    {
      requestId: log.requestId,
      method: log.method,
      path: log.path,
      statusCode: log.statusCode,
      responseTime: `${log.responseTime}ms`,
      query: Object.keys(log.query).length > 0 ? log.query : undefined,
      userAgent: log.userAgent,
      ip: log.ip,
    }
  );
}

/**
 * Generate unique request ID
 */
function generateRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Verbose request logger
 * Logs request body and headers (for development)
 */
export function verboseRequestLogger(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  if (process.env.NODE_ENV === 'development') {
    logger.debug(`${req.method} ${req.path}`, {
      headers: sanitizeHeaders(req.headers),
      body: sanitizeBody(req.body),
      query: req.query,
    });
  }

  next();
}

/**
 * Sanitize headers for logging
 * Removes sensitive information
 */
function sanitizeHeaders(headers: Record<string, unknown>): Record<string, unknown> {
  const sanitized = { ...headers };
  const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key'];

  sensitiveHeaders.forEach((header) => {
    if (sanitized[header]) {
      sanitized[header] = '[REDACTED]';
    }
  });

  return sanitized;
}

/**
 * Sanitize request body for logging
 * Removes sensitive information
 */
function sanitizeBody(body: unknown): unknown {
  if (!body || typeof body !== 'object') {
    return body;
  }

  const sanitized = { ...body as Record<string, unknown> };
  const sensitiveFields = ['password', 'password_hash', 'token', 'secret'];

  sensitiveFields.forEach((field) => {
    if (field in sanitized) {
      sanitized[field] = '[REDACTED]';
    }
  });

  return sanitized;
}

/**
 * Request performance tracker
 * Tracks slow requests
 */
export function performanceTracker(slowRequestThreshold: number = 1000) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const startTime = Date.now();

    res.on('finish', () => {
      const responseTime = Date.now() - startTime;

      if (responseTime > slowRequestThreshold) {
        logger.warn(`Slow request detected: ${req.method} ${req.path} (${responseTime}ms)`);
      }
    });

    next();
  };
}
