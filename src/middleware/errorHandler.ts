/**
 * Error Handler Middleware
 * Global error handling for Express application
 */

import { Request, Response, NextFunction } from 'express';
import { log } from '../services/loggingService';
import { AuthenticatedRequest } from './authMiddleware';

/**
 * Custom application error class
 */
export class AppError extends Error {
  constructor(
    public statusCode: number = 500,
    message: string = 'Internal server error',
    public isOperational: boolean = true,
  ) {
    super(message);
    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * Validation error class
 */
export class ValidationError extends AppError {
  constructor(
    public errors: any[],
    message: string = 'Validation failed',
  ) {
    super(400, message);
  }
}

/**
 * Authentication error class
 */
export class AuthenticationError extends AppError {
  constructor(message: string = 'Authentication failed') {
    super(401, message);
  }
}

/**
 * Authorization error class
 */
export class AuthorizationError extends AppError {
  constructor(message: string = 'Insufficient permissions') {
    super(403, message);
  }
}

/**
 * Not found error class
 */
export class NotFoundError extends AppError {
  constructor(resource: string = 'Resource') {
    super(404, `${resource} not found`);
  }
}

/**
 * Conflict error class
 */
export class ConflictError extends AppError {
  constructor(message: string = 'Resource already exists') {
    super(409, message);
  }
}

/**
 * Rate limit error class
 */
export class RateLimitError extends AppError {
  constructor(message: string = 'Too many requests') {
    super(429, message);
  }
}

/**
 * Service unavailable error class
 */
export class ServiceUnavailableError extends AppError {
  constructor(service: string = 'Service') {
    super(503, `${service} is temporarily unavailable`);
  }
}

/**
 * Format error response
 */
interface ErrorResponse {
  error: {
    message: string;
    statusCode: number;
    requestId?: string;
    timestamp: string;
    path?: string;
    details?: any;
    stack?: string;
  };
}

/**
 * Global error handler middleware
 * Must be the last middleware registered
 */
export const errorHandler = (
  error: Error | AppError,
  req: AuthenticatedRequest,
  res: Response,
  _next: NextFunction,
): void => {
  const isDevelopment = process.env.NODE_ENV !== 'production';

  let appError: AppError;

  // Convert error to AppError if necessary
  if (error instanceof AppError) {
    appError = error;
  } else if (error instanceof SyntaxError && 'body' in error) {
    // JSON parse error
    appError = new AppError(400, 'Invalid JSON in request body');
  } else {
    // Unknown error
    appError = new AppError(500, 'Internal server error', false);

    log.error('Unhandled error', error as Error, {
      requestId: req.requestId,
      path: req.path,
      method: req.method,
      userId: req.userId,
    });
  }

  // Prepare error response
  const errorResponse: ErrorResponse = {
    error: {
      message: appError.message,
      statusCode: appError.statusCode,
      requestId: req.requestId,
      timestamp: new Date().toISOString(),
      path: req.path,
    },
  };

  // Add details for validation errors
  if (error instanceof ValidationError) {
    errorResponse.error.details = error.errors;
  }

  // Add stack trace in development
  if (isDevelopment && process.env.LOG_LEVEL === 'debug') {
    errorResponse.error.stack = appError.stack;
  }

  // Log the error
  if (appError.statusCode >= 500) {
    log.error(
      `[${appError.statusCode}] ${appError.message}`,
      error instanceof AppError ? new Error(error.message) : (error as Error),
      {
        requestId: req.requestId,
        userId: req.userId,
        path: req.path,
        method: req.method,
        statusCode: appError.statusCode,
      },
    );

    log.security('Server error', 'medium', {
      requestId: req.requestId,
      statusCode: appError.statusCode,
      message: appError.message,
    });
  } else if (appError.statusCode >= 400) {
    log.warn(`[${appError.statusCode}] ${appError.message}`, {
      requestId: req.requestId,
      userId: req.userId,
      path: req.path,
      method: req.method,
      statusCode: appError.statusCode,
    });
  }

  // Send error response
  res.status(appError.statusCode).json(errorResponse);
};

/**
 * Async error handler wrapper for route handlers
 * Catches errors in async functions and passes them to error handler
 */
export const asyncHandler = (
  fn: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<any>,
) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

/**
 * 404 Not Found handler
 * Should be last middleware before error handler
 */
export const notFoundHandler = (req: Request, res: Response): void => {
  const message = `Cannot ${req.method} ${req.path}`;

  log.warn(`404 Not Found: ${message}`, {
    method: req.method,
    path: req.path,
    ip: req.ip,
  });

  res.status(404).json({
    error: {
      message,
      statusCode: 404,
      timestamp: new Date().toISOString(),
      path: req.path,
    },
  });
};

/**
 * Keep original ApiError for backward compatibility
 */
export class ApiError extends AppError {
  constructor(statusCode: number, message: string, isOperational = true) {
    super(statusCode, message, isOperational);
  }
}
