import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';
import { config } from '../config';

/**
 * Custom error class for API errors
 */
export class ApiError extends Error {
  statusCode: number;
  isOperational: boolean;
  
  constructor(statusCode: number, message: string, isOperational = true) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    
    // Ensure the correct prototype chain is maintained
    Object.setPrototypeOf(this, ApiError.prototype);
    
    // Capture stack trace
    Error.captureStackTrace(this, this.constructor);
    
    // Log all non-operational errors (system/programming errors)
    if (!isOperational) {
      logger.error(`Non-operational error: ${message}`, { 
        statusCode, 
        stack: this.stack 
      });
    }
  }
}

/**
 * Global error handling middleware
 */
export const errorHandler = (
  err: Error | ApiError,
  req: Request,
  res: Response,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  next: NextFunction
): void => {
  // Default status code and error message
  let statusCode = 500;
  let message = 'Internal Server Error';
  let isOperational = false;
  
  // If the error is an instance of our ApiError class
  if (err instanceof ApiError) {
    statusCode = err.statusCode;
    message = err.message;
    isOperational = err.isOperational;
  }
  
  // Create a standardized error response
  const errorResponse = {
    status: 'error',
    message,
    ...(config.server.isDevelopment && { stack: err.stack }),
    ...(config.server.isDevelopment && !isOperational && { type: 'non-operational' }),
  };
  
  // Log error details
  const logMethod = statusCode >= 500 ? 'error' : 'warn';
  logger[logMethod](`${req.method} ${req.path} - ${statusCode}: ${message}`, {
    ip: req.ip,
    method: req.method,
    path: req.path,
    statusCode,
    userId: req.user?.id, // If authentication middleware adds user info
    requestId: req.headers['x-request-id'] || null,
    isOperational,
  });
  
  // If HIPAA is enabled, perform additional logging for compliance
  if (config.hipaa.enabled) {
    // Add to audit log for compliance
    logger.info('HIPAA Audit Log Entry', {
      action: 'ERROR',
      userId: req.user?.id || 'unauthenticated',
      resource: req.path,
      outcome: 'FAILURE',
      statusCode,
      timestamp: new Date().toISOString(),
    });
  }
  
  // Send error response
  res.status(statusCode).json(errorResponse);
};
