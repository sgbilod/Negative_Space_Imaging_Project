/**
 * Custom Error Classes
 * Standardized error handling across application
 */

/**
 * Base application error
 * All custom errors should extend this class
 */
export class AppError extends Error {
  constructor(
    public statusCode: number,
    public message: string,
    public details?: Record<string, any>,
  ) {
    super(message);
    Object.setPrototypeOf(this, AppError.prototype);
  }
}

/**
 * Validation error for invalid input
 */
export class ValidationError extends AppError {
  constructor(
    message: string = 'Validation failed',
    public fields?: Array<{ field: string; message: string }>,
    details?: Record<string, any>,
  ) {
    super(400, message, details);
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}

/**
 * Authentication error for invalid credentials
 */
export class AuthenticationError extends AppError {
  constructor(message: string = 'Authentication failed', details?: Record<string, any>) {
    super(401, message, details);
    Object.setPrototypeOf(this, AuthenticationError.prototype);
  }
}

/**
 * Authorization error for insufficient permissions
 */
export class AuthorizationError extends AppError {
  constructor(message: string = 'Access denied', details?: Record<string, any>) {
    super(403, message, details);
    Object.setPrototypeOf(this, AuthorizationError.prototype);
  }
}

/**
 * Not found error for missing resources
 */
export class NotFoundError extends AppError {
  constructor(resource: string = 'Resource', details?: Record<string, any>) {
    super(404, `${resource} not found`, details);
    Object.setPrototypeOf(this, NotFoundError.prototype);
  }
}

/**
 * Conflict error for duplicate resources
 */
export class ConflictError extends AppError {
  constructor(message: string = 'Resource already exists', details?: Record<string, any>) {
    super(409, message, details);
    Object.setPrototypeOf(this, ConflictError.prototype);
  }
}

/**
 * Rate limit error for too many requests
 */
export class RateLimitError extends AppError {
  constructor(message: string = 'Too many requests', details?: Record<string, any>) {
    super(429, message, details);
    Object.setPrototypeOf(this, RateLimitError.prototype);
  }
}

/**
 * Service unavailable error
 */
export class ServiceUnavailableError extends AppError {
  constructor(service: string = 'Service', details?: Record<string, any>) {
    super(503, `${service} is temporarily unavailable`, details);
    Object.setPrototypeOf(this, ServiceUnavailableError.prototype);
  }
}

/**
 * File handling error
 */
export class FileError extends AppError {
  constructor(message: string = 'File operation failed', details?: Record<string, any>) {
    super(400, message, details);
    Object.setPrototypeOf(this, FileError.prototype);
  }
}

/**
 * Database error
 */
export class DatabaseError extends AppError {
  constructor(message: string = 'Database operation failed', details?: Record<string, any>) {
    super(500, message, details);
    Object.setPrototypeOf(this, DatabaseError.prototype);
  }
}

/**
 * Check if error is an AppError
 */
export function isAppError(error: unknown): error is AppError {
  return error instanceof AppError;
}

/**
 * Convert error to AppError
 */
export function toAppError(error: unknown): AppError {
  if (isAppError(error)) {
    return error;
  }

  if (error instanceof Error) {
    return new AppError(500, error.message);
  }

  return new AppError(500, 'Unknown error occurred');
}
