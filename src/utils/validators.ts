/**
 * Validation Utilities
 * Request and data validation logic
 */

import Joi from 'joi';
import { ValidationError } from './errors';

/**
 * Validation result interface
 */
export interface ValidationResult<T> {
  isValid: boolean;
  data?: T;
  errors?: Array<{ field: string; message: string }>;
}

/**
 * Validate data against Joi schema
 * @param data - Data to validate
 * @param schema - Joi validation schema
 * @returns Validation result with errors if any
 */
export function validateData<T>(data: unknown, schema: Joi.ObjectSchema): ValidationResult<T> {
  const { error, value } = schema.validate(data, {
    abortEarly: false,
    allowUnknown: false,
  });

  if (error) {
    const errors = error.details.map((detail) => ({
      field: detail.path.join('.'),
      message: detail.message,
    }));

    return {
      isValid: false,
      errors,
    };
  }

  return {
    isValid: true,
    data: value as T,
  };
}

/**
 * Assert validation result and throw error if invalid
 * @param result - Validation result
 * @throws ValidationError if validation failed
 */
export function assertValidation<T>(result: ValidationResult<T>): T {
  if (!result.isValid) {
    throw new ValidationError('Validation failed', result.errors);
  }

  return result.data as T;
}

/**
 * Email validation schema
 */
export const emailSchema = Joi.string().email().lowercase().required().max(255);

/**
 * Password validation schema (min 12 chars)
 */
export const passwordSchema = Joi.string().min(12).required().messages({
  'string.min': 'Password must be at least 12 characters',
});

/**
 * Strong password validation (special requirements)
 */
export const strongPasswordSchema = Joi.string()
  .min(12)
  .pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/)
  .required()
  .messages({
    'string.pattern.base':
      'Password must contain uppercase, lowercase, number, and special character',
  });

/**
 * Name validation schema
 */
export const nameSchema = Joi.string().min(1).max(50).required();

/**
 * URL validation schema
 */
export const urlSchema = Joi.string().uri().required();

/**
 * File type validation
 */
export interface FileValidationOptions {
  maxSize?: number;
  allowedMimes?: string[];
  allowedExtensions?: string[];
}

/**
 * Validate file properties
 * @param filename - Original filename
 * @param mimeType - MIME type of file
 * @param fileSize - File size in bytes
 * @param options - Validation options
 * @throws ValidationError if file is invalid
 */
export function validateFile(
  filename: string,
  mimeType: string,
  fileSize: number,
  options: FileValidationOptions,
): void {
  const errors: Array<{ field: string; message: string }> = [];

  // Check file size
  if (options.maxSize && fileSize > options.maxSize) {
    errors.push({
      field: 'file',
      message: `File size exceeds maximum of ${options.maxSize} bytes`,
    });
  }

  // Check MIME type
  if (options.allowedMimes && !options.allowedMimes.includes(mimeType)) {
    errors.push({
      field: 'file',
      message: `File type ${mimeType} is not allowed`,
    });
  }

  // Check extension
  if (options.allowedExtensions) {
    const ext = filename.substring(filename.lastIndexOf('.')).toLowerCase();
    if (!options.allowedExtensions.includes(ext)) {
      errors.push({
        field: 'file',
        message: `File extension ${ext} is not allowed`,
      });
    }
  }

  if (errors.length > 0) {
    throw new ValidationError('File validation failed', errors);
  }
}

/**
 * Pagination validation schema
 */
export const paginationSchema = Joi.object({
  limit: Joi.number().integer().min(1).max(100).default(20),
  offset: Joi.number().integer().min(0).default(0),
  sort: Joi.string().optional(),
  order: Joi.string().valid('asc', 'desc').default('desc'),
});

/**
 * Validate email format
 * @param email - Email to validate
 * @throws ValidationError if invalid
 */
export function validateEmail(email: string): void {
  const { error } = emailSchema.validate(email);
  if (error) {
    throw new ValidationError('Invalid email format');
  }
}

/**
 * Validate password strength
 * @param password - Password to validate
 * @param strong - Require strong password
 * @throws ValidationError if invalid
 */
export function validatePassword(password: string, strong = false): void {
  const schema = strong ? strongPasswordSchema : passwordSchema;
  const { error } = schema.validate(password);
  if (error) {
    throw new ValidationError(error.message);
  }
}

/**
 * Sanitize string input (basic XSS prevention)
 * @param input - String to sanitize
 * @returns Sanitized string
 */
export function sanitizeInput(input: string): string {
  return input
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/[&]/g, '&amp;') // Escape ampersand
    .trim();
}

/**
 * Validate JSON Web Token format
 * @param token - Token to validate
 * @throws ValidationError if invalid
 */
export function validateTokenFormat(token: string): void {
  const parts = token.split('.');
  if (parts.length !== 3) {
    throw new ValidationError('Invalid token format');
  }

  try {
    Buffer.from(parts[0], 'base64').toString('utf-8');
    Buffer.from(parts[1], 'base64').toString('utf-8');
  } catch {
    throw new ValidationError('Invalid token encoding');
  }
}

/**
 * Validate numeric ID
 * @param id - ID to validate
 * @throws ValidationError if invalid
 */
export function validateId(id: string | unknown): void {
  if (typeof id !== 'string' || id.trim().length === 0) {
    throw new ValidationError('Invalid ID format');
  }
}

/**
 * Validate object has required fields
 * @param obj - Object to validate
 * @param requiredFields - Required field names
 * @throws ValidationError if missing required fields
 */
export function validateRequiredFields(
  obj: Record<string, unknown>,
  requiredFields: string[],
): void {
  const missing = requiredFields.filter((field) => !(field in obj));
  if (missing.length > 0) {
    throw new ValidationError(
      'Missing required fields',
      missing.map((field) => ({
        field,
        message: `${field} is required`,
      })),
    );
  }
}
