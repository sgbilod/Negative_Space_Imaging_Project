/**
 * Input Validation Middleware
 *
 * Validates incoming request data using Joi schemas.
 * Provides middleware factory for schema-based validation.
 *
 * @module middleware/validation
 */

import { Response, NextFunction } from 'express';
import Joi from 'joi';
import { AuthRequest, ServiceError } from '../types';
import logger from '../utils/logger';

/**
 * Validation schemas for common inputs
 */
export const validationSchemas = {
  // User validation
  userRegister: Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().min(8).required(),
    first_name: Joi.string().min(2).required(),
    last_name: Joi.string().min(2).required(),
  }),

  userLogin: Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().required(),
  }),

  // Image validation
  imageUpload: Joi.object({
    filename: Joi.string().required(),
    original_filename: Joi.string().required(),
    file_size: Joi.number().min(1).required(),
    storage_path: Joi.string().required(),
  }),

  // Analysis result validation
  analysisResult: Joi.object({
    image_id: Joi.number().integer().required(),
    negative_space_percentage: Joi.number().min(0).max(100).required(),
    regions_count: Joi.number().integer().min(0).required(),
    processing_time_ms: Joi.number().min(0).required(),
    raw_data: Joi.object().required(),
  }),

  // Pagination validation
  pagination: Joi.object({
    page: Joi.number().integer().min(1).default(1),
    limit: Joi.number().integer().min(1).max(100).default(10),
  }),
};

/**
 * Validate request body
 * Factory function that returns middleware for a given schema
 */
export function validateBody(schema: Joi.ObjectSchema) {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    const { error, value } = schema.validate(req.body, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      logger.warn(`Validation error: ${error.message}`);
      const details = error.details.map((d) => ({
        field: d.path.join('.'),
        message: d.message,
      }));

      throw new ServiceError(400, 'Validation failed', 'VALIDATION_ERROR');
    }

    req.body = value;
    logger.debug('Request body validation passed');
    next();
  };
}

/**
 * Validate request query parameters
 */
export function validateQuery(schema: Joi.ObjectSchema) {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    const { error, value } = schema.validate(req.query, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      logger.warn(`Query validation error: ${error.message}`);
      throw new ServiceError(400, 'Query validation failed', 'QUERY_VALIDATION_ERROR');
    }

    req.query = value;
    logger.debug('Query validation passed');
    next();
  };
}

/**
 * Validate request parameters
 */
export function validateParams(schema: Joi.ObjectSchema) {
  return (req: AuthRequest, res: Response, next: NextFunction): void => {
    const { error, value } = schema.validate(req.params, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      logger.warn(`Params validation error: ${error.message}`);
      throw new ServiceError(400, 'Parameter validation failed', 'PARAMS_VALIDATION_ERROR');
    }

    req.params = value;
    logger.debug('Parameter validation passed');
    next();
  };
}

/**
 * Combined validation middleware
 */
export function validate(
  bodySchema?: Joi.ObjectSchema,
  querySchema?: Joi.ObjectSchema,
  paramsSchema?: Joi.ObjectSchema
) {
  return async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
    try {
      if (bodySchema && req.body) {
        const { error, value } = bodySchema.validate(req.body, {
          abortEarly: false,
          stripUnknown: true,
        });

        if (error) {
          throw new ServiceError(400, 'Body validation failed', 'VALIDATION_ERROR');
        }

        req.body = value;
      }

      if (querySchema && req.query) {
        const { error, value } = querySchema.validate(req.query, {
          abortEarly: false,
          stripUnknown: true,
        });

        if (error) {
          throw new ServiceError(400, 'Query validation failed', 'QUERY_VALIDATION_ERROR');
        }

        req.query = value;
      }

      if (paramsSchema && req.params) {
        const { error, value } = paramsSchema.validate(req.params, {
          abortEarly: false,
          stripUnknown: true,
        });

        if (error) {
          throw new ServiceError(400, 'Parameter validation failed', 'PARAMS_VALIDATION_ERROR');
        }

        req.params = value;
      }

      logger.debug('All validations passed');
      next();
    } catch (error) {
      throw error;
    }
  };
}
