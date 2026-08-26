/**
 * Authentication Routes Module
 *
 * Defines authentication endpoints for user registration, login,
 * logout, and token refresh operations.
 *
 * @module routes/auth
 */

import express, { Router, Request, Response, NextFunction } from 'express';
import UserService from '../services/userService';
import { validateRequest } from '../middleware/validation';
import { authenticateToken } from '../middleware/auth';
import {
  generateTokenPair,
  extractTokenFromHeader,
  verifyRefreshToken,
} from '../utils/tokenUtils';
import { ServiceError } from '../types';
import logger from '../utils/logger';
import Joi from 'joi';

const router: Router = express.Router();

/**
 * Validation schemas
 */
const registerSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().min(8).required(),
  first_name: Joi.string().required(),
  last_name: Joi.string().required(),
});

const loginSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().required(),
});

const refreshSchema = Joi.object({
  refreshToken: Joi.string().required(),
});

/**
 * POST /auth/register
 * Register a new user
 */
router.post(
  '/register',
  validateRequest(registerSchema),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { email, password, first_name, last_name } = req.body;

      logger.info(`Registration attempt for: ${email}`);

      const result = await UserService.registerUser(
        email,
        password,
        first_name,
        last_name
      );

      const { user, token } = result;

      // Generate token pair
      const tokens = generateTokenPair({
        id: user.id,
        email: user.email,
      });

      logger.info(`User registered successfully: ${user.id}`);

      res.status(201).json({
        success: true,
        data: {
          user,
          tokens,
        },
        message: 'Registration successful',
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * POST /auth/login
 * Login user with email and password
 */
router.post(
  '/login',
  validateRequest(loginSchema),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { email, password } = req.body;

      logger.info(`Login attempt for: ${email}`);

      const result = await UserService.loginUser(email, password);
      const { user, token } = result;

      // Generate token pair
      const tokens = generateTokenPair({
        id: user.id,
        email: user.email,
      });

      logger.info(`User logged in: ${user.id}`);

      res.status(200).json({
        success: true,
        data: {
          user,
          tokens,
        },
        message: 'Login successful',
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * POST /auth/refresh
 * Refresh access token using refresh token
 */
router.post(
  '/refresh',
  validateRequest(refreshSchema),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { refreshToken } = req.body;

      logger.debug('Token refresh attempt');

      // Verify refresh token
      const decoded = verifyRefreshToken(refreshToken);

      // Generate new token pair
      const tokens = generateTokenPair({
        id: decoded.id,
        email: decoded.email,
      });

      logger.info(`Token refreshed for user: ${decoded.id}`);

      res.status(200).json({
        success: true,
        data: { tokens },
        message: 'Token refreshed',
      });
    } catch (error) {
      logger.warn('Token refresh failed', error);
      res.status(401).json({
        success: false,
        error: {
          code: 'INVALID_REFRESH_TOKEN',
          message: 'Invalid or expired refresh token',
        },
      });
    }
  }
);

/**
 * POST /auth/logout
 * Logout user (token invalidation on client)
 */
router.post(
  '/logout',
  authenticateToken,
  (req: Request, res: Response) => {
    try {
      const userId = (req as any).user?.id;
      logger.info(`User logged out: ${userId}`);

      res.status(200).json({
        success: true,
        message: 'Logout successful',
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'LOGOUT_ERROR',
          message: 'Logout failed',
        },
      });
    }
  }
);

/**
 * POST /auth/verify
 * Verify current token
 */
router.post('/verify', authenticateToken, (req: Request, res: Response) => {
  const user = (req as any).user;

  res.status(200).json({
    success: true,
    data: { user },
    message: 'Token is valid',
  });
});

/**
 * GET /auth/me
 * Get current user info
 */
router.get('/me', authenticateToken, async (req: Request, res: Response, next: NextFunction) => {
  try {
    const userId = (req as any).user?.id;

    if (!userId) {
      return res.status(401).json({
        success: false,
        error: {
          code: 'UNAUTHORIZED',
          message: 'User ID not found in token',
        },
      });
    }

    const user = await UserService.getUserById(userId);

    res.status(200).json({
      success: true,
      data: { user },
    });
  } catch (error) {
    if (error instanceof ServiceError) {
      return res.status(error.statusCode).json({
        success: false,
        error: {
          code: error.code,
          message: error.message,
        },
      });
    }
    next(error);
  }
});

export default router;
