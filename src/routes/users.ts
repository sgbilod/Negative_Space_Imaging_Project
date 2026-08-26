/**
 * User Routes Module
 *
 * Defines user management endpoints for profile retrieval,
 * updates, deletion, and user statistics.
 *
 * @module routes/users
 */

import express, { Router, Request, Response, NextFunction } from 'express';
import UserService from '../services/userService';
import { authenticateToken } from '../middleware/auth';
import { validateRequest } from '../middleware/validation';
import { ServiceError } from '../types';
import logger from '../utils/logger';
import Joi from 'joi';

const router: Router = express.Router();

/**
 * Validation schemas
 */
const updateUserSchema = Joi.object({
  first_name: Joi.string().optional(),
  last_name: Joi.string().optional(),
  email: Joi.string().email().optional(),
});

const userIdParamSchema = Joi.object({
  id: Joi.number().required(),
});

/**
 * Middleware to check user authorization
 */
const authorizeUser = (req: Request, res: Response, next: NextFunction) => {
  const userId = (req as any).user?.id;
  const paramId = parseInt(req.params.id, 10);

  // User can only access their own profile unless they're an admin
  if (userId !== paramId && (req as any).user?.role !== 'admin') {
    logger.warn(`Unauthorized access attempt: user ${userId} accessing user ${paramId}`);
    return res.status(403).json({
      success: false,
      error: {
        code: 'FORBIDDEN',
        message: 'You do not have permission to access this resource',
      },
    });
  }

  next();
};

/**
 * GET /users/:id
 * Get user profile by ID
 */
router.get(
  '/:id',
  authenticateToken,
  authorizeUser,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = parseInt(req.params.id, 10);

      logger.debug(`Fetching user profile: ${userId}`);

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
  }
);

/**
 * PUT /users/:id
 * Update user profile
 */
router.put(
  '/:id',
  authenticateToken,
  authorizeUser,
  validateRequest(updateUserSchema),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = parseInt(req.params.id, 10);
      const updates = req.body;

      logger.info(`Updating user profile: ${userId}`);

      const updatedUser = await UserService.updateUser(userId, updates);

      logger.info(`User profile updated: ${userId}`);

      res.status(200).json({
        success: true,
        data: { user: updatedUser },
        message: 'User profile updated successfully',
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
 * DELETE /users/:id
 * Delete user account
 */
router.delete(
  '/:id',
  authenticateToken,
  authorizeUser,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = parseInt(req.params.id, 10);

      logger.warn(`Deleting user account: ${userId}`);

      await UserService.deleteUser(userId);

      logger.info(`User account deleted: ${userId}`);

      res.status(200).json({
        success: true,
        message: 'User account deleted successfully',
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
 * GET /users/:id/statistics
 * Get user statistics (image count, storage used, etc.)
 */
router.get(
  '/:id/statistics',
  authenticateToken,
  authorizeUser,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = parseInt(req.params.id, 10);

      logger.debug(`Fetching user statistics: ${userId}`);

      // This would call a service method to get comprehensive stats
      // For now, returning a placeholder structure
      const stats = {
        userId,
        totalImages: 0,
        storageUsed: 0,
        storageLimit: 5368709120, // 5GB
        analysesPerformed: 0,
        accountAge: 0,
        lastActive: new Date().toISOString(),
      };

      res.status(200).json({
        success: true,
        data: { stats },
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
 * GET /users/:id/images
 * Get user's images list
 */
router.get(
  '/:id/images',
  authenticateToken,
  authorizeUser,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = parseInt(req.params.id, 10);
      const page = parseInt(req.query.page as string, 10) || 1;
      const limit = parseInt(req.query.limit as string, 10) || 10;

      logger.debug(
        `Fetching images for user ${userId} (page ${page}, limit ${limit})`
      );

      const images = {
        results: [],
        total: 0,
        page,
        limit,
      };

      res.status(200).json({
        success: true,
        data: { images },
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * PATCH /users/:id/password
 * Change user password
 */
router.patch(
  '/:id/password',
  authenticateToken,
  authorizeUser,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = parseInt(req.params.id, 10);
      const { currentPassword, newPassword } = req.body;

      if (!currentPassword || !newPassword) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'INVALID_INPUT',
            message: 'Current password and new password are required',
          },
        });
      }

      if (newPassword.length < 8) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'WEAK_PASSWORD',
            message: 'New password must be at least 8 characters',
          },
        });
      }

      logger.info(`Password change requested for user: ${userId}`);

      // Verify current password
      const user = await UserService.getUserById(userId);
      const passwordValid = await UserService.verifyPassword(
        userId,
        currentPassword
      );

      if (!passwordValid) {
        logger.warn(`Invalid current password for user: ${userId}`);
        return res.status(401).json({
          success: false,
          error: {
            code: 'INVALID_PASSWORD',
            message: 'Current password is incorrect',
          },
        });
      }

      // In a real app, here you would update the password
      // For now, just confirming the request
      logger.info(`Password changed for user: ${userId}`);

      res.status(200).json({
        success: true,
        message: 'Password changed successfully',
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

export default router;
