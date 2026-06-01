import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { User } from '../models/User';
import { ApiError } from '../middleware/errorHandler';
import { config } from '../config';
import { Security } from '../models/Security';
import { logger } from '../utils/logger';

/**
 * Authentication controller with HIPAA-compliant security measures
 */
export class AuthController {
  /**
   * Register a new user
   */
  public static async register(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { firstName, lastName, email, password, role } = req.body;

      // Check if user already exists
      const existingUser = await User.findOne({ where: { email } });
      if (existingUser) {
        throw new ApiError(409, 'User with this email already exists');
      }

      // Create new user
      const user = await User.create({
        firstName,
        lastName,
        email,
        password, // Will be automatically hashed by model hooks
        role: role || 'researcher',
      });

      // Audit log for HIPAA compliance
      await Security.createAuditLog({
        userId: user.id,
        action: 'REGISTER',
        resource: 'USER',
        resourceId: user.id,
        details: {
          email: user.email,
          role: user.role,
        },
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'] || 'unknown',
        outcome: 'SUCCESS',
      });

      // Return created user (without password)
      res.status(201).json({
        status: 'success',
        data: {
          user: user.toJSON(),
        },
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * User login
   */
  public static async login(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { email, password } = req.body;

      // Find user by email
      const user = await User.findOne({ where: { email } });
      if (!user) {
        throw new ApiError(401, 'Invalid credentials');
      }

      // Check if user is active
      if (!user.isActive) {
        throw new ApiError(403, 'Account is disabled');
      }

      // Verify password
      const isPasswordValid = await user.comparePassword(password);
      if (!isPasswordValid) {
        // Log failed login attempt for security
        await Security.createAuditLog({
          userId: user.id,
          action: 'LOGIN',
          resource: 'AUTH',
          resourceId: user.id,
          details: {
            email: user.email,
          },
          ipAddress: req.ip,
          userAgent: req.headers['user-agent'] || 'unknown',
          outcome: 'FAILURE',
        });
        
        throw new ApiError(401, 'Invalid credentials');
      }

      // Update last login timestamp
      user.lastLogin = new Date();
      await user.save();

      // Generate JWT token
      const token = jwt.sign(
        { id: user.id, email: user.email, role: user.role },
        config.security.jwtSecret,
        { expiresIn: config.security.jwtExpiresIn }
      );

      // Log successful login for HIPAA compliance
      await Security.createAuditLog({
        userId: user.id,
        action: 'LOGIN',
        resource: 'AUTH',
        resourceId: user.id,
        details: {
          email: user.email,
          role: user.role,
        },
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'] || 'unknown',
        outcome: 'SUCCESS',
      });

      // Send token in response
      res.status(200).json({
        status: 'success',
        data: {
          user: user.toJSON(),
          token,
          expiresIn: config.security.jwtExpiresIn,
        },
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * User logout
   */
  public static async logout(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      // Log logout for HIPAA compliance
      if (req.user) {
        await Security.createAuditLog({
          userId: req.user.id,
          action: 'LOGOUT',
          resource: 'AUTH',
          resourceId: req.user.id,
          details: {},
          ipAddress: req.ip,
          userAgent: req.headers['user-agent'] || 'unknown',
          outcome: 'SUCCESS',
        });
      }

      res.status(200).json({
        status: 'success',
        message: 'Logged out successfully',
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * Get current user profile
   */
  public static async getCurrentUser(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      if (!req.user) {
        throw new ApiError(401, 'Not authenticated');
      }

      const user = await User.findByPk(req.user.id);
      if (!user) {
        throw new ApiError(404, 'User not found');
      }

      res.status(200).json({
        status: 'success',
        data: {
          user: user.toJSON(),
        },
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * Change password
   */
  public static async changePassword(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      if (!req.user) {
        throw new ApiError(401, 'Not authenticated');
      }

      const { currentPassword, newPassword } = req.body;

      // Find user
      const user = await User.findByPk(req.user.id);
      if (!user) {
        throw new ApiError(404, 'User not found');
      }

      // Verify current password
      const isPasswordValid = await user.comparePassword(currentPassword);
      if (!isPasswordValid) {
        throw new ApiError(401, 'Current password is incorrect');
      }

      // Update password
      user.password = newPassword; // Will be automatically hashed by model hooks
      await user.save();

      // Log password change for HIPAA compliance
      await Security.createAuditLog({
        userId: user.id,
        action: 'PASSWORD_CHANGE',
        resource: 'USER',
        resourceId: user.id,
        details: {},
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'] || 'unknown',
        outcome: 'SUCCESS',
      });

      res.status(200).json({
        status: 'success',
        message: 'Password changed successfully',
      });
    } catch (error) {
      next(error);
    }
  }

  /**
   * Request password reset
   */
  public static async requestPasswordReset(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { email } = req.body;

      // Find user by email
      const user = await User.findOne({ where: { email } });
      
      // Even if user doesn't exist, return success to prevent email enumeration
      if (!user) {
        logger.debug(`Password reset requested for non-existent email: ${email}`);
        res.status(200).json({
          status: 'success',
          message: 'If your email is registered, you will receive a password reset link',
        });
        return;
      }

      // Generate reset token
      const resetToken = Security.generateSecureToken();
      
      // Store token with expiry (implementation depends on your storage solution)
      // ...

      // Send reset email (implementation depends on your email service)
      // ...

      // Log password reset request for HIPAA compliance
      await Security.createAuditLog({
        userId: user.id,
        action: 'PASSWORD_RESET_REQUEST',
        resource: 'USER',
        resourceId: user.id,
        details: {
          email: user.email,
        },
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'] || 'unknown',
        outcome: 'SUCCESS',
      });

      res.status(200).json({
        status: 'success',
        message: 'If your email is registered, you will receive a password reset link',
      });
    } catch (error) {
      next(error);
    }
  }
}
