/**
 * Authentication Controller
 * Handles HTTP layer for authentication endpoints
 * Bridges routes with authService business logic
 */

import { Request, Response } from 'express';
import Joi from 'joi';
import { AuthenticatedRequest } from '../middleware/authMiddleware';
import { asyncHandler, ValidationError } from '../middleware/errorHandler';
import { authService } from '../services/authService';
import { log } from '../services/loggingService';
import { validateData, assertValidation } from '../utils/validators';

/**
 * Request body types for validation
 */
interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
}

interface LoginRequest {
  email: string;
  password: string;
}

interface RefreshRequest {
  refreshToken: string;
}

/**
 * Validation schemas for controller layer
 */
const registerSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().min(12).required(),
  firstName: Joi.string().min(1).max(50).required(),
  lastName: Joi.string().min(1).max(50).required(),
});

const loginSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().required(),
});

const refreshSchema = Joi.object({
  refreshToken: Joi.string().required(),
});

/**
 * Authentication Controller
 * Static methods for auth endpoint handlers
 */
export class AuthController {
  /**
   * Register new user
   * POST /auth/register
   *
   * @param req - Express Request with body {email, password, firstName, lastName}
   * @param res - Express Response
   */
  static register = asyncHandler(async (req: Request, res: Response) => {
    // Validate request body
    const result = validateData<RegisterRequest>(req.body, registerSchema);
    const data = assertValidation<RegisterRequest>(result);
    const { email, password, firstName, lastName } = data;

    // Call authentication service
    const auth = await authService.register(email, password, firstName, lastName);

    log.info('User registration successful', { email });

    // Send success response with 201 Created
    res.status(201).json({
      status: 'success',
      message: 'User registered successfully',
      data: auth,
      timestamp: new Date().toISOString(),
    });
  });

  /**
   * Login user
   * POST /auth/login
   *
   * @param req - Express Request with body {email, password}
   * @param res - Express Response
   */
  static login = asyncHandler(async (req: Request, res: Response) => {
    // Validate request body
    const result = validateData<LoginRequest>(req.body, loginSchema);
    const data = assertValidation<LoginRequest>(result);
    const { email, password } = data;

    // Get client IP address for audit logging
    const ipAddress = (req.ip || req.socket.remoteAddress) as string;

    // Call authentication service
    const auth = await authService.login(email, password, ipAddress);

    log.info('User login successful', { email, ipAddress });

    // Send success response
    res.status(200).json({
      status: 'success',
      message: 'User logged in successfully',
      data: auth,
      timestamp: new Date().toISOString(),
    });
  });

  /**
   * Refresh access token
   * POST /auth/refresh
   *
   * @param req - Express Request with body {refreshToken}
   * @param res - Express Response
   */
  static refresh = asyncHandler(async (req: Request, res: Response) => {
    // Validate request body
    const result = validateData<RefreshRequest>(req.body, refreshSchema);
    const data = assertValidation<RefreshRequest>(result);
    const { refreshToken } = data;

    // Call authentication service
    const tokens = await authService.refreshToken(refreshToken);

    log.info('Token refreshed successfully');

    // Send success response
    res.status(200).json({
      status: 'success',
      message: 'Token refreshed successfully',
      data: tokens,
      timestamp: new Date().toISOString(),
    });
  });

  /**
   * Logout user - revoke all tokens
   * POST /auth/logout
   *
   * @param req - Authenticated Request with userId from JWT
   * @param res - Express Response
   */
  static logout = asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = req.userId;

    if (!userId) {
      throw new ValidationError([{ field: 'token', message: 'User ID not found in token' }]);
    }

    // Call authentication service
    await authService.logout(userId);

    log.info('User logged out successfully', { userId });

    // Send success response
    res.status(200).json({
      status: 'success',
      message: 'User logged out successfully',
      timestamp: new Date().toISOString(),
    });
  });

  /**
   * Verify token validity
   * GET /auth/verify
   *
   * @param req - Authenticated Request with Bearer token in Authorization header
   * @param res - Express Response
   */
  static verify = asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const userId = req.userId;

    if (!userId) {
      throw new ValidationError([{ field: 'token', message: 'User ID not found in token' }]);
    }

    // Extract token from Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      throw new ValidationError([
        { field: 'authorization', message: 'Authorization header missing or invalid' },
      ]);
    }

    const token = authHeader.substring(7);

    // Call authentication service to verify
    const payload = await authService.verifyToken(token);

    log.info('Token verified successfully', { userId });

    // Send success response with token details
    res.status(200).json({
      status: 'success',
      message: 'Token is valid',
      data: {
        userId: payload.userId,
        email: payload.email,
        roles: payload.roles,
        issuedAt: payload.iat,
        expiresAt: payload.exp,
        valid: true,
      },
      timestamp: new Date().toISOString(),
    });
  });
}

export default AuthController;
