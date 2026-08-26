/**
 * Authentication Routes
 * Handles user registration, login, token refresh, logout, and verification
 *
 * Routes:
 * POST   /register    - Register new user
 * POST   /login       - User login
 * POST   /refresh     - Refresh JWT token
 * POST   /logout      - User logout
 * GET    /verify      - Verify token validity
 */

import { Router, Response } from 'express';
import Joi from 'joi';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';

import {
  AuthenticatedRequest,
  authenticateToken,
  validateRequest,
} from '../middleware/authMiddleware';
import {
  asyncHandler,
  ValidationError,
  AuthenticationError,
  ConflictError,
} from '../middleware/errorHandler';
import { configService } from '../config/serverConfig';
import { log } from '../services/loggingService';

const router = Router();
const config = configService.getConfig();

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

/**
 * Schema for user registration
 */
const registerSchema = Joi.object({
  email: Joi.string().email().required().messages({
    'string.email': 'Invalid email format',
    'any.required': 'Email is required',
  }),
  password: Joi.string()
    .min(config.security.passwordMinLength)
    .required()
    .messages({
      'string.min': `Password must be at least ${config.security.passwordMinLength} characters`,
      'any.required': 'Password is required',
    }),
  firstName: Joi.string().min(1).max(50).required().messages({
    'string.min': 'First name is required',
  }),
  lastName: Joi.string().min(1).max(50).required().messages({
    'string.min': 'Last name is required',
  }),
});

/**
 * Schema for user login
 */
const loginSchema = Joi.object({
  email: Joi.string().email().required().messages({
    'string.email': 'Invalid email format',
    'any.required': 'Email is required',
  }),
  password: Joi.string().required().messages({
    'any.required': 'Password is required',
  }),
});

/**
 * Schema for token refresh
 */
const refreshSchema = Joi.object({
  refreshToken: Joi.string().required().messages({
    'any.required': 'Refresh token is required',
  }),
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Hash password with bcrypt
 * @param password - Plain text password
 * @returns Hashed password
 */
async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, config.security.bcryptRounds);
}

/**
 * Compare password with hash
 * @param password - Plain text password
 * @param hash - Hashed password
 * @returns True if passwords match
 */
async function comparePassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

/**
 * Generate JWT token
 * @param payload - Token payload
 * @returns JWT token
 */
function generateToken(payload: Record<string, any>): string {
  return jwt.sign(payload, config.jwt.secret, {
    expiresIn: config.jwt.expiresIn,
    algorithm: config.jwt.algorithm as any,
  });
}

/**
 * Generate refresh token (longer expiration)
 * @param payload - Token payload
 * @returns Refresh token
 */
function generateRefreshToken(payload: Record<string, any>): string {
  return jwt.sign(payload, config.jwt.secret, {
    expiresIn: config.jwt.refreshExpiresIn || '7d',
    algorithm: config.jwt.algorithm as any,
  });
}

/**
 * Standard token response format
 */
interface TokenResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: string;
  user: {
    id: string;
    email: string;
    firstName: string;
    lastName: string;
  };
}

// ============================================================================
// ROUTES
// ============================================================================

/**
 * POST /register
 * Register new user with email and password
 *
 * @body {string} email - User email
 * @body {string} password - User password (min 12 chars)
 * @body {string} firstName - User first name
 * @body {string} lastName - User last name
 *
 * @returns {Object} Token response with user info
 * @throws {ValidationError} - Invalid input
 * @throws {ConflictError} - Email already registered
 *
 * @example
 * POST /api/v1/auth/register
 * {
 *   "email": "user@example.com",
 *   "password": "SecurePassword123!",
 *   "firstName": "John",
 *   "lastName": "Doe"
 * }
 *
 * Response: 201 Created
 * {
 *   "accessToken": "eyJhbGciOiJIUzI1NiIs...",
 *   "refreshToken": "eyJhbGciOiJIUzI1NiIs...",
 *   "expiresIn": "24h",
 *   "user": {
 *     "id": "user-id",
 *     "email": "user@example.com",
 *     "firstName": "John",
 *     "lastName": "Doe"
 *   }
 * }
 */
router.post(
  '/register',
  validateRequest(registerSchema),
  asyncHandler(async (req, res: Response<TokenResponse>) => {
    const { email, password, firstName, lastName } = req.body;

    // Check if user already exists (in real app, query database)
    // For now, using a mock check
    const existingUser = await findUserByEmail(email);
    if (existingUser) {
      throw new ConflictError('Email already registered');
    }

    // Hash password
    const hashedPassword = await hashPassword(password);

    // Create user (in real app, save to database)
    const userId = generateUserId();
    const user = {
      id: userId,
      email,
      firstName,
      lastName,
      password: hashedPassword,
      createdAt: new Date().toISOString(),
      roles: ['user'],
    };

    // Save user to database (mock)
    await saveUser(user);

    // Generate tokens
    const accessToken = generateToken({ userId, email, roles: ['user'] });
    const refreshToken = generateRefreshToken({ userId, email });

    log.auth('User registered', userId, true, { email });

    res.status(201).json({
      accessToken,
      refreshToken,
      expiresIn: config.jwt.expiresIn,
      user: {
        id: userId,
        email,
        firstName,
        lastName,
      },
    });
  }),
);

/**
 * POST /login
 * Login user with email and password
 *
 * @body {string} email - User email
 * @body {string} password - User password
 *
 * @returns {Object} Token response with user info
 * @throws {AuthenticationError} - Invalid credentials
 *
 * @example
 * POST /api/v1/auth/login
 * {
 *   "email": "user@example.com",
 *   "password": "SecurePassword123!"
 * }
 *
 * Response: 200 OK
 * {
 *   "accessToken": "eyJhbGciOiJIUzI1NiIs...",
 *   "refreshToken": "eyJhbGciOiJIUzI1NiIs...",
 *   "expiresIn": "24h",
 *   "user": {
 *     "id": "user-id",
 *     "email": "user@example.com",
 *     "firstName": "John",
 *     "lastName": "Doe"
 *   }
 * }
 */
router.post(
  '/login',
  validateRequest(loginSchema),
  asyncHandler(async (req, res: Response<TokenResponse>) => {
    const { email, password } = req.body;

    // Find user by email (mock)
    const user = await findUserByEmail(email);
    if (!user) {
      throw new AuthenticationError('Invalid email or password');
    }

    // Compare password
    const passwordMatch = await comparePassword(password, user.password);
    if (!passwordMatch) {
      log.security('Failed login attempt', 'medium', { email });
      throw new AuthenticationError('Invalid email or password');
    }

    // Generate tokens
    const accessToken = generateToken({
      userId: user.id,
      email: user.email,
      roles: user.roles || ['user'],
    });
    const refreshToken = generateRefreshToken({ userId: user.id, email: user.email });

    log.auth('User logged in', user.id, true, { email });

    res.status(200).json({
      accessToken,
      refreshToken,
      expiresIn: config.jwt.expiresIn,
      user: {
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
      },
    });
  }),
);

/**
 * POST /refresh
 * Refresh expired JWT token
 *
 * @body {string} refreshToken - Valid refresh token
 *
 * @returns {Object} New access token
 * @throws {AuthenticationError} - Invalid refresh token
 *
 * @example
 * POST /api/v1/auth/refresh
 * {
 *   "refreshToken": "eyJhbGciOiJIUzI1NiIs..."
 * }
 *
 * Response: 200 OK
 * {
 *   "accessToken": "eyJhbGciOiJIUzI1NiIs...",
 *   "expiresIn": "24h"
 * }
 */
router.post(
  '/refresh',
  validateRequest(refreshSchema),
  asyncHandler(async (req, res) => {
    const { refreshToken } = req.body;

    try {
      // Verify refresh token
      const decoded = jwt.verify(refreshToken, config.jwt.secret) as any;

      // Generate new access token
      const accessToken = generateToken({
        userId: decoded.userId,
        email: decoded.email,
        roles: decoded.roles || ['user'],
      });

      log.info('Token refreshed', { userId: decoded.userId });

      res.status(200).json({
        accessToken,
        expiresIn: config.jwt.expiresIn,
      });
    } catch (error) {
      throw new AuthenticationError('Invalid or expired refresh token');
    }
  }),
);

/**
 * POST /logout
 * Logout user (invalidate tokens)
 * Requires authentication
 *
 * @returns {Object} Success message
 * @throws {AuthenticationError} - Not authenticated
 *
 * @example
 * POST /api/v1/auth/logout
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * {
 *   "message": "Successfully logged out"
 * }
 */
router.post(
  '/logout',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res) => {
    const userId = req.userId;

    // Invalidate tokens in database (mock)
    await invalidateUserTokens(userId);

    log.auth('User logged out', userId, true, {});

    res.status(200).json({
      message: 'Successfully logged out',
    });
  }),
);

/**
 * GET /verify
 * Verify JWT token validity
 * Requires authentication
 *
 * @returns {Object} Token validity and user info
 * @throws {AuthenticationError} - Invalid token
 *
 * @example
 * GET /api/v1/auth/verify
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * {
 *   "valid": true,
 *   "userId": "user-id",
 *   "email": "user@example.com",
 *   "roles": ["user"]
 * }
 */
router.get(
  '/verify',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res) => {
    const userId = req.userId;
    const user = req.user || (await findUserById(userId));

    res.status(200).json({
      valid: true,
      userId: req.userId,
      email: req.user?.email || user?.email,
      roles: req.user?.roles || user?.roles || ['user'],
      timestamp: new Date().toISOString(),
    });
  }),
);

// ============================================================================
// MOCK DATABASE FUNCTIONS
// ============================================================================
// These are placeholders. In production, implement actual database queries.

/**
 * Find user by email (mock)
 */
async function findUserByEmail(email: string): Promise<any> {
  // TODO: Implement database query
  return null;
}

/**
 * Find user by ID (mock)
 */
async function findUserById(id: string): Promise<any> {
  // TODO: Implement database query
  return null;
}

/**
 * Save user to database (mock)
 */
async function saveUser(user: any): Promise<void> {
  // TODO: Implement database save
}

/**
 * Invalidate user tokens (mock)
 */
async function invalidateUserTokens(userId: string): Promise<void> {
  // TODO: Implement token invalidation (add to blacklist)
}

/**
 * Generate unique user ID (mock)
 */
function generateUserId(): string {
  return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

export default router;
