/**
 * User Routes
 * Handles user profile management and admin user operations
 *
 * Routes:
 * GET    /profile     - Get current user profile
 * PUT    /profile     - Update user profile
 * GET    /:id         - Get user by ID (admin only)
 * DELETE /:id         - Delete user (admin only)
 * GET                 - List all users (admin only)
 */

import { Router, Response } from 'express';
import Joi from 'joi';

import { AuthenticatedRequest, authenticateToken, authorize } from '../middleware/authMiddleware';
import { asyncHandler, NotFoundError, AppError } from '../middleware/errorHandler';
import { log } from '../services/loggingService';

const router = Router();

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

/**
 * Schema for updating user profile
 */
const updateProfileSchema = Joi.object({
  firstName: Joi.string().min(1).max(50).optional(),
  lastName: Joi.string().min(1).max(50).optional(),
  avatar: Joi.string().uri().optional(),
  bio: Joi.string().max(500).optional(),
  preferences: Joi.object({
    emailNotifications: Joi.boolean().optional(),
    darkMode: Joi.boolean().optional(),
    defaultAnalysisType: Joi.string().optional(),
  }).optional(),
}).min(1);

/**
 * Schema for pagination query parameters
 */
const paginationSchema = Joi.object({
  limit: Joi.number().integer().min(1).max(100).default(20),
  offset: Joi.number().integer().min(0).default(0),
  sort: Joi.string().valid('created', 'email', 'name').default('created'),
  order: Joi.string().valid('asc', 'desc').default('desc'),
}).unknown(true);

// ============================================================================
// TYPES
// ============================================================================

interface UserProfile {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  avatar?: string;
  bio?: string;
  roles: string[];
  createdAt: string;
  updatedAt: string;
  preferences?: {
    emailNotifications: boolean;
    darkMode: boolean;
    defaultAnalysisType?: string;
  };
}

interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    limit: number;
    offset: number;
    total: number;
    totalPages: number;
  };
  timestamp: string;
}

// ============================================================================
// ROUTES
// ============================================================================

/**
 * GET /profile
 * Get current authenticated user's profile
 * Requires authentication
 *
 * @returns {Object} User profile data
 * @throws {AuthenticationError} - Not authenticated
 *
 * @example
 * GET /api/v1/users/profile
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * {
 *   "id": "user-id",
 *   "email": "user@example.com",
 *   "firstName": "John",
 *   "lastName": "Doe",
 *   "roles": ["user"],
 *   "createdAt": "2025-10-17T10:00:00Z",
 *   "updatedAt": "2025-10-17T10:00:00Z"
 * }
 */
router.get(
  '/profile',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res: Response<UserProfile>) => {
    const userId = req.userId;

    // Fetch user profile (mock)
    const user = await findUserById(userId);
    if (!user) {
      throw new NotFoundError('User');
    }

    log.info('User profile retrieved', { userId });

    res.status(200).json({
      id: user.id,
      email: user.email,
      firstName: user.firstName,
      lastName: user.lastName,
      avatar: user.avatar,
      bio: user.bio,
      roles: user.roles,
      createdAt: user.createdAt,
      updatedAt: user.updatedAt,
      preferences: user.preferences,
    });
  }),
);

/**
 * PUT /profile
 * Update current user's profile
 * Requires authentication
 *
 * @body {string} [firstName] - First name
 * @body {string} [lastName] - Last name
 * @body {string} [avatar] - Avatar URL
 * @body {string} [bio] - User bio (max 500 chars)
 * @body {Object} [preferences] - User preferences
 *
 * @returns {Object} Updated user profile
 * @throws {ValidationError} - Invalid input
 * @throws {NotFoundError} - User not found
 *
 * @example
 * PUT /api/v1/users/profile
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 * {
 *   "firstName": "Jane",
 *   "bio": "Interested in astronomical imaging",
 *   "preferences": {
 *     "emailNotifications": true,
 *     "darkMode": true
 *   }
 * }
 *
 * Response: 200 OK
 * {
 *   "id": "user-id",
 *   "email": "user@example.com",
 *   "firstName": "Jane",
 *   "lastName": "Doe",
 *   "bio": "Interested in astronomical imaging",
 *   "updatedAt": "2025-10-17T11:00:00Z",
 *   ...
 * }
 */
router.put(
  '/profile',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res: Response<UserProfile>) => {
    const userId = req.userId;
    const { firstName, lastName, avatar, bio, preferences } = req.body;

    // Validate update data
    const { error, value } = updateProfileSchema.validate(req.body);
    if (error) {
      throw new AppError(400, `Validation error: ${error.message}`);
    }

    // Update user profile (mock)
    const user = await updateUserProfile(userId, value);
    if (!user) {
      throw new NotFoundError('User');
    }

    log.info('User profile updated', { userId });

    res.status(200).json({
      id: user.id,
      email: user.email,
      firstName: user.firstName,
      lastName: user.lastName,
      avatar: user.avatar,
      bio: user.bio,
      roles: user.roles,
      createdAt: user.createdAt,
      updatedAt: user.updatedAt,
      preferences: user.preferences,
    });
  }),
);

/**
 * GET /:id
 * Get user by ID (admin only)
 * Requires authentication and admin role
 *
 * @param {string} id - User ID
 *
 * @returns {Object} User profile data
 * @throws {AuthenticationError} - Not authenticated
 * @throws {AuthorizationError} - Not admin
 * @throws {NotFoundError} - User not found
 *
 * @example
 * GET /api/v1/users/user-id
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * {
 *   "id": "user-id",
 *   "email": "user@example.com",
 *   "firstName": "John",
 *   "lastName": "Doe",
 *   "roles": ["user"],
 *   "createdAt": "2025-10-17T10:00:00Z"
 * }
 */
router.get(
  '/:id',
  authenticateToken,
  authorize(['admin']),
  asyncHandler(async (req: AuthenticatedRequest, res: Response<UserProfile>) => {
    const userId = req.params.id;

    // Fetch user by ID (mock)
    const user = await findUserById(userId);
    if (!user) {
      throw new NotFoundError('User');
    }

    log.info('Admin retrieved user profile', { adminId: req.userId, userId });

    res.status(200).json({
      id: user.id,
      email: user.email,
      firstName: user.firstName,
      lastName: user.lastName,
      avatar: user.avatar,
      bio: user.bio,
      roles: user.roles,
      createdAt: user.createdAt,
      updatedAt: user.updatedAt,
      preferences: user.preferences,
    });
  }),
);

/**
 * DELETE /:id
 * Delete user (admin only)
 * Requires authentication and admin role
 *
 * @param {string} id - User ID
 *
 * @returns {Object} Success message
 * @throws {AuthenticationError} - Not authenticated
 * @throws {AuthorizationError} - Not admin
 * @throws {NotFoundError} - User not found
 *
 * @example
 * DELETE /api/v1/users/user-id
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * {
 *   "message": "User deleted successfully",
 *   "deletedId": "user-id"
 * }
 */
router.delete(
  '/:id',
  authenticateToken,
  authorize(['admin']),
  asyncHandler(async (req: AuthenticatedRequest, res) => {
    const userId = req.params.id;

    // Delete user (mock)
    const success = await deleteUser(userId);
    if (!success) {
      throw new NotFoundError('User');
    }

    log.security('User deleted by admin', 'info', { adminId: req.userId, deletedUserId: userId });

    res.status(200).json({
      message: 'User deleted successfully',
      deletedId: userId,
    });
  }),
);

/**
 * GET /
 * List all users with pagination (admin only)
 * Requires authentication and admin role
 *
 * @query {number} [limit=20] - Results per page (1-100)
 * @query {number} [offset=0] - Results offset
 * @query {string} [sort=created] - Sort field (created, email, name)
 * @query {string} [order=desc] - Sort order (asc, desc)
 *
 * @returns {Object} Paginated user list
 * @throws {AuthenticationError} - Not authenticated
 * @throws {AuthorizationError} - Not admin
 *
 * @example
 * GET /api/v1/users?limit=10&offset=0&sort=created&order=desc
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * {
 *   "data": [
 *     {
 *       "id": "user-1",
 *       "email": "user1@example.com",
 *       "firstName": "John",
 *       "lastName": "Doe",
 *       "roles": ["user"],
 *       "createdAt": "2025-10-17T10:00:00Z"
 *     },
 *     ...
 *   ],
 *   "pagination": {
 *     "limit": 10,
 *     "offset": 0,
 *     "total": 150,
 *     "totalPages": 15
 *   },
 *   "timestamp": "2025-10-17T11:00:00Z"
 * }
 */
router.get(
  '/',
  authenticateToken,
  authorize(['admin']),
  asyncHandler(async (req: AuthenticatedRequest, res: Response<PaginatedResponse<UserProfile>>) => {
    // Validate pagination parameters
    const { error, value } = paginationSchema.validate(req.query);
    if (error) {
      throw new AppError(400, `Invalid pagination parameters: ${error.message}`);
    }

    const { limit, offset, sort, order } = value;

    // Fetch users (mock)
    const { users, total } = await listUsers({
      limit,
      offset,
      sort,
      order,
    });

    log.info('Admin listed users', { adminId: req.userId, limit, offset, total });

    res.status(200).json({
      data: users.map((user) => ({
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        avatar: user.avatar,
        bio: user.bio,
        roles: user.roles,
        createdAt: user.createdAt,
        updatedAt: user.updatedAt,
      })),
      pagination: {
        limit,
        offset,
        total,
        totalPages: Math.ceil(total / limit),
      },
      timestamp: new Date().toISOString(),
    });
  }),
);

// ============================================================================
// MOCK DATABASE FUNCTIONS
// ============================================================================
// These are placeholders. In production, implement actual database queries.

/**
 * Find user by ID (mock)
 */
async function findUserById(id: string): Promise<any> {
  // TODO: Implement database query
  return null;
}

/**
 * Update user profile (mock)
 */
async function updateUserProfile(userId: string, updates: any): Promise<any> {
  // TODO: Implement database update
  return null;
}

/**
 * Delete user (mock)
 */
async function deleteUser(userId: string): Promise<boolean> {
  // TODO: Implement database delete
  return false;
}

/**
 * List users with pagination (mock)
 */
async function listUsers(options: {
  limit: number;
  offset: number;
  sort: string;
  order: string;
}): Promise<{ users: any[]; total: number }> {
  // TODO: Implement database query
  return { users: [], total: 0 };
}

export default router;
