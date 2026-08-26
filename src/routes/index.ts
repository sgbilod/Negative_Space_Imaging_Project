/**
 * API Routes Index
 * Aggregates all route modules and mounts them on the API router
 *
 * Route Structure:
 * /api/v{version}/auth      - Authentication routes
 * /api/v{version}/users     - User management routes
 * /api/v{version}/images    - Image management routes
 * /api/v{version}/analysis  - Analysis results routes
 * /api/v{version}/admin     - Admin management routes (admin-only)
 */

import { Router } from 'express';

// Route imports
import authRoutes from './authRoutes';
import userRoutes from './userRoutes';
import imageRoutes from './imageRoutes';
import analysisRoutes from './analysisRoutes';
import adminRoutes from './adminRoutes';

/**
 * Create API v1 router with all route modules
 *
 * @returns {Router} Express router with all API routes mounted
 *
 * @example
 * // In server.ts:
 * import apiRoutes from '../routes';
 * app.use('/api/v1', apiRoutes);
 */
const router = Router();

/**
 * Authentication Routes
 * POST   /auth/register    - User registration
 * POST   /auth/login       - User login
 * POST   /auth/refresh     - Token refresh
 * POST   /auth/logout      - User logout
 * GET    /auth/verify      - Token verification
 */
router.use('/auth', authRoutes);

/**
 * User Routes
 * GET    /users/profile    - Get current user profile
 * PUT    /users/profile    - Update user profile
 * GET    /users            - List all users (admin only)
 * GET    /users/:id        - Get user by ID (admin only)
 * DELETE /users/:id        - Delete user (admin only)
 */
router.use('/users', userRoutes);

/**
 * Image Routes
 * POST   /images/upload    - Upload image
 * GET    /images           - List user's images
 * GET    /images/:id       - Get image details
 * DELETE /images/:id       - Delete image
 * GET    /images/:id/download - Download image
 * POST   /images/:id/analyze  - Trigger analysis
 */
router.use('/images', imageRoutes);

/**
 * Analysis Routes
 * GET    /analysis/:imageId         - Get analysis results
 * GET    /analysis/:imageId/export  - Export analysis results
 * POST   /analysis/:imageId/compare - Compare analyses
 * GET    /analysis                  - List user's analyses
 */
router.use('/analysis', analysisRoutes);

/**
 * Admin Routes (admin-only)
 * GET    /admin/stats              - System statistics
 * GET    /admin/queue              - View processing queue
 * POST   /admin/queue/:jobId/priority - Update job priority
 * GET    /admin/logs               - Access system logs
 * POST   /admin/config             - Update system configuration
 */
router.use('/admin', adminRoutes);

/**
 * Health check endpoint (should not require authentication)
 * Can be removed if health checks are handled elsewhere
 */
router.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    api: 'v1',
  });
});

export default router;
