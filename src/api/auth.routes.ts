import { Router } from 'express';
import { AuthController } from '../controllers/auth.controller';
import { authenticate } from '../middleware/auth';
import { validateRequest } from '../middleware/validator';
import { authSchemas } from '../schemas/auth.schema';

const router = Router();

/**
 * Authentication routes
 */

// Register a new user
router.post(
  '/register',
  validateRequest(authSchemas.register),
  AuthController.register
);

// Login
router.post(
  '/login',
  validateRequest(authSchemas.login),
  AuthController.login
);

// Logout (requires authentication)
router.post(
  '/logout',
  authenticate,
  AuthController.logout
);

// Get current user profile (requires authentication)
router.get(
  '/me',
  authenticate,
  AuthController.getCurrentUser
);

// Change password (requires authentication)
router.post(
  '/change-password',
  authenticate,
  validateRequest(authSchemas.changePassword),
  AuthController.changePassword
);

// Request password reset
router.post(
  '/request-reset',
  validateRequest(authSchemas.requestReset),
  AuthController.requestPasswordReset
);

export const authRoutes = router;
