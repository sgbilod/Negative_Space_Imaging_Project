/**
 * User Service Module
 *
 * Handles user-related business logic including registration, authentication,
 * profile management, and JWT token generation.
 *
 * @module services/userService
 */

import bcrypt from 'bcrypt';
import { User } from '../models/User';
import { generateToken } from '../utils/tokenUtils';
import { ServiceError } from '../types';
import logger from '../utils/logger';

/**
 * UserService class
 * Provides methods for user management and authentication
 */
export class UserService {
  /**
   * Register a new user
   * Hashes password and creates user record
   */
  async registerUser(
    email: string,
    password: string,
    first_name: string,
    last_name: string
  ): Promise<{ user: Partial<User>; token: string }> {
    try {
      // Check if user already exists
      const existingUser = await User.findOne({ where: { email } });
      if (existingUser) {
        logger.warn(`Registration attempt with existing email: ${email}`);
        throw new ServiceError(409, 'Email already registered', 'EMAIL_EXISTS');
      }

      // Hash password
      const passwordHash = await bcrypt.hash(password, 10);
      logger.debug(`Password hashed for new user: ${email}`);

      // Create user
      const user = await User.create({
        email,
        password_hash: passwordHash,
        first_name,
        last_name,
      });

      logger.info(`User registered successfully: ${email}`);

      // Generate JWT token
      const token = generateToken({
        id: user.id,
        email: user.email,
      });

      return {
        user: user.toJSON(),
        token,
      };
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('User registration failed', error);
      throw new ServiceError(500, 'Registration failed', 'REGISTRATION_ERROR');
    }
  }

  /**
   * Authenticate user and generate token
   */
  async loginUser(
    email: string,
    password: string
  ): Promise<{ user: Partial<User>; token: string }> {
    try {
      // Find user
      const user = await User.findOne({ where: { email } });
      if (!user) {
        logger.warn(`Login attempt with non-existent email: ${email}`);
        throw new ServiceError(401, 'Invalid credentials', 'INVALID_CREDENTIALS');
      }

      // Verify password
      const passwordMatch = await bcrypt.compare(password, user.password_hash);
      if (!passwordMatch) {
        logger.warn(`Failed login attempt for user: ${email}`);
        throw new ServiceError(401, 'Invalid credentials', 'INVALID_CREDENTIALS');
      }

      logger.info(`User logged in: ${email}`);

      // Generate token
      const token = generateToken({
        id: user.id,
        email: user.email,
      });

      return {
        user: user.toJSON(),
        token,
      };
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Login failed', error);
      throw new ServiceError(500, 'Login failed', 'LOGIN_ERROR');
    }
  }

  /**
   * Get user by ID
   */
  async getUserById(id: number): Promise<Partial<User>> {
    try {
      const user = await User.findByPk(id);
      if (!user) {
        logger.warn(`User not found: ${id}`);
        throw new ServiceError(404, 'User not found', 'USER_NOT_FOUND');
      }

      logger.debug(`Retrieved user: ${id}`);
      return user.toJSON();
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to get user', error);
      throw new ServiceError(500, 'Failed to get user', 'GET_USER_ERROR');
    }
  }

  /**
   * Update user profile
   */
  async updateUser(
    id: number,
    data: { first_name?: string; last_name?: string; email?: string }
  ): Promise<Partial<User>> {
    try {
      const user = await User.findByPk(id);
      if (!user) {
        throw new ServiceError(404, 'User not found', 'USER_NOT_FOUND');
      }

      // Update fields
      if (data.first_name) user.first_name = data.first_name;
      if (data.last_name) user.last_name = data.last_name;
      if (data.email) {
        // Check if email is already taken
        const existingUser = await User.findOne({
          where: { email: data.email },
        });
        if (existingUser && existingUser.id !== id) {
          throw new ServiceError(409, 'Email already in use', 'EMAIL_IN_USE');
        }
        user.email = data.email;
      }

      await user.save();
      logger.info(`User profile updated: ${id}`);

      return user.toJSON();
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to update user', error);
      throw new ServiceError(500, 'Failed to update user', 'UPDATE_USER_ERROR');
    }
  }

  /**
   * Delete user account
   */
  async deleteUser(id: number): Promise<void> {
    try {
      const user = await User.findByPk(id);
      if (!user) {
        throw new ServiceError(404, 'User not found', 'USER_NOT_FOUND');
      }

      await user.destroy();
      logger.info(`User deleted: ${id}`);
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to delete user', error);
      throw new ServiceError(500, 'Failed to delete user', 'DELETE_USER_ERROR');
    }
  }

  /**
   * Verify password for user
   */
  async verifyPassword(userId: number, password: string): Promise<boolean> {
    try {
      const user = await User.findByPk(userId);
      if (!user) {
        return false;
      }

      return await bcrypt.compare(password, user.password_hash);
    } catch (error) {
      logger.error('Password verification failed', error);
      return false;
    }
  }
}

export default new UserService();
