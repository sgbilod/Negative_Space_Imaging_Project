/**
 * Authentication Service
 * Handles user registration, login, token management, and authentication logic
 */

import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { configService } from '../config/serverConfig';
import { log } from '../services/loggingService';
import {
  AuthenticationError,
  ConflictError,
  NotFoundError,
  ValidationError,
} from '../utils/errors';
import { validatePassword, validateEmail } from '../utils/validators';

const config = configService.getConfig();

/**
 * Authentication service interface
 */
export interface IAuthService {
  register(
    email: string,
    password: string,
    firstName: string,
    lastName: string,
  ): Promise<AuthResponse>;
  login(email: string, password: string, ipAddress?: string): Promise<AuthResponse>;
  refreshToken(refreshToken: string): Promise<{ accessToken: string }>;
  logout(userId: string): Promise<void>;
  verifyToken(token: string): Promise<TokenPayload>;
}

/**
 * Authentication response with tokens
 */
export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
  user: UserInfo;
}

/**
 * User information in token
 */
export interface TokenPayload {
  userId: string;
  email: string;
  roles: string[];
  iat: number;
  exp: number;
}

/**
 * User information response
 */
export interface UserInfo {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  roles: string[];
  createdAt: string;
}

/**
 * Login attempt tracking for rate limiting
 */
interface LoginAttempt {
  count: number;
  timestamp: number;
}

class AuthService implements IAuthService {
  private loginAttempts: Map<string, LoginAttempt> = new Map();
  private readonly MAX_LOGIN_ATTEMPTS = 5;
  private readonly LOGIN_ATTEMPT_WINDOW = 15 * 60 * 1000; // 15 minutes

  /**
   * Register new user
   * @param email - User email
   * @param password - User password
   * @param firstName - First name
   * @param lastName - Last name
   * @returns Authentication response with tokens
   * @throws ConflictError if email already exists
   * @throws ValidationError if input invalid
   */
  async register(
    email: string,
    password: string,
    firstName: string,
    lastName: string,
  ): Promise<AuthResponse> {
    // Validate inputs
    validateEmail(email);
    validatePassword(password);

    if (firstName.trim().length === 0) {
      throw new ValidationError('First name cannot be empty');
    }

    if (lastName.trim().length === 0) {
      throw new ValidationError('Last name cannot be empty');
    }

    // Check if user already exists (mock - replace with DB query)
    const existingUser = await findUserByEmail(email);
    if (existingUser) {
      log.warn('Registration attempt with existing email', { email });
      throw new ConflictError('Email already registered');
    }

    // Hash password
    const hashedPassword = await this.hashPassword(password);

    // Create user (mock - replace with DB insert)
    const user = await createUser({
      email: email.toLowerCase(),
      password: hashedPassword,
      firstName: firstName.trim(),
      lastName: lastName.trim(),
      roles: ['user'],
    });

    log.info('User registered', { userId: user.id, email });

    // Generate tokens
    return this.generateAuthResponse(user);
  }

  /**
   * Login user
   * @param email - User email
   * @param password - User password
   * @param ipAddress - Client IP address (optional)
   * @returns Authentication response with tokens
   * @throws AuthenticationError if credentials invalid
   * @throws ValidationError if input invalid
   */
  async login(email: string, password: string, ipAddress?: string): Promise<AuthResponse> {
    // Validate inputs
    validateEmail(email);

    if (!password || password.length === 0) {
      throw new ValidationError('Password is required');
    }

    // Check rate limiting
    this.checkLoginAttempts(email);

    // Find user by email (mock - replace with DB query)
    const user = await findUserByEmail(email);
    if (!user) {
      // Record failed attempt
      this.recordFailedLoginAttempt(email);
      log.warn('Login attempt with non-existent email', { email });
      throw new AuthenticationError('Invalid email or password');
    }

    // Compare password
    const passwordValid = await this.comparePassword(password, user.password);
    if (!passwordValid) {
      // Record failed attempt
      this.recordFailedLoginAttempt(email);
      log.warn('Login failed - invalid password', { userId: user.id, email });
      throw new AuthenticationError('Invalid email or password');
    }

    // Reset login attempts on successful login
    this.loginAttempts.delete(email);

    log.info('User logged in', { userId: user.id, email, ipAddress });

    // Generate tokens
    return this.generateAuthResponse(user);
  }

  /**
   * Refresh access token
   * @param refreshToken - Refresh token
   * @returns New access token
   * @throws AuthenticationError if token invalid or expired
   */
  async refreshToken(refreshToken: string): Promise<{ accessToken: string }> {
    try {
      // Verify refresh token (same secret as access token)
      const decoded = jwt.verify(refreshToken, config.jwt.secret) as TokenPayload;

      // Check if token is in revoked list (mock implementation)
      const isRevoked = await isTokenRevoked(decoded.userId, refreshToken);
      if (isRevoked) {
        log.warn('Attempt to use revoked refresh token', { userId: decoded.userId });
        throw new AuthenticationError('Refresh token has been revoked');
      }

      // Get user (mock - replace with DB query)
      const user = await findUserById(decoded.userId);
      if (!user) {
        throw new NotFoundError('User');
      }

      // Generate new access token
      const accessToken = this.generateAccessToken({
        id: user.id,
        email: user.email,
        roles: user.roles,
      });

      log.info('Token refreshed', { userId: user.id });

      return { accessToken };
    } catch (error) {
      if (error instanceof AuthenticationError) {
        throw error;
      }

      if (error instanceof jwt.TokenExpiredError) {
        throw new AuthenticationError('Refresh token has expired');
      }

      if (error instanceof jwt.JsonWebTokenError) {
        throw new AuthenticationError('Invalid refresh token');
      }

      throw error;
    }
  }

  /**
   * Logout user (invalidate refresh tokens)
   * @param userId - User ID
   */
  async logout(userId: string): Promise<void> {
    // Add user's tokens to revoked list (mock implementation)
    await revokeUserTokens(userId);
    log.info('User logged out', { userId });
  }

  /**
   * Verify token validity
   * @param token - JWT token
   * @returns Token payload
   * @throws AuthenticationError if token invalid
   */
  async verifyToken(token: string): Promise<TokenPayload> {
    try {
      const decoded = jwt.verify(token, config.jwt.secret) as TokenPayload;

      return decoded;
    } catch (error) {
      if (error instanceof jwt.TokenExpiredError) {
        throw new AuthenticationError('Token has expired');
      }

      if (error instanceof jwt.JsonWebTokenError) {
        throw new AuthenticationError('Invalid token');
      }

      throw error;
    }
  }

  /**
   * Hash password with bcrypt
   * @param password - Plain text password
   * @returns Hashed password
   */
  private async hashPassword(password: string): Promise<string> {
    return bcrypt.hash(password, config.security.bcryptRounds);
  }

  /**
   * Compare password with hash
   * @param password - Plain text password
   * @param hash - Hashed password
   * @returns True if password matches
   */
  private async comparePassword(password: string, hash: string): Promise<boolean> {
    return bcrypt.compare(password, hash);
  }

  /**
   * Generate access token
   * @param user - User data
   * @returns JWT access token
   */
  private generateAccessToken(user: { id: string; email: string; roles: string[] }): string {
    return jwt.sign(
      {
        userId: user.id,
        email: user.email,
        roles: user.roles,
      },
      config.jwt.secret,
      { expiresIn: config.jwt.expiresIn, algorithm: config.jwt.algorithm },
    );
  }

  /**
   * Generate refresh token
   * @param user - User data
   * @returns JWT refresh token
   */
  private generateRefreshToken(user: { id: string; email: string }): string {
    return jwt.sign(
      {
        userId: user.id,
        email: user.email,
      },
      config.jwt.secret,
      { expiresIn: '7d', algorithm: config.jwt.algorithm },
    );
  }

  /**
   * Generate full authentication response
   * @param user - User data
   * @returns Auth response with tokens
   */
  private generateAuthResponse(user: any): AuthResponse {
    const accessToken = this.generateAccessToken(user);
    const refreshToken = this.generateRefreshToken(user);

    return {
      accessToken,
      refreshToken,
      user: {
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        roles: user.roles,
        createdAt: user.createdAt || new Date().toISOString(),
      },
    };
  }

  /**
   * Check if user has exceeded login attempts
   * @param email - User email
   * @throws ValidationError if too many attempts
   */
  private checkLoginAttempts(email: string): void {
    const attempt = this.loginAttempts.get(email);

    if (!attempt) {
      return;
    }

    // Check if attempt window has expired
    if (Date.now() - attempt.timestamp > this.LOGIN_ATTEMPT_WINDOW) {
      this.loginAttempts.delete(email);
      return;
    }

    // Check if max attempts exceeded
    if (attempt.count >= this.MAX_LOGIN_ATTEMPTS) {
      throw new ValidationError('Too many login attempts. Please try again later.');
    }
  }

  /**
   * Record failed login attempt
   * @param email - User email
   */
  private recordFailedLoginAttempt(email: string): void {
    const attempt = this.loginAttempts.get(email);

    if (!attempt) {
      this.loginAttempts.set(email, { count: 1, timestamp: Date.now() });
    } else {
      attempt.count += 1;
    }
  }
}

// ============================================================================
// MOCK DATABASE FUNCTIONS
// ============================================================================

/**
 * Find user by email (mock - replace with DB query)
 */
async function findUserByEmail(email: string): Promise<any> {
  // TODO: Implement database query
  // Example: return await db.query('SELECT * FROM users WHERE email = ?', [email])
  return null;
}

/**
 * Find user by ID (mock - replace with DB query)
 */
async function findUserById(id: string): Promise<any> {
  // TODO: Implement database query
  return null;
}

/**
 * Create user (mock - replace with DB insert)
 */
async function createUser(userData: {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  roles: string[];
}): Promise<any> {
  // TODO: Implement database insert
  return {
    id: `user-${Date.now()}`,
    ...userData,
    createdAt: new Date().toISOString(),
  };
}

/**
 * Check if token is revoked (mock - replace with cache/DB query)
 */
async function isTokenRevoked(userId: string, token: string): Promise<boolean> {
  // TODO: Implement with Redis or database
  return false;
}

/**
 * Revoke all user tokens (mock - replace with cache/DB update)
 */
async function revokeUserTokens(userId: string): Promise<void> {
  // TODO: Implement with Redis or database
}

// Export service instance
export const authService = new AuthService();

// Export for testing
export { AuthService };
