/**
 * Token Utilities Module
 *
 * Handles JWT token generation, verification, and refresh operations.
 * Provides centralized token management for authentication.
 *
 * @module utils/tokenUtils
 */

import jwt from 'jsonwebtoken';
import logger from './logger';

/**
 * JWT payload interface
 */
export interface JwtPayload {
  id: number;
  email: string;
  iat?: number;
  exp?: number;
}

/**
 * Token configuration
 */
const TOKEN_CONFIG = {
  accessTokenSecret:
    process.env.JWT_SECRET || 'your-secret-key-change-in-production',
  accessTokenExpiry: process.env.JWT_EXPIRY || '24h',
  refreshTokenSecret:
    process.env.JWT_REFRESH_SECRET || 'your-refresh-secret-change-in-production',
  refreshTokenExpiry: process.env.JWT_REFRESH_EXPIRY || '7d',
};

/**
 * Generate access token
 */
export function generateAccessToken(payload: Partial<JwtPayload>): string {
  try {
    const token = jwt.sign(payload, TOKEN_CONFIG.accessTokenSecret, {
      expiresIn: TOKEN_CONFIG.accessTokenExpiry,
      algorithm: 'HS256',
    });

    logger.debug(`Access token generated for user: ${payload.id}`);
    return token;
  } catch (error) {
    logger.error('Failed to generate access token', error);
    throw error;
  }
}

/**
 * Generate refresh token
 */
export function generateRefreshToken(payload: Partial<JwtPayload>): string {
  try {
    const token = jwt.sign(payload, TOKEN_CONFIG.refreshTokenSecret, {
      expiresIn: TOKEN_CONFIG.refreshTokenExpiry,
      algorithm: 'HS256',
    });

    logger.debug(`Refresh token generated for user: ${payload.id}`);
    return token;
  } catch (error) {
    logger.error('Failed to generate refresh token', error);
    throw error;
  }
}

/**
 * Generate both access and refresh tokens
 */
export function generateTokenPair(
  payload: Partial<JwtPayload>
): { accessToken: string; refreshToken: string } {
  return {
    accessToken: generateAccessToken(payload),
    refreshToken: generateRefreshToken(payload),
  };
}

/**
 * Verify access token
 */
export function verifyAccessToken(token: string): JwtPayload {
  try {
    const decoded = jwt.verify(
      token,
      TOKEN_CONFIG.accessTokenSecret,
      { algorithms: ['HS256'] }
    );

    logger.debug(`Access token verified for user: ${(decoded as JwtPayload).id}`);
    return decoded as JwtPayload;
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      logger.warn('Access token expired');
      throw new Error('Token expired');
    }
    if (error instanceof jwt.JsonWebTokenError) {
      logger.warn('Invalid access token');
      throw new Error('Invalid token');
    }
    throw error;
  }
}

/**
 * Verify refresh token
 */
export function verifyRefreshToken(token: string): JwtPayload {
  try {
    const decoded = jwt.verify(
      token,
      TOKEN_CONFIG.refreshTokenSecret,
      { algorithms: ['HS256'] }
    );

    logger.debug(`Refresh token verified for user: ${(decoded as JwtPayload).id}`);
    return decoded as JwtPayload;
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      logger.warn('Refresh token expired');
      throw new Error('Refresh token expired');
    }
    if (error instanceof jwt.JsonWebTokenError) {
      logger.warn('Invalid refresh token');
      throw new Error('Invalid refresh token');
    }
    throw error;
  }
}

/**
 * Refresh access token using refresh token
 */
export function refreshAccessToken(refreshToken: string): string {
  try {
    const decoded = verifyRefreshToken(refreshToken);

    // Generate new access token with same user info
    const newAccessToken = generateAccessToken({
      id: decoded.id,
      email: decoded.email,
    });

    logger.info(`Access token refreshed for user: ${decoded.id}`);
    return newAccessToken;
  } catch (error) {
    logger.error('Failed to refresh access token', error);
    throw error;
  }
}

/**
 * Decode token without verification (for debugging)
 */
export function decodeToken(token: string): Partial<JwtPayload> | null {
  try {
    const decoded = jwt.decode(token);
    return decoded as Partial<JwtPayload>;
  } catch (error) {
    logger.warn('Failed to decode token', error);
    return null;
  }
}

/**
 * Extract token from authorization header
 */
export function extractTokenFromHeader(
  authHeader: string | undefined
): string | null {
  if (!authHeader) {
    return null;
  }

  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') {
    return null;
  }

  return parts[1];
}

/**
 * Check if token is expired
 */
export function isTokenExpired(token: string): boolean {
  try {
    const decoded = jwt.decode(token) as any;
    if (!decoded || !decoded.exp) {
      return true;
    }

    const expirationTime = decoded.exp * 1000; // Convert to milliseconds
    return Date.now() >= expirationTime;
  } catch (error) {
    return true;
  }
}

/**
 * Get time until token expiration
 */
export function getTimeUntilExpiration(token: string): number | null {
  try {
    const decoded = jwt.decode(token) as any;
    if (!decoded || !decoded.exp) {
      return null;
    }

    const expirationTime = decoded.exp * 1000;
    const timeUntilExpiration = expirationTime - Date.now();

    return Math.max(0, timeUntilExpiration);
  } catch (error) {
    return null;
  }
}

/**
 * Validate token structure
 */
export function isValidTokenStructure(token: string): boolean {
  if (!token || typeof token !== 'string') {
    return false;
  }

  const parts = token.split('.');
  return parts.length === 3;
}

/**
 * Create anonymous token (optional feature)
 */
export function generateAnonymousToken(sessionId: string): string {
  return jwt.sign(
    { sessionId, isAnonymous: true },
    TOKEN_CONFIG.accessTokenSecret,
    {
      expiresIn: '1h',
      algorithm: 'HS256',
    }
  );
}

export default {
  generateAccessToken,
  generateRefreshToken,
  generateTokenPair,
  verifyAccessToken,
  verifyRefreshToken,
  refreshAccessToken,
  decodeToken,
  extractTokenFromHeader,
  isTokenExpired,
  getTimeUntilExpiration,
  isValidTokenStructure,
  generateAnonymousToken,
};
