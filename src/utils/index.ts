/**
 * Utilities Index Module
 *
 * Central export point for all utility functions and helpers.
 * Simplifies imports throughout the application.
 *
 * @module utils/index
 */

// Export file utilities
export {
  ensureUploadDir,
  generateUniqueFilename,
  validateFileType,
  validateFileExtension,
  validateFileSize,
  saveUploadedFile,
  deleteFile,
  getFileStats,
  fileExists,
  getRelativePath,
  getAbsolutePath,
  calculateFileHash,
  cleanupOldFiles,
  getUploadDirInfo,
} from './fileUtils';

// Export Python bridge utilities
export {
  callPythonAnalysis,
  spawnPythonProcess,
  batchAnalysis,
  checkPythonAvailability,
  getAnalysisMetadata,
} from './pythonBridge';

// Export token utilities
export {
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
  type JwtPayload,
} from './tokenUtils';

// Common helpers
export function getCurrentTimestamp(): string {
  return new Date().toISOString();
}

export function getUnixTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function generateRandomId(length: number = 16): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

export function sanitizeInput(input: string): string {
  return input
    .trim()
    .replace(/[<>]/g, '')
    .slice(0, 255);
}

export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

export default {
  // File utilities
  ensureUploadDir,
  generateUniqueFilename,
  validateFileType,
  validateFileExtension,
  validateFileSize,
  saveUploadedFile,
  deleteFile,
  getFileStats,
  fileExists,
  getRelativePath,
  getAbsolutePath,
  calculateFileHash,
  cleanupOldFiles,
  getUploadDirInfo,

  // Python bridge
  callPythonAnalysis,
  spawnPythonProcess,
  batchAnalysis,
  checkPythonAvailability,
  getAnalysisMetadata,

  // Token utilities
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

  // Common helpers
  getCurrentTimestamp,
  getUnixTimestamp,
  delay,
  generateRandomId,
  sanitizeInput,
  isValidEmail,
  isValidUrl,
};
