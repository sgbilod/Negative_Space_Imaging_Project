/**
 * File Utilities Module
 *
 * Handles file operations including upload handling, deletion,
 * validation, and storage path management.
 *
 * @module utils/fileUtils
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { promises as fsPromises } from 'fs';
import logger from './logger';

// Configuration
const UPLOAD_DIR = path.join(process.cwd(), 'uploads', 'images');
const ALLOWED_MIME_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/tiff',
];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'];

/**
 * Ensure upload directory exists
 */
export async function ensureUploadDir(): Promise<void> {
  try {
    await fsPromises.mkdir(UPLOAD_DIR, { recursive: true });
    logger.debug(`Upload directory ready: ${UPLOAD_DIR}`);
  } catch (error) {
    logger.error('Failed to create upload directory', error);
    throw error;
  }
}

/**
 * Generate unique filename
 */
export function generateUniqueFilename(originalFilename: string): string {
  const ext = path.extname(originalFilename).toLowerCase();
  const timestamp = Date.now();
  const random = crypto.randomBytes(6).toString('hex');
  return `${timestamp}-${random}${ext}`;
}

/**
 * Validate file type
 */
export function validateFileType(mimeType: string): boolean {
  const isAllowed = ALLOWED_MIME_TYPES.includes(mimeType.toLowerCase());
  if (!isAllowed) {
    logger.warn(`Invalid MIME type: ${mimeType}`);
  }
  return isAllowed;
}

/**
 * Validate file extension
 */
export function validateFileExtension(filename: string): boolean {
  const ext = path.extname(filename).toLowerCase();
  const isAllowed = ALLOWED_EXTENSIONS.includes(ext);
  if (!isAllowed) {
    logger.warn(`Invalid file extension: ${ext}`);
  }
  return isAllowed;
}

/**
 * Validate file size
 */
export function validateFileSize(sizeInBytes: number): boolean {
  const isValid = sizeInBytes <= MAX_FILE_SIZE;
  if (!isValid) {
    logger.warn(
      `File size exceeds limit: ${sizeInBytes} > ${MAX_FILE_SIZE}`
    );
  }
  return isValid;
}

/**
 * Save uploaded file to disk
 * Expects file object from multer middleware
 */
export async function saveUploadedFile(
  file: Express.Multer.File | any
): Promise<string> {
  try {
    // Validate inputs
    if (!file) {
      throw new Error('No file provided');
    }

    if (!validateFileType(file.mimetype)) {
      throw new Error(`Invalid file type: ${file.mimetype}`);
    }

    if (!validateFileExtension(file.originalname)) {
      throw new Error(
        `Invalid file extension: ${path.extname(file.originalname)}`
      );
    }

    if (!validateFileSize(file.size)) {
      throw new Error(
        `File size exceeds limit: ${file.size} bytes (max: ${MAX_FILE_SIZE})`
      );
    }

    // Ensure upload directory
    await ensureUploadDir();

    // Generate unique filename
    const uniqueFilename = generateUniqueFilename(file.originalname);
    const storagePath = path.join(UPLOAD_DIR, uniqueFilename);

    // Save file
    await fsPromises.writeFile(storagePath, file.buffer);

    logger.info(
      `File saved: ${uniqueFilename} (${file.size} bytes)`
    );

    return storagePath;
  } catch (error) {
    logger.error('File upload failed', error);
    throw error;
  }
}

/**
 * Delete file from disk
 */
export async function deleteFile(filePath: string): Promise<void> {
  try {
    if (!filePath) {
      throw new Error('No file path provided');
    }

    // Security check: ensure file is in upload directory
    const resolvedPath = path.resolve(filePath);
    const resolvedUploadDir = path.resolve(UPLOAD_DIR);

    if (!resolvedPath.startsWith(resolvedUploadDir)) {
      throw new Error('Invalid file path: access denied');
    }

    // Check if file exists
    try {
      await fsPromises.access(resolvedPath, fs.constants.F_OK);
    } catch {
      logger.warn(`File not found for deletion: ${filePath}`);
      return;
    }

    // Delete file
    await fsPromises.unlink(resolvedPath);
    logger.info(`File deleted: ${filePath}`);
  } catch (error) {
    logger.error('File deletion failed', error);
    throw error;
  }
}

/**
 * Get file stats
 */
export async function getFileStats(
  filePath: string
): Promise<fs.Stats | null> {
  try {
    if (!filePath) {
      return null;
    }

    const stats = await fsPromises.stat(filePath);
    return stats;
  } catch (error) {
    logger.warn(`Failed to get file stats: ${filePath}`, error);
    return null;
  }
}

/**
 * Check if file exists
 */
export async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fsPromises.access(filePath, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get relative path from upload directory
 */
export function getRelativePath(absolutePath: string): string {
  return path.relative(UPLOAD_DIR, absolutePath);
}

/**
 * Get absolute path from relative path
 */
export function getAbsolutePath(relativePath: string): string {
  return path.join(UPLOAD_DIR, relativePath);
}

/**
 * Calculate file hash
 */
export async function calculateFileHash(filePath: string): Promise<string> {
  try {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filePath);

    return new Promise((resolve, reject) => {
      stream.on('data', (data) => {
        hash.update(data);
      });
      stream.on('end', () => {
        resolve(hash.digest('hex'));
      });
      stream.on('error', (error) => {
        reject(error);
      });
    });
  } catch (error) {
    logger.error('Hash calculation failed', error);
    throw error;
  }
}

/**
 * Clean up old files (for maintenance)
 */
export async function cleanupOldFiles(maxAgeMs: number): Promise<number> {
  try {
    const now = Date.now();
    let deletedCount = 0;

    const files = await fsPromises.readdir(UPLOAD_DIR);

    for (const filename of files) {
      const filePath = path.join(UPLOAD_DIR, filename);
      const stats = await fsPromises.stat(filePath);

      if (now - stats.mtimeMs > maxAgeMs) {
        await fsPromises.unlink(filePath);
        deletedCount++;
        logger.debug(`Cleaned up old file: ${filename}`);
      }
    }

    if (deletedCount > 0) {
      logger.info(`Cleaned up ${deletedCount} old files`);
    }

    return deletedCount;
  } catch (error) {
    logger.error('Cleanup failed', error);
    return 0;
  }
}

/**
 * Get upload directory info
 */
export async function getUploadDirInfo(): Promise<{
  path: string;
  fileCount: number;
  totalSize: number;
}> {
  try {
    const files = await fsPromises.readdir(UPLOAD_DIR);
    let totalSize = 0;

    for (const filename of files) {
      const filePath = path.join(UPLOAD_DIR, filename);
      const stats = await fsPromises.stat(filePath);
      totalSize += stats.size;
    }

    return {
      path: UPLOAD_DIR,
      fileCount: files.length,
      totalSize,
    };
  } catch (error) {
    logger.error('Failed to get upload directory info', error);
    throw error;
  }
}
