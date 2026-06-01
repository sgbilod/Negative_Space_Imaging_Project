/**
 * Image Service Module
 *
 * Handles image upload, storage, retrieval, and metadata management.
 * Integrates with file system and database.
 *
 * @module services/imageService
 */

import path from 'path';
import { Image } from '../models/Image';
import { User } from '../models/User';
import { saveUploadedFile, deleteFile } from '../utils/fileUtils';
import { ServiceError } from '../types';
import logger from '../utils/logger';

/**
 * ImageService class
 * Provides methods for image management
 */
export class ImageService {
  private uploadDir = process.env.UPLOAD_DIR || './uploads';

  /**
   * Upload and store image file
   */
  async uploadImage(
    userId: number,
    file: Express.Multer.File
  ): Promise<Partial<Image>> {
    try {
      // Verify user exists
      const user = await User.findByPk(userId);
      if (!user) {
        throw new ServiceError(404, 'User not found', 'USER_NOT_FOUND');
      }

      // Save file to storage
      const storagePath = await saveUploadedFile(file, this.uploadDir);
      logger.debug(`File saved to: ${storagePath}`);

      // Create image record
      const image = await Image.create({
        user_id: userId,
        filename: path.basename(storagePath),
        original_filename: file.originalname,
        file_size: file.size,
        storage_path: storagePath,
      });

      logger.info(`Image uploaded: ${image.id} by user ${userId}`);

      return image.toJSON();
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Image upload failed', error);
      throw new ServiceError(500, 'Image upload failed', 'UPLOAD_ERROR');
    }
  }

  /**
   * Get all images for user with pagination
   */
  async getImagesByUserId(
    userId: number,
    page: number = 1,
    limit: number = 10
  ): Promise<{ images: Partial<Image>[]; total: number }> {
    try {
      const offset = (page - 1) * limit;

      const { rows, count } = await Image.findAndCountAll({
        where: { user_id: userId },
        offset,
        limit,
        order: [['uploaded_at', 'DESC']],
      });

      logger.debug(`Retrieved ${rows.length} images for user ${userId}`);

      return {
        images: rows.map((img) => img.toJSON()),
        total: count,
      };
    } catch (error) {
      logger.error('Failed to retrieve images', error);
      throw new ServiceError(500, 'Failed to retrieve images', 'FETCH_IMAGES_ERROR');
    }
  }

  /**
   * Get image by ID
   */
  async getImageById(id: number): Promise<Partial<Image>> {
    try {
      const image = await Image.findByPk(id);
      if (!image) {
        logger.warn(`Image not found: ${id}`);
        throw new ServiceError(404, 'Image not found', 'IMAGE_NOT_FOUND');
      }

      logger.debug(`Retrieved image: ${id}`);
      return image.toJSON();
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to get image', error);
      throw new ServiceError(500, 'Failed to get image', 'GET_IMAGE_ERROR');
    }
  }

  /**
   * Delete image and associated file
   */
  async deleteImage(id: number): Promise<void> {
    try {
      const image = await Image.findByPk(id);
      if (!image) {
        throw new ServiceError(404, 'Image not found', 'IMAGE_NOT_FOUND');
      }

      // Delete file from storage
      await deleteFile(image.storage_path);
      logger.debug(`File deleted: ${image.storage_path}`);

      // Delete database record (cascade will remove analysis results)
      await image.destroy();
      logger.info(`Image deleted: ${id}`);
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to delete image', error);
      throw new ServiceError(500, 'Failed to delete image', 'DELETE_IMAGE_ERROR');
    }
  }

  /**
   * Get image file path
   */
  async getImagePath(id: number): Promise<string> {
    try {
      const image = await Image.findByPk(id);
      if (!image) {
        throw new ServiceError(404, 'Image not found', 'IMAGE_NOT_FOUND');
      }

      return image.storage_path;
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to get image path', error);
      throw new ServiceError(500, 'Failed to get image path', 'GET_PATH_ERROR');
    }
  }

  /**
   * Verify user owns image
   */
  async verifyImageOwnership(imageId: number, userId: number): Promise<boolean> {
    try {
      const image = await Image.findOne({
        where: { id: imageId, user_id: userId },
      });
      return !!image;
    } catch (error) {
      logger.error('Ownership verification failed', error);
      return false;
    }
  }

  /**
   * Get image statistics for user
   */
  async getUserImageStats(userId: number): Promise<object> {
    try {
      const { count, sum } = await Image.findOne({
        attributes: [
          ['COUNT(*)', 'count'],
          ['SUM(file_size)', 'totalSize'],
        ],
        where: { user_id: userId },
        raw: true,
      }) as any;

      return {
        totalImages: count || 0,
        totalSize: sum || 0,
      };
    } catch (error) {
      logger.error('Failed to get image stats', error);
      throw new ServiceError(500, 'Failed to get image stats', 'STATS_ERROR');
    }
  }
}

export default new ImageService();
