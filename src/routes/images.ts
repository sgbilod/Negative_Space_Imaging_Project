/**
 * Image Routes Module
 *
 * Defines image management endpoints for upload, retrieval,
 * deletion, and image analysis workflows.
 *
 * @module routes/images
 */

import express, { Router, Request, Response, NextFunction } from 'express';
import multer from 'multer';
import ImageService from '../services/imageService';
import AnalysisService from '../services/analysisService';
import { authenticateToken } from '../middleware/auth';
import { ServiceError } from '../types';
import logger from '../utils/logger';
import { saveUploadedFile, deleteFile } from '../utils/fileUtils';

const router: Router = express.Router();

/**
 * Configure multer for file uploads
 */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB
  },
  fileFilter: (req, file, cb) => {
    const allowedMimes = [
      'image/jpeg',
      'image/png',
      'image/webp',
      'image/tiff',
    ];

    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type: ${file.mimetype}`));
    }
  },
});

/**
 * GET /images
 * Get user's images with pagination
 */
router.get(
  '/',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = (req as any).user?.id;
      const page = parseInt(req.query.page as string, 10) || 1;
      const limit = parseInt(req.query.limit as string, 10) || 10;

      logger.debug(
        `Fetching images for user ${userId} (page ${page}, limit ${limit})`
      );

      const { results, total } = await ImageService.getImagesByUserId(
        userId,
        page,
        limit
      );

      res.status(200).json({
        success: true,
        data: {
          images: results,
          pagination: {
            page,
            limit,
            total,
            pages: Math.ceil(total / limit),
          },
        },
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * GET /images/:id
 * Get image details
 */
router.get(
  '/:id',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const imageId = parseInt(req.params.id, 10);
      const userId = (req as any).user?.id;

      logger.debug(`Fetching image: ${imageId}`);

      const image = await ImageService.getImageById(imageId);

      // Verify ownership
      await ImageService.verifyImageOwnership(imageId, userId);

      res.status(200).json({
        success: true,
        data: { image },
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * POST /images
 * Upload new image
 */
router.post(
  '/',
  authenticateToken,
  upload.single('image'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = (req as any).user?.id;
      const file = req.file;

      if (!file) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'NO_FILE',
            message: 'No file uploaded',
          },
        });
      }

      logger.info(`Image upload started for user ${userId}`);

      // Save file
      const savedFile = await saveUploadedFile(file);

      // Create database record
      const image = await ImageService.uploadImage(userId, {
        ...file,
        path: savedFile,
      });

      logger.info(`Image uploaded successfully: ${image.id}`);

      res.status(201).json({
        success: true,
        data: { image },
        message: 'Image uploaded successfully',
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * DELETE /images/:id
 * Delete image
 */
router.delete(
  '/:id',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const imageId = parseInt(req.params.id, 10);
      const userId = (req as any).user?.id;

      logger.info(`Image deletion requested: ${imageId}`);

      // Verify ownership
      await ImageService.verifyImageOwnership(imageId, userId);

      // Get image path for file deletion
      const imagePath = await ImageService.getImagePath(imageId);

      // Delete from database
      await ImageService.deleteImage(imageId);

      // Delete from file system
      if (imagePath) {
        await deleteFile(imagePath);
      }

      logger.info(`Image deleted: ${imageId}`);

      res.status(200).json({
        success: true,
        message: 'Image deleted successfully',
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * POST /images/:id/analyze
 * Start analysis on image
 */
router.post(
  '/:id/analyze',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const imageId = parseInt(req.params.id, 10);
      const userId = (req as any).user?.id;

      logger.info(`Analysis started for image: ${imageId}`);

      // Verify ownership
      await ImageService.verifyImageOwnership(imageId, userId);

      // Get image path
      const imagePath = await ImageService.getImagePath(imageId);

      if (!imagePath) {
        return res.status(404).json({
          success: false,
          error: {
            code: 'IMAGE_PATH_NOT_FOUND',
            message: 'Image path not found',
          },
        });
      }

      // Start analysis
      const result = await AnalysisService.startAnalysis(imageId, imagePath);

      logger.info(`Analysis completed for image: ${imageId}`);

      res.status(200).json({
        success: true,
        data: { analysis: result },
        message: 'Analysis completed',
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * GET /images/:id/analyses
 * Get all analyses for image
 */
router.get(
  '/:id/analyses',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const imageId = parseInt(req.params.id, 10);
      const userId = (req as any).user?.id;
      const page = parseInt(req.query.page as string, 10) || 1;
      const limit = parseInt(req.query.limit as string, 10) || 10;

      logger.debug(`Fetching analyses for image: ${imageId}`);

      // Verify ownership
      await ImageService.verifyImageOwnership(imageId, userId);

      // Get analyses
      const { results, total } = await AnalysisService.getAnalysesByImageId(
        imageId,
        page,
        limit
      );

      res.status(200).json({
        success: true,
        data: {
          analyses: results,
          pagination: {
            page,
            limit,
            total,
            pages: Math.ceil(total / limit),
          },
        },
      });
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

/**
 * GET /images/:id/download
 * Download image file
 */
router.get(
  '/:id/download',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const imageId = parseInt(req.params.id, 10);
      const userId = (req as any).user?.id;

      logger.debug(`Download requested for image: ${imageId}`);

      // Verify ownership
      await ImageService.verifyImageOwnership(imageId, userId);

      // Get image path
      const imagePath = await ImageService.getImagePath(imageId);

      if (!imagePath) {
        return res.status(404).json({
          success: false,
          error: {
            code: 'IMAGE_NOT_FOUND',
            message: 'Image not found',
          },
        });
      }

      // Send file
      res.download(imagePath);
    } catch (error) {
      if (error instanceof ServiceError) {
        return res.status(error.statusCode).json({
          success: false,
          error: {
            code: error.code,
            message: error.message,
          },
        });
      }
      next(error);
    }
  }
);

export default router;
