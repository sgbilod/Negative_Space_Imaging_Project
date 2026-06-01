/**
 * Analysis Routes Module
 *
 * Defines analysis endpoints for retrieving results,
 * batch analysis, and real-time status tracking.
 *
 * @module routes/analysis
 */

import express, { Router, Request, Response, NextFunction } from 'express';
import AnalysisService from '../services/analysisService';
import ImageService from '../services/imageService';
import { authenticateToken } from '../middleware/auth';
import { ServiceError } from '../types';
import logger from '../utils/logger';

const router: Router = express.Router();

/**
 * GET /analysis/:id
 * Get analysis result by ID
 */
router.get(
  '/:id',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const analysisId = parseInt(req.params.id, 10);

      logger.debug(`Fetching analysis: ${analysisId}`);

      const result = await AnalysisService.getAnalysisResult(analysisId);

      res.status(200).json({
        success: true,
        data: { analysis: result },
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
 * GET /analysis/:id/status
 * Check analysis status (real-time status check)
 */
router.get(
  '/:id/status',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const analysisId = parseInt(req.params.id, 10);

      logger.debug(`Checking analysis status: ${analysisId}`);

      const result = await AnalysisService.getAnalysisResult(analysisId);

      const status = {
        id: analysisId,
        completed: true,
        processing_time_ms: (result as any).processing_time_ms || 0,
        quality_acceptable: await AnalysisService.isQualityAcceptable(
          analysisId
        ),
        timestamp: new Date().toISOString(),
      };

      res.status(200).json({
        success: true,
        data: { status },
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
 * POST /analysis/batch
 * Submit batch analysis for multiple images
 */
router.post(
  '/batch',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = (req as any).user?.id;
      const { imageIds } = req.body;

      if (!imageIds || !Array.isArray(imageIds) || imageIds.length === 0) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'INVALID_INPUT',
            message: 'imageIds must be a non-empty array',
          },
        });
      }

      if (imageIds.length > 100) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'TOO_MANY_IMAGES',
            message: 'Maximum 100 images per batch',
          },
        });
      }

      logger.info(
        `Batch analysis started for user ${userId} with ${imageIds.length} images`
      );

      // Verify all images belong to user and get paths
      const imagePaths: { id: number; path: string }[] = [];

      for (const imageId of imageIds) {
        try {
          await ImageService.verifyImageOwnership(imageId, userId);
          const imagePath = await ImageService.getImagePath(imageId);
          if (imagePath) {
            imagePaths.push({ id: imageId, path: imagePath });
          }
        } catch (error) {
          logger.warn(
            `Skipping image ${imageId}: not found or not owned by user`
          );
        }
      }

      if (imagePaths.length === 0) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'NO_VALID_IMAGES',
            message: 'No valid images found for analysis',
          },
        });
      }

      // Start batch analysis
      const results = await AnalysisService.batchAnalysis(
        imagePaths.map((ip) => ip.path)
      );

      const successCount = results.size;
      const failureCount = imagePaths.length - successCount;

      logger.info(
        `Batch analysis completed: ${successCount} succeeded, ${failureCount} failed`
      );

      res.status(200).json({
        success: true,
        data: {
          batch: {
            totalRequested: imageIds.length,
            totalAnalyzed: successCount,
            totalFailed: failureCount,
            results: Array.from(results.entries()).map(([path, analysis]) => ({
              path,
              analysis,
            })),
          },
        },
        message: `Batch analysis completed: ${successCount}/${imagePaths.length}`,
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
 * GET /analysis/image/:imageId
 * Get all analyses for specific image
 */
router.get(
  '/image/:imageId',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const imageId = parseInt(req.params.imageId, 10);
      const userId = (req as any).user?.id;
      const page = parseInt(req.query.page as string, 10) || 1;
      const limit = parseInt(req.query.limit as string, 10) || 10;

      logger.debug(
        `Fetching analyses for image ${imageId} (page ${page}, limit ${limit})`
      );

      // Verify image ownership
      await ImageService.verifyImageOwnership(imageId, userId);

      // Get analyses
      const { results, total } = await AnalysisService.getAnalysesByImageId(
        imageId,
        page,
        limit
      );

      // Get statistics
      const stats = await AnalysisService.getAnalysisStats(imageId);

      res.status(200).json({
        success: true,
        data: {
          analyses: results,
          statistics: stats,
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
 * DELETE /analysis/:id
 * Delete analysis result
 */
router.delete(
  '/:id',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const analysisId = parseInt(req.params.id, 10);

      logger.info(`Deleting analysis: ${analysisId}`);

      await AnalysisService.deleteAnalysis(analysisId);

      logger.info(`Analysis deleted: ${analysisId}`);

      res.status(200).json({
        success: true,
        message: 'Analysis deleted successfully',
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
 * GET /analysis/stats/user
 * Get analysis statistics for current user
 */
router.get(
  '/stats/user',
  authenticateToken,
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userId = (req as any).user?.id;

      logger.debug(`Fetching analysis stats for user: ${userId}`);

      const stats = {
        userId,
        totalAnalysesPerformed: 0,
        averageProcessingTime: 0,
        averageNegativeSpace: 0,
        totalImagesAnalyzed: 0,
        lastAnalysisTime: new Date().toISOString(),
      };

      res.status(200).json({
        success: true,
        data: { stats },
      });
    } catch (error) {
      next(error);
    }
  }
);

export default router;
