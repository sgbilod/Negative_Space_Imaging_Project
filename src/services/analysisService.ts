/**
 * Analysis Service Module
 *
 * Handles image analysis orchestration, bridging between Express API
 * and Python analysis engine, and stores results in database.
 *
 * @module services/analysisService
 */

import { AnalysisResult } from '../models/AnalysisResult';
import { Image } from '../models/Image';
import { callPythonAnalysis } from '../utils/pythonBridge';
import { ServiceError } from '../types';
import logger from '../utils/logger';

/**
 * AnalysisService class
 * Provides methods for image analysis
 */
export class AnalysisService {
  /**
   * Start analysis on image
   * Calls Python backend and stores results
   */
  async startAnalysis(imageId: number, imagePath: string): Promise<Partial<AnalysisResult>> {
    try {
      // Verify image exists
      const image = await Image.findByPk(imageId);
      if (!image) {
        throw new ServiceError(404, 'Image not found', 'IMAGE_NOT_FOUND');
      }

      logger.info(`Starting analysis for image: ${imageId}`);

      // Call Python analysis engine
      const startTime = Date.now();
      const analysisData = await callPythonAnalysis(imagePath);
      const processingTime = Date.now() - startTime;

      logger.debug(`Analysis completed in ${processingTime}ms for image ${imageId}`);

      // Extract metrics from analysis data
      const negativeSpacePercentage =
        analysisData.negative_space_percentage || 0;
      const regionsCount = analysisData.contours_count || 0;

      // Store results in database
      const result = await AnalysisResult.create({
        image_id: imageId,
        negative_space_percentage: negativeSpacePercentage,
        regions_count: regionsCount,
        processing_time_ms: processingTime,
        raw_data: analysisData,
      });

      logger.info(
        `Analysis result stored: ${result.id} for image ${imageId}`
      );

      return result.toJSON();
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Analysis failed', error);
      throw new ServiceError(500, 'Analysis failed', 'ANALYSIS_ERROR');
    }
  }

  /**
   * Get analysis result by ID
   */
  async getAnalysisResult(id: number): Promise<Partial<AnalysisResult>> {
    try {
      const result = await AnalysisResult.findByPk(id);
      if (!result) {
        logger.warn(`Analysis result not found: ${id}`);
        throw new ServiceError(404, 'Analysis not found', 'ANALYSIS_NOT_FOUND');
      }

      logger.debug(`Retrieved analysis result: ${id}`);
      return result.serialize();
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to get analysis result', error);
      throw new ServiceError(500, 'Failed to get analysis', 'GET_ANALYSIS_ERROR');
    }
  }

  /**
   * Get all analyses for an image
   */
  async getAnalysesByImageId(
    imageId: number,
    page: number = 1,
    limit: number = 10
  ): Promise<{ results: Partial<AnalysisResult>[]; total: number }> {
    try {
      const offset = (page - 1) * limit;

      const { rows, count } = await AnalysisResult.findAndCountAll({
        where: { image_id: imageId },
        offset,
        limit,
        order: [['created_at', 'DESC']],
      });

      logger.debug(`Retrieved ${rows.length} analyses for image ${imageId}`);

      return {
        results: rows.map((result) => result.serialize()),
        total: count,
      };
    } catch (error) {
      logger.error('Failed to retrieve analyses', error);
      throw new ServiceError(
        500,
        'Failed to retrieve analyses',
        'FETCH_ANALYSES_ERROR'
      );
    }
  }

  /**
   * Get analysis statistics
   */
  async getAnalysisStats(imageId: number): Promise<object> {
    try {
      const results = await AnalysisResult.findAll({
        where: { image_id: imageId },
        attributes: [
          ['negative_space_percentage', 'negativeSpace'],
          ['regions_count', 'regionsCount'],
          ['processing_time_ms', 'processingTime'],
        ],
      });

      if (results.length === 0) {
        return {
          count: 0,
          average_negative_space: 0,
          average_regions: 0,
          average_processing_time: 0,
        };
      }

      const totalNegativeSpace = results.reduce(
        (sum, r) => sum + (r.get('negativeSpace') as number),
        0
      );
      const totalRegions = results.reduce(
        (sum, r) => sum + (r.get('regionsCount') as number),
        0
      );
      const totalTime = results.reduce(
        (sum, r) => sum + (r.get('processingTime') as number),
        0
      );

      return {
        count: results.length,
        average_negative_space: totalNegativeSpace / results.length,
        average_regions: totalRegions / results.length,
        average_processing_time: totalTime / results.length,
      };
    } catch (error) {
      logger.error('Failed to get analysis stats', error);
      throw new ServiceError(500, 'Failed to get stats', 'STATS_ERROR');
    }
  }

  /**
   * Batch analysis for multiple images
   */
  async batchAnalysis(imageIds: number[]): Promise<Partial<AnalysisResult>[]> {
    try {
      const results: Partial<AnalysisResult>[] = [];

      for (const imageId of imageIds) {
        try {
          const image = await Image.findByPk(imageId);
          if (!image) {
            logger.warn(`Image not found in batch: ${imageId}`);
            continue;
          }

          const result = await this.startAnalysis(
            imageId,
            image.storage_path
          );
          results.push(result);
        } catch (error) {
          logger.error(`Batch analysis failed for image ${imageId}`, error);
          continue;
        }
      }

      logger.info(`Batch analysis completed: ${results.length} of ${imageIds.length}`);
      return results;
    } catch (error) {
      logger.error('Batch analysis failed', error);
      throw new ServiceError(500, 'Batch analysis failed', 'BATCH_ERROR');
    }
  }

  /**
   * Check analysis quality
   */
  async isQualityAcceptable(
    resultId: number,
    minConfidence: number = 0.5
  ): Promise<boolean> {
    try {
      const result = await AnalysisResult.findByPk(resultId);
      if (!result) {
        return false;
      }

      return result.isQualityAcceptable(minConfidence);
    } catch (error) {
      logger.error('Quality check failed', error);
      return false;
    }
  }

  /**
   * Delete analysis result
   */
  async deleteAnalysis(id: number): Promise<void> {
    try {
      const result = await AnalysisResult.findByPk(id);
      if (!result) {
        throw new ServiceError(404, 'Analysis not found', 'ANALYSIS_NOT_FOUND');
      }

      await result.destroy();
      logger.info(`Analysis deleted: ${id}`);
    } catch (error) {
      if (error instanceof ServiceError) {
        throw error;
      }
      logger.error('Failed to delete analysis', error);
      throw new ServiceError(500, 'Failed to delete analysis', 'DELETE_ERROR');
    }
  }
}

export default new AnalysisService();
