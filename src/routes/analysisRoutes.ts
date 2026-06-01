/**
 * Analysis Results Routes
 * Handles retrieval, export, and comparison of image analysis results
 *
 * Routes:
 * GET    /:imageId              - Get analysis results for image
 * GET    /:imageId/export       - Export results as JSON/CSV
 * POST   /:imageId/compare      - Compare with other analyses
 * GET                           - List all user's analyses
 */

import { Router, Response } from 'express';
import Joi from 'joi';
import { json2csv } from 'json2csv';

import { AuthenticatedRequest, authenticateToken } from '../middleware/authMiddleware';
import { asyncHandler, NotFoundError, ValidationError, AppError } from '../middleware/errorHandler';
import { log } from '../services/loggingService';

const router = Router();

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

/**
 * Schema for export options
 */
const exportSchema = Joi.object({
  format: Joi.string().valid('json', 'csv').default('json'),
  fields: Joi.array().items(Joi.string()).optional(),
  includeMetadata: Joi.boolean().default(true),
});

/**
 * Schema for comparison
 */
const compareSchema = Joi.object({
  compareImageIds: Joi.array().items(Joi.string().required()).min(1).required(),
  metrics: Joi.array().items(Joi.string()).optional(),
});

/**
 * Schema for pagination
 */
const paginationSchema = Joi.object({
  limit: Joi.number().integer().min(1).max(100).default(20),
  offset: Joi.number().integer().min(0).default(0),
  sort: Joi.string().valid('created', 'algorithm', 'status').default('created'),
  order: Joi.string().valid('asc', 'desc').default('desc'),
  status: Joi.string().valid('queued', 'processing', 'completed', 'failed').optional(),
}).unknown(true);

// ============================================================================
// TYPES
// ============================================================================

interface AnalysisResult {
  id: string;
  imageId: string;
  userId: string;
  algorithmType: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  results: {
    spatialMetrics?: Record<string, any>;
    frequencyAnalysis?: Record<string, any>;
    statistics?: Record<string, any>;
    detectionsCount?: number;
    confidence?: number;
  };
  quality?: {
    signalNoiseRatio: number;
    contrast: number;
    brightness: number;
    sharpness: number;
  };
  metadata?: Record<string, any>;
  error?: string;
}

interface ComparisonResult {
  baseImageId: string;
  comparisonImageIds: string[];
  similarityScores: Record<string, number>;
  differences: {
    spatialMetrics: Record<string, any>;
    frequencyAnalysis: Record<string, any>;
  };
  recommendations: string[];
  timestamp: string;
}

// ============================================================================
// ROUTES
// ============================================================================

/**
 * GET /:imageId
 * Get all analysis results for a specific image
 * Requires authentication (owner or admin)
 *
 * @param {string} imageId - Image ID
 *
 * @returns {Array} Array of analysis results
 * @throws {NotFoundError} - Image not found
 *
 * @example
 * GET /api/v1/analysis/img-123
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * [
 *   {
 *     "id": "result-1",
 *     "imageId": "img-123",
 *     "algorithmType": "advanced",
 *     "status": "completed",
 *     "completedAt": "2025-10-17T10:30:00Z",
 *     "duration": 125000,
 *     "results": {
 *       "spatialMetrics": { ... },
 *       "statistics": { ... },
 *       "detectionsCount": 42,
 *       "confidence": 0.95
 *     },
 *     "quality": {
 *       "signalNoiseRatio": 12.5,
 *       "contrast": 0.8,
 *       "brightness": 0.7,
 *       "sharpness": 0.85
 *     }
 *   },
 *   ...
 * ]
 */
router.get(
  '/:imageId',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res: Response<AnalysisResult[]>) => {
    const imageId = req.params.imageId;
    const userId = req.userId;

    // Check permissions (mock - verify user owns image)
    await verifyImageOwnership(imageId, userId);

    // Fetch analysis results (mock)
    const results = await getAnalysisResultsByImageId(imageId);

    log.info('Analysis results retrieved', { userId, imageId, count: results.length });

    res.status(200).json(results);
  }),
);

/**
 * GET /:imageId/export
 * Export analysis results as JSON or CSV
 * Requires authentication (owner or admin)
 *
 * @param {string} imageId - Image ID
 * @query {string} [format=json] - Export format (json, csv)
 * @query {string[]} [fields] - Fields to include
 * @query {boolean} [includeMetadata=true] - Include metadata
 *
 * @returns {File|Object} Exported data
 * @throws {NotFoundError} - Image not found
 * @throws {ValidationError} - Invalid export parameters
 *
 * @example
 * GET /api/v1/analysis/img-123/export?format=csv&includeMetadata=true
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK (CSV file)
 * id,algorithm,status,duration,detectionsCount,confidence...
 * result-1,advanced,completed,125000,42,0.95...
 */
router.get(
  '/:imageId/export',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res) => {
    const imageId = req.params.imageId;
    const userId = req.userId;

    // Validate export options
    const { error, value } = exportSchema.validate(req.query);
    if (error) {
      throw new ValidationError(
        [{ field: 'export', message: error.message }],
        'Invalid export parameters',
      );
    }

    const { format, fields, includeMetadata } = value;

    // Check permissions (mock)
    await verifyImageOwnership(imageId, userId);

    // Fetch analysis results (mock)
    const results = await getAnalysisResultsByImageId(imageId);

    if (results.length === 0) {
      throw new NotFoundError('Analysis results');
    }

    log.info('Analysis results exported', { userId, imageId, format });

    if (format === 'csv') {
      try {
        const csv = json2csv({ data: formatResultsForExport(results, fields, includeMetadata) });
        res.header('Content-Type', 'text/csv');
        res.header('Content-Disposition', `attachment; filename="analysis-${imageId}.csv"`);
        res.send(csv);
      } catch (error) {
        throw new AppError(500, 'Failed to export as CSV');
      }
    } else {
      // JSON export
      res.header('Content-Type', 'application/json');
      res.header('Content-Disposition', `attachment; filename="analysis-${imageId}.json"`);
      res.json({
        imageId,
        exportedAt: new Date().toISOString(),
        resultCount: results.length,
        results: formatResultsForExport(results, fields, includeMetadata),
      });
    }
  }),
);

/**
 * POST /:imageId/compare
 * Compare analysis results of multiple images
 * Requires authentication (owner or admin)
 *
 * @param {string} imageId - Base image ID
 * @body {string[]} compareImageIds - Image IDs to compare with (required)
 * @body {string[]} [metrics] - Specific metrics to compare
 *
 * @returns {Object} Comparison results and similarity scores
 * @throws {NotFoundError} - Image not found
 * @throws {ValidationError} - Invalid comparison parameters
 *
 * @example
 * POST /api/v1/analysis/img-123/compare
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 * {
 *   "compareImageIds": ["img-124", "img-125"],
 *   "metrics": ["contrast", "sharpness", "detectionsCount"]
 * }
 *
 * Response: 200 OK
 * {
 *   "baseImageId": "img-123",
 *   "comparisonImageIds": ["img-124", "img-125"],
 *   "similarityScores": {
 *     "img-124": 0.92,
 *     "img-125": 0.78
 *   },
 *   "differences": {
 *     "spatialMetrics": { ... },
 *     "frequencyAnalysis": { ... }
 *   },
 *   "recommendations": [
 *     "img-123 and img-124 are highly similar in spatial characteristics",
 *     "img-125 shows different frequency components"
 *   ]
 * }
 */
router.post(
  '/:imageId/compare',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res: Response<ComparisonResult>) => {
    const imageId = req.params.imageId;
    const userId = req.userId;

    // Validate comparison parameters
    const { error, value } = compareSchema.validate(req.body);
    if (error) {
      throw new ValidationError(
        [{ field: 'compare', message: error.message }],
        'Invalid comparison parameters',
      );
    }

    const { compareImageIds, metrics } = value;

    // Check permissions (mock - verify user owns all images)
    await verifyImageOwnership(imageId, userId);
    for (const id of compareImageIds) {
      await verifyImageOwnership(id, userId);
    }

    // Fetch base image results (mock)
    const baseResults = await getAnalysisResultsByImageId(imageId);
    if (baseResults.length === 0) {
      throw new NotFoundError('Analysis results for base image');
    }

    // Fetch comparison image results (mock)
    const comparisonResults: Record<string, AnalysisResult[]> = {};
    for (const id of compareImageIds) {
      const results = await getAnalysisResultsByImageId(id);
      if (results.length === 0) {
        throw new NotFoundError(`Analysis results for image ${id}`);
      }
      comparisonResults[id] = results;
    }

    // Calculate comparison (mock implementation)
    const comparison = calculateComparison(
      baseResults[0],
      comparisonResults,
      compareImageIds,
      metrics,
    );

    log.info('Analysis comparison generated', {
      userId,
      imageId,
      compareCount: compareImageIds.length,
    });

    res.status(200).json({
      baseImageId: imageId,
      comparisonImageIds: compareImageIds,
      similarityScores: comparison.similarityScores,
      differences: comparison.differences,
      recommendations: comparison.recommendations,
      timestamp: new Date().toISOString(),
    });
  }),
);

/**
 * GET /
 * List all user's analyses with pagination and filtering
 * Requires authentication
 *
 * @query {number} [limit=20] - Results per page (1-100)
 * @query {number} [offset=0] - Results offset
 * @query {string} [sort=created] - Sort field (created, algorithm, status)
 * @query {string} [order=desc] - Sort order (asc, desc)
 * @query {string} [status] - Filter by status (queued, processing, completed, failed)
 *
 * @returns {Object} Paginated analysis results
 *
 * @example
 * GET /api/v1/analysis?limit=10&offset=0&status=completed&sort=created&order=desc
 * Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
 *
 * Response: 200 OK
 * {
 *   "data": [
 *     {
 *       "id": "result-1",
 *       "imageId": "img-123",
 *       "algorithmType": "advanced",
 *       "status": "completed",
 *       "completedAt": "2025-10-17T10:30:00Z",
 *       "duration": 125000
 *     },
 *     ...
 *   ],
 *   "pagination": {
 *     "limit": 10,
 *     "offset": 0,
 *     "total": 125,
 *     "totalPages": 13
 *   },
 *   "timestamp": "2025-10-17T10:00:00Z"
 * }
 */
router.get(
  '/',
  authenticateToken,
  asyncHandler(async (req: AuthenticatedRequest, res) => {
    // Validate pagination
    const { error, value } = paginationSchema.validate(req.query);
    if (error) {
      throw new AppError(400, `Invalid parameters: ${error.message}`);
    }

    const { limit, offset, sort, order, status } = value;
    const userId = req.userId;

    // Fetch analyses (mock)
    const { analyses, total } = await listUserAnalyses(userId, {
      limit,
      offset,
      sort,
      order,
      status,
    });

    log.info('User analyses listed', { userId, limit, offset, total });

    res.status(200).json({
      data: analyses,
      pagination: {
        limit,
        offset,
        total,
        totalPages: Math.ceil(total / limit),
      },
      timestamp: new Date().toISOString(),
    });
  }),
);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Verify that user owns the image (mock)
 */
async function verifyImageOwnership(imageId: string, userId: string): Promise<void> {
  // TODO: Implement database query
  const image = await getImageById(imageId);
  if (!image || image.userId !== userId) {
    throw new NotFoundError('Image');
  }
}

/**
 * Get analysis results by image ID (mock)
 */
async function getAnalysisResultsByImageId(imageId: string): Promise<AnalysisResult[]> {
  // TODO: Implement database query
  return [];
}

/**
 * Get image by ID (mock)
 */
async function getImageById(imageId: string): Promise<any> {
  // TODO: Implement database query
  return null;
}

/**
 * List user analyses (mock)
 */
async function listUserAnalyses(
  userId: string,
  options: {
    limit: number;
    offset: number;
    sort: string;
    order: string;
    status?: string;
  },
): Promise<{ analyses: any[]; total: number }> {
  // TODO: Implement database query
  return { analyses: [], total: 0 };
}

/**
 * Format results for export
 */
function formatResultsForExport(
  results: AnalysisResult[],
  fields?: string[],
  includeMetadata?: boolean,
): any[] {
  return results.map((result) => {
    const formatted: any = {
      id: result.id,
      imageId: result.imageId,
      algorithm: result.algorithmType,
      status: result.status,
      duration: result.duration,
      ...result.results,
    };

    if (includeMetadata && result.quality) {
      formatted.quality = result.quality;
    }

    if (fields && fields.length > 0) {
      const filtered: any = {};
      fields.forEach((field) => {
        if (field in formatted) {
          filtered[field] = formatted[field];
        }
      });
      return filtered;
    }

    return formatted;
  });
}

/**
 * Calculate comparison between analyses
 */
function calculateComparison(
  baseResult: AnalysisResult,
  comparisonResults: Record<string, AnalysisResult[]>,
  compareImageIds: string[],
  metrics?: string[],
): Partial<ComparisonResult> {
  const similarityScores: Record<string, number> = {};
  const differences: any = {
    spatialMetrics: {},
    frequencyAnalysis: {},
  };
  const recommendations: string[] = [];

  // Mock calculation of similarity scores
  compareImageIds.forEach((id) => {
    similarityScores[id] = (Math.random() * 100) / 100; // 0-1 scale
  });

  // Generate recommendations based on similarity
  Object.entries(similarityScores).forEach(([imageId, score]) => {
    if (score > 0.8) {
      recommendations.push(`${imageId} is highly similar to the base image`);
    } else if (score < 0.5) {
      recommendations.push(`${imageId} is significantly different from the base image`);
    }
  });

  return {
    similarityScores,
    differences,
    recommendations,
  };
}

export default router;
