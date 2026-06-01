/**
 * Health Check Routes
 * Endpoints for service health monitoring and load balancing
 */

import { Router, Response } from 'express';
import { AuthenticatedRequest, requestIdMiddleware } from '../middleware/authMiddleware';
import { asyncHandler } from '../middleware/errorHandler';
import { configService } from '../config/serverConfig';
import { log } from '../services/loggingService';

export const healthRouter = Router();

/**
 * GET /health
 * Basic health check endpoint
 */
healthRouter.get(
  '/health',
  requestIdMiddleware,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const health = {
      status: 'ok',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      version: process.env.npm_package_version || '1.0.0',
      requestId: req.requestId,
    };

    res.json(health);
  }),
);

/**
 * GET /health/detailed
 * Detailed health check with database and Redis status
 */
healthRouter.get(
  '/health/detailed',
  requestIdMiddleware,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const startTime = Date.now();
    const config = configService.getConfig();

    try {
      // Check service health
      const serviceHealth = await configService.checkHealth();

      const detailedHealth = {
        status: serviceHealth.database && !serviceHealth.errors.length ? 'healthy' : 'degraded',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: config.server.nodeEnv,
        version: process.env.npm_package_version || '1.0.0',
        requestId: req.requestId,
        services: {
          database: {
            status: serviceHealth.database ? 'connected' : 'disconnected',
          },
          redis: {
            status: serviceHealth.redis ? 'connected' : 'disconnected',
            enabled: config.redis.enabled,
          },
        },
        errors: serviceHealth.errors,
        responseTime: Date.now() - startTime,
      };

      const statusCode = serviceHealth.errors.length > 0 ? 503 : 200;

      log.performance('Health check', Date.now() - startTime, 'ms');

      res.status(statusCode).json(detailedHealth);
    } catch (error) {
      log.error('Health check failed', error as Error, {
        requestId: req.requestId,
      });

      res.status(503).json({
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: (error as Error).message,
        requestId: req.requestId,
      });
    }
  }),
);

/**
 * GET /health/ready
 * Kubernetes readiness probe
 */
healthRouter.get(
  '/health/ready',
  requestIdMiddleware,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    try {
      const serviceHealth = await configService.checkHealth();

      if (serviceHealth.database) {
        res.status(200).json({ ready: true });
      } else {
        res.status(503).json({ ready: false, reason: 'Database not ready' });
      }
    } catch (error) {
      res.status(503).json({
        ready: false,
        reason: (error as Error).message,
      });
    }
  }),
);

/**
 * GET /health/live
 * Kubernetes liveness probe
 */
healthRouter.get(
  '/health/live',
  requestIdMiddleware,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    // Simple check - if server is responding, it's alive
    res.status(200).json({
      alive: true,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
    });
  }),
);

/**
 * GET /health/startup
 * Kubernetes startup probe
 */
healthRouter.get(
  '/health/startup',
  requestIdMiddleware,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    try {
      const serviceHealth = await configService.checkHealth();

      if (serviceHealth.database) {
        log.info('Startup probe passed', {
          requestId: req.requestId,
        });

        res.status(200).json({ started: true });
      } else {
        res.status(503).json({ started: false });
      }
    } catch (error) {
      res.status(503).json({
        started: false,
        error: (error as Error).message,
      });
    }
  }),
);

/**
 * GET /health/metrics
 * Performance and resource metrics
 */
healthRouter.get(
  '/health/metrics',
  requestIdMiddleware,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const memUsage = process.memoryUsage();

    const metrics = {
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: {
        rss: `${Math.round((memUsage.rss / 1024 / 1024) * 100) / 100} MB`,
        heapTotal: `${Math.round((memUsage.heapTotal / 1024 / 1024) * 100) / 100} MB`,
        heapUsed: `${Math.round((memUsage.heapUsed / 1024 / 1024) * 100) / 100} MB`,
        external: `${Math.round((memUsage.external / 1024 / 1024) * 100) / 100} MB`,
      },
      process: {
        pid: process.pid,
        version: process.version,
        platform: process.platform,
      },
    };

    res.json(metrics);
  }),
);

/**
 * GET /health/status
 * Current service status
 */
healthRouter.get(
  '/health/status',
  requestIdMiddleware,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const config = configService.getConfig();
    const serviceHealth = await configService.checkHealth();

    const status = {
      environment: config.server.nodeEnv,
      timestamp: new Date().toISOString(),
      services: {
        database: {
          configured: !!config.database.url,
          status: serviceHealth.database ? 'connected' : 'disconnected',
          pool: {
            min: config.database.pool.min,
            max: config.database.pool.max,
          },
        },
        redis: {
          enabled: config.redis.enabled,
          status: serviceHealth.redis ? 'connected' : 'disconnected',
        },
      },
      requestId: req.requestId,
    };

    res.json(status);
  }),
);

export default healthRouter;
