/**
 * Express Application Server
 * Main server setup with middleware, routing, and error handling
 */

import express, {
  Express,
  Request,
  Response,
  NextFunction,
  json,
  urlencoded,
  static as expressStatic,
} from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import morgan from 'morgan';
import path from 'path';

import apiRoutes from '../routes';
import { log, logger } from '../services/loggingService';
import { configService, AppConfig } from '../config/serverConfig';
import {
  requestIdMiddleware,
  requestLoggingMiddleware,
  authenticateToken,
  AuthenticatedRequest,
} from '../middleware/authMiddleware';
import { errorHandler, notFoundHandler, AppError } from '../middleware/errorHandler';
import healthRouter from '../routes/healthRoutes';

export interface ServerOptions {
  config?: AppConfig;
  customMiddleware?: Array<(app: Express) => void>;
  customRoutes?: Array<(app: Express) => void>;
}

/**
 * Create Express application with production-ready middleware stack
 */
export function createApp(options: ServerOptions = {}): Express {
  const app = express();
  const config = options.config || configService.getConfig();

  // ============================================================================
  // SECURITY MIDDLEWARE
  // ============================================================================

  // Helmet helps secure Express apps by setting various HTTP headers
  app.use(
    helmet({
      contentSecurityPolicy: false, // Disable for API
      frameguard: { action: 'deny' },
      noSniff: true,
      xssFilter: true,
    }),
  );

  // ============================================================================
  // CORS MIDDLEWARE
  // ============================================================================

  app.use(
    cors({
      origin: config.cors.origin,
      credentials: config.cors.credentials,
      maxAge: config.cors.maxAge,
      methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    }),
  );

  // ============================================================================
  // COMPRESSION MIDDLEWARE
  // ============================================================================

  app.use(
    compression({
      filter: (req, res) => {
        if (req.headers['x-no-compression']) {
          return false;
        }
        return compression.filter(req, res);
      },
      level: 6, // Balance between compression ratio and speed
    }),
  );

  // ============================================================================
  // REQUEST PARSING MIDDLEWARE
  // ============================================================================

  app.use(json({ limit: '50mb' }));
  app.use(urlencoded({ limit: '50mb', extended: true }));

  // ============================================================================
  // REQUEST TRACKING MIDDLEWARE
  // ============================================================================

  // Add unique ID to each request
  app.use(requestIdMiddleware);

  // Log all requests
  if (config.logging.requestLogging) {
    app.use(requestLoggingMiddleware);

    // Morgan for HTTP request logging
    app.use(
      morgan((tokens, req, res) => {
        const statusCode = parseInt(tokens.status(req, res) || '200', 10);
        const color = statusCode >= 400 ? '\x1b[31m' : '\x1b[32m'; // Red for errors, green for success
        const reset = '\x1b[0m';

        return `${color}${statusCode}${reset} ${tokens.method(req, res)} ${tokens.url(req, res)} ${tokens.response_time(req, res)}ms`;
      }),
    );
  }

  // ============================================================================
  // RATE LIMITING MIDDLEWARE
  // ============================================================================

  const limiter = rateLimit({
    windowMs: config.rateLimit.windowMs,
    max: config.rateLimit.max,
    message: config.rateLimit.message,
    standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
    legacyHeaders: false, // Disable the `X-RateLimit-*` headers
    skip: (req) => {
      // Skip rate limiting for health checks
      return req.path.startsWith('/health');
    },
    keyGenerator: (req) => {
      // Use user ID if authenticated, otherwise use IP
      const authReq = req as AuthenticatedRequest;
      return authReq.userId || req.ip || 'unknown';
    },
  });

  app.use(limiter);

  // ============================================================================
  // STATIC FILES MIDDLEWARE
  // ============================================================================

  app.use(expressStatic(path.join(process.cwd(), 'public')));

  // ============================================================================
  // APPLY CUSTOM MIDDLEWARE
  // ============================================================================

  if (options.customMiddleware) {
    options.customMiddleware.forEach((middleware) => {
      middleware(app);
    });
  }

  // ============================================================================
  // API ROUTES
  // ============================================================================

  // Health check endpoints (no authentication required)
  app.use('/health', healthRouter);
  app.use('/', healthRouter); // Also available at root for Kubernetes probes

  // API v1 routes with all endpoints
  app.use(`/api/v${config.server.apiVersion}`, apiRoutes);

  // ============================================================================
  // APPLY CUSTOM ROUTES
  // ============================================================================

  if (options.customRoutes) {
    options.customRoutes.forEach((routes) => {
      routes(app);
    });
  }

  // ============================================================================
  // 404 HANDLER
  // ============================================================================

  app.use(notFoundHandler);

  // ============================================================================
  // ERROR HANDLING MIDDLEWARE
  // ============================================================================

  // Global error handler (MUST be last)
  app.use(errorHandler);

  return app;
}

/**
 * Start the Express server
 */
export async function startServer(options: ServerOptions = {}): Promise<void> {
  const app = createApp(options);
  const config = options.config || configService.getConfig();

  try {
    // Initialize database
    log.info('Initializing database connection pool...');
    await configService.initializeDatabase();

    // Initialize Redis (optional)
    if (config.redis.enabled) {
      log.info('Initializing Redis connection...');
      await configService.initializeRedis();
    }

    // Start HTTP server
    const server = app.listen(config.server.port, config.server.host, () => {
      log.info(`🚀 Server started successfully`, {
        host: config.server.host,
        port: config.server.port,
        environment: config.server.nodeEnv,
      });
    });

    // Graceful shutdown
    const gracefulShutdown = async () => {
      log.info('Shutting down gracefully...');

      // Stop accepting new requests
      server.close(async () => {
        log.info('HTTP server closed');

        // Close database connections
        await configService.closeDatabasePool();

        // Close Redis connection
        await configService.closeRedisClient();

        log.info('All connections closed');
        process.exit(0);
      });

      // Force shutdown after 30 seconds
      setTimeout(() => {
        log.error('Forced shutdown after 30 seconds');
        process.exit(1);
      }, 30000);
    };

    // Handle shutdown signals
    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      log.error('Uncaught Exception', error);
      gracefulShutdown();
    });

    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason) => {
      log.error('Unhandled Rejection', new Error(String(reason)));
      gracefulShutdown();
    });
  } catch (error) {
    log.error('Failed to start server', error as Error);
    process.exit(1);
  }
}

export default createApp;
