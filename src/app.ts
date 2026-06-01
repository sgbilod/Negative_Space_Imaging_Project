import express, { Application, Request, Response } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import { configureRoutes } from './api/routes';
import { errorHandler } from './middleware/errorHandler';
import { morganLogFormat } from './utils/logger';
import { dbConnect } from './database/connection';

import { config } from './config';
import { setupSwagger } from './swagger';
import redisClient from './redisClient';

/**
 * Initialize Express application with security and performance middlewares
 */
export const createApp = async (): Promise<Application> => {
  // Create Express application
  const app: Application = express();

  // Connect to database
  await dbConnect();

  // Security middlewares
  app.use(helmet()); // Set security-focused HTTP headers
  app.use(
    cors({
      origin: config.cors.allowedOrigins,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
      allowedHeaders: ['Content-Type', 'Authorization'],
      credentials: true,
      maxAge: 86400, // 24 hours
    }),
  );

  // Swagger API docs
  setupSwagger(app);

  // Rate limiting to prevent abuse
  app.use(
    rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // Limit each IP to 100 requests per windowMs
      standardHeaders: true,
      legacyHeaders: false,
      message: 'Too many requests from this IP, please try again later',
    }),
  );

  // Performance middlewares
  app.use(compression()); // Compress responses
  app.use(express.json({ limit: '1mb' })); // Parse JSON bodies with size limit
  app.use(express.urlencoded({ extended: true, limit: '1mb' })); // Parse URL-encoded bodies

  // Logging middleware
  app.use(morgan(morganLogFormat));

  // Health check endpoint
  app.get('/health', (_req: Request, res: Response) => {
    res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
  });

  // Example Redis caching usage in API
  app.get('/api/data/:id', async (req: Request, res: Response) => {
    const key = `data:${req.params.id}`;
    redisClient.get(key, (data: any) => {
      if (data) return res.json(JSON.parse(data));
      // Simulate DB fetch and cache
      const dbData = { id: req.params.id, value: 'example' };
      redisClient.set(key, JSON.stringify(dbData));
      return res.json(dbData);
    });
  });

  // Configure API routes
  configureRoutes(app);

  // Error handling middleware (must be last)
  app.use(errorHandler);

  return app;
};
