import { Application } from 'express';
import { authRoutes } from './auth.routes';
import { userRoutes } from './user.routes';
import { imageRoutes } from './image.routes';
import { analysisRoutes } from './analysis.routes';
import { reportsRoutes } from './reports.routes';
import { hipaaRoutes } from './hipaa.routes';
import { adminRoutes } from './admin.routes';

/**
 * Configure all API routes for the application
 */
export const configureRoutes = (app: Application): void => {
  // API version prefix
  const apiPrefix = '/api/v1';
  
  // Register all route groups
  app.use(`${apiPrefix}/auth`, authRoutes);
  app.use(`${apiPrefix}/users`, userRoutes);
  app.use(`${apiPrefix}/images`, imageRoutes);
  app.use(`${apiPrefix}/analysis`, analysisRoutes);
  app.use(`${apiPrefix}/reports`, reportsRoutes);
  app.use(`${apiPrefix}/hipaa`, hipaaRoutes);
  app.use(`${apiPrefix}/admin`, adminRoutes);
  
  // API documentation route
  app.get(`${apiPrefix}/docs`, (_req, res) => {
    res.redirect('/api-docs');
  });
};
