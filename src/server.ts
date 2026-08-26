/**
 * Server Bootstrap Module
 *
 * Initializes database, creates Express app, and starts the server.
 * Entry point for the Express API backend.
 *
 * @module server
 */

import app from './app';
import { sequelize } from './config/database';
import logger from './utils/logger';
import { checkPythonAvailability } from './utils/pythonBridge';
import { ensureUploadDir } from './utils/fileUtils';

const PORT = process.env.PORT || 5000;
const HOST = process.env.HOST || 'localhost';

/**
 * Initialize and start server
 */
async function startServer(): Promise<void> {
  try {
    // ============================================================
    // DATABASE CONNECTION
    // ============================================================

    logger.info('Initializing database connection...');

    try {
      await sequelize.authenticate();
      logger.info('Database connection established');

      // Sync models
      await sequelize.sync({ alter: process.env.NODE_ENV !== 'production' });
      logger.info('Database models synchronized');
    } catch (dbError) {
      logger.error('Database initialization failed', dbError);
      throw new Error('Failed to connect to database');
    }

    // ============================================================
    // EXTERNAL SERVICES CHECK
    // ============================================================

    logger.info('Checking external services...');

    // Check Python availability
    const pythonAvailable = await checkPythonAvailability();
    if (pythonAvailable) {
      logger.info('Python environment available for analysis');
    } else {
      logger.warn(
        'Python environment not available - analysis features will fail'
      );
    }

    // Ensure upload directory exists
    await ensureUploadDir();
    logger.info('Upload directory ready');

    // ============================================================
    // START HTTP SERVER
    // ============================================================

    const server = app.listen(PORT, HOST, () => {
      logger.info(
        `🚀 Server running at http://${HOST}:${PORT}`
      );
      logger.info(`📚 API Documentation: http://${HOST}:${PORT}/api/v1`);
      logger.info(`💚 Health check: http://${HOST}:${PORT}/health`);
      logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
    });

    // ============================================================
    // GRACEFUL SHUTDOWN
    // ============================================================

    const gracefulShutdown = async (signal: string) => {
      logger.info(`\n${signal} received, starting graceful shutdown...`);

      // Stop accepting new requests
      server.close(() => {
        logger.info('HTTP server closed');
      });

      // Close database connection
      try {
        await sequelize.close();
        logger.info('Database connection closed');
      } catch (error) {
        logger.error('Error closing database connection', error);
      }

      // Exit process
      process.exit(0);
    };

    // Handle termination signals
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception', error);
      gracefulShutdown('uncaughtException');
    });

    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('Unhandled rejection at', promise, 'reason:', reason);
    });
  } catch (error) {
    logger.error('Failed to start server', error);
    process.exit(1);
  }
}

// ============================================================
// STARTUP
// ============================================================

// Only start server if this file is executed directly
if (require.main === module) {
  startServer();
}

export { startServer };
