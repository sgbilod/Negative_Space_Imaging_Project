import { createApp } from './app';
import { config } from './config';
import { logger } from './utils/logger';

/**
 * Application entry point
 */
const startServer = async (): Promise<void> => {
  try {
    const app = await createApp();
    
    // Start the server
    const server = app.listen(config.server.port, () => {
      logger.info(
        `Server started in ${config.server.env} mode on port ${config.server.port}`
      );
      logger.info(`HIPAA compliance mode: ${config.hipaa.enabled ? 'Enabled' : 'Disabled'}`);
      logger.info(`Visit: http://localhost:${config.server.port}`);
    });

    // Handle server shutdown gracefully
    const shutdownGracefully = async (signal: string): Promise<void> => {
      logger.info(`${signal} received, shutting down gracefully`);
      server.close(() => {
        logger.info('HTTP server closed');
        process.exit(0);
      });

      // Force close after 10 seconds
      setTimeout(() => {
        logger.error('Could not close connections in time, forcefully shutting down');
        process.exit(1);
      }, 10000);
    };

    // Listen for termination signals
    process.on('SIGTERM', () => shutdownGracefully('SIGTERM'));
    process.on('SIGINT', () => shutdownGracefully('SIGINT'));
    
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
};

// Start the application
startServer().catch((error) => {
  console.error('Unhandled error during startup:', error);
  process.exit(1);
});
