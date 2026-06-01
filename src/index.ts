/**
 * Application Entry Point
 * Initialize and start the Negative Space Imaging API server
 */

import dotenv from 'dotenv';
import { startServer } from './server/server';
import { log } from './services/loggingService';

// Load environment variables
dotenv.config();

/**
 * Application entry point
 */
async function main(): Promise<void> {
  try {
    log.info('🌟 Negative Space Imaging API Server');
    log.info(`Environment: ${process.env.NODE_ENV || 'development'}`);

    // Start the server
    await startServer();
  } catch (error) {
    log.error('Failed to start application', error as Error);
    process.exit(1);
  }
}

// Execute entry point
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
