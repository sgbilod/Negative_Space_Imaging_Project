import { Sequelize } from 'sequelize';
import { logger } from '../utils/logger';
import { config } from '../config';

/**
 * Database connection instance
 */
export const sequelize = new Sequelize(config.database.url, {
  logging: config.database.logging ? (msg) => logger.debug(msg) : false,
  dialect: 'postgres',
  dialectOptions: {
    ssl: config.database.ssl ? {
      require: true,
      rejectUnauthorized: false,
    } : undefined,
  },
  pool: {
    max: 20,
    min: 0,
    acquire: 30000,
    idle: 10000,
  },
});

/**
 * Initialize database connection
 */
export const dbConnect = async (): Promise<void> => {
  try {
    await sequelize.authenticate();
    logger.info('Database connection established successfully');
    
    // Sync models with database if configured to do so
    if (config.database.synchronize) {
      logger.info('Synchronizing database models...');
      await sequelize.sync({ alter: true });
      logger.info('Database models synchronized successfully');
    }
  } catch (error) {
    logger.error('Unable to connect to the database:', error);
    throw new Error('Database connection failed');
  }
};
