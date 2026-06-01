/**
 * Database Configuration Module
 *
 * Configures and manages Sequelize ORM connection to PostgreSQL database.
 * Supports environment-based configuration with connection pooling.
 *
 * @module config/database
 */

import { Sequelize, SequelizeOptions } from 'sequelize';
import dotenv from 'dotenv';
import logger from '../utils/logger';

dotenv.config();

/**
 * Database configuration based on environment
 */
const dbConfig: SequelizeOptions = {
  dialect: 'postgres',
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432', 10),
  username: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME || 'negative_space_imaging',
  logging: process.env.NODE_ENV === 'development' ? console.log : false,
  pool: {
    max: parseInt(process.env.DB_POOL_MAX || '10', 10),
    min: parseInt(process.env.DB_POOL_MIN || '5', 10),
    acquire: parseInt(process.env.DB_POOL_ACQUIRE || '30000', 10),
    idle: parseInt(process.env.DB_POOL_IDLE || '10000', 10),
  },
  retry: {
    max: 3,
    delay: 3000,
  },
  define: {
    timestamps: true,
    underscored: true,
  },
};

/**
 * Initialize Sequelize instance
 */
let sequelize: Sequelize;

export function initializeDatabase(): Sequelize {
  try {
    sequelize = new Sequelize(dbConfig);
    logger.info('Database configuration initialized');
    return sequelize;
  } catch (error) {
    logger.error('Failed to initialize database configuration', error);
    throw error;
  }
}

/**
 * Test database connection
 */
export async function testConnection(): Promise<void> {
  try {
    await sequelize.authenticate();
    logger.info('Database connection established successfully');
  } catch (error) {
    logger.error('Failed to connect to database', error);
    throw error;
  }
}

/**
 * Sync database models with schema
 * WARNING: Use with caution in production
 */
export async function syncDatabase(force: boolean = false): Promise<void> {
  try {
    await sequelize.sync({ force });
    logger.info(`Database synchronized (force: ${force})`);
  } catch (error) {
    logger.error('Failed to synchronize database', error);
    throw error;
  }
}

/**
 * Close database connection
 */
export async function closeConnection(): Promise<void> {
  try {
    await sequelize.close();
    logger.info('Database connection closed');
  } catch (error) {
    logger.error('Failed to close database connection', error);
    throw error;
  }
}

/**
 * Get Sequelize instance
 */
export function getSequelize(): Sequelize {
  if (!sequelize) {
    throw new Error('Database not initialized. Call initializeDatabase() first.');
  }
  return sequelize;
}

export default sequelize;
