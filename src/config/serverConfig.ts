/**
 * Server Configuration Service
 * Environment-aware configuration for Express server initialization
 */

import dotenv from 'dotenv';
import { Pool } from 'pg';
import { createClient } from 'redis';
import { logger } from '../services/loggingService';

// Load environment variables
dotenv.config();

/**
 * Application configuration interface
 */
export interface AppConfig {
  server: {
    port: number;
    host: string;
    nodeEnv: 'development' | 'staging' | 'production';
    apiVersion: string;
  };
  database: {
    url: string;
    pool: {
      max: number;
      min: number;
      idleTimeoutMillis: number;
      connectionTimeoutMillis: number;
    };
  };
  redis: {
    url: string;
    enabled: boolean;
  };
  jwt: {
    secret: string;
    expiresIn: string;
    algorithm: 'HS256' | 'HS512';
  };
  cors: {
    origin: string[];
    credentials: boolean;
    maxAge: number;
  };
  rateLimit: {
    windowMs: number;
    max: number;
    message: string;
  };
  logging: {
    level: string;
    requestLogging: boolean;
    errorStack: boolean;
  };
  security: {
    bcryptRounds: number;
    passwordMinLength: number;
    sessionTimeout: number; // in minutes
    enableHttpsOnly: boolean;
  };
}

/**
 * Validate required environment variables
 */
function validateEnvironment(): void {
  const requiredVars = ['NODE_ENV', 'PORT', 'DATABASE_URL', 'JWT_SECRET'];

  const missing = requiredVars.filter((varName) => !process.env[varName]);

  if (missing.length > 0) {
    logger.warn(`Missing environment variables: ${missing.join(', ')}`);
  }
}

/**
 * Build application configuration from environment variables
 */
export function buildConfig(): AppConfig {
  validateEnvironment();

  const config: AppConfig = {
    server: {
      port: parseInt(process.env.PORT || '3000', 10),
      host: process.env.HOST || '0.0.0.0',
      nodeEnv: (process.env.NODE_ENV as any) || 'development',
      apiVersion: 'v1',
    },

    database: {
      url: process.env.DATABASE_URL || 'postgresql://localhost/negative_space',
      pool: {
        max: parseInt(process.env.DB_POOL_MAX || '20', 10),
        min: parseInt(process.env.DB_POOL_MIN || '2', 10),
        idleTimeoutMillis: parseInt(process.env.DB_IDLE_TIMEOUT || '30000', 10),
        connectionTimeoutMillis: parseInt(process.env.DB_CONNECTION_TIMEOUT || '2000', 10),
      },
    },

    redis: {
      url: process.env.REDIS_URL || 'redis://localhost:6379',
      enabled: process.env.REDIS_ENABLED !== 'false',
    },

    jwt: {
      secret: process.env.JWT_SECRET || 'your-secret-key-change-in-production',
      expiresIn: process.env.JWT_EXPIRES_IN || '24h',
      algorithm: 'HS256',
    },

    cors: {
      origin: (process.env.CORS_ORIGIN || 'http://localhost:3000,http://localhost:3001').split(','),
      credentials: true,
      maxAge: 86400, // 24 hours
    },

    rateLimit: {
      windowMs: parseInt(process.env.RATE_LIMIT_WINDOW || '900000', 10), // 15 minutes
      max: parseInt(process.env.RATE_LIMIT_MAX || '100', 10),
      message: 'Too many requests from this IP, please try again later.',
    },

    logging: {
      level: process.env.LOG_LEVEL || 'debug',
      requestLogging: process.env.REQUEST_LOGGING !== 'false',
      errorStack: process.env.NODE_ENV !== 'production',
    },

    security: {
      bcryptRounds: parseInt(process.env.BCRYPT_ROUNDS || '10', 10),
      passwordMinLength: parseInt(process.env.PASSWORD_MIN_LENGTH || '12', 10),
      sessionTimeout: parseInt(process.env.SESSION_TIMEOUT || '60', 10), // 60 minutes
      enableHttpsOnly: process.env.NODE_ENV === 'production',
    },
  };

  return config;
}

/**
 * Server configuration service for initialization
 */
export class ServerConfigService {
  private static instance: ServerConfigService;
  private config: AppConfig;
  private dbPool: Pool | null = null;
  private redisClient: any = null;

  private constructor() {
    this.config = buildConfig();
  }

  /**
   * Get or create singleton instance
   */
  static getInstance(): ServerConfigService {
    if (!ServerConfigService.instance) {
      ServerConfigService.instance = new ServerConfigService();
    }
    return ServerConfigService.instance;
  }

  /**
   * Get configuration object
   */
  getConfig(): AppConfig {
    return this.config;
  }

  /**
   * Initialize database connection pool
   */
  async initializeDatabase(): Promise<Pool> {
    if (this.dbPool) {
      return this.dbPool;
    }

    try {
      this.dbPool = new Pool({
        connectionString: this.config.database.url,
        max: this.config.database.pool.max,
        min: this.config.database.pool.min,
        idleTimeoutMillis: this.config.database.pool.idleTimeoutMillis,
        connectionTimeoutMillis: this.config.database.pool.connectionTimeoutMillis,
      });

      // Test connection
      const client = await this.dbPool.connect();
      await client.query('SELECT NOW()');
      client.release();

      logger.info('Database connection pool initialized', {
        context: {
          database: this.config.database.url.split('@')[1] || 'unknown',
          poolSize: this.config.database.pool.max,
        },
      });

      return this.dbPool;
    } catch (error) {
      logger.error('Failed to initialize database connection pool', error as Error);
      throw error;
    }
  }

  /**
   * Get database pool
   */
  getDatabasePool(): Pool {
    if (!this.dbPool) {
      throw new Error('Database pool not initialized. Call initializeDatabase() first.');
    }
    return this.dbPool;
  }

  /**
   * Initialize Redis connection
   */
  async initializeRedis(): Promise<any> {
    if (!this.config.redis.enabled) {
      logger.info('Redis disabled in configuration');
      return null;
    }

    if (this.redisClient) {
      return this.redisClient;
    }

    try {
      this.redisClient = createClient({
        url: this.config.redis.url,
        socket: {
          reconnectStrategy: (retries: number) => {
            if (retries > 10) {
              logger.error('Redis: Max reconnection attempts reached');
              return new Error('Redis max retries reached');
            }
            return retries * 100;
          },
        },
      });

      this.redisClient.on('error', (err: any) => {
        logger.error('Redis client error', err);
      });

      this.redisClient.on('connect', () => {
        logger.info('Redis client connected');
      });

      await this.redisClient.connect();

      // Test connection
      await this.redisClient.ping();

      logger.info('Redis connection established', {
        context: {
          url: this.config.redis.url.split('//')[1] || 'unknown',
        },
      });

      return this.redisClient;
    } catch (error) {
      logger.warn('Failed to initialize Redis connection', error as Error);
      // Redis is optional, so we don't throw
      return null;
    }
  }

  /**
   * Get Redis client
   */
  getRedisClient(): any {
    return this.redisClient;
  }

  /**
   * Close database connection pool
   */
  async closeDatabasePool(): Promise<void> {
    if (this.dbPool) {
      await this.dbPool.end();
      logger.info('Database connection pool closed');
      this.dbPool = null;
    }
  }

  /**
   * Close Redis connection
   */
  async closeRedisClient(): Promise<void> {
    if (this.redisClient) {
      await this.redisClient.quit();
      logger.info('Redis connection closed');
      this.redisClient = null;
    }
  }

  /**
   * Check health of all services
   */
  async checkHealth(): Promise<{
    database: boolean;
    redis: boolean;
    errors: string[];
  }> {
    const errors: string[] = [];
    let dbHealth = false;
    let redisHealth = false;

    // Check database
    try {
      if (this.dbPool) {
        const client = await this.dbPool.connect();
        await client.query('SELECT 1');
        client.release();
        dbHealth = true;
      }
    } catch (error) {
      errors.push(`Database health check failed: ${(error as Error).message}`);
    }

    // Check Redis
    try {
      if (this.redisClient) {
        await this.redisClient.ping();
        redisHealth = true;
      }
    } catch (error) {
      errors.push(`Redis health check failed: ${(error as Error).message}`);
    }

    return {
      database: dbHealth,
      redis: redisHealth,
      errors,
    };
  }
}

/**
 * Export configuration utility functions
 */
export const configService = ServerConfigService.getInstance();

export default ServerConfigService;
