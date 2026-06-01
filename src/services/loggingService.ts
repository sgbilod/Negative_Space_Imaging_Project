/**
 * Logging Service
 * Structured logging using Winston with multiple transports and log levels
 */

import winston from 'winston';
import path from 'path';

interface LogContext {
  requestId?: string;
  userId?: string;
  endpoint?: string;
  method?: string;
  timestamp?: string;
  [key: string]: any;
}

/**
 * Create structured logger instance with file and console output
 */
export const createLogger = (serviceName: string): winston.Logger => {
  const logsDir = path.join(process.cwd(), 'logs');

  const colorizer = winston.format.colorize();
  const timestampFormat = winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' });
  const errorStackFormat = winston.format.errors({ stack: true });

  /**
   * Custom format for readable console output
   */
  const consoleFormat = winston.format.combine(
    errorStackFormat,
    timestampFormat,
    winston.format.printf((info) => {
      const level = colorizer.colorize(info.level, info.level.toUpperCase());
      const timestamp = info.timestamp;
      const service = serviceName;
      const message = info.message;
      const context = info.context ? ` | ${JSON.stringify(info.context)}` : '';
      const error = info.stack ? `\n${info.stack}` : '';

      return `[${timestamp}] ${level} [${service}]${context}: ${message}${error}`;
    }),
  );

  /**
   * JSON format for structured logging to files
   */
  const jsonFormat = winston.format.combine(
    errorStackFormat,
    timestampFormat,
    winston.format.json(),
  );

  /**
   * Configure transports based on environment
   */
  const transports: winston.transport[] = [
    // Console output
    new winston.transports.Console({
      format: consoleFormat,
      level: process.env.LOG_LEVEL || 'debug',
    }),

    // Error log file
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      format: jsonFormat,
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),

    // Combined log file
    new winston.transports.File({
      filename: path.join(logsDir, 'combined.log'),
      format: jsonFormat,
      maxsize: 5242880, // 5MB
      maxFiles: 10,
    }),
  ];

  // Add debug log file in development
  if (process.env.NODE_ENV !== 'production') {
    transports.push(
      new winston.transports.File({
        filename: path.join(logsDir, 'debug.log'),
        level: 'debug',
        format: jsonFormat,
        maxsize: 5242880, // 5MB
        maxFiles: 5,
      }),
    );
  }

  const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'debug',
    format: jsonFormat,
    defaultMeta: { service: serviceName },
    transports,
    exceptionHandlers: [
      new winston.transports.File({
        filename: path.join(logsDir, 'exceptions.log'),
        format: jsonFormat,
      }),
    ],
    rejectionHandlers: [
      new winston.transports.File({
        filename: path.join(logsDir, 'rejections.log'),
        format: jsonFormat,
      }),
    ],
  });

  return logger;
};

/**
 * Logger instance for the application
 */
export const logger = createLogger('NegativeSpaceAPI');

/**
 * Structured logging methods
 */
export const log = {
  /**
   * Log info level message
   */
  info: (message: string, context?: LogContext) => {
    logger.info(message, { context });
  },

  /**
   * Log warning level message
   */
  warn: (message: string, context?: LogContext) => {
    logger.warn(message, { context });
  },

  /**
   * Log error level message
   */
  error: (message: string, error?: Error, context?: LogContext) => {
    logger.error(message, {
      context,
      ...(error && { error: error.message, stack: error.stack }),
    });
  },

  /**
   * Log debug level message
   */
  debug: (message: string, context?: LogContext) => {
    logger.debug(message, { context });
  },

  /**
   * Log HTTP request details
   */
  request: (
    method: string,
    endpoint: string,
    statusCode: number,
    duration: number,
    requestId: string,
  ) => {
    logger.info('HTTP Request', {
      context: {
        method,
        endpoint,
        statusCode,
        duration: `${duration}ms`,
        requestId,
      },
    });
  },

  /**
   * Log database operations
   */
  database: (operation: string, table: string, duration: number, rowsAffected?: number) => {
    logger.debug('Database Operation', {
      context: {
        operation,
        table,
        duration: `${duration}ms`,
        ...(rowsAffected !== undefined && { rowsAffected }),
      },
    });
  },

  /**
   * Log authentication events
   */
  auth: (event: string, userId?: string, success: boolean = true, context?: LogContext) => {
    const level = success ? 'info' : 'warn';
    logger[level as 'info' | 'warn'](`Auth Event: ${event}`, {
      context: {
        userId,
        success,
        ...context,
      },
    });
  },

  /**
   * Log security events
   */
  security: (
    event: string,
    severity: 'low' | 'medium' | 'high' | 'critical' = 'medium',
    context?: LogContext,
  ) => {
    const level = severity === 'critical' ? 'error' : severity === 'high' ? 'warn' : 'info';
    logger[level as 'error' | 'warn' | 'info'](`Security Event: ${event}`, {
      context: {
        severity,
        ...context,
      },
    });
  },

  /**
   * Log performance metrics
   */
  performance: (metric: string, value: number, unit: string = 'ms') => {
    logger.debug('Performance Metric', {
      context: {
        metric,
        value,
        unit,
      },
    });
  },
};

export default logger;
