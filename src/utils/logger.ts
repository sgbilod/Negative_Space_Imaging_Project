import winston from 'winston';
import fs from 'fs';
import path from 'path';
import { config } from '../config';

// Create logs directory if it doesn't exist
const logDir = config.logging.directory;
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

// Define log format
const logFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.splat(),
  winston.format.json()
);

// Morgan format for HTTP logging
export const morganLogFormat = config.server.isDevelopment
  ? 'dev' // Colored output for development
  : ':remote-addr - :remote-user [:date[clf]] ":method :url HTTP/:http-version" :status :res[content-length] ":referrer" ":user-agent" - :response-time ms';

/**
 * Application logger with multiple transports
 */
export const logger = winston.createLogger({
  level: config.logging.level,
  format: logFormat,
  defaultMeta: { service: 'negative-space-imaging' },
  transports: [
    // Write logs to console
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf(({ timestamp, level, message, ...meta }) => {
          // Add metadata only if it exists and isn't empty
          const metaString = Object.keys(meta).length
            ? `\n${JSON.stringify(meta, null, 2)}`
            : '';
            
          return `${timestamp} ${level}: ${message}${metaString}`;
        })
      ),
    }),

    // Write all logs to a file
    new winston.transports.File({
      filename: path.join(logDir, config.logging.filename),
      maxsize: parseInt(config.logging.maxSize, 10),
      maxFiles: config.logging.maxFiles,
      tailable: true,
    }),

    // Write error logs to a separate file
    new winston.transports.File({
      filename: path.join(logDir, 'error.log'),
      level: 'error',
      maxsize: parseInt(config.logging.maxSize, 10),
      maxFiles: config.logging.maxFiles,
      tailable: true,
    }),
  ],
  exceptionHandlers: [
    new winston.transports.File({
      filename: path.join(logDir, 'exceptions.log'),
      maxsize: parseInt(config.logging.maxSize, 10),
      maxFiles: config.logging.maxFiles,
    }),
  ],
  rejectionHandlers: [
    new winston.transports.File({
      filename: path.join(logDir, 'rejections.log'),
      maxsize: parseInt(config.logging.maxSize, 10),
      maxFiles: config.logging.maxFiles,
    }),
  ],
  exitOnError: false,
});

// If we're not in production, add a stream for Winston to Morgan
export const morganStream = {
  write: (message: string): void => {
    logger.http(message.trim());
  },
};
