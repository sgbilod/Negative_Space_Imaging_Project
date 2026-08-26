/**
 * TypeScript Type Definitions
 *
 * Central location for all TypeScript interfaces and types used throughout
 * the Negative Space Imaging API.
 *
 * @module types
 */

import { Request, Response, NextFunction } from 'express';

/**
 * JWT Payload Interface
 * Contains claims stored in the JWT token
 */
export interface AuthPayload {
  id: number;
  email: string;
  iat?: number;
  exp?: number;
}

/**
 * Authenticated Express Request
 * Extends Express Request with user information from JWT
 */
export interface AuthRequest extends Request {
  user?: AuthPayload;
  token?: string;
}

/**
 * Generic API Response Interface
 */
export interface ApiResponse<T = unknown> {
  success: boolean;
  message: string;
  data?: T;
  error?: string | object;
  timestamp: Date;
}

/**
 * Pagination Query Interface
 */
export interface PaginationQuery {
  page?: number;
  limit?: number;
  offset?: number;
}

/**
 * Paginated Response Interface
 */
export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

/**
 * Image Upload Request Body
 */
export interface ImageUploadRequest {
  filename: string;
  original_filename: string;
  file_size: number;
  storage_path: string;
}

/**
 * Analysis Result Request Body
 */
export interface AnalysisResultRequest {
  image_id: number;
  negative_space_percentage: number;
  regions_count: number;
  processing_time_ms: number;
  raw_data: Record<string, unknown>;
}

/**
 * User Registration Request
 */
export interface UserRegisterRequest {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
}

/**
 * User Login Request
 */
export interface UserLoginRequest {
  email: string;
  password: string;
}

/**
 * Middleware Function Type
 */
export type Middleware = (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => void | Promise<void>;

/**
 * Service Error Class
 */
export class ServiceError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public code?: string
  ) {
    super(message);
    this.name = 'ServiceError';
  }
}

/**
 * Database Model Timestamps
 */
export interface Timestamps {
  created_at: Date;
  updated_at: Date;
}
