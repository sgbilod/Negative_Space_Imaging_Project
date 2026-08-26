/**
 * Types Barrel Export
 * Centralized type definitions for the Negative Space Imaging Project
 */

import React from 'react';

// ============================================================================
// User & Authentication Types
// ============================================================================

/**
 * Represents a user in the system
 */
export interface User {
  id: string;
  email: string;
  username?: string;
  firstName?: string;
  lastName?: string;
  roles?: string[];
  createdAt?: string;
  updatedAt?: string;
}

/**
 * Authentication state for the application
 */
export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

/**
 * Response from authentication endpoints
 */
export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
  user: User;
}

/**
 * Credentials required for user login
 */
export interface LoginCredentials {
  email: string;
  password: string;
}

/**
 * Credentials required for user registration
 */
export interface RegisterCredentials extends LoginCredentials {
  firstName?: string;
  lastName?: string;
  confirmPassword: string;
}

// ============================================================================
// Image & Analysis Types
// ============================================================================

/**
 * Supported image formats for the application
 */
export type ImageFormat = 'jpg' | 'jpeg' | 'png' | 'tiff' | 'bmp' | 'webp' | 'dicom' | 'fits' | 'raw';

/**
 * Metadata associated with an uploaded image
 */
export interface ImageMetadata {
  id: string;
  filename: string;
  originalName: string;
  mimeType: string;
  size: number;
  width: number;
  height: number;
  format: ImageFormat;
  uploadedAt: string;
  uploadedBy: string;
}

/**
 * Represents a 2D point coordinate
 */
export interface Point {
  x: number;
  y: number;
}

/**
 * Represents a bounding box for a region
 */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Represents a detected region in an image
 */
export interface Region {
  id: string;
  type: 'negative' | 'positive';
  area: number;
  centroid: Point;
  boundingBox: BoundingBox;
  contourPoints: Point[];
}

/**
 * Data for a detected contour
 */
export interface ContourData {
  id: string;
  points: Point[];
  area: number;
  perimeter: number;
  isNegativeSpace: boolean;
}

/**
 * Extracted feature data from image analysis
 */
export interface FeatureData {
  edges: number;
  corners: number;
  blobs: number;
  textureScore: number;
}

/**
 * Statistical data from image analysis
 */
export interface AnalysisStatistics {
  mean: number;
  median: number;
  stdDev: number;
  min: number;
  max: number;
  histogram: number[];
}

/**
 * Complete result of an image analysis
 */
export interface AnalysisResult {
  id: string;
  imageId: string;
  negativeSpacePercentage: number;
  regions: Region[];
  contours: ContourData[];
  features: FeatureData;
  statistics: AnalysisStatistics;
  processingTime: number;
  algorithm: string;
  createdAt: string;
}

// ============================================================================
// Upload & Progress Types
// ============================================================================

/**
 * Progress information during file upload
 */
export interface UploadProgress {
  loaded: number;
  total: number;
  percent: number;
}

/**
 * Result of an image upload operation
 */
export interface UploadResult {
  success: boolean;
  imageId: string;
  analysisId?: string;
  error?: string;
}

/**
 * Possible states during the upload process
 */
export type UploadStatus = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';

// ============================================================================
// Notification Types
// ============================================================================

/**
 * Types of notifications that can be displayed
 */
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

/**
 * A notification message to be displayed to the user
 */
export interface Notification {
  id: string;
  message: string;
  type: NotificationType;
  duration?: number;
  dismissible?: boolean;
  createdAt: string;
}

// ============================================================================
// API Response Types
// ============================================================================

/**
 * Error information from API responses
 */
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * Metadata for paginated responses
 */
export interface ApiMeta {
  page?: number;
  pageSize?: number;
  totalCount?: number;
  totalPages?: number;
}

/**
 * Standard API response wrapper
 * @template T The type of data contained in the response
 */
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  meta?: ApiMeta;
}

/**
 * Paginated API response
 * @template T The type of items in the paginated array
 */
export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  meta: ApiMeta;
}

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * Available analysis algorithms
 */
export type AnalysisAlgorithm = 'standard' | 'advanced' | 'quantum';

/**
 * Application-wide configuration settings
 */
export interface AppConfig {
  apiBaseUrl: string;
  maxFileSize: number;
  allowedFileTypes: ImageFormat[];
  defaultAlgorithm: AnalysisAlgorithm;
  enableGpu: boolean;
}

/**
 * Configuration options for image analysis
 */
export interface AnalysisConfig {
  algorithm: AnalysisAlgorithm;
  threshold: number;
  minRegionSize: number;
  detectContours: boolean;
  extractFeatures: boolean;
}

// ============================================================================
// Component Props Types
// ============================================================================

/**
 * Props for the Button component
 */
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}

/**
 * Props for the Input component
 */
export interface InputProps {
  label?: string;
  type?: 'text' | 'email' | 'password' | 'number';
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  error?: string;
  placeholder?: string;
  disabled?: boolean;
  required?: boolean;
}

/**
 * Props for the Alert component
 */
export interface AlertProps {
  type: NotificationType;
  message: string;
  dismissible?: boolean;
  onClose?: () => void;
}
