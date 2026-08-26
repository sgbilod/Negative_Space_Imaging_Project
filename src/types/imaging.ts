/**
 * Type definitions for the Negative Space Imaging System
 * 
 * This file contains interfaces and types related to image processing, analysis,
 * and the core negative space detection algorithms.
 */

/**
 * Supported image formats
 */
export enum ImageFormat {
  JPEG = 'jpeg',
  PNG = 'png',
  TIFF = 'tiff',
  DICOM = 'dicom',
  FITS = 'fits',
  RAW = 'raw',
  BMP = 'bmp',
  WEBP = 'webp'
}

/**
 * Base image representation
 */
export interface Image {
  /** Unique identifier for the image */
  id: string;
  
  /** Width of the image in pixels */
  width: number;
  
  /** Height of the image in pixels */
  height: number;
  
  /** Original filename of the image */
  filename: string;
  
  /** Format of the image */
  format: ImageFormat;
  
  /** Optional path to the image file */
  path?: string;
  
  /** Optional buffer containing the image data */
  data?: Buffer;
  
  /** Optional URL to the image */
  url?: string;
  
  /** Additional metadata associated with the image */
  metadata?: ImageMetadata;
  
  /** Timestamp when the image was created/uploaded */
  createdAt: Date;
  
  /** Timestamp when the image was last updated */
  updatedAt: Date;
}

/**
 * Metadata associated with an image
 */
export interface ImageMetadata {
  /** Width of the image in pixels */
  width: number;
  
  /** Height of the image in pixels */
  height: number;
  
  /** Format of the image */
  format: ImageFormat;
  
  /** Original filename */
  filename: string;
  
  /** File size in bytes */
  size?: number;
  
  /** Color space of the image */
  colorSpace?: string;
  
  /** Bits per pixel */
  bitsPerPixel?: number;
  
  /** Original creation date from EXIF data if available */
  originalDate?: Date;
  
  /** Camera or device information if available */
  device?: string;
  
  /** Exposure information if available */
  exposure?: string;
  
  /** Focal length if available */
  focalLength?: string;
  
  /** ISO setting if available */
  iso?: number;
  
  /** Whether the image has been encrypted */
  isEncrypted?: boolean;
  
  /** Custom properties specific to the application */
  custom?: Record<string, any>;
}

/**
 * A 2D point in an image
 */
export interface Point {
  /** X coordinate */
  x: number;
  
  /** Y coordinate */
  y: number;
}

/**
 * A bounding box in an image
 */
export interface BoundingBox {
  /** X coordinate of the top-left corner */
  x: number;
  
  /** Y coordinate of the top-left corner */
  y: number;
  
  /** Width of the bounding box */
  width: number;
  
  /** Height of the bounding box */
  height: number;
}

/**
 * A detected region in an image
 */
export interface Region {
  /** Unique identifier for the region */
  id: string;
  
  /** Bounding box of the region */
  boundingBox: BoundingBox;
  
  /** Area of the region in pixels */
  area: number;
  
  /** Perimeter of the region in pixels */
  perimeter: number;
  
  /** Centroid of the region */
  centroid: Point;
  
  /** Contour points of the region */
  contour?: Point[];
  
  /** Whether this region is classified as negative space */
  isNegativeSpace?: boolean;
  
  /** Confidence score for the region classification */
  confidence?: number;
  
  /** Additional properties of the region */
  properties?: Record<string, any>;
}

/**
 * A feature detected in negative space
 */
export interface NegativeSpaceFeature {
  /** Unique identifier for the feature */
  id: string;
  
  /** ID of the region containing this feature */
  regionId: string;
  
  /** Type of the feature */
  type: string;
  
  /** Confidence score for the feature detection */
  confidence: number;
  
  /** Significance score of the feature */
  significance: number;
  
  /** Points defining the feature */
  points: Point[];
  
  /** Additional properties of the feature */
  properties?: {
    /** Type of symmetry if present */
    symmetry?: 'bilateral' | 'radial' | 'none';
    
    /** Contrast measure relative to surroundings */
    contrast?: number;
    
    /** Complexity measure of the feature */
    complexity?: number;
    
    /** Additional custom properties */
    [key: string]: any;
  };
}

/**
 * Result of a negative space analysis
 */
export interface AnalysisResult {
  /** Unique identifier for the analysis result */
  id: string;
  
  /** Timestamp when the analysis was performed */
  timestamp: Date;
  
  /** ID of the analyzed image */
  imageId: string;
  
  /** Metadata of the analyzed image */
  imageMetadata: ImageMetadata;
  
  /** Detected regions classified as negative space */
  detectedRegions: Region[];
  
  /** Features detected within negative spaces */
  features: NegativeSpaceFeature[];
  
  /** Statistical analysis of the results */
  statistics: Record<string, number>;
  
  /** Processing time in milliseconds */
  processingTimeMs: number;
  
  /** Version of the algorithm used */
  algorithmVersion: string;
  
  /** Options used for the analysis */
  options: Record<string, any>;
}

/**
 * Parameters for image processing operations
 */
export interface ImageProcessingParameters {
  /** Target image width */
  width?: number;
  
  /** Target image height */
  height?: number;
  
  /** Whether to maintain aspect ratio when resizing */
  maintainAspectRatio?: boolean;
  
  /** Crop parameters */
  crop?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  
  /** Brightness adjustment (-1 to 1) */
  brightness?: number;
  
  /** Contrast adjustment (-1 to 1) */
  contrast?: number;
  
  /** Sharpening level (0 to 1) */
  sharpen?: number;
  
  /** Blur level (0 to 1) */
  blur?: number;
  
  /** Rotation in degrees */
  rotation?: number;
  
  /** Whether to flip horizontally */
  flipHorizontal?: boolean;
  
  /** Whether to flip vertically */
  flipVertical?: boolean;
  
  /** Format to convert to */
  format?: ImageFormat;
  
  /** Quality level for lossy formats (0 to 100) */
  quality?: number;
  
  /** Additional parameters for specific operations */
  [key: string]: any;
}

/**
 * Error types specific to image processing
 */
export enum ImageProcessingErrorType {
  FILE_NOT_FOUND = 'FILE_NOT_FOUND',
  INVALID_FORMAT = 'INVALID_FORMAT',
  PROCESSING_FAILED = 'PROCESSING_FAILED',
  UNSUPPORTED_OPERATION = 'UNSUPPORTED_OPERATION',
  ANALYSIS_FAILED = 'ANALYSIS_FAILED',
  INSUFFICIENT_MEMORY = 'INSUFFICIENT_MEMORY',
  TIMEOUT = 'TIMEOUT',
  PERMISSION_DENIED = 'PERMISSION_DENIED',
  ENCRYPTION_ERROR = 'ENCRYPTION_ERROR',
}

/**
 * Image processing error
 */
export class ImageProcessingError extends Error {
  type: ImageProcessingErrorType;
  details?: Record<string, any>;

  constructor(message: string, type: ImageProcessingErrorType, details?: Record<string, any>) {
    super(message);
    this.name = 'ImageProcessingError';
    this.type = type;
    this.details = details;
  }
}

/**
 * Image filter types
 */
export enum ImageFilter {
  GRAYSCALE = 'grayscale',
  SEPIA = 'sepia',
  INVERT = 'invert',
  BLUR = 'blur',
  SHARPEN = 'sharpen',
  EDGE_DETECTION = 'edge-detection',
  NOISE_REDUCTION = 'noise-reduction',
  THRESHOLD = 'threshold',
  POSTERIZE = 'posterize',
  CUSTOM = 'custom',
}

/**
 * Input parameter requirements for image analysis
 */
export interface AnalysisParameters {
  /** Sensitivity threshold for detecting negative spaces (0-1) */
  sensitivityThreshold?: number;
  
  /** Whether to detect patterns in negative spaces */
  detectPatterns?: boolean;
  
  /** Whether to analyze shapes of negative spaces */
  analyzeShapes?: boolean;
  
  /** Minimum size of regions to consider (in pixels) */
  minRegionSize?: number;
  
  /** Maximum number of regions to detect */
  maxRegions?: number;
  
  /** Additional parameters specific to analysis algorithms */
  [key: string]: any;
}

/**
 * Image acquisition options
 */
export interface AcquisitionOptions {
  /** Target resolution */
  resolution?: {
    width: number;
    height: number;
  };
  
  /** Target format */
  format?: ImageFormat;
  
  /** Whether to compress the image */
  compress?: boolean;
  
  /** Compression quality (0-100) */
  quality?: number;
  
  /** Whether to include metadata */
  includeMetadata?: boolean;
  
  /** Whether to encrypt the image */
  encrypt?: boolean;
  
  /** Additional acquisition options */
  [key: string]: any;
}

/**
 * Image visualization options
 */
export interface VisualizationOptions {
  /** Whether to highlight negative spaces */
  highlightNegativeSpaces?: boolean;
  
  /** Color to use for highlighting (hex code) */
  highlightColor?: string;
  
  /** Whether to show region boundaries */
  showBoundaries?: boolean;
  
  /** Whether to label regions */
  labelRegions?: boolean;
  
  /** Whether to show feature points */
  showFeatures?: boolean;
  
  /** Whether to include a legend */
  includeLegend?: boolean;
  
  /** Whether to include statistics */
  includeStatistics?: boolean;
  
  /** Additional visualization options */
  [key: string]: any;
}

/**
 * Image storage location
 */
export enum StorageLocation {
  LOCAL = 'local',
  S3 = 'aws-s3',
  AZURE = 'azure-blob',
  GCP = 'gcp-storage',
  DATABASE = 'database',
}

/**
 * Image permission level
 */
export enum ImagePermission {
  READ = 'read',
  WRITE = 'write',
  DELETE = 'delete',
  PROCESS = 'process',
  ANALYZE = 'analyze',
  SHARE = 'share',
  ADMIN = 'admin',
}
