export interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'analyst' | 'physician' | 'auditor';
  isActive: boolean;
  createdAt: string;
  lastLogin?: string;
}

export interface ImageMetadata {
  id: string;
  filename: string;
  fileHash: string;
  fileSize: number;
  mimeType: string;
  width: number;
  height: number;
  createdAt: string;
  processed: boolean;
  exposureTime?: number;
  isoSpeed?: number;
  focalLength?: number;
}

export interface ProcessingJob {
  id: string;
  imageId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  startedAt: string;
  completedAt?: string;
  algorithm: string;
  parameters: Record<string, any>;
  result?: string;
}

export interface AuditLog {
  id: string;
  userId: string;
  action: string;
  timestamp: string;
  details?: string;
}

export interface Signature {
  id: string;
  signerId: string;
  signature: string;
  createdAt: string;
  verified: boolean;
}

// ...add additional interfaces as needed for full schema coverage
