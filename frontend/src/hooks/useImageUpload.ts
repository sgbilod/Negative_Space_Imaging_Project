/**
 * Image Upload Hook
 * Handles image file uploads and processing
 */

import { useState, useCallback, useRef } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

export interface UploadProgress {
  loaded: number;
  total: number;
  percent: number;
}

interface UploadResponse {
  success: boolean;
  data?: {
    analysisId?: string;
    imageId?: string;
  };
  error?: {
    message?: string;
  };
}

export const useImageUpload = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [progress, setProgress] = useState<UploadProgress>({ loaded: 0, total: 0, percent: 0 });
  const abortControllerRef = useRef<AbortController | null>(null);

  const upload = useCallback(async (file: File, accessToken?: string): Promise<string> => {
    setIsLoading(true);
    setError(null);
    setProgress({ loaded: 0, total: file.size, percent: 0 });

    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController();

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('filename', file.name);
      formData.append('mimeType', file.type);

      // Use XMLHttpRequest for progress tracking
      const response = await new Promise<Response>((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (event: ProgressEvent) => {
          if (event.lengthComputable) {
            const progressData: UploadProgress = {
              loaded: event.loaded,
              total: event.total,
              percent: Math.round((event.loaded / event.total) * 100),
            };
            setProgress(progressData);
          }
        });

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(
              new Response(xhr.responseText, {
                status: xhr.status,
                statusText: xhr.statusText,
              }),
            );
          } else {
            reject(new Error(xhr.statusText || 'Upload failed'));
          }
        });

        xhr.addEventListener('error', () => {
          reject(new Error('Network error during upload'));
        });

        xhr.addEventListener('abort', () => {
          reject(new Error('Upload cancelled'));
        });

        xhr.open('POST', `${API_BASE_URL}/images/upload`);

        if (accessToken) {
          xhr.setRequestHeader('Authorization', `Bearer ${accessToken}`);
        }

        xhr.send(formData);

        // Handle abort
        const abortHandler = (): void => {
          xhr.abort();
        };
        abortControllerRef.current?.signal.addEventListener('abort', abortHandler);
      });

      const data = (await response.json()) as UploadResponse;

      if (!data.success) {
        throw new Error(data.error?.message || 'Upload failed');
      }

      return data.data?.analysisId || data.data?.imageId || '';
    } catch (err) {
      const uploadError = err instanceof Error ? err : new Error('Upload failed');
      setError(uploadError);
      throw uploadError;
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, []);

  const cancelUpload = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setProgress({ loaded: 0, total: 0, percent: 0 });
  }, []);

  return {
    upload,
    uploadImage: upload,
    uploading: isLoading,
    isLoading,
    error,
    progress,
    cancelUpload,
    reset,
  };
};
