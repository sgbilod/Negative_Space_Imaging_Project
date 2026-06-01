import { apiClient } from '@/utils/api';

export interface ImageResponse {
  id: string;
  name: string;
  filePath: string;
  fileSize: number;
  uploadedAt: string;
  mimeType: string;
}

export interface ImageDetailResponse extends ImageResponse {
  width: number;
  height: number;
  format: string;
}

export interface ImageListResponse {
  images: ImageResponse[];
  total: number;
  page: number;
  pageSize: number;
}

export interface UploadImageRequest {
  file: File;
  name?: string;
}

export const imageService = {
  async getImages(page: number = 1, pageSize: number = 10): Promise<ImageListResponse> {
    const response = await apiClient.get<ImageListResponse>(
      `/images?page=${page}&pageSize=${pageSize}`
    );
    return response.data;
  },

  async getImageById(imageId: string): Promise<ImageDetailResponse> {
    const response = await apiClient.get<ImageDetailResponse>(`/images/${imageId}`);
    return response.data;
  },

  async uploadImage(file: File, name?: string): Promise<ImageResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (name) {
      formData.append('name', name);
    }

    const response = await apiClient.post<ImageResponse>('/images/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async deleteImage(imageId: string): Promise<void> {
    await apiClient.delete(`/images/${imageId}`);
  },

  async searchImages(query: string, page: number = 1): Promise<ImageListResponse> {
    const response = await apiClient.get<ImageListResponse>(`/images/search?q=${query}&page=${page}`);
    return response.data;
  },

  getImageUrl(imageId: string): string {
    return `${import.meta.env.VITE_API_URL}/images/${imageId}`;
  },
};
