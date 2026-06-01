import { apiClient } from '@/utils/api';

export interface AnalysisResult {
  id: string;
  imageId: string;
  negativeSpacePercentage: number;
  positiveSpacePercentage: number;
  regionsCount: number;
  confidence: number;
  processingTime: number;
  createdAt: string;
  completedAt: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  errorMessage?: string;
}

export interface AnalysisListResponse {
  analyses: AnalysisResult[];
  total: number;
  page: number;
  pageSize: number;
}

export interface AnalysisResponse {
  analysisId: string;
  status: string;
}

export const analysisService = {
  async startAnalysis(imageId: string): Promise<AnalysisResponse> {
    const response = await apiClient.post<AnalysisResponse>('/analysis/start', {
      imageId,
    });
    return response.data;
  },

  async getAnalysisResult(analysisId: string): Promise<AnalysisResult> {
    const response = await apiClient.get<AnalysisResult>(`/analysis/${analysisId}`);
    return response.data;
  },

  async getAnalyses(page: number = 1, pageSize: number = 10): Promise<AnalysisListResponse> {
    const response = await apiClient.get<AnalysisListResponse>(
      `/analysis?page=${page}&pageSize=${pageSize}`
    );
    return response.data;
  },

  async getImageAnalyses(imageId: string): Promise<AnalysisResult[]> {
    const response = await apiClient.get<AnalysisResult[]>(`/analysis/image/${imageId}`);
    return response.data;
  },

  async waitForAnalysisCompletion(
    analysisId: string,
    maxWaitTime: number = 300000 // 5 minutes
  ): Promise<AnalysisResult> {
    const startTime = Date.now();
    const pollInterval = 2000; // 2 seconds

    while (Date.now() - startTime < maxWaitTime) {
      try {
        const result = await this.getAnalysisResult(analysisId);

        if (result.status === 'completed' || result.status === 'failed') {
          return result;
        }

        await new Promise((resolve) => setTimeout(resolve, pollInterval));
      } catch {
        await new Promise((resolve) => setTimeout(resolve, pollInterval));
      }
    }

    throw new Error('Analysis polling timeout');
  },
};
