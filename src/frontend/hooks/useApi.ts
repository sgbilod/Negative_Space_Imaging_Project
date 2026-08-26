import { useState, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import { apiClient } from '@/utils/api';

export interface UseApiOptions {
  immediate?: boolean;
}

export interface UseApiReturn<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  get: <R = T>(url: string) => Promise<R>;
  post: <R = T>(url: string, data: unknown) => Promise<R>;
  put: <R = T>(url: string, data: unknown) => Promise<R>;
  delete: <R = T>(url: string) => Promise<R>;
}

export const useApi = <T = unknown>(_options?: UseApiOptions): UseApiReturn<T> => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleError = (err: unknown): string => {
    if (axios.isAxiosError(err)) {
      return err.response?.data?.message || err.message;
    }
    return err instanceof Error ? err.message : 'An error occurred';
  };

  const get = useCallback(async <R = T,>(url: string): Promise<R> => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get<R>(url);
      setData(response.data as unknown as T);
      return response.data;
    } catch (err) {
      const errorMessage = handleError(err);
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const post = useCallback(async <R = T,>(url: string, payload: unknown): Promise<R> => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.post<R>(url, payload);
      setData(response.data as unknown as T);
      return response.data;
    } catch (err) {
      const errorMessage = handleError(err);
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const put = useCallback(async <R = T,>(url: string, payload: unknown): Promise<R> => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.put<R>(url, payload);
      setData(response.data as unknown as T);
      return response.data;
    } catch (err) {
      const errorMessage = handleError(err);
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const delete_ = useCallback(async <R = T,>(url: string): Promise<R> => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.delete<R>(url);
      setData(response.data as unknown as T);
      return response.data;
    } catch (err) {
      const errorMessage = handleError(err);
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, get, post, put, delete: delete_ };
};
