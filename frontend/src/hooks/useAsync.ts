/**
 * Async Hook
 * Handles async operations with loading and error states
 */

import { useState, useCallback } from 'react';

export const useAsync = <T>(asyncFunction: () => Promise<T>) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<T | null>(null);

  const execute = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await asyncFunction();
      setData(result);
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [asyncFunction]);

  return { isLoading, error, data, execute };
};
