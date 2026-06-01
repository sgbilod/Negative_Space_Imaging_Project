/**
 * Debounce Hook
 * Debounces a value after a specified delay
 */

import { useState, useEffect } from 'react';

export const useDebounce = <T>(value: T, delayMs: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delayMs);

    return () => clearTimeout(handler);
  }, [value, delayMs]);

  return debouncedValue;
};
