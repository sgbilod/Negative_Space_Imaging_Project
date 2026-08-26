import { useState, useEffect, useCallback } from 'react';

export interface UseLocalStorageReturn<T> {
  value: T | null;
  setValue: (value: T | null) => void;
  removeValue: () => void;
  isLoaded: boolean;
}

export const useLocalStorage = <T,>(key: string, initialValue?: T): UseLocalStorageReturn<T> => {
  const [value, setValue] = useState<T | null>(initialValue ?? null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const item = window.localStorage.getItem(key);
      if (item) {
        setValue(JSON.parse(item));
      }
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
    } finally {
      setIsLoaded(true);
    }
  }, [key]);

  const updateValue = useCallback(
    (newValue: T | null) => {
      try {
        if (newValue === null) {
          window.localStorage.removeItem(key);
        } else {
          window.localStorage.setItem(key, JSON.stringify(newValue));
        }
        setValue(newValue);
      } catch (error) {
        console.error(`Error setting localStorage key "${key}":`, error);
      }
    },
    [key]
  );

  const removeValue = useCallback(() => {
    try {
      window.localStorage.removeItem(key);
      setValue(null);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key]);

  return { value, setValue: updateValue, removeValue, isLoaded };
};
