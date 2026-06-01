/**
 * Authentication Hook
 * Manages user authentication state and actions
 */

import { useCallback, useState } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Token storage keys
const ACCESS_TOKEN_KEY = 'accessToken';
const REFRESH_TOKEN_KEY = 'refreshToken';
const USER_KEY = 'user';

export interface User {
  id: string;
  email: string;
  username?: string;
  firstName?: string;
  lastName?: string;
  roles?: string[];
}

export interface UseAuthReturn {
  user: User | null;
  isLoading: boolean;
  loading: boolean; // Alias for isLoading
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (
    email: string,
    password: string,
    firstName?: string,
    lastName?: string,
  ) => Promise<void>;
  refreshAccessToken: () => Promise<boolean>;
  getAccessToken: () => string | null;
  clearError: () => void;
  isAuthenticated: boolean;
}

export const useAuth = (): UseAuthReturn => {
  const [user, setUser] = useState<User | null>(() => {
    // Initialize from localStorage
    const storedUser = localStorage.getItem(USER_KEY);
    try {
      return storedUser ? (JSON.parse(storedUser) as User) : null;
    } catch {
      return null;
    }
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (email: string, password: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const errorData = (await response.json()) as { message?: string };
        throw new Error(errorData.message || 'Login failed');
      }

      const data = (await response.json()) as {
        accessToken: string;
        refreshToken: string;
        user: User;
      };

      // Store tokens
      localStorage.setItem(ACCESS_TOKEN_KEY, data.accessToken);
      localStorage.setItem(REFRESH_TOKEN_KEY, data.refreshToken);
      localStorage.setItem(USER_KEY, JSON.stringify(data.user));

      setUser(data.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setIsLoading(true);
    try {
      const token = localStorage.getItem(ACCESS_TOKEN_KEY);

      if (token) {
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
      }
    } catch (err) {
      // Log but don't throw - we still want to clear local state
      console.error('Logout API call failed:', err);
    } finally {
      // Clear local storage and state
      localStorage.removeItem(ACCESS_TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      setUser(null);
      setIsLoading(false);
    }
  }, []);

  const register = useCallback(
    async (email: string, password: string, firstName?: string, lastName?: string) => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/auth/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ email, password, firstName, lastName }),
        });

        if (!response.ok) {
          const errorData = (await response.json()) as { message?: string };
          throw new Error(errorData.message || 'Registration failed');
        }

        const data = (await response.json()) as {
          accessToken: string;
          refreshToken: string;
          user: User;
        };

        // Store tokens
        localStorage.setItem(ACCESS_TOKEN_KEY, data.accessToken);
        localStorage.setItem(REFRESH_TOKEN_KEY, data.refreshToken);
        localStorage.setItem(USER_KEY, JSON.stringify(data.user));

        setUser(data.user);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Registration failed');
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  const refreshAccessToken = useCallback(async (): Promise<boolean> => {
    const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);

    if (!refreshToken) {
      return false;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refreshToken }),
      });

      if (!response.ok) {
        // Refresh failed - clear tokens and user
        localStorage.removeItem(ACCESS_TOKEN_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        setUser(null);
        return false;
      }

      const data = (await response.json()) as {
        accessToken: string;
        refreshToken?: string;
      };
      localStorage.setItem(ACCESS_TOKEN_KEY, data.accessToken);

      if (data.refreshToken) {
        localStorage.setItem(REFRESH_TOKEN_KEY, data.refreshToken);
      }

      return true;
    } catch (err) {
      console.error('Token refresh failed:', err);
      return false;
    }
  }, []);

  const getAccessToken = useCallback((): string | null => {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    user,
    isLoading,
    loading: isLoading,
    error,
    login,
    logout,
    register,
    refreshAccessToken,
    getAccessToken,
    clearError,
    isAuthenticated: !!user,
  };
};
