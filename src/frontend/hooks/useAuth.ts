import { useContext, useCallback } from 'react';
import { AuthContext } from '@/context/AuthContext';
import { authService } from '@/services/authService';

export interface AuthUser {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
}

export interface UseAuthReturn {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, firstName: string, lastName: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

export const useAuth = (): UseAuthReturn => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }

  const { user, token, loading, error, setUser, setToken, setLoading, setError } = context;

  const login = useCallback(
    async (email: string, password: string) => {
      setLoading(true);
      setError(null);
      try {
        const response = await authService.login(email, password);
        setUser(response.user);
        setToken(response.token);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Login failed');
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [setUser, setToken, setLoading, setError]
  );

  const register = useCallback(
    async (email: string, password: string, firstName: string, lastName: string) => {
      setLoading(true);
      setError(null);
      try {
        const response = await authService.register(email, password, firstName, lastName);
        setUser(response.user);
        setToken(response.token);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Registration failed');
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [setUser, setToken, setLoading, setError]
  );

  const logout = useCallback(() => {
    authService.logout();
    setUser(null);
    setToken(null);
    setError(null);
  }, [setUser, setToken, setError]);

  const checkAuth = useCallback(async () => {
    setLoading(true);
    try {
      const response = await authService.getCurrentUser();
      setUser(response.user);
      setToken(response.token);
    } catch {
      setUser(null);
      setToken(null);
    } finally {
      setLoading(false);
    }
  }, [setUser, setToken, setLoading]);

  return { user, token, loading, error, login, register, logout, checkAuth };
};
