/**
 * Auth Context
 * Provides global authentication state and methods to all components
 * Manages user authentication state, tokens, and auth operations
 */

import React, { createContext, useMemo } from 'react';
import { useAuth, User } from '../hooks/useAuth';

/**
 * Auth context state shape
 */
interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, firstName: string, lastName: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<boolean>;
  getAccessToken: () => string | null;
  clearError: () => void;
}

/**
 * Create auth context
 */
export const AuthContext = createContext<AuthContextType | undefined>(undefined);

/**
 * Auth Context Provider Props
 */
interface AuthProviderProps {
  children: React.ReactNode;
}

/**
 * Auth Provider Component
 * Wraps the app with authentication state
 *
 * @param props - Provider props
 * @returns Auth provider component
 *
 * @example
 * <AuthProvider>
 *   <App />
 * </AuthProvider>
 */
export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const {
    user,
    isAuthenticated,
    loading,
    error,
    login,
    register,
    logout,
    refreshAccessToken,
    getAccessToken,
    clearError,
  } = useAuth();

  /**
   * Memoized context value to prevent unnecessary re-renders
   */
  const value = useMemo<AuthContextType>(
    () => ({
      user,
      isAuthenticated,
      loading,
      error,
      login,
      register,
      logout,
      refreshToken: refreshAccessToken,
      getAccessToken,
      clearError,
    }),
    [
      user,
      isAuthenticated,
      loading,
      error,
      login,
      register,
      logout,
      refreshAccessToken,
      getAccessToken,
      clearError,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

/**
 * useAuthContext - Custom hook to use auth context
 * Must be used within AuthProvider
 *
 * @returns {AuthContextType} Auth context value
 * @throws {Error} If used outside of AuthProvider
 *
 * @example
 * const { user, login, logout } = useAuthContext();
 */
export const useAuthContext = (): AuthContextType => {
  const context = React.useContext(AuthContext);

  if (context === undefined) {
    throw new Error('useAuthContext must be used within an AuthProvider');
  }

  return context;
};

export default AuthProvider;
