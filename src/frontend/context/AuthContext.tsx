import React, { createContext, useState, useCallback, ReactNode } from 'react';

export interface AuthContextType {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
  error: string | null;
  setUser: (user: AuthUser | null) => void;
  setToken: (token: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export interface AuthUser {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(() => {
    // Load token from localStorage on initialization
    if (typeof window !== 'undefined') {
      return localStorage.getItem('auth_token');
    }
    return null;
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateUser = useCallback((newUser: AuthUser | null) => {
    setUser(newUser);
  }, []);

  const updateToken = useCallback((newToken: string | null) => {
    setToken(newToken);
    if (typeof window !== 'undefined') {
      if (newToken) {
        localStorage.setItem('auth_token', newToken);
      } else {
        localStorage.removeItem('auth_token');
      }
    }
  }, []);

  const updateLoading = useCallback((newLoading: boolean) => {
    setLoading(newLoading);
  }, []);

  const updateError = useCallback((newError: string | null) => {
    setError(newError);
  }, []);

  const value: AuthContextType = {
    user,
    token,
    loading,
    error,
    setUser: updateUser,
    setToken: updateToken,
    setLoading: updateLoading,
    setError: updateError,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
