import { useState, useEffect } from 'react';
import jwt_decode from 'jwt-decode';

interface AuthTokenPayload {
  sub: string;
  exp: number;
}

export const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [user, setUser] = useState<string | null>(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const decoded = jwt_decode<AuthTokenPayload>(token);
        const currentTime = Date.now() / 1000;

        if (decoded.exp > currentTime) {
          setIsAuthenticated(true);
          setUser(decoded.sub);
        } else {
          // Token expired
          localStorage.removeItem('token');
          setIsAuthenticated(false);
          setUser(null);
        }
      } catch (error) {
        // Invalid token
        localStorage.removeItem('token');
        setIsAuthenticated(false);
        setUser(null);
      }
    }
  }, []);

  const login = (token: string) => {
    localStorage.setItem('token', token);
    const decoded = jwt_decode<AuthTokenPayload>(token);
    setIsAuthenticated(true);
    setUser(decoded.sub);
  };

  const logout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setUser(null);
  };

  return {
    isAuthenticated,
    user,
    login,
    logout,
  };
};

export default useAuth;
