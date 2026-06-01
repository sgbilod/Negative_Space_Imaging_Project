import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../hooks';

interface PrivateRouteProps {
  children: React.ReactNode;
  requiredRoles?: string[];
}

/**
 * PrivateRoute Component
 *
 * Protected route wrapper that:
 * - Checks authentication status
 * - Enforces role-based access control
 * - Redirects unauthenticated users to login
 * - Shows loading spinner during auth check
 * - Preserves attempted location for post-login redirect
 */
const PrivateRoute: React.FC<PrivateRouteProps> = ({ children, requiredRoles = [] }) => {
  const { user, isAuthenticated } = useAuth();
  const location = useLocation();

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Check role-based access control
  if (requiredRoles.length > 0) {
    const hasRequiredRole = requiredRoles.some((role) => user?.roles?.includes(role));

    if (!hasRequiredRole) {
      // Redirect to unauthorized page or dashboard
      return <Navigate to="/unauthorized" replace />;
    }
  }

  // User is authenticated and authorized
  return <>{children}</>;
};

export default PrivateRoute;
