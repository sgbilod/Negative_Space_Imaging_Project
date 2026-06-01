/**
 * ProtectedRoute Component
 *
 * Route wrapper component that protects routes requiring authentication.
 *
 * @component
 */

import React from 'react';
import { Navigate } from 'react-router-dom';

interface ProtectedRouteProps {
  children: React.ReactNode;
  isAuthenticated: boolean;
  requiredRole?: string;
  userRole?: string;
}

/**
 * ProtectedRoute component
 */
export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  isAuthenticated,
  requiredRole,
  userRole,
}) => {
  // Check if user is authenticated
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Check if user has required role
  if (requiredRole && userRole !== requiredRole) {
    return <Navigate to="/unauthorized" replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
