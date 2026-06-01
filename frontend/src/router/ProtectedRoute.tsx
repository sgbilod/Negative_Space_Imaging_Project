/**
 * Protected Route Component
 *
 * Wrapper component that enforces route protection based on authentication
 * and user role. Redirects to login if not authenticated, or to error page
 * if insufficient permissions.
 *
 * Features:
 * - Authentication checks
 * - Role-based access control (user vs admin)
 * - Automatic redirects
 * - Optional loading state display
 * - TypeScript strict typing
 */

import type { ReactNode, FC } from 'react';
import { Navigate } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import { useAuth } from '../hooks/useAuth';

/**
 * User roles for access control
 */
export type UserRole = 'user' | 'admin' | 'guest';

/**
 * Props for ProtectedRoute component
 */
interface ProtectedRouteProps {
  /** Component to render if access is allowed */
  children: ReactNode;

  /** Required user role ('user' | 'admin') */
  requiredRole?: UserRole;

  /** Whether to show loading spinner while checking auth */
  showLoadingSpinner?: boolean;

  /** Fallback component to show while checking auth */
  fallback?: ReactNode;
}

/**
 * ProtectedRoute Component
 *
 * Checks authentication and authorization before rendering protected content.
 * Automatically redirects unauthorized users to login or error pages.
 *
 * Usage:
 * ```tsx
 * <ProtectedRoute requiredRole="user">
 *   <DashboardPage />
 * </ProtectedRoute>
 *
 * <ProtectedRoute requiredRole="admin">
 *   <AdminPanel />
 * </ProtectedRoute>
 * ```
 *
 * @param children - Component to render if access is allowed
 * @param requiredRole - Required role ('user' or 'admin'), defaults to 'user'
 * @param showLoadingSpinner - Show loading while checking auth (default: true)
 * @param fallback - Custom fallback component while checking auth
 * @returns Protected component or redirect/loading state
 */
export function ProtectedRoute({
  children,
  requiredRole = 'user',
  showLoadingSpinner = true,
  fallback,
}: ProtectedRouteProps): JSX.Element {
  const { isAuthenticated, loading, user } = useAuth();

  // Show loading state while checking authentication
  if (loading) {
    if (fallback) {
      return fallback as JSX.Element;
    }

    if (showLoadingSpinner) {
      return (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
          <CircularProgress />
        </Box>
      );
    }

    return <></>;
  }

  // Not authenticated - redirect to login
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Check role-based access
  if (requiredRole === 'admin' && !user?.roles?.includes('admin')) {
    // Not admin - redirect to error page
    return <Navigate to="/error" replace />;
  }

  // Access granted - render children
  return <>{children}</>;
}

/**
 * Higher-order component for protecting routes
 *
 * Usage:
 * ```tsx
 * const ProtectedDashboard = withProtection(DashboardPage, 'user');
 *
 * <Routes>
 *   <Route path="/dashboard" element={<ProtectedDashboard />} />
 * </Routes>
 * ```
 */
export function withProtection<P extends object>(
  Component: React.ComponentType<P>,
  requiredRole: UserRole = 'user',
) {
  return function ProtectedComponent(props: P) {
    return (
      <ProtectedRoute requiredRole={requiredRole}>
        <Component {...props} />
      </ProtectedRoute>
    );
  };
}

/**
 * Custom hook to check if current user has access to a route
 *
 * Usage:
 * ```tsx
 * function MyComponent() {
 *   const hasAccess = useRouteAccess('admin');
 *
 *   if (!hasAccess) {
 *     return <div>Access Denied</div>;
 *   }
 *
 *   return <AdminPanel />;
 * }
 * ```
 */
export function useRouteAccess(requiredRole: UserRole = 'user'): boolean {
  const { isAuthenticated, user, loading } = useAuth();

  // Still loading
  if (loading) {
    return false;
  }

  // Not authenticated
  if (!isAuthenticated) {
    return false;
  }

  // Check role
  if (requiredRole === 'admin' && !user?.roles?.includes('admin')) {
    return false;
  }

  return true;
}
