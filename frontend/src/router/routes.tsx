/**
 * Route Definitions & Configuration
 *
 * Defines all application routes with lazy loading, meta information,
 * and access control requirements.
 */

import { lazy } from 'react';
import type { RouteObject } from 'react-router-dom';
import LoginPage from '../pages/LoginPage';
import RegisterPage from '../pages/RegisterPage';

const DashboardPage = lazy(() => import('../pages/DashboardPage'));
const UploadPage = lazy(() => import('../pages/UploadPage'));
const AnalysisResultsPage = lazy(() => import('../pages/AnalysisResultsPage'));
const SettingsPage = lazy(() => import('../pages/SettingsPage'));
const NotFoundPage = lazy(() => import('../pages/NotFoundPage'));
const ErrorPage = lazy(() => import('../pages/ErrorPage'));

/**
 * Route metadata for enhanced route information
 */
export interface RouteMeta {
  title: string;
  description?: string;
  access: 'public' | 'protected' | 'admin';
  showInNav?: boolean;
  icon?: string;
  parent?: string;
}

/**
 * Extended route object with metadata
 */
export type ExtendedRouteObject = RouteObject & {
  meta?: RouteMeta;
};

/**
 * Route metadata registry
 */
export const routeMetadata: Record<string, RouteMeta> = {
  '/': {
    title: 'Dashboard',
    description: 'Main application dashboard',
    access: 'protected',
    showInNav: true,
    icon: 'Dashboard',
  },
  '/login': {
    title: 'Login',
    description: 'User authentication',
    access: 'public',
    showInNav: false,
  },
  '/register': {
    title: 'Register',
    description: 'Create new account',
    access: 'public',
    showInNav: false,
  },
  '/dashboard': {
    title: 'Dashboard',
    description: 'Main application dashboard',
    access: 'protected',
    showInNav: true,
    icon: 'Dashboard',
  },
  '/upload': {
    title: 'Upload',
    description: 'Upload new images for analysis',
    access: 'protected',
    showInNav: true,
    icon: 'CloudUpload',
    parent: '/dashboard',
  },
  '/images': {
    title: 'Images',
    description: 'Browse uploaded images',
    access: 'protected',
    showInNav: true,
    icon: 'Image',
    parent: '/dashboard',
  },
  '/analysis': {
    title: 'Analysis',
    description: 'View analysis results',
    access: 'protected',
    showInNav: true,
    icon: 'Analytics',
    parent: '/dashboard',
  },
  '/settings': {
    title: 'Settings',
    description: 'User account settings',
    access: 'protected',
    showInNav: true,
    icon: 'Settings',
  },
  '/admin': {
    title: 'Admin',
    description: 'Administration panel',
    access: 'admin',
    showInNav: true,
    icon: 'AdminPanelSettings',
  },
  '/admin/users': {
    title: 'User Management',
    description: 'Manage system users',
    access: 'admin',
    showInNav: true,
    icon: 'People',
    parent: '/admin',
  },
  '/admin/stats': {
    title: 'System Statistics',
    description: 'View system statistics and metrics',
    access: 'admin',
    showInNav: true,
    icon: 'BarChart',
    parent: '/admin',
  },
  '*': {
    title: 'Not Found',
    description: 'Page not found',
    access: 'public',
    showInNav: false,
  },
};

/**
 * Complete route configuration
 */
export const routes: ExtendedRouteObject[] = [
  {
    path: '/',
    element: <DashboardPage />,
    meta: routeMetadata['/'],
  },
  {
    path: '/login',
    element: <LoginPage />,
    meta: routeMetadata['/login'],
  },
  {
    path: '/register',
    element: <RegisterPage />,
    meta: routeMetadata['/register'],
  },
  {
    path: '/dashboard',
    element: <DashboardPage />,
    meta: routeMetadata['/dashboard'],
  },
  {
    path: '/upload',
    element: <UploadPage />,
    meta: routeMetadata['/upload'],
  },
  {
    path: '/analysis/:id',
    element: <AnalysisResultsPage />,
    meta: routeMetadata['/analysis'],
  },
  {
    path: '/settings',
    element: <SettingsPage />,
    meta: routeMetadata['/settings'],
  },
  {
    path: '/admin',
    element: <div>Admin Dashboard - Coming Soon</div>,
    meta: routeMetadata['/admin'],
  },
  {
    path: '/admin/users',
    element: <div>User Management - Coming Soon</div>,
    meta: routeMetadata['/admin/users'],
  },
  {
    path: '/admin/stats',
    element: <div>System Statistics - Coming Soon</div>,
    meta: routeMetadata['/admin/stats'],
  },
  {
    path: '/error',
    element: <ErrorPage />,
    meta: routeMetadata['*'],
  },
  {
    path: '*',
    element: <NotFoundPage />,
    meta: routeMetadata['*'],
  },
];

/**
 * Get route metadata by path
 */
export function getRouteMeta(path: string): RouteMeta | undefined {
  return routeMetadata[path];
}

/**
 * Check if route is public
 */
export function isPublicRoute(path: string): boolean {
  const meta = getRouteMeta(path);
  return meta?.access === 'public';
}

/**
 * Check if route requires protection
 */
export function isProtectedRoute(path: string): boolean {
  const meta = getRouteMeta(path);
  return meta?.access === 'protected' || meta?.access === 'admin';
}

/**
 * Check if route requires admin role
 */
export function isAdminRoute(path: string): boolean {
  const meta = getRouteMeta(path);
  return meta?.access === 'admin';
}

/**
 * Get all navigable routes
 */
export function getNavigableRoutes(): Array<{
  path: string;
  title: string;
  icon?: string;
  access: string;
}> {
  return Object.entries(routeMetadata)
    .filter(([, meta]) => meta.showInNav)
    .map(([path, meta]) => ({
      path,
      title: meta.title,
      icon: meta.icon,
      access: meta.access,
    }));
}

/**
 * Get breadcrumb path for navigation
 */
export function getBreadcrumbs(path: string): Array<{ path: string; title: string }> {
  const breadcrumbs: Array<{ path: string; title: string }> = [];
  const segments = path.split('/').filter(Boolean);

  // Add home
  breadcrumbs.push({ path: '/dashboard', title: 'Dashboard' });

  // Build breadcrumb path
  let currentPath = '';
  for (const segment of segments) {
    currentPath += `/${segment}`;
    const meta = getRouteMeta(currentPath);
    if (meta && meta.showInNav) {
      breadcrumbs.push({ path: currentPath, title: meta.title });
    }
  }

  return breadcrumbs;
}
