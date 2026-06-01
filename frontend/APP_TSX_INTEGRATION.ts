/**
 * APP.TSX INTEGRATION GUIDE
 * How to integrate Phase 4 pages into your React Router setup
 *
 * This file shows the exact changes needed to App.tsx to support the new pages
 */

// ============================================================================
// UPDATED App.tsx EXAMPLE
// ============================================================================

import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Import context providers
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider as CustomThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Import components
import Layout from './components/Layout';
import PrivateRoute from './components/PrivateRoute';
import ErrorBoundary from './components/common/ErrorBoundary';
import LoadingSpinner from './components/common/LoadingSpinner';

// Lazy load pages for code splitting
// Phase 3 pages
const Login = lazy(() => import('./pages/Login'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ImageProcessing = lazy(() => import('./pages/ImageProcessing'));
const SecurityMonitor = lazy(() => import('./pages/SecurityMonitor'));
const AuditLogs = lazy(() => import('./pages/AuditLogs'));
const UserManagement = lazy(() => import('./pages/UserManagement'));

// Phase 4 pages (NEW)
const LoginPage = lazy(() => import('./pages/LoginPage'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const AnalysisResultsPage = lazy(() => import('./pages/AnalysisResultsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));

/**
 * Material-UI theme configuration
 */
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#90caf9' },
    secondary: { main: '#f48fb1' },
    background: { default: '#121212', paper: '#1e1e1e' },
  },
});

/**
 * Root App Component
 * Provides global state management, theme, and routing
 */
const App: React.FC = () => (
  <ErrorBoundary showDetails={process.env.NODE_ENV === 'development'}>
    <CustomThemeProvider defaultTheme="dark">
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AuthProvider>
          <NotificationProvider>
            <Router>
              <Suspense fallback={<LoadingSpinner fullScreen />}>
                <Routes>
                  {/* ============================================================ */}
                  {/* AUTHENTICATION ROUTES (Public) */}
                  {/* ============================================================ */}

                  {/* Phase 4: New Login Page */}
                  <Route path="/login" element={<LoginPage />} />

                  {/* Phase 4: New Register Page */}
                  <Route path="/register" element={<RegisterPage />} />

                  {/* Phase 3: Keep existing login for backward compatibility */}
                  <Route path="/auth/login" element={<Login />} />

                  {/* ============================================================ */}
                  {/* AUTHENTICATED ROUTES (Private) */}
                  {/* ============================================================ */}

                  {/* Dashboard */}
                  <Route
                    path="/dashboard"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <DashboardPage />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  {/* Phase 4: Upload Page */}
                  <Route
                    path="/upload"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <UploadPage />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  {/* Phase 4: Analysis Results Page */}
                  <Route
                    path="/analysis/:imageId"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <AnalysisResultsPage />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  {/* Phase 4: Settings Page */}
                  <Route
                    path="/settings"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <SettingsPage />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  {/* Phase 3: Existing pages */}
                  <Route
                    path="/dashboard-old"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <Dashboard />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  <Route
                    path="/imaging"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <ImageProcessing />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  <Route
                    path="/security"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <SecurityMonitor />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  <Route
                    path="/audit"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <AuditLogs />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  <Route
                    path="/users"
                    element={
                      <PrivateRoute>
                        <Layout>
                          <UserManagement />
                        </Layout>
                      </PrivateRoute>
                    }
                  />

                  {/* ============================================================ */}
                  {/* DEFAULT ROUTES */}
                  {/* ============================================================ */}

                  {/* Root path: redirect to dashboard */}
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />

                  {/* 404 Not Found - add this last */}
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Suspense>
            </Router>
          </NotificationProvider>
        </AuthProvider>
      </ThemeProvider>
    </CustomThemeProvider>
  </ErrorBoundary>
);

export default App;

// ============================================================================
// NAVIGATION COMPONENT EXAMPLE
// ============================================================================

/**
 * Example Navbar component for navigation
 *
 * This should be placed in src/components/Layout.tsx or similar
 */

import { Box, Button, Link, List, ListItem } from '@mui/material';
import { useNavigate } from 'react-router-dom';

interface NavItem {
  label: string;
  path: string;
  authenticated: boolean;
}

const navigationItems: NavItem[] = [
  // Public routes
  { label: 'Login', path: '/login', authenticated: false },
  { label: 'Register', path: '/register', authenticated: false },

  // Authenticated routes
  { label: 'Dashboard', path: '/dashboard', authenticated: true },
  { label: 'Upload', path: '/upload', authenticated: true },
  { label: 'Settings', path: '/settings', authenticated: true },
];

const NavigationExample: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box sx={{ display: 'flex', gap: 2 }}>
      {navigationItems.map(item => (
        <Button
          key={item.path}
          onClick={() => navigate(item.path)}
          sx={{
            textTransform: 'none',
            fontSize: '1rem',
            '&:hover': {
              backgroundColor: 'rgba(102, 126, 234, 0.1)',
            },
          }}
        >
          {item.label}
        </Button>
      ))}
    </Box>
  );
};

// ============================================================================
// PRIVATE ROUTE COMPONENT
// ============================================================================

/**
 * PrivateRoute component to protect authenticated pages
 * Should be in src/components/PrivateRoute.tsx
 */

interface PrivateRouteProps {
  children: React.ReactNode;
}

const PrivateRouteExample: React.FC<PrivateRouteProps> = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();

  if (loading) {
    return <LoadingSpinner fullScreen />;
  }

  if (!isAuthenticated) {
    navigate('/login', { state: { from: window.location.pathname } });
    return null;
  }

  return <>{children}</>;
};

// ============================================================================
// ENVIRONMENT VARIABLES
// ============================================================================

/**
 * Required environment variables in .env.local
 *
 * REACT_APP_API_URL=http://localhost:3000/api
 * REACT_APP_ENV=development
 * REACT_APP_DEBUG=true
 */

// ============================================================================
// ROUTING STRATEGY
// ============================================================================

/**
 * Route Organization:
 *
 * 1. PUBLIC ROUTES (No authentication required)
 *    - /login (LoginPage)
 *    - /register (RegisterPage)
 *    - /forgot-password (Future)
 *    - /reset-password (Future)
 *
 * 2. AUTHENTICATED ROUTES (Require login)
 *    - /dashboard (DashboardPage)
 *    - /upload (UploadPage)
 *    - /analysis/:imageId (AnalysisResultsPage)
 *    - /settings (SettingsPage)
 *    - /security (SecurityMonitor)
 *    - /audit (AuditLogs)
 *    - /users (UserManagement)
 *
 * 3. DEFAULT ROUTES
 *    - / redirects to /dashboard (or /login if not authenticated)
 *    - /* (404) redirects to /
 *
 * 4. LAYOUT HIERARCHY
 *    - Public pages: No layout (full width, gradient background)
 *    - Authenticated pages: Wrapped in Layout component (navbar, sidebar)
 */

// ============================================================================
// TESTING ROUTES
// ============================================================================

/**
 * Testing checklist:
 *
 * 1. Public Routes
 *    ✓ Navigate to /login - should show LoginPage
 *    ✓ Navigate to /register - should show RegisterPage
 *    ✓ Unauthenticated /dashboard redirects to /login
 *
 * 2. Authentication Flow
 *    ✓ Click Register button on login page
 *    ✓ Fill registration form and submit
 *    ✓ Auto-login and redirect to dashboard
 *    ✓ Dashboard shows user name and data
 *
 * 3. Authenticated Routes
 *    ✓ Upload page accessible and working
 *    ✓ Analysis results page loads properly
 *    ✓ Settings page tabs function correctly
 *    ✓ Logout redirects to login
 *
 * 4. Error Handling
 *    ✓ Invalid login shows error
 *    ✓ Weak password rejected on register
 *    ✓ Network errors handled gracefully
 *    ✓ 404 pages redirect to home
 */

// ============================================================================
// MIGRATION PATH
// ============================================================================

/**
 * Upgrade from Phase 3 to Phase 4:
 *
 * 1. Add new page imports
 * 2. Add new routes to Routes section
 * 3. Update navigation components
 * 4. Keep old routes for backward compatibility
 * 5. Test all flows
 * 6. Update documentation
 * 7. Deploy to staging
 * 8. Gradually migrate users to new pages
 * 9. Deprecate old pages after transition period
 * 10. Remove old code in cleanup phase
 */

export {};
