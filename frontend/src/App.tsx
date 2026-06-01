/**
 * Root Application Component
 * Configures providers, routing, and global theme
 */

import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Import context providers
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider as CustomThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Import components
import Layout from './components/layout-root';
import PrivateRoute from './components/PrivateRoute';
import ErrorBoundary from './components/common/ErrorBoundary';
import LoadingSpinner from './components/common/LoadingSpinner';

// Lazy load pages for code splitting
const Login = lazy(() => import('./pages/Login'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ImageProcessing = lazy(() => import('./pages/ImageProcessing'));
const SecurityMonitor = lazy(() => import('./pages/SecurityMonitor'));
const AuditLogs = lazy(() => import('./pages/AuditLogs'));
const UserManagement = lazy(() => import('./pages/UserManagement'));

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
 * Provides global state management and routing
 */
const App: React.FC = () => (
  <ErrorBoundary showDetails={process.env.NODE_ENV === 'development'}>
    <CustomThemeProvider defaultTheme="dark">
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AuthProvider>
          <NotificationProvider maxNotifications={5}>
            <Router>
              <Suspense fallback={<LoadingSpinner fullScreen message="Loading application..." />}>
                <Routes>
                  <Route path="/login" element={<Login />} />
                  <Route path="/" element={<Layout />}>
                    <Route
                      index
                      element={
                        <PrivateRoute>
                          <Dashboard />
                        </PrivateRoute>
                      }
                    />
                    <Route
                      path="processing"
                      element={
                        <PrivateRoute>
                          <ImageProcessing />
                        </PrivateRoute>
                      }
                    />
                    <Route
                      path="security"
                      element={
                        <PrivateRoute>
                          <SecurityMonitor />
                        </PrivateRoute>
                      }
                    />
                    <Route
                      path="audit"
                      element={
                        <PrivateRoute>
                          <AuditLogs />
                        </PrivateRoute>
                      }
                    />
                    <Route
                      path="users"
                      element={
                        <PrivateRoute>
                          <UserManagement />
                        </PrivateRoute>
                      }
                    />
                  </Route>
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
