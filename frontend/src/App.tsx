import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './components/Layout';
import PrivateRoute from './components/PrivateRoute';

const Login = lazy(() => import('./pages/Login'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ImageProcessing = lazy(() => import('./pages/ImageProcessing'));
const SecurityMonitor = lazy(() => import('./pages/SecurityMonitor'));
const AuditLogs = lazy(() => import('./pages/AuditLogs'));
const UserManagement = lazy(() => import('./pages/UserManagement'));

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#90caf9' },
    secondary: { main: '#f48fb1' },
    background: { default: '#121212', paper: '#1e1e1e' },
  },
});

const App: React.FC = () => (
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <Router>
      <Suspense fallback={<div>Loading...</div>}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Layout />}>
            <Route index element={<PrivateRoute><Dashboard /></PrivateRoute>} />
            <Route path="processing" element={<PrivateRoute><ImageProcessing /></PrivateRoute>} />
            <Route path="security" element={<PrivateRoute><SecurityMonitor /></PrivateRoute>} />
            <Route path="audit" element={<PrivateRoute><AuditLogs /></PrivateRoute>} />
            <Route path="users" element={<PrivateRoute><UserManagement /></PrivateRoute>} />
          </Route>
        </Routes>
      </Suspense>
    </Router>
  </ThemeProvider>
);

export default App;
