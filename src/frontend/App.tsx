import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '@/context/AuthContext';
import { NotificationProvider } from '@/context/NotificationContext';
import MainLayout from '@/layouts/MainLayout';
import AuthLayout from '@/layouts/AuthLayout';
import LoginPage from '@/pages/LoginPage';
import RegisterPage from '@/pages/RegisterPage';
import DashboardPage from '@/pages/DashboardPage';
import ImagesPage from '@/pages/ImagesPage';
import ImageDetailPage from '@/pages/ImageDetailPage';
import UploadPage from '@/pages/UploadPage';
import AnalysisPage from '@/pages/AnalysisPage';
import ProfilePage from '@/pages/ProfilePage';
import NotFoundPage from '@/pages/NotFoundPage';
import ProtectedRoute from '@/components/ProtectedRoute';
import '@/styles/globals.css';

export default function App() {
  return (
    <NotificationProvider>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Auth Routes */}
            <Route
              path="/login"
              element={
                <AuthLayout>
                  <LoginPage />
                </AuthLayout>
              }
            />
            <Route
              path="/register"
              element={
                <AuthLayout>
                  <RegisterPage />
                </AuthLayout>
              }
            />

            {/* Protected Routes */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <DashboardPage />
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <DashboardPage />
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/images"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <ImagesPage />
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/images/:imageId"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <ImageDetailPage />
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/upload"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <UploadPage />
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/analysis/:analysisId"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <AnalysisPage />
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <ProfilePage />
                  </MainLayout>
                </ProtectedRoute>
              }
            />

            {/* Error Routes */}
            <Route path="/404" element={<NotFoundPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Router>
      </AuthProvider>
    </NotificationProvider>
  );
}
