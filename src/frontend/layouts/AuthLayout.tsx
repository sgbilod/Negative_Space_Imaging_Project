/**
 * Auth Layout Component
 *
 * Centered layout for authentication pages (login, register).
 * No navigation bar or sidebar.
 *
 * @component
 */

import React from 'react';
import { Outlet } from 'react-router-dom';

/**
 * Auth layout component
 */
export const AuthLayout: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        {/* Logo/Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Negative Space
          </h1>
          <p className="text-gray-600">
            Image Analysis Platform
          </p>
        </div>

        {/* Content */}
        <div className="bg-white rounded-lg shadow-md p-8">
          <Outlet />
        </div>

        {/* Footer */}
        <div className="text-center mt-6 text-sm text-gray-600">
          <p>© 2025 Negative Space Imaging. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
};

export default AuthLayout;
