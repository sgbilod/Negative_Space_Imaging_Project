/**
 * NotFoundPage Component
 *
 * 404 error page for non-existent routes.
 *
 * @component
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '../components/Button';

/**
 * NotFoundPage component
 */
export const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 px-4">
      <div className="text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-gray-900">404</h1>
          <p className="text-2xl font-semibold text-gray-800 mt-4">Page Not Found</p>
        </div>

        <p className="text-gray-600 text-lg mb-8 max-w-md mx-auto">
          Sorry, the page you're looking for doesn't exist. It might have been moved or deleted.
        </p>

        <div className="space-y-4">
          <Link to="/" className="inline-block">
            <Button variant="primary" size="md">
              Go Home
            </Button>
          </Link>
          <p className="text-gray-600">
            <Link to="/dashboard" className="text-blue-600 hover:text-blue-700 underline">
              Return to Dashboard
            </Link>
          </p>
        </div>

        <div className="mt-12 text-6xl">🔍</div>
      </div>
    </div>
  );
};

export default NotFoundPage;
