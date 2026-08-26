/**
 * DashboardPage Component
 *
 * Main dashboard page showing user statistics and recent activity.
 *
 * @component
 */

import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Spinner } from '../components/Spinner';
import { ImageCard } from '../components/ImageCard';

interface DashboardStats {
  totalImages: number;
  totalAnalyses: number;
  averageProcessingTime: number;
}

/**
 * DashboardPage component
 */
export const DashboardPage: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentImages, setRecentImages] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);

        // TODO: Replace with actual API calls
        const statsResponse = await Promise.resolve({
          totalImages: 24,
          totalAnalyses: 18,
          averageProcessingTime: 2.34,
        });

        const imagesResponse = await Promise.resolve([
          {
            id: '1',
            name: 'Mountain Landscape',
            imageUrl: 'https://via.placeholder.com/300x200?text=Mountain',
            uploadedAt: new Date().toISOString(),
            fileSize: 2048576,
          },
          {
            id: '2',
            name: 'City Architecture',
            imageUrl: 'https://via.placeholder.com/300x200?text=City',
            uploadedAt: new Date(Date.now() - 86400000).toISOString(),
            fileSize: 1536000,
          },
          {
            id: '3',
            name: 'Forest Scene',
            imageUrl: 'https://via.placeholder.com/300x200?text=Forest',
            uploadedAt: new Date(Date.now() - 172800000).toISOString(),
            fileSize: 3145728,
          },
        ]);

        setStats(statsResponse);
        setRecentImages(imagesResponse);
      } catch (err) {
        setError('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">Welcome back! Here's your analysis overview.</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm font-medium">Total Images</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalImages}</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center text-xl">
                🖼️
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm font-medium">Total Analyses</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalAnalyses}</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center text-xl">
                ✓
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm font-medium">Avg Processing Time</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {stats.averageProcessingTime}s
                </p>
              </div>
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center text-xl">
                ⏱️
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Images Section */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Recent Images</h2>
          <Link to="/images" className="text-blue-600 hover:text-blue-700 font-semibold">
            View All
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {recentImages.map((image) => (
            <ImageCard
              key={image.id}
              id={image.id}
              name={image.name}
              imageUrl={image.imageUrl}
              uploadedAt={image.uploadedAt}
              fileSize={image.fileSize}
            />
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Link
            to="/upload"
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition text-center font-semibold"
          >
            📤 Upload New Image
          </Link>
          <Link
            to="/images"
            className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition text-center font-semibold"
          >
            🖼️ Browse Gallery
          </Link>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
