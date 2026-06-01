/**
 * Main Layout Component
 *
 * Primary layout wrapper for authenticated pages with navigation,
 * sidebar, and main content area.
 *
 * @component
 */

import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from '../components/Navbar';

/**
 * Main layout component
 */
export const MainLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'w-64' : 'w-20'
        } bg-gray-900 text-white transition-all duration-300 ease-in-out overflow-y-auto`}
      >
        <div className="p-4">
          {/* Logo */}
          <div className="flex items-center justify-center mb-8">
            <div className="text-2xl font-bold">
              {sidebarOpen ? 'NSI' : 'N'}
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-2">
            <a
              href="/dashboard"
              className="block px-4 py-2 rounded-lg hover:bg-gray-800 transition"
            >
              <span className="inline-block">📊</span>
              {sidebarOpen && <span className="ml-3">Dashboard</span>}
            </a>
            <a
              href="/images"
              className="block px-4 py-2 rounded-lg hover:bg-gray-800 transition"
            >
              <span className="inline-block">🖼️</span>
              {sidebarOpen && <span className="ml-3">Images</span>}
            </a>
            <a
              href="/upload"
              className="block px-4 py-2 rounded-lg hover:bg-gray-800 transition"
            >
              <span className="inline-block">📤</span>
              {sidebarOpen && <span className="ml-3">Upload</span>}
            </a>
            <a
              href="/profile"
              className="block px-4 py-2 rounded-lg hover:bg-gray-800 transition"
            >
              <span className="inline-block">👤</span>
              {sidebarOpen && <span className="ml-3">Profile</span>}
            </a>
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
        <main className="flex-1 overflow-y-auto">
          <div className="p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
