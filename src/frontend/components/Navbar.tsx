/**
 * Navbar Component
 *
 * Top navigation bar with logo, navigation links, and user menu.
 *
 * @component
 */

import React, { useState } from 'react';
import { Link } from 'react-router-dom';

interface NavbarProps {
  onToggleSidebar?: () => void;
}

/**
 * Navbar component
 */
export const Navbar: React.FC<NavbarProps> = ({ onToggleSidebar }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  return (
    <nav className="bg-white shadow-md px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left Section: Toggle & Logo */}
        <div className="flex items-center gap-4">
          {onToggleSidebar && (
            <button
              onClick={onToggleSidebar}
              className="text-gray-600 hover:text-gray-900 lg:hidden"
              aria-label="Toggle sidebar"
            >
              ☰
            </button>
          )}
          <Link to="/" className="text-xl font-bold text-blue-600">
            NSI
          </Link>
        </div>

        {/* Center Section: Navigation Links (Desktop) */}
        <div className="hidden md:flex gap-6">
          <Link
            to="/dashboard"
            className="text-gray-700 hover:text-blue-600 transition"
          >
            Dashboard
          </Link>
          <Link
            to="/images"
            className="text-gray-700 hover:text-blue-600 transition"
          >
            Images
          </Link>
          <Link
            to="/upload"
            className="text-gray-700 hover:text-blue-600 transition"
          >
            Upload
          </Link>
        </div>

        {/* Right Section: Mobile Menu & User Menu */}
        <div className="flex items-center gap-4">
          {/* Mobile Menu Toggle */}
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="md:hidden text-gray-600 hover:text-gray-900"
            aria-label="Toggle menu"
          >
            {menuOpen ? '✕' : '☰'}
          </button>

          {/* User Menu Dropdown */}
          <div className="relative">
            <button
              onClick={() => setUserMenuOpen(!userMenuOpen)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition"
              aria-expanded={userMenuOpen}
            >
              <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-semibold">
                U
              </div>
              <span className="hidden sm:inline text-sm text-gray-700">User</span>
              <span className="text-gray-600">▼</span>
            </button>

            {/* Dropdown Menu */}
            {userMenuOpen && (
              <div
                className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg py-2 z-50"
                role="menu"
              >
                <Link
                  to="/profile"
                  className="block px-4 py-2 text-gray-700 hover:bg-gray-100"
                  onClick={() => setUserMenuOpen(false)}
                >
                  Profile
                </Link>
                <button
                  className="w-full text-left px-4 py-2 text-gray-700 hover:bg-gray-100"
                  onClick={() => {
                    setUserMenuOpen(false);
                    // TODO: Handle logout
                  }}
                >
                  Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="md:hidden mt-4 space-y-2 border-t border-gray-200 pt-4">
          <Link
            to="/dashboard"
            className="block px-4 py-2 text-gray-700 hover:bg-gray-100 rounded"
            onClick={() => setMenuOpen(false)}
          >
            Dashboard
          </Link>
          <Link
            to="/images"
            className="block px-4 py-2 text-gray-700 hover:bg-gray-100 rounded"
            onClick={() => setMenuOpen(false)}
          >
            Images
          </Link>
          <Link
            to="/upload"
            className="block px-4 py-2 text-gray-700 hover:bg-gray-100 rounded"
            onClick={() => setMenuOpen(false)}
          >
            Upload
          </Link>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
