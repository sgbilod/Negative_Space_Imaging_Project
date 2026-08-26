/**
 * Spinner Component
 *
 * Loading spinner component with different sizes.
 *
 * @component
 */

import React from 'react';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

/**
 * Spinner component
 */
export const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  className = '',
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  return (
    <div
      className={`inline-block animate-spin rounded-full border-4 border-gray-300 border-t-blue-600 ${sizeClasses[size]} ${className}`}
      aria-label="Loading"
    />
  );
};

export default Spinner;
