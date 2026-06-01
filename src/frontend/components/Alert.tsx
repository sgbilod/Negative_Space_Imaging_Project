/**
 * Alert Component
 *
 * Alert message component with different types and dismissible option.
 *
 * @component
 */

import React, { useState } from 'react';

interface AlertProps {
  type?: 'success' | 'error' | 'warning' | 'info';
  message: string;
  onClose?: () => void;
  dismissible?: boolean;
}

/**
 * Alert component
 */
export const Alert: React.FC<AlertProps> = ({
  type = 'info',
  message,
  onClose,
  dismissible = true,
}) => {
  const [isVisible, setIsVisible] = useState(true);

  const handleClose = () => {
    setIsVisible(false);
    if (onClose) {
      onClose();
    }
  };

  if (!isVisible) {
    return null;
  }

  const typeStyles = {
    success: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      text: 'text-green-800',
      icon: '✓',
      iconColor: 'text-green-600',
    },
    error: {
      bg: 'bg-red-50',
      border: 'border-red-200',
      text: 'text-red-800',
      icon: '✕',
      iconColor: 'text-red-600',
    },
    warning: {
      bg: 'bg-yellow-50',
      border: 'border-yellow-200',
      text: 'text-yellow-800',
      icon: '⚠',
      iconColor: 'text-yellow-600',
    },
    info: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-800',
      icon: 'ℹ',
      iconColor: 'text-blue-600',
    },
  };

  const style = typeStyles[type];

  return (
    <div
      className={`${style.bg} ${style.border} ${style.text} rounded-lg border px-4 py-3`}
      role="alert"
    >
      <div className="flex items-center gap-3">
        <span className={`text-lg ${style.iconColor}`}>{style.icon}</span>
        <span className="flex-1">{message}</span>
        {dismissible && (
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600"
            aria-label="Close alert"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
};

export default Alert;
