/**
 * Loading Spinner Component
 * Displays a loading spinner for async operations
 * Supports different sizes and variants
 */

import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

/**
 * Loading spinner props
 */
interface LoadingSpinnerProps {
  /**
   * Size of the spinner
   */
  size?: 'small' | 'medium' | 'large';

  /**
   * Loading message to display below spinner
   */
  message?: string;

  /**
   * Whether to display as full page overlay
   */
  fullScreen?: boolean;

  /**
   * Color of the spinner
   */
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success';

  /**
   * Additional CSS classes
   */
  className?: string;
}

/**
 * Size to pixel mapping
 */
const sizeMap = {
  small: 24,
  medium: 40,
  large: 56,
};

/**
 * LoadingSpinner Component
 * Displays a customizable loading indicator
 *
 * @param props - Component props
 * @returns Loading spinner element
 *
 * @example
 * <LoadingSpinner size="large" message="Loading..." />
 */
export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  message = 'Loading...',
  fullScreen = false,
  color = 'primary',
  className,
}) => {
  const containerStyles: React.CSSProperties = fullScreen
    ? {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        zIndex: 9999,
      }
    : {};

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
        ...containerStyles,
      }}
      className={className}
    >
      <CircularProgress size={sizeMap[size]} color={color} />
      {message && (
        <Typography variant="body2" color="textSecondary">
          {message}
        </Typography>
      )}
    </Box>
  );
};

export default LoadingSpinner;
