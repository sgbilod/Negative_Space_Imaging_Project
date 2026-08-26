/**
 * Error Boundary Component
 * Catches React component errors and displays fallback UI
 * Prevents entire app from crashing on component errors
 */

import React, { Component, ReactNode, ErrorInfo } from 'react';
import { Box, Typography, Button, Container } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

/**
 * Error boundary state
 */
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error boundary props
 */
interface ErrorBoundaryProps {
  children: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  fallback?: ReactNode;
  showDetails?: boolean;
}

/**
 * Error Boundary Component
 * Catches errors in child components and displays error UI
 *
 * @example
 * <ErrorBoundary fallback={<CustomError />}>
 *   <App />
 * </ErrorBoundary>
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  /**
   * Update state when error is caught
   */
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  /**
   * Log error details
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);

    this.setState({ errorInfo });

    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  /**
   * Reset error state
   */
  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Container maxWidth="sm">
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '100vh',
              textAlign: 'center',
              gap: 2,
            }}
          >
            <ErrorOutlineIcon sx={{ fontSize: 64, color: 'error.main' }} />

            <Typography variant="h4" gutterBottom>
              Oops! Something went wrong
            </Typography>

            <Typography variant="body1" color="textSecondary" gutterBottom>
              We&apos;re sorry for the inconvenience. The application encountered an unexpected
              error.
            </Typography>

            {this.props.showDetails && this.state.error && (
              <Box
                sx={{
                  width: '100%',
                  backgroundColor: 'error.light',
                  borderRadius: 1,
                  padding: 2,
                  marginY: 2,
                  textAlign: 'left',
                  maxHeight: '200px',
                  overflowY: 'auto',
                }}
              >
                <Typography
                  variant="subtitle2"
                  color="error"
                  component="pre"
                  sx={{ whiteSpace: 'pre-wrap' }}
                >
                  {this.state.error.toString()}
                </Typography>
                {this.state.errorInfo && (
                  <Typography
                    variant="caption"
                    color="error"
                    component="pre"
                    sx={{ whiteSpace: 'pre-wrap', marginTop: 1 }}
                  >
                    {this.state.errorInfo.componentStack}
                  </Typography>
                )}
              </Box>
            )}

            <Button variant="contained" onClick={this.handleReset} sx={{ marginTop: 2 }}>
              Try Again
            </Button>
          </Box>
        </Container>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
