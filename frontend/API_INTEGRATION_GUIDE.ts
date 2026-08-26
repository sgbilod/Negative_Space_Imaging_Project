// API Integration Guide - Negative Space Imaging Project
// Complete reference for connecting frontend to backend APIs

/**
 * AUTHENTICATION FLOW
 * ==================
 *
 * 1. User logs in via POST /api/auth/login
 * 2. Backend returns accessToken and refreshToken
 * 3. Frontend stores tokens in localStorage
 * 4. accessToken included in Authorization header for all requests
 * 5. On 401, frontend automatically refreshes token
 * 6. Retry original request with new token
 */

import { useAuthContext } from '@/contexts';
import { apiClient } from '@/services/apiClient';

// LOGIN EXAMPLE
export const LoginExample = () => {
  const { login, isAuthenticated, error, loading } = useAuthContext();

  const handleLogin = async (email: string, password: string) => {
    try {
      await login(email, password);
      // Redirects to dashboard on success
    } catch (err) {
      console.error('Login failed:', err);
    }
  };

  return (
    <div>
      {error && <div className="error">{error}</div>}
      <button onClick={() => handleLogin('user@example.com', 'password')}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </div>
  );
};

/**
 * IMAGE UPLOAD FLOW
 * ================
 *
 * Endpoint: POST /api/images/upload
 * Method: multipart/form-data
 * Headers: Authorization: Bearer <accessToken>
 *
 * Request Body:
 * - file: File object
 * - metadata: Optional JSON metadata
 *
 * Response:
 * {
 *   id: string,
 *   filename: string,
 *   url: string,
 *   size: number,
 *   mimeType: string,
 *   uploadedAt: string
 * }
 */

import { useImageUpload } from '@/hooks';

export const ImageUploadExample = () => {
  const { uploadImage, progress, uploading, error } = useImageUpload();
  const { getAccessToken } = useAuthContext();

  const handleFileSelect = async (file: File) => {
    try {
      const token = getAccessToken();
      if (!token) {
        console.error('Not authenticated');
        return;
      }

      const result = await uploadImage(file, token);
      console.log('Upload successful:', result);
      // result.id can be used for analysis
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  return (
    <div>
      {progress && (
        <progress value={progress.percent} max={100}>
          {Math.round(progress.percent)}%
        </progress>
      )}
      {uploading && <p>Uploading...</p>}
      {error && <div className="error">{error}</div>}
      <input
        type="file"
        accept="image/*"
        onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
        disabled={uploading}
      />
    </div>
  );
};

/**
 * IMAGE ANALYSIS FLOW
 * ==================
 *
 * Step 1: Get available analysis modes
 * Endpoint: GET /api/analysis/modes
 * Response: { modes: ['negative_space', 'composition', 'object_detection'] }
 *
 * Step 2: Start analysis job
 * Endpoint: POST /api/analysis/analyze
 * Request: { imageId: string, mode: string }
 * Response: { jobId: string, status: 'processing' }
 *
 * Step 3: Poll for results
 * Endpoint: GET /api/analysis/results/:imageId
 * Response:
 * {
 *   imageId: string,
 *   status: 'completed' | 'processing' | 'failed',
 *   results: {
 *     negative_space_ratio: number,
 *     composition_score: number,
 *     dominant_colors: string[],
 *     objects: { name: string, confidence: number }[]
 *   }
 * }
 */

import { useAnalysisResults } from '@/hooks';
import { useNotificationContext } from '@/contexts';

export const ImageAnalysisExample = () => {
  const { getAnalysisResults, selectedResult, loading, error } = useAnalysisResults();
  const { success, error: showError } = useNotificationContext();
  const { getAccessToken } = useAuthContext();

  const analyzeImage = async (imageId: string) => {
    try {
      const token = getAccessToken();
      if (!token) return;

      // Poll for results (backend handles job creation)
      const results = await getAnalysisResults(imageId, token);
      success('Analysis complete');
      return results;
    } catch (err) {
      showError('Analysis failed');
    }
  };

  return (
    <div>
      {loading && <p>Analyzing image...</p>}
      {selectedResult && (
        <div>
          <h3>Analysis Results</h3>
          <p>Negative Space Ratio: {selectedResult.results?.negative_space_ratio}%</p>
          <p>Composition Score: {selectedResult.results?.composition_score}/100</p>
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
};

/**
 * USER MANAGEMENT FLOW
 * ===================
 *
 * Get Current User:
 * GET /api/users/me
 * Response: { id, email, firstName, lastName, role, createdAt }
 *
 * Update User Profile:
 * PUT /api/users/me
 * Request: { firstName?, lastName?, preferences? }
 *
 * Get All Users (Admin):
 * GET /api/users?page=1&limit=10
 * Response: { users: User[], total, pages }
 *
 * Create User (Admin):
 * POST /api/users
 * Request: { email, password, firstName, lastName, role }
 *
 * Update User (Admin):
 * PUT /api/users/:id
 * Request: { firstName?, lastName?, role? }
 *
 * Delete User (Admin):
 * DELETE /api/users/:id
 */

import { useFetch } from '@/hooks';

export const UserManagementExample = () => {
  const { data: users, loading, refetch } = useFetch('/users', {
    method: 'GET',
  });

  const createUser = async (userData: any) => {
    try {
      const newUser = await apiClient.post('/users', userData);
      refetch();
      return newUser;
    } catch (err) {
      console.error('Failed to create user:', err);
    }
  };

  return (
    <div>
      {loading && <p>Loading users...</p>}
      {users?.map((user: any) => (
        <div key={user.id}>{user.email}</div>
      ))}
    </div>
  );
};

/**
 * SECURITY MONITORING FLOW
 * =======================
 *
 * Get Security Metrics:
 * GET /api/security/metrics
 * Response: { failedLogins, suspiciousActivity, lastScan, threatLevel }
 *
 * Get Security Events:
 * GET /api/security/events?limit=50
 * Response: { events: SecurityEvent[] }
 *
 * Audit Logs:
 * GET /api/audit?resource=users&action=create&limit=50
 * Response: { logs: AuditLog[], total }
 */

export const SecurityMonitoringExample = () => {
  const { data: metrics, loading } = useFetch('/security/metrics', {
    cacheTime: 60000, // Cache for 1 minute
  });

  return (
    <div>
      {loading && <p>Loading security metrics...</p>}
      {metrics && (
        <div>
          <p>Threat Level: {metrics.threatLevel}</p>
          <p>Failed Logins: {metrics.failedLogins}</p>
        </div>
      )}
    </div>
  );
};

/**
 * ERROR HANDLING PATTERN
 * ====================
 *
 * ApiClient automatically:
 * - Catches and normalizes errors
 * - Refreshes token on 401
 * - Retries failed requests (max 3 times)
 * - Dispatches 'unauthorized' event on auth failure
 */

export const ErrorHandlingExample = () => {
  const handleApiCall = async () => {
    try {
      const data = await apiClient.get('/protected-resource');
      // Success
    } catch (error) {
      if (error.status === 404) {
        console.error('Resource not found');
      } else if (error.status === 403) {
        console.error('Access denied');
      } else if (error.status === 401) {
        console.error('Unauthorized');
        // App redirects to login via context
      } else {
        console.error('Unexpected error:', error.message);
      }
    }
  };

  return <button onClick={handleApiCall}>Make API Call</button>;
};

/**
 * REAL-TIME UPDATES WITH WEBSOCKET
 * ================================
 *
 * useWebSocket hook connects to WebSocket server
 * Provides real-time analysis updates, security alerts, etc.
 */

import { useWebSocket } from '@/hooks';

export const RealtimeUpdatesExample = () => {
  const { data, isConnected, error } = useWebSocket('/ws/analysis');

  return (
    <div>
      {!isConnected && <p>Reconnecting...</p>}
      {error && <p>Connection error: {error}</p>}
      {data && <p>Update: {JSON.stringify(data)}</p>}
    </div>
  );
};

/**
 * COMPLETE FLOW EXAMPLE: IMAGE UPLOAD → ANALYSIS → DISPLAY
 * ==========================================================
 */

export const CompleteWorkflowExample = () => {
  const { uploadImage } = useImageUpload();
  const { getAnalysisResults } = useAnalysisResults();
  const { success, error: showError } = useNotificationContext();
  const { getAccessToken } = useAuthContext();

  const handleCompleteWorkflow = async (file: File) => {
    try {
      const token = getAccessToken();
      if (!token) throw new Error('Not authenticated');

      // Step 1: Upload image
      success('Uploading image...');
      const imageResult = await uploadImage(file, token);
      success(`Image uploaded: ${imageResult.filename}`);

      // Step 2: Analyze image
      success('Analyzing image...');
      const analysisResults = await getAnalysisResults(imageResult.id, token);
      success('Analysis complete');

      return {
        image: imageResult,
        analysis: analysisResults,
      };
    } catch (err) {
      showError(`Workflow failed: ${err.message}`);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => e.target.files && handleCompleteWorkflow(e.target.files[0])}
      />
    </div>
  );
};

/**
 * API ENDPOINTS SUMMARY
 * ===================
 *
 * Authentication:
 *   POST   /auth/login              - User login
 *   POST   /auth/register           - User registration
 *   POST   /auth/logout             - User logout
 *   POST   /auth/refresh            - Token refresh
 *
 * Images:
 *   POST   /images/upload           - Upload image
 *   GET    /images                  - List images
 *   GET    /images/:id              - Get image details
 *   DELETE /images/:id              - Delete image
 *
 * Analysis:
 *   GET    /analysis/modes          - Available modes
 *   POST   /analysis/analyze        - Start analysis
 *   GET    /analysis/results/:id    - Get results
 *   GET    /analysis/status/:jobId  - Job status
 *
 * Users:
 *   GET    /users/me                - Current user
 *   PUT    /users/me                - Update profile
 *   GET    /users                   - List users (admin)
 *   POST   /users                   - Create user (admin)
 *   PUT    /users/:id               - Update user (admin)
 *   DELETE /users/:id               - Delete user (admin)
 *
 * Security:
 *   GET    /security/metrics        - Security metrics
 *   GET    /security/events         - Security events
 *   GET    /audit                   - Audit logs
 *
 * Health:
 *   GET    /health                  - Health check
 *   GET    /health/ready            - Readiness probe
 */

/**
 * CONFIGURATION
 * =============
 */

// Environment variables needed in .env.local:
// REACT_APP_API_URL=http://localhost:3000/api
// REACT_APP_WS_URL=ws://localhost:3000/ws

// Or with production URLs:
// REACT_APP_API_URL=https://api.example.com
// REACT_APP_WS_URL=wss://api.example.com

/**
 * DEBUGGING TIPS
 * ==============
 *
 * 1. Check localStorage for tokens:
 *    console.log(localStorage.getItem('accessToken'))
 *
 * 2. Monitor API calls:
 *    Open DevTools → Network tab
 *    Filter by "Fetch/XHR"
 *    Check Authorization headers
 *
 * 3. Check context values:
 *    Use React DevTools Extension
 *    Inspect context providers
 *
 * 4. Enable request logging in apiClient:
 *    Uncomment console.log statements
 *
 * 5. Watch for token expiration:
 *    Tokens expire in backend (check JWT payload)
 *    Frontend automatically refreshes
 */

/**
 * TESTING API INTEGRATION
 * ======================
 *
 * Unit test with mock data:
 *
 * import { renderHook, act } from '@testing-library/react';
 * import { useFetch } from '@/hooks';
 *
 * test('useFetch loads data', async () => {
 *   const { result } = renderHook(() => useFetch('/test'));
 *
 *   await act(async () => {
 *     await result.current.performFetch();
 *   });
 *
 *   expect(result.current.data).toBeDefined();
 * });
 */

export default {};
