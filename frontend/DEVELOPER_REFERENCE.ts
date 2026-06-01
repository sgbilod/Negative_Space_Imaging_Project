// Developer Reference Card
// Quick copy-paste code examples for common tasks

// ============================================================================
// 1. AUTHENTICATION
// ============================================================================

// Get current user
import { useAuthContext } from '@/contexts';

export function MyComponent() {
  const { user, isAuthenticated } = useAuthContext();
  return <div>{isAuthenticated && user?.email}</div>;
}

// Login
const { login, loading, error } = useAuthContext();
await login('user@example.com', 'password');

// Logout
const { logout } = useAuthContext();
await logout();

// Get access token for manual API calls
const { getAccessToken } = useAuthContext();
const token = getAccessToken();

// ============================================================================
// 2. API REQUESTS
// ============================================================================

// Simple GET request
import { useFetch } from '@/hooks';

const { data, loading, error } = useFetch('/api/items');

// GET with options
const { data, loading, error, refetch } = useFetch('/api/items', {
  method: 'GET',
  cacheTime: 5 * 60 * 1000, // 5 minutes
});

// POST request
const { data } = useFetch('/api/items', {
  method: 'POST',
  body: { name: 'New Item' },
});

// Manual API call with error handling
import { apiClient } from '@/services/apiClient';

try {
  const result = await apiClient.get('/api/items');
  console.log(result);
} catch (error) {
  if (error.status === 401) {
    // Unauthorized - should redirect to login
  } else if (error.status === 404) {
    // Not found
  } else {
    console.error('Error:', error.message);
  }
}

// POST with data
const newItem = await apiClient.post('/api/items', {
  name: 'Item Name',
  description: 'Description',
});

// PUT to update
await apiClient.put('/api/items/1', {
  name: 'Updated Name',
});

// DELETE
await apiClient.delete('/api/items/1');

// ============================================================================
// 3. FILE UPLOAD
// ============================================================================

import { useImageUpload } from '@/hooks';
import { useAuthContext } from '@/contexts';

export function ImageUploader() {
  const { uploadImage, progress, uploading, error } = useImageUpload();
  const { getAccessToken } = useAuthContext();

  const handleUpload = async (file: File) => {
    const token = getAccessToken();
    const result = await uploadImage(file, token);
    console.log('Uploaded:', result);
  };

  return (
    <div>
      {progress && (
        <progress value={progress.percent} max={100} />
      )}
      <input
        type="file"
        accept="image/*"
        onChange={(e) => e.target.files && handleUpload(e.target.files[0])}
        disabled={uploading}
      />
      {error && <div>{error}</div>}
    </div>
  );
}

// ============================================================================
// 4. NOTIFICATIONS
// ============================================================================

import { useNotificationContext } from '@/contexts';

export function MyForm() {
  const { success, error, warning, info } = useNotificationContext();

  const handleSubmit = async (data) => {
    try {
      await saveData(data);
      success('Data saved successfully', 3000); // 3 second duration
    } catch (err) {
      error('Failed to save data');
    }
  };

  const handleDelete = () => {
    warning('Are you sure?', 5000);
  };

  const handleInfo = () => {
    info('This is informational', 3000);
  };

  return (
    <div>
      <button onClick={() => handleSubmit({})}>Save</button>
      <button onClick={handleDelete}>Delete</button>
      <button onClick={handleInfo}>Info</button>
    </div>
  );
}

// With action button
const { addNotification } = useNotificationContext();

addNotification(
  'Undo action?',
  'success',
  5000,
  {
    label: 'Undo',
    onClick: () => undoAction(),
  }
);

// ============================================================================
// 5. THEME MANAGEMENT
// ============================================================================

import { useThemeContext } from '@/contexts';

export function ThemeToggle() {
  const { isDark, toggleTheme, setTheme, mode } = useThemeContext();

  return (
    <div>
      <button onClick={toggleTheme}>
        {isDark ? '☀️ Light' : '🌙 Dark'}
      </button>
      <p>Current theme: {mode}</p>
      <button onClick={() => setTheme('light')}>Force Light</button>
      <button onClick={() => setTheme('dark')}>Force Dark</button>
      <button onClick={() => setTheme('auto')}>Auto (System)</button>
    </div>
  );
}

// ============================================================================
// 6. LOADING STATES
// ============================================================================

import { LoadingSpinner } from '@/components';

// Small loading spinner
<LoadingSpinner size="small" />

// Medium (default)
<LoadingSpinner message="Loading data..." />

// Large full-screen overlay
<LoadingSpinner size="large" fullScreen message="Processing..." />

// Custom color
<LoadingSpinner color="success" message="Uploading..." />

// ============================================================================
// 7. ERROR HANDLING
// ============================================================================

import { ErrorBoundary } from '@/components';

// Wrap components to catch errors
<ErrorBoundary
  onError={(error, errorInfo) => console.error(error)}
  showDetails={true} // Show in dev mode
>
  <YourComponent />
</ErrorBoundary>

// Custom fallback
<ErrorBoundary
  fallback={
    <div>
      <h2>Something went wrong</h2>
      <p>Please refresh the page</p>
    </div>
  }
>
  <YourComponent />
</ErrorBoundary>

// ============================================================================
// 8. FORM HANDLING
// ============================================================================

// Store form data in localStorage
import { useLocalStorage } from '@/hooks';

export function MyForm() {
  const [formData, setFormData] = useLocalStorage('myForm', {
    email: '',
    password: '',
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // formData is persisted in localStorage
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={formData.email}
        onChange={(e) =>
          setFormData({ ...formData, email: e.target.value })
        }
      />
    </form>
  );
}

// ============================================================================
// 9. DEBOUNCED SEARCH
// ============================================================================

import { useDebounce } from '@/hooks';
import { useFetch } from '@/hooks';

export function SearchUsers() {
  const [searchTerm, setSearchTerm] = useState('');
  const debouncedTerm = useDebounce(searchTerm, 500);

  const { data: results } = useFetch('/api/users/search', {
    method: 'POST',
    body: { query: debouncedTerm },
    cacheTime: 0, // Disable cache for search
  });

  return (
    <div>
      <input
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search users..."
      />
      {results && results.map((user) => <div key={user.id}>{user.name}</div>)}
    </div>
  );
}

// ============================================================================
// 10. ANALYSIS RESULTS
// ============================================================================

import { useAnalysisResults } from '@/hooks';
import { useAuthContext } from '@/contexts';

export function AnalysisViewer({ imageId }) {
  const { getAnalysisResults, selectedResult, loading, error } =
    useAnalysisResults();
  const { getAccessToken } = useAuthContext();

  useEffect(() => {
    const fetchResults = async () => {
      const token = getAccessToken();
      const results = await getAnalysisResults(imageId, token);
      console.log('Analysis complete:', results);
    };

    fetchResults();
  }, [imageId]);

  if (loading) return <LoadingSpinner message="Analyzing..." />;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      {selectedResult && (
        <div>
          <p>Status: {selectedResult.status}</p>
          <p>Negative Space Ratio: {selectedResult.results?.negative_space_ratio}%</p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// 11. ASYNC OPERATIONS
// ============================================================================

import { useAsync } from '@/hooks';

export function DataFetcher() {
  const { data, loading, error, execute, reset } = useAsync(
    async () => {
      return await apiClient.get('/api/expensive-operation');
    },
    { immediate: false } // Manual execution
  );

  return (
    <div>
      <button onClick={execute}>Fetch Data</button>
      <button onClick={reset}>Clear</button>
      {loading && <LoadingSpinner />}
      {error && <div>Error: {error}</div>}
      {data && <div>Result: {JSON.stringify(data)}</div>}
    </div>
  );
}

// ============================================================================
// 12. PROTECTED ROUTES
// ============================================================================

import { useAuthContext } from '@/contexts';
import { Navigate } from 'react-router-dom';

export function PrivateRoute({ element }) {
  const { isAuthenticated, loading } = useAuthContext();

  if (loading) return <LoadingSpinner fullScreen />;

  return isAuthenticated ? element : <Navigate to="/login" />;
}

// Usage in App.tsx
<Route
  path="/dashboard"
  element={<PrivateRoute element={<Dashboard />} />}
/>

// ============================================================================
// 13. CONDITIONAL RENDERING WITH AUTH
// ============================================================================

import { useAuthContext } from '@/contexts';

export function AdminPanel() {
  const { user, isAuthenticated } = useAuthContext();

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  if (user?.role !== 'admin') {
    return <Navigate to="/unauthorized" />;
  }

  return <div>Admin panel content</div>;
}

// ============================================================================
// 14. ENVIRONMENT VARIABLES
// ============================================================================

// Access in code
const apiUrl = import.meta.env.VITE_API_URL;

// In .env.local
VITE_API_URL=http://localhost:3000/api
VITE_WS_URL=ws://localhost:3000
VITE_APP_NAME="Negative Space Imaging"

// ============================================================================
// 15. ERROR PATTERNS
// ============================================================================

// Try-catch with typed error
import { apiClient } from '@/services/apiClient';

async function saveData(data) {
  try {
    const result = await apiClient.post('/api/data', data);
    return result;
  } catch (error) {
    // error is instance of ApiError
    if (error.status === 400) {
      throw new Error('Invalid data: ' + error.response?.message);
    } else if (error.status === 401) {
      // Auth context handles this automatically
      throw new Error('Session expired');
    } else if (error.status === 403) {
      throw new Error('Permission denied');
    } else if (error.status === 500) {
      throw new Error('Server error');
    } else {
      throw error;
    }
  }
}

// ============================================================================
// 16. QUICK COMPONENT TEMPLATE
// ============================================================================

import { useState, useEffect } from 'react';
import { useFetch } from '@/hooks';
import { useNotificationContext } from '@/contexts';
import { LoadingSpinner, ErrorBoundary } from '@/components';

interface MyComponentProps {
  id: string;
}

export function MyComponent({ id }: MyComponentProps) {
  const [localState, setLocalState] = useState('');
  const { data, loading, error } = useFetch(`/api/resource/${id}`);
  const { success, error: showError } = useNotificationContext();

  useEffect(() => {
    if (error) {
      showError('Failed to load data');
    }
  }, [error]);

  const handleSubmit = async () => {
    try {
      // Handle submit
      success('Saved successfully');
    } catch (err) {
      showError('Failed to save');
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div>Error: {error}</div>;

  return <ErrorBoundary>{/* Content */}</ErrorBoundary>;
}

// ============================================================================
// 17. DATA TRANSFORMATION
// ============================================================================

// Transform API response
const { data: rawData } = useFetch('/api/items');

const transformedData = useMemo(() => {
  if (!rawData) return null;

  return rawData.map((item) => ({
    ...item,
    displayName: `${item.firstName} ${item.lastName}`,
    isActive: item.status === 'active',
  }));
}, [rawData]);

// ============================================================================
// 18. CACHE MANAGEMENT
// ============================================================================

// Refetch with cache bypass
const { data, refetch } = useFetch('/api/items');

const handleRefresh = () => {
  refetch(); // Gets fresh data from server
};

// Manual cache clear
const { clearCache } = useFetch('/api/items');

clearCache();

// ============================================================================
// 19. WEBSOCKET INTEGRATION
// ============================================================================

import { useWebSocket } from '@/hooks';

export function RealtimeMonitor() {
  const { data, isConnected, error } = useWebSocket('/ws/monitor');

  return (
    <div>
      <div>{isConnected ? '🟢 Connected' : '🔴 Disconnected'}</div>
      {error && <div>Connection error: {error}</div>}
      {data && <div>Latest: {JSON.stringify(data)}</div>}
    </div>
  );
}

// ============================================================================
// 20. CLEANUP AND BEST PRACTICES
// ============================================================================

// Always cleanup in useEffect
useEffect(() => {
  const timer = setTimeout(() => {
    // cleanup after 5 seconds
  }, 5000);

  return () => clearTimeout(timer);
}, [dependencies]);

// Use memo for expensive renders
const ExpensiveComponent = memo(({ data }) => {
  return <div>{/* expensive render */}</div>;
});

// Use callback for event handlers
const handleClick = useCallback(() => {
  // handler
}, [dependencies]);

// Proper dependency arrays
useEffect(() => {
  // effect
}, [id, userId]); // Include all used variables

// ============================================================================
