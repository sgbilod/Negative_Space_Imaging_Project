# Phase 5: Integration Guide

## Quick Start (5 Minutes)

### 1. Create Store Index File

Create `frontend/src/store/index.ts`:

```typescript
export { useUserStore, UserStoreProvider } from './userStore';
export type { UserState, UserAction } from './userStore';

export { useImageStore, ImageStoreProvider } from './imageStore';
export type { ImageItem, ImageState, ImageAction } from './imageStore';

export { useAnalysisStore, AnalysisStoreProvider } from './analysisStore';
export type { AnalysisResult, AnalysisState, AnalysisAction } from './analysisStore';

export { useUIStore, UIStoreProvider } from './uiStore';
export type { UIState, UIAction } from './uiStore';
```

### 2. Create Router Index File

Create `frontend/src/router/index.ts`:

```typescript
export { routes, routeMetadata } from './routes';
export type { RouteMeta, ExtendedRouteObject } from './routes';
export {
  isPublicRoute,
  isProtectedRoute,
  isAdminRoute,
  getRouteMeta,
  getNavigableRoutes,
  getBreadcrumbs,
} from './routes';

export { ProtectedRoute, withProtection, useRouteAccess } from './ProtectedRoute';
export type { UserRole } from './ProtectedRoute';
```

### 3. Update App.tsx

Replace your current App.tsx with:

```typescript
import { Suspense, ReactNode } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import { AuthProvider } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { ThemeProvider } from './contexts/ThemeContext';

import { UIStoreProvider, UserStoreProvider, ImageStoreProvider, AnalysisStoreProvider } from './store';
import { routes } from './router';
import { ProtectedRoute } from './router/ProtectedRoute';
import Navigation from './components/navigation/Navigation';

interface LoadingSpinnerProps {
  fullHeight?: boolean;
}

function LoadingSpinner({ fullHeight = true }: LoadingSpinnerProps): ReactNode {
  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight={fullHeight ? '100vh' : '400px'}
    >
      <CircularProgress />
    </Box>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <NotificationProvider>
          <UIStoreProvider>
            <UserStoreProvider>
              <ImageStoreProvider>
                <AnalysisStoreProvider>
                  <Router>
                    <Navigation />
                    <Box sx={{ pl: { xs: 0, md: '250px' }, pt: '112px' }}>
                      <Suspense fallback={<LoadingSpinner />}>
                        <Routes>
                          {routes.map((route) => {
                            const meta = route.meta;

                            if (meta?.access === 'protected' && meta.access !== 'admin') {
                              return (
                                <Route
                                  key={route.path}
                                  path={route.path}
                                  element={
                                    <ProtectedRoute requiredRole="user">
                                      {route.element}
                                    </ProtectedRoute>
                                  }
                                />
                              );
                            }

                            if (meta?.access === 'admin') {
                              return (
                                <Route
                                  key={route.path}
                                  path={route.path}
                                  element={
                                    <ProtectedRoute requiredRole="admin">
                                      {route.element}
                                    </ProtectedRoute>
                                  }
                                />
                              );
                            }

                            return (
                              <Route
                                key={route.path}
                                path={route.path}
                                element={route.element}
                              />
                            );
                          })}
                        </Routes>
                      </Suspense>
                    </Box>
                  </Router>
                </AnalysisStoreProvider>
              </ImageStoreProvider>
            </UserStoreProvider>
          </UIStoreProvider>
        </NotificationProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
```

### 4. Update main.tsx

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

## Detailed Integration Guide

### Using Stores in Components

#### Example 1: Upload Image with Progress

```typescript
import { useImageStore } from '../store';
import { useAppNotification } from '../store/useAppStore';

export default function UploadComponent() {
  const { addImage, updateProgress, updateImage } = useImageStore();
  const { show: showNotification } = useAppNotification();

  const handleFileUpload = async (file: File) => {
    const imageId = crypto.randomUUID();

    // Add image to store
    addImage({
      id: imageId,
      name: file.name,
      uploadDate: new Date().toISOString(),
      size: file.size,
      format: file.type,
      status: 'uploading',
      progress: 0,
    });

    try {
      // Simulate upload with progress
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const percentComplete = (e.loaded / e.total) * 100;
          updateProgress(imageId, percentComplete);
        }
      });

      xhr.addEventListener('load', () => {
        // Update image status to uploaded
        updateImage({
          id: imageId,
          name: file.name,
          uploadDate: new Date().toISOString(),
          size: file.size,
          format: file.type,
          status: 'uploaded',
          progress: 100,
        });

        showNotification(`${file.name} uploaded successfully!`, 'success');
      });

      xhr.addEventListener('error', () => {
        showNotification('Upload failed', 'error');
        updateImage({
          id: imageId,
          name: file.name,
          uploadDate: new Date().toISOString(),
          size: file.size,
          format: file.type,
          status: 'failed',
        });
      });

      xhr.open('POST', '/api/images/upload');
      xhr.send(formData);
    } catch (error) {
      showNotification('Upload error: ' + (error as Error).message, 'error');
    }
  };

  return (
    <input
      type="file"
      onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
    />
  );
}
```

#### Example 2: Theme Switching

```typescript
import { useUIStore } from '../store';
import { IconButton } from '@mui/material';
import { Brightness4 as DarkIcon, Brightness7 as LightIcon } from '@mui/icons-material';

export default function ThemeSwitcher() {
  const { state: { theme }, setTheme } = useUIStore();

  return (
    <IconButton onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
      {theme === 'light' ? <DarkIcon /> : <LightIcon />}
    </IconButton>
  );
}
```

#### Example 3: Display Analysis Results

```typescript
import { useAnalysisStore } from '../store';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';

export default function AnalysisTable() {
  const { state: { analyses } } = useAnalysisStore();

  return (
    <TableContainer>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>ID</TableCell>
            <TableCell>Image ID</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Confidence</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {analyses.map((analysis) => (
            <TableRow key={analysis.id}>
              <TableCell>{analysis.id}</TableCell>
              <TableCell>{analysis.imageId}</TableCell>
              <TableCell>{analysis.status}</TableCell>
              <TableCell>{(analysis.confidence * 100).toFixed(1)}%</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
```

### Route Protection Examples

#### Example 1: Protected Component

```typescript
import { useRouteAccess } from '../router';
import { Box, Alert } from '@mui/material';

export default function AdminStats() {
  const hasAccess = useRouteAccess('admin');

  if (!hasAccess) {
    return <Alert severity="error">You do not have access to this page.</Alert>;
  }

  return <Box>{/* Admin content */}</Box>;
}
```

#### Example 2: HOC Pattern

```typescript
import { withProtection } from '../router';

function AdminPanel() {
  return <div>Admin content</div>;
}

export default withProtection(AdminPanel, 'admin');
```

## Architecture Patterns

### Pattern 1: Store + Component Integration

```
Component
  ↓ (useImageStore)
Store Context
  ↓ (dispatch)
Reducer
  ↓ (new state)
Component Re-renders
```

### Pattern 2: Route Protection Flow

```
Navigate to /admin
  ↓
Router matches route
  ↓
ProtectedRoute checks auth
  ↓
ProtectedRoute checks role
  ↓
Access granted → Render
         OR
Access denied → Redirect
```

### Pattern 3: localStorage Persistence

```
App Loads
  ↓
Read localStorage
  ↓
Hydrate initial state
  ↓
Subscribe to state changes
  ↓
Write to localStorage on update
  ↓
User reloads
  ↓
localStorage restored
```

## Testing Store Actions

### Unit Test Example

```typescript
import { renderHook, act } from '@testing-library/react';
import { useImageStore, ImageStoreProvider } from '../store/imageStore';

describe('useImageStore', () => {
  it('should add image to store', () => {
    const wrapper = ({ children }: any) => (
      <ImageStoreProvider>{children}</ImageStoreProvider>
    );

    const { result } = renderHook(() => useImageStore(), { wrapper });

    const newImage = {
      id: '1',
      name: 'test.jpg',
      uploadDate: new Date().toISOString(),
      size: 1000,
      format: 'image/jpeg',
      status: 'uploaded' as const,
    };

    act(() => {
      result.current.addImage(newImage);
    });

    expect(result.current.state.images).toContainEqual(newImage);
    expect(result.current.state.totalCount).toBe(1);
  });
});
```

## Troubleshooting

### Issue 1: Context is undefined

**Solution:** Make sure component is wrapped in provider:

```typescript
<UIStoreProvider>
  <YourComponent />
</UIStoreProvider>
```

### Issue 2: State not persisting

**Solution:** Check localStorage is enabled in browser. Use try-catch in provider:

```typescript
useEffect(() => {
  try {
    localStorage.setItem('key', JSON.stringify(state));
  } catch {
    // localStorage disabled
  }
}, [state]);
```

### Issue 3: Protected route redirects to login immediately

**Solution:** Ensure useAuth hook properly checks token:

```typescript
const { isAuthenticated } = useAuth();
// Check if token exists in localStorage
```

## Performance Tips

1. **Memoize Selectors:** Use separate hooks for each store slice
2. **Lazy Load Routes:** Already implemented with React.lazy()
3. **Suspense Boundaries:** Wrap routes in Suspense
4. **Callback Dependencies:** All dispatch functions are memoized
5. **localStorage Limits:** Keep state serialized and small

## Next Steps

1. Test integration in local development
2. Verify all routes protect correctly
3. Test localStorage persistence
4. Build and test production bundle
5. Deploy to staging environment

---

**Phase 5 Integration Complete** ✅
