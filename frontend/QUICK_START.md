# React Frontend - Quick Start Guide

## Installation & Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint
```

## Project Structure

```
frontend/src/
├── App.tsx                 # Root component with providers
├── index.tsx              # Entry point
├── hooks/                 # Custom React hooks
├── services/              # API client and utilities
├── contexts/              # Context providers
├── components/            # Reusable components
├── pages/                 # Page components
├── styles/                # Global styles
└── types/                 # TypeScript types
```

## Core Features

### 1. Authentication
```typescript
import { useAuthContext } from '@/contexts';

export const MyComponent = () => {
  const { user, login, logout, isAuthenticated } = useAuthContext();

  return (
    <div>
      {isAuthenticated && <p>Welcome, {user?.email}</p>}
      <button onClick={logout}>Logout</button>
    </div>
  );
};
```

### 2. API Requests
```typescript
import { useFetch } from '@/hooks';

export const DataList = () => {
  const { data, loading, error } = useFetch('/api/data', {
    cacheTime: 5 * 60 * 1000, // 5 minutes
  });

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return <div>{data?.map(item => <div key={item.id}>{item.name}</div>)}</div>;
};
```

### 3. Image Upload
```typescript
import { useImageUpload } from '@/hooks';
import { useAuthContext } from '@/contexts';

export const ImageUpload = () => {
  const { uploadImage, progress } = useImageUpload();
  const { getAccessToken } = useAuthContext();

  const handleUpload = async (file: File) => {
    const token = getAccessToken();
    const result = await uploadImage(file, token);
    console.log('Uploaded:', result);
  };

  return (
    <div>
      <input type="file" onChange={e => e.target.files && handleUpload(e.target.files[0])} />
      {progress && <progress value={progress.percent} max={100} />}
    </div>
  );
};
```

### 4. Notifications
```typescript
import { useNotificationContext } from '@/contexts';

export const MyForm = () => {
  const { success, error } = useNotificationContext();

  const handleSubmit = async (data) => {
    try {
      await saveData(data);
      success('Saved successfully!');
    } catch (err) {
      error('Save failed');
    }
  };

  return <form onSubmit={handleSubmit}>...</form>;
};
```

### 5. Theme Toggle
```typescript
import { useThemeContext } from '@/contexts';

export const ThemeButton = () => {
  const { isDark, toggleTheme } = useThemeContext();

  return (
    <button onClick={toggleTheme}>
      {isDark ? 'Light Mode' : 'Dark Mode'}
    </button>
  );
};
```

## Environment Variables

Create `.env.local`:
```
REACT_APP_API_URL=http://localhost:3000/api
REACT_APP_WS_URL=ws://localhost:3000
```

Production:
```
REACT_APP_API_URL=https://api.example.com
REACT_APP_WS_URL=wss://api.example.com
```

## API Integration

All API calls go through `apiClient` which:
- Automatically injects JWT tokens
- Handles 401 errors with token refresh
- Retries failed requests
- Normalizes error responses

### Direct API Calls
```typescript
import { apiClient } from '@/services/apiClient';

// GET
const data = await apiClient.get('/api/resource');

// POST
const result = await apiClient.post('/api/resource', { name: 'John' });

// PUT
await apiClient.put('/api/resource/1', { name: 'Jane' });

// DELETE
await apiClient.delete('/api/resource/1');

// File Upload
await apiClient.uploadFile('/api/images/upload', file, { userId: 123 });
```

## Debugging

### Check Authentication
```typescript
// In browser console
localStorage.getItem('accessToken')
localStorage.getItem('refreshToken')
```

### Monitor Network Requests
1. Open DevTools (F12)
2. Go to Network tab
3. Filter by "Fetch/XHR"
4. Check Authorization headers

### Check Context Values
1. Install React DevTools Chrome Extension
2. Click on Components tab
3. Search for "Auth" or "Theme"
4. View context values

## Common Tasks

### Add a New Hook
1. Create file in `src/hooks/useMyHook.ts`
2. Implement your hook
3. Export from `src/hooks/index.ts`
4. Use in components: `import { useMyHook } from '@/hooks';`

### Add a New Page
1. Create file in `src/pages/MyPage.tsx`
2. Implement page component
3. Add route in `src/App.tsx`:
```typescript
const MyPage = lazy(() => import('@/pages/MyPage'));

// In Routes:
<Route path="/my-page" element={<MyPage />} />
```

### Add API Request
```typescript
const { data, loading } = useFetch('/api/my-data', {
  method: 'GET',
  cacheTime: 5 * 60 * 1000,
});
```

### Handle Errors
Use ErrorBoundary for component errors and try-catch for async:
```typescript
try {
  await apiClient.post('/api/data', data);
} catch (error) {
  if (error.status === 401) {
    // Unauthorized
  } else if (error.status === 404) {
    // Not found
  }
}
```

## Performance Tips

1. **Lazy Load Pages**
```typescript
const Dashboard = lazy(() => import('@/pages/Dashboard'));
```

2. **Memoize Components**
```typescript
export const MyComponent = memo(({ data }) => {
  return <div>{data}</div>;
});
```

3. **Use useCallback**
```typescript
const handleClick = useCallback(() => {
  // handler
}, [dependencies]);
```

4. **Cache API Responses**
```typescript
const { data } = useFetch('/api/data', {
  cacheTime: 10 * 60 * 1000, // 10 minutes
});
```

## Testing

### Run Tests
```bash
npm test
```

### Write a Test
```typescript
import { render, screen } from '@testing-library/react';
import { MyComponent } from '@/components/MyComponent';

test('renders component', () => {
  render(<MyComponent />);
  expect(screen.getByText(/hello/i)).toBeInTheDocument();
});
```

## Deployment

### Build for Production
```bash
npm run build
```

### Output
- Production build: `frontend/dist/`
- Optimized bundle
- Ready for hosting

### Deploy to Vercel/Netlify
```bash
# Vercel
npm i -g vercel
vercel

# Netlify
npm i -g netlify-cli
netlify deploy
```

## Troubleshooting

### API Requests Failing
- Check network tab in DevTools
- Verify API URL in `.env.local`
- Check Bearer token in headers
- Look for CORS errors

### Authentication Not Working
- Check localStorage for tokens
- Verify token refresh endpoint
- Check for 401 responses
- Look in browser console for errors

### Styles Not Applied
- Check if component is wrapped by ThemeProvider
- Verify MUI theme configuration
- Check CSS class names
- Look for conflicting styles

### Memory Leaks
- Check for proper cleanup in useEffect
- Verify components unmount correctly
- Look for unaborted requests
- Check for event listener cleanup

## Resources

- [React Documentation](https://react.dev)
- [React Router Docs](https://reactrouter.com)
- [Material-UI Docs](https://mui.com)
- [Axios Docs](https://axios-http.com)
- [TypeScript Docs](https://www.typescriptlang.org)

## Support

For issues or questions:
1. Check the documentation
2. Search GitHub issues
3. Contact the development team
4. Open an issue in the repository

---

**Last Updated:** 2024
**Status:** ✅ Production Ready
