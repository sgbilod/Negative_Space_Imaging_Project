# Phase 5: Quick Reference Guide

## File Locations

```
frontend/src/
├── router/
│   ├── routes.tsx              ← Route definitions & metadata
│   ├── ProtectedRoute.tsx      ← Route guard component
│   └── index.ts                ← Route exports (create this)
├── store/
│   ├── userStore.ts            ← User authentication state
│   ├── imageStore.ts           ← Image management state
│   ├── analysisStore.ts        ← Analysis results state
│   ├── uiStore.ts              ← UI state (theme, sidebar, etc)
│   └── index.ts                ← Store exports (create this)
├── pages/
│   ├── LoginPage.tsx           ← Existing (Phase 4)
│   ├── RegisterPage.tsx        ← Existing (Phase 4)
│   ├── DashboardPage.tsx       ← Existing (Phase 4)
│   ├── UploadPage.tsx          ← Existing (Phase 4)
│   ├── AnalysisResultsPage.tsx ← Existing (Phase 4)
│   ├── SettingsPage.tsx        ← Existing (Phase 4)
│   ├── NotFoundPage.tsx        ← NEW (404 page)
│   └── ErrorPage.tsx           ← NEW (error page)
├── components/
│   └── navigation/
│       └── Navigation.tsx       ← NEW (app bar + sidebar)
├── contexts/
│   ├── AuthContext.tsx         ← Existing (Phase 3)
│   ├── ThemeContext.tsx        ← Existing (Phase 3)
│   └── NotificationContext.tsx ← Existing (Phase 3)
└── App.tsx                     ← UPDATE with new routing
```

## 1-Minute Integration

### Step 1: Wrap with Providers

```typescript
// App.tsx
<UIStoreProvider>
  <UserStoreProvider>
    <ImageStoreProvider>
      <AnalysisStoreProvider>
        <Router>
          {/* routes here */}
        </Router>
      </AnalysisStoreProvider>
    </ImageStoreProvider>
  </UserStoreProvider>
</UIStoreProvider>
```

### Step 2: Protect Routes

```typescript
{routes.map((route) => (
  route.meta?.access === 'protected' ? (
    <Route
      path={route.path}
      element={<ProtectedRoute>{route.element}</ProtectedRoute>}
    />
  ) : (
    <Route path={route.path} element={route.element} />
  )
))}
```

### Step 3: Use Stores

```typescript
const { state, addImage } = useImageStore();
const { state: ui, setTheme } = useUIStore();
const { state: user } = useUserStore();
const { state: analyses } = useAnalysisStore();
```

## Store Cheat Sheet

### useImageStore()

```typescript
// State
const { images, totalCount, isLoading } = state;

// Actions
addImage(image);
updateImage(image);
removeImage(id);
setImages(images);
setLoading(true / false);
updateProgress(id, percentage);
clearImages();
```

### useUserStore()

```typescript
// State
const { userId, email, name, role, avatar } = state;

// Actions
setUser(userData)
updateProfile({ name, email, ... })
setRole('user' | 'admin')
logout()
```

### useAnalysisStore()

```typescript
// State
const { analyses, currentAnalysis, isLoading } = state;

// Actions
addAnalysis(result);
updateAnalysis(result);
setCurrentAnalysis(result);
setAnalyses(results);
removeAnalysis(id);
setLoading(true / false);
clearAnalyses();
```

### useUIStore()

```typescript
// State
const { sidebarOpen, theme, drawerOpen, modalOpen } = state;

// Actions
toggleSidebar();
setTheme('light' | 'dark');
openDrawer() / closeDrawer();
openModal(content) / closeModal();
reset();
```

## Route Protection Examples

### Protect a Route

```typescript
<Route
  path="/admin"
  element={
    <ProtectedRoute requiredRole="admin">
      <AdminPanel />
    </ProtectedRoute>
  }
/>
```

### Check Access in Component

```typescript
const hasAccess = useRouteAccess('admin');
if (!hasAccess) return <AccessDenied />;
```

### Protect Component with HOC

```typescript
const ProtectedAdmin = withProtection(AdminPanel, 'admin');
<Route path="/admin" element={<ProtectedAdmin />} />
```

## Common Patterns

### Upload File with Progress

```typescript
const { addImage, updateProgress } = useImageStore();

addImage({ id, name, status: 'uploading', progress: 0 });
// ...
updateProgress(id, 50);
// ...
updateProgress(id, 100);
```

### Toggle Theme

```typescript
const { state: { theme }, setTheme } = useUIStore();

onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
```

### Display User Info

```typescript
const { state: { name, role } } = useUserStore();

<Typography>{name} ({role})</Typography>
```

### Show Notification

```typescript
import { useNotification } from '../hooks/useNotification';

const { success, error } = useNotification();

success('Done!');
error('Oops!');
```

## Route Structure

### Public Routes

- `/login` - Login page
- `/register` - Registration page
- `/error` - Error page
- `/*` - 404 Not Found

### Protected Routes (User)

- `/dashboard` - Main dashboard
- `/upload` - Upload images
- `/analysis/:id` - View analysis
- `/settings` - User settings

### Admin Routes (Admin Only)

- `/admin` - Admin dashboard
- `/admin/users` - User management
- `/admin/stats` - System statistics

## TypeScript Types

### UserRole

```typescript
type UserRole = 'user' | 'admin' | 'guest';
```

### ImageStatus

```typescript
type ImageStatus = 'uploading' | 'uploaded' | 'processing' | 'ready' | 'failed';
```

### AnalysisStatus

```typescript
type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed';
```

### RouteAccess

```typescript
type RouteAccess = 'public' | 'protected' | 'admin';
```

## Storage Keys (localStorage)

```typescript
'user-store'; // User state (persisted on login, cleared on logout)
'app-store'; // App theme preference
```

## Testing Store

```typescript
import { renderHook, act } from '@testing-library/react';
import { useImageStore, ImageStoreProvider } from '../store';

const wrapper = ({ children }: any) => (
  <ImageStoreProvider>{children}</ImageStoreProvider>
);

const { result } = renderHook(() => useImageStore(), { wrapper });

act(() => {
  result.current.addImage({ /* ... */ });
});

expect(result.current.state.totalCount).toBe(1);
```

## Debugging

### Check Store State

```typescript
const { state } = useImageStore();
console.log('Images:', state);
```

### Check Route Access

```typescript
import { isAdminRoute, getRouteMeta } from '../router';

console.log(isAdminRoute('/admin')); // true
console.log(getRouteMeta('/dashboard')); // { title: 'Dashboard', ... }
```

### Verify Authentication

```typescript
const { isAuthenticated, user } = useAuth();
console.log('Authenticated:', isAuthenticated);
console.log('User:', user);
```

## Performance Tips

✅ Use individual store hooks (don't import whole store)
✅ Memoize selectors with useCallback
✅ Use React.lazy() for routes (already done)
✅ Keep localStorage state small
✅ Use Suspense boundaries for loading states

## Common Issues & Fixes

| Issue                    | Fix                            |
| ------------------------ | ------------------------------ |
| Context undefined        | Wrap with provider             |
| State not persisting     | Check localStorage permissions |
| Route redirects to login | Verify token in useAuth        |
| Store not updating       | Check action dispatch          |
| Performance slow         | Check store subscriptions      |

---

**Quick Reference Complete** ✅ Ready for integration!
