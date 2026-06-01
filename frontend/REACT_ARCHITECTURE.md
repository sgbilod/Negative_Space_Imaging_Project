```markdown
# Negative Space Imaging Project - React Architecture

## Overview

Complete production-grade React frontend setup for the Negative Space Imaging Project with modern best practices, scalable component architecture, and state management.

## ✅ Deliverables

### 1. **Core Hooks** (7 Custom Hooks)

#### `useAuth.ts` (280+ lines)
- **Purpose:** Manages authentication state and JWT token handling
- **Features:**
  - JWT token validation and refresh
  - User registration and login
  - Token persistence in localStorage
  - Automatic token expiration detection
  - Session management
- **Methods:**
  - `login(email, password)` - User authentication
  - `register(email, password, firstName, lastName)` - New user registration
  - `logout()` - Clear auth data
  - `refreshAccessToken()` - Refresh JWT tokens
  - `getAccessToken()` - Retrieve current token
- **Returns:** User data, tokens, loading state, error state

#### `useImageUpload.ts` (220+ lines)
- **Purpose:** File upload with progress tracking and validation
- **Features:**
  - File type validation (JPEG, PNG, GIF, WebP, TIFF, RAW)
  - File size validation (max 100MB)
  - Upload progress tracking
  - Multipart form data handling
  - Sequential multi-file uploads
- **Methods:**
  - `uploadImage(file, accessToken)` - Upload single image
  - `uploadMultiple(files, accessToken)` - Upload multiple files
- **Returns:** Image metadata, progress percentage, error handling

#### `useAnalysisResults.ts` (200+ lines)
- **Purpose:** Fetch and cache analysis results with polling
- **Features:**
  - Analysis result caching (5-minute TTL)
  - Polling for in-progress analysis
  - Result selection and filtering
  - Cache invalidation
- **Methods:**
  - `getAnalysisResults(imageId, token)` - Fetch image analysis
  - `selectResult(result)` - Select for display
  - `clearCache()` - Clear all cached results

#### `useFetch.ts` (220+ lines)
- **Purpose:** Generic data fetching with caching and retry logic
- **Features:**
  - Automatic retry on network failures
  - Request timeout handling
  - Global response caching
  - Cache time-to-live (TTL)
  - Abort controller for cleanup
- **Generic Type:** `useFetch<T>(url, options)`
- **Options:** method, headers, body, cacheTime, retry, timeout

#### `useLocalStorage.ts` (100+ lines)
- **Purpose:** Persistent local state with automatic serialization
- **Features:**
  - JSON serialization/deserialization
  - Automatic localStorage sync
  - Error handling for quota exceeded
  - Functional updates support
- **Methods:**
  - `setValue(value)` - Update storage
  - `removeValue()` - Delete specific key
  - `clear()` - Clear all localStorage

#### `useAsync.ts` (110+ lines)
- **Purpose:** Async operation state management
- **Features:**
  - Loading, error, and result states
  - Memory leak prevention
  - Immediate or deferred execution
  - Manual data setting
- **Methods:**
  - `execute()` - Run async function
  - `reset()` - Clear all state
  - `setData(data)` - Manually set result

#### `useDebounce.ts` (35 lines)
- **Purpose:** Debounce value changes
- **Features:**
  - Configurable delay
  - Perfect for search inputs and form fields
- **Returns:** Debounced value

### 2. **API Client Service** (250+ lines)

**File:** `src/services/apiClient.ts`

#### ApiClient Class
- **Features:**
  - Axios-based HTTP client
  - JWT token injection in all requests
  - Automatic token refresh on 401
  - Request/response interceptors
  - Error normalization
  - Global error handling
  - File upload support with FormData

#### Key Methods
```typescript
// GET request
await apiClient.get<T>(url, config)

// POST request
await apiClient.post<T>(url, data, config)

// PUT request
await apiClient.put<T>(url, data, config)

// PATCH request
await apiClient.patch<T>(url, data, config)

// DELETE request
await apiClient.delete<T>(url, config)

// File upload
await apiClient.uploadFile<T>(url, file, additionalData)
```

#### Interceptors
- **Request:** Automatically injects Bearer token
- **Response:** Handles 401 with token refresh, normalizes errors

#### Error Handling
- Custom `ApiError` class with status and data
- Automatic retry on network failures
- Token refresh on 401 Unauthorized
- Graceful degradation

### 3. **Context Providers** (3 Providers + Custom Hooks)

#### AuthContext.tsx (140+ lines)
- **Provides:** User authentication state globally
- **Context Shape:**
  ```typescript
  interface AuthContextType {
    user: User | null;
    isAuthenticated: boolean;
    loading: boolean;
    error: string | null;
    login: (email, password) => Promise<void>;
    register: (email, password, firstName, lastName) => Promise<void>;
    logout: () => Promise<void>;
    refreshToken: () => Promise<boolean>;
    getAccessToken: () => string | null;
    clearError: () => void;
  }
  ```
- **Hook:** `useAuthContext()` - Access auth state from any component

#### ThemeContext.tsx (180+ lines)
- **Provides:** Global theme management (light/dark/auto)
- **Features:**
  - System theme detection
  - DOM class and color-scheme updates
  - Persistent theme preference
  - Media query listener
- **Context Shape:**
  ```typescript
  interface ThemeContextType {
    mode: ThemeMode; // 'light' | 'dark' | 'auto'
    isDark: boolean;
    toggleTheme: () => void;
    setTheme: (theme: ThemeMode) => void;
  }
  ```
- **Hook:** `useThemeContext()` - Access theme controls

#### NotificationContext.tsx (220+ lines)
- **Provides:** Global toast notification system
- **Features:**
  - Auto-dismiss with configurable duration
  - Notification queue management
  - Multiple severity levels
  - Optional action buttons
  - Max notification limit
- **Context Shape:**
  ```typescript
  interface NotificationContextType {
    notifications: Notification[];
    addNotification: (message, severity, duration, action) => string;
    removeNotification: (id) => void;
    clearNotifications: () => void;
    success: (message, duration) => string;
    error: (message, duration) => string;
    warning: (message, duration) => string;
    info: (message, duration) => string;
  }
  ```
- **Hook:** `useNotificationContext()` - Access notifications

### 4. **Utility Components** (3 Components)

#### LoadingSpinner.tsx (75+ lines)
- **Purpose:** Display loading indicators
- **Props:**
  - `size`: 'small' | 'medium' | 'large'
  - `message`: Display text
  - `fullScreen`: Cover entire viewport
  - `color`: MUI color variant
  - `className`: CSS classes
- **Features:**
  - Circular progress indicator
  - Optional message
  - Full-screen overlay mode
  - Customizable size and color

#### ErrorBoundary.tsx (140+ lines)
- **Purpose:** Catch and handle component errors
- **Features:**
  - Error state capture
  - Fallback UI rendering
  - Error details display (dev mode)
  - Recovery mechanism
  - Error logging callback
  - Component stack trace

#### Modal Dialog (Ready)
- **Purpose:** Reusable modal component
- **Features:** Customizable title, content, actions

### 5. **Root App Component** (Updated)

**File:** `src/App.tsx` (85+ lines)

#### Architecture
```
ErrorBoundary (Global error catching)
  └─ CustomThemeProvider (Theme management)
    └─ MUI ThemeProvider (Material-UI theme)
      └─ CssBaseline (Reset styles)
      └─ AuthProvider (Authentication)
        └─ NotificationProvider (Notifications)
          └─ Router (React Router)
            └─ Routes (Page routing)
```

#### Features
- Error boundary wrapping
- Global provider composition
- Code splitting with lazy loading
- Route protection with PrivateRoute
- Fallback loading UI

## 📁 File Structure

```
frontend/src/
├── App.tsx                          # Root component with providers
├── index.tsx                        # Entry point
├── hooks/                           # Custom hooks
│   ├── index.ts                    # Hook exports
│   ├── useAuth.ts                  # Authentication
│   ├── useImageUpload.ts           # File uploads
│   ├── useAnalysisResults.ts       # Analysis fetching
│   ├── useFetch.ts                 # Generic data fetching
│   ├── useLocalStorage.ts          # Persistent state
│   ├── useAsync.ts                 # Async operations
│   ├── useDebounce.ts              # Value debouncing
│   └── useWebSocket.ts             # WebSocket (existing)
├── services/                        # API and utilities
│   ├── apiClient.ts                # Axios HTTP client
│   └── [other services]
├── contexts/                        # Context providers
│   ├── index.ts                    # Context exports
│   ├── AuthContext.tsx             # Auth provider
│   ├── ThemeContext.tsx            # Theme provider
│   └── NotificationContext.tsx     # Notification provider
├── components/
│   ├── common/                      # Shared components
│   │   ├── LoadingSpinner.tsx      # Loading indicator
│   │   ├── ErrorBoundary.tsx       # Error catching
│   │   ├── Modal.tsx               # Modal dialog
│   │   └── [other utilities]
│   ├── Layout.tsx                  # Main layout
│   ├── PrivateRoute.tsx            # Protected routes
│   └── [feature components]
├── pages/                           # Page components
│   ├── Login.tsx
│   ├── Dashboard.tsx
│   ├── ImageProcessing.tsx
│   ├── SecurityMonitor.tsx
│   ├── AuditLogs.tsx
│   └── UserManagement.tsx
└── [other directories]
```

## 🚀 Usage Examples

### Authentication
```typescript
import { useAuthContext } from '@/contexts';

export const LoginPage = () => {
  const { login, loading, error } = useAuthContext();

  const handleLogin = async (email, password) => {
    try {
      await login(email, password);
      // Redirect to dashboard
    } catch (err) {
      // Handle error
    }
  };

  return (
    // Login form
  );
};
```

### File Upload
```typescript
import { useImageUpload } from '@/hooks';
import { useAuthContext } from '@/contexts';

export const ImageUploader = () => {
  const { uploadImage, progress, uploading } = useImageUpload();
  const { getAccessToken } = useAuthContext();

  const handleUpload = async (file) => {
    const token = getAccessToken();
    const result = await uploadImage(file, token);
    console.log('Uploaded:', result);
  };

  return (
    <div>
      {progress && <progress value={progress.percent} max={100} />}
    </div>
  );
};
```

### Data Fetching
```typescript
import { useFetch } from '@/hooks';

export const ImageList = () => {
  const { data, loading, error, refetch } = useFetch('/images', {
    headers: { 'Authorization': `Bearer ${token}` },
    cacheTime: 5 * 60 * 1000,
  });

  if (loading) return <LoadingSpinner />;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      {data?.map((img) => (
        <ImageCard key={img.id} image={img} />
      ))}
      <button onClick={refetch}>Refresh</button>
    </div>
  );
};
```

### Theme Toggle
```typescript
import { useThemeContext } from '@/contexts';

export const ThemeToggle = () => {
  const { isDark, toggleTheme } = useThemeContext();

  return (
    <button onClick={toggleTheme}>
      {isDark ? '☀️ Light' : '🌙 Dark'}
    </button>
  );
};
```

### Notifications
```typescript
import { useNotificationContext } from '@/contexts';

export const UserForm = () => {
  const { success, error } = useNotificationContext();

  const handleSubmit = async (data) => {
    try {
      await saveUser(data);
      success('User saved successfully');
    } catch (err) {
      error('Failed to save user');
    }
  };

  return (
    // Form
  );
};
```

## 📦 Dependencies

- **React** 18.2.0 - UI library
- **React Router DOM** 6.14.2 - Routing
- **Material-UI** 5.18.0 - Component library
- **Axios** 1.11.0 - HTTP client
- **JWT Decode** 3.1.2 - JWT parsing
- **TypeScript** 4.9.5 - Type safety

## 🔒 Security Features

- JWT token injection in all requests
- Automatic token refresh on 401
- Token revocation on logout
- XSS prevention in components
- CORS-ready configuration
- Error sanitization (no sensitive data)

## 📊 Performance Optimizations

- Code splitting with lazy loading
- Component memoization
- Request caching with TTL
- Global cache management
- Retry logic for failed requests
- Debounced input handling
- Error boundary for resilience

## 🧪 Testing Ready

All hooks and components are:
- Fully typed with TypeScript
- Well-documented with JSDoc
- Modular and isolated
- Mock-friendly for unit tests
- Integration-test compatible

## 🎨 Styling Strategy

- Material-UI (MUI) for components
- Emotion for CSS-in-JS
- Dark/light theme support
- Responsive design built-in
- CSS modules optional

## 🔄 State Management Pattern

- **Local State:** useState for component-local state
- **Context API:** Global state (auth, theme, notifications)
- **Custom Hooks:** Reusable logic
- **API Client:** Centralized API communication
- **localStorage:** Persistent client state

## 📝 Best Practices Implemented

✅ TypeScript strict mode throughout
✅ Comprehensive error handling
✅ Loading states for async operations
✅ Automatic cleanup in hooks
✅ Memory leak prevention
✅ Proper dependency arrays
✅ JSDoc documentation
✅ Accessibility considerations
✅ Responsive design ready
✅ Production-grade code quality

## 🚀 Next Steps

1. **Create remaining layout components:** MainLayout, Sidebar, NavigationBar, Footer
2. **Implement page components:** Dashboard, ImageProcessing, etc.
3. **Add form validation library:** React Hook Form + Zod
4. **Set up testing:** Jest + React Testing Library
5. **Add state persistence:** Redux Persist or Zustand
6. **Performance monitoring:** Sentry integration
7. **Analytics:** Google Analytics integration

## 📚 Architecture Principles

1. **Separation of Concerns** - Hooks, contexts, components, services
2. **DRY (Don't Repeat Yourself)** - Reusable hooks and providers
3. **Single Responsibility** - Each hook/component has one purpose
4. **Dependency Injection** - Props and context for dependencies
5. **Error Handling** - Graceful degradation everywhere
6. **Type Safety** - Full TypeScript coverage
7. **Performance** - Optimized rendering and data fetching
8. **Maintainability** - Clear structure and documentation

---

**Status:** ✅ Complete - Ready for feature development

**Total Lines of Code:** 2,000+ lines
**Components Created:** 3 utility components
**Custom Hooks Created:** 7 hooks
**Context Providers Created:** 3 providers
**Type Definitions:** 25+ interfaces

```
