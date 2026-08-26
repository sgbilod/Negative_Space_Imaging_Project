# ⚛️ MASTER PROMPT 03B: REACT FRONTEND - HOOKS & SERVICES
**Weeks 3-4 Execution (Part B) | Dec 6-13, 2025**

**Copy this entire prompt and paste into GitHub Copilot Chat, then request:**
```
"Generate files 21-41 (useAuth.ts through globals.css including app root)"
```

---

## DETAILED PROMPT FOR COPILOT

```
TASK: Create React hooks, services, and configuration for Negative Space Imaging Project frontend.

PART B: Hooks, Services, Context, Utilities, and Styling

FILES TO GENERATE (Part B - 21-41):

21. src/frontend/hooks/useAuth.ts (80 lines)
    - Hook for authentication
    - Methods: login, register, logout, checkAuth
    - State: user, token, loading, error
    - Persists token to localStorage

22. src/frontend/hooks/useApi.ts (100 lines)
    - Hook for API calls
    - Methods: get, post, put, delete
    - State: data, loading, error
    - Handles JWT token in headers
    - Error handling

23. src/frontend/hooks/useForm.ts (100 lines)
    - Hook for form state management
    - Methods: handleChange, handleSubmit, reset
    - State: values, errors, touched
    - Zod validation integration

24. src/frontend/hooks/usePagination.ts (80 lines)
    - Hook for pagination
    - State: page, pageSize, total
    - Methods: nextPage, prevPage, goToPage

25. src/frontend/hooks/useLocalStorage.ts (60 lines)
    - Hook for localStorage persistence
    - Getter and setter
    - Type-safe

26. src/frontend/hooks/useNotification.ts (70 lines)
    - Hook for notifications
    - Methods: showSuccess, showError, showWarning, showInfo

27. src/frontend/context/AuthContext.tsx (100 lines)
    - React Context for auth state
    - Provider component
    - useAuth consumer hook

28. src/frontend/context/NotificationContext.tsx (80 lines)
    - React Context for notifications
    - Provider component
    - useNotification consumer hook

29. src/frontend/services/authService.ts (100 lines)
    - Functions: login, register, logout, refresh, getCurrentUser
    - API calls to backend
    - Token management
    - localStorage persistence

30. src/frontend/services/imageService.ts (120 lines)
    - Functions: getImages, getImageById, uploadImage, deleteImage
    - API calls to backend
    - File upload handling
    - Pagination support

31. src/frontend/services/analysisService.ts (100 lines)
    - Functions: startAnalysis, getAnalysisResult, getAnalyses
    - API calls to backend
    - Status polling for long-running analyses

32. src/frontend/services/userService.ts (80 lines)
    - Functions: getProfile, updateProfile, changePassword, deleteAccount
    - API calls to backend

33. src/frontend/utils/api.ts (80 lines)
    - Axios instance configuration
    - Request/response interceptors
    - Token injection
    - Error handling

34. src/frontend/utils/validation.ts (100 lines)
    - Zod schemas for validation
    - Schema: LoginSchema
    - Schema: RegisterSchema
    - Schema: ImageUploadSchema
    - Schema: ProfileSchema

35. src/frontend/utils/formatters.ts (60 lines)
    - Function: formatFileSize(bytes): string
    - Function: formatDate(date): string
    - Function: formatPercentage(value): string

36. src/frontend/utils/constants.ts (50 lines)
    - API endpoints
    - Error messages
    - Success messages
    - Constants for UI

37. src/frontend/styles/globals.css (100 lines)
    - Tailwind imports
    - Global styles
    - CSS variables
    - Typography defaults
    - Dark mode configuration

38. src/frontend/styles/tailwind.css (50 lines)
    - Tailwind CSS directives

39. src/frontend/App.tsx (80 lines)
    - Main app component
    - Route definitions
    - Providers setup
    - Layout wrapping

40. src/frontend/main.tsx (50 lines)
    - App entry point
    - React DOM render
    - Router setup

41. public/index.html (40 lines)
    - HTML template
    - Meta tags
    - Root div for React

STRUCTURE EXAMPLE:
```typescript
// useAuth.ts
import { useContext } from 'react';
import { AuthContext } from '@/context/AuthContext';

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

// authService.ts
import axios from '@/utils/api';

export const authService = {
  async login(email: string, password: string) {
    const response = await axios.post('/auth/login', { email, password });
    localStorage.setItem('token', response.data.token);
    return response.data;
  },
};

// App.tsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from '@/context/AuthContext';
import MainLayout from '@/layouts/MainLayout';
import LoginPage from '@/pages/LoginPage';
import ProtectedRoute from '@/components/ProtectedRoute';

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <MainLayout />
              </ProtectedRoute>
            }
          />
        </Routes>
      </Router>
    </AuthProvider>
  );
}
```

Generate files 21-41 now.
```

---

## 📋 EXECUTION CHECKLIST

After Copilot generates the files:

- [ ] All 21 files generated successfully
- [ ] Files saved to correct directories
- [ ] No TypeScript compilation errors
- [ ] All hooks work correctly
- [ ] Services connect to API
- [ ] Context providers wrap app properly
- [ ] App builds without errors
- [ ] Files committed to git

## 📍 FILE STRUCTURE

```
src/frontend/
├── hooks/
│   ├── useAuth.ts
│   ├── useApi.ts
│   ├── useForm.ts
│   ├── usePagination.ts
│   ├── useLocalStorage.ts
│   └── useNotification.ts
├── context/
│   ├── AuthContext.tsx
│   └── NotificationContext.tsx
├── services/
│   ├── authService.ts
│   ├── imageService.ts
│   ├── analysisService.ts
│   └── userService.ts
├── utils/
│   ├── api.ts
│   ├── validation.ts
│   ├── formatters.ts
│   └── constants.ts
├── styles/
│   ├── globals.css
│   └── tailwind.css
├── App.tsx
├── main.tsx
└── public/
    └── index.html
```

## ✅ TESTING REACT FRONTEND

After generating all 41 files (Part A + Part B):

```bash
# Install dependencies
npm install

# Build React
npm run build

# Start dev server
npm run dev

# Visit: http://localhost:5173
```

## 🔄 NEXT: MASTER PROMPT 04

After completing Part A & B of Weeks 3-4:
1. Review all 41 files for correctness
2. Ensure frontend builds and loads
3. Commit changes to git
4. Use **MASTER_PROMPT_04** for database & testing

---

**Status:** Ready for execution
**Created:** November 8, 2025
**Time to Generate:** 15-20 minutes
**Part:** B of 2 for Weeks 3-4
**Prerequisites:** Complete Master Prompt 03A first
