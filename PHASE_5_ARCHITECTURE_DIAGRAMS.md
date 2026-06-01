# Phase 5: Architecture & Data Flow Diagrams

## 1. Application Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    React Application                      │ │
│  │                                                           │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │            Store Providers (Layers)             │   │ │
│  │  │                                                  │   │ │
│  │  │  1. UIStoreProvider                             │   │ │
│  │  │  2. UserStoreProvider                           │   │ │
│  │  │  3. ImageStoreProvider                          │   │ │
│  │  │  4. AnalysisStoreProvider                       │   │ │
│  │  │                                                  │   │ │
│  │  └──────────────────────────────────────────────────┘   │ │
│  │                        ↓                                  │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │         Context Providers (Layers)              │   │ │
│  │  │                                                  │   │ │
│  │  │  1. AuthProvider                                │   │ │
│  │  │  2. ThemeProvider                               │   │ │
│  │  │  3. NotificationProvider                        │   │ │
│  │  │                                                  │   │ │
│  │  └──────────────────────────────────────────────────┘   │ │
│  │                        ↓                                  │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │           React Router (v6)                      │   │ │
│  │  │                                                  │   │ │
│  │  │  ┌────────────────────────────────────────────┐ │   │ │
│  │  │  │     Navigation Component                  │ │   │ │
│  │  │  │  (AppBar + Sidebar + Breadcrumbs)        │ │   │ │
│  │  │  └────────────────────────────────────────────┘ │   │ │
│  │  │                        ↓                         │   │ │
│  │  │  ┌────────────────────────────────────────────┐ │   │ │
│  │  │  │      Route Matching                       │ │   │ │
│  │  │  │  (Protected/Public/Admin)                 │ │   │ │
│  │  │  └────────────────────────────────────────────┘ │   │ │
│  │  │                        ↓                         │   │ │
│  │  │  ┌────────────────────────────────────────────┐ │   │ │
│  │  │  │   ProtectedRoute Component                │ │   │ │
│  │  │  │  - Check Authentication                   │ │   │ │
│  │  │  │  - Check Role (Admin)                     │ │   │ │
│  │  │  │  - Redirect if Unauthorized               │ │   │ │
│  │  │  └────────────────────────────────────────────┘ │   │ │
│  │  │                        ↓                         │   │ │
│  │  │  ┌────────────────────────────────────────────┐ │   │ │
│  │  │  │     Lazy Route Components                 │ │   │ │
│  │  │  │  (Code Splitting with React.lazy)        │ │   │ │
│  │  │  └────────────────────────────────────────────┘ │   │ │
│  │  │                                                  │   │ │
│  │  └──────────────────────────────────────────────────┘   │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        localStorage
                    (Session Persistence)
```

## 2. Authentication Flow

```
User Navigates to /dashboard
            ↓
Route Matcher: /dashboard → protected
            ↓
ProtectedRoute Component Renders
            ↓
Check: isAuthenticated === true?
            ↓ NO
     useAuth() returns null/false
            ↓
  Navigate to /login
            ↓
User Enters Credentials
            ↓
AuthContext.login()
            ↓
API Call: POST /auth/login
            ↓
Store Token in localStorage
            ↓
Update useAuth() hook
            ↓
Re-navigate to /dashboard
            ↓
Check: isAuthenticated === true?
            ↓ YES
Check: requiredRole === user.role?
            ↓ YES
  Render Protected Component
```

## 3. State Management Flow

```
Component
    ↓
Import Store Hook
    ↓
useImageStore() {
    const context = useContext(ImageStoreContext)
    return context
}
    ↓
Access State or Dispatch Action
    ↓
State: const { images, isLoading } = useImageStore()
    ↓
Action: addImage(newImage)
    ↓
Dispatch Action to Reducer
    ↓
Reducer Updates State
    ↓
localStorage Updated
    ↓
Context Notifies Subscribers
    ↓
Component Re-renders
    ↓
UI Updated
```

## 4. Protected Route Logic

```
┌─────────────────────────────────────────┐
│   <ProtectedRoute>                      │
│     <DashboardPage />                   │
│   </ProtectedRoute>                     │
└─────────────────────────────────────────┘
              ↓
    Check Props & Conditions
              ↓
        ┌─────┴─────┐
        │           │
     YES│           │NO
        ↓           ↓
    ┌────────────────────┐
    │   Loading State?   │
    │   isLoading=true   │
    └────────────────────┘
        │           │
       NO           YES
        ↓           ↓
    ┌────────────────────┐
    │ Show Loading       │
    │ (Fallback)         │
    │ return fallback    │
    └────────────────────┘
        ↓
    ┌────────────────────────────────┐
    │ isAuthenticated === false?      │
    │ <Navigate to="/login" />        │
    └────────────────────────────────┘
        ↓
    ┌────────────────────────────────┐
    │ requiredRole === 'admin'?       │
    │ user.role !== 'admin'?          │
    │ <Navigate to="/error" />        │
    └────────────────────────────────┘
        ↓
    ┌────────────────────────────────┐
    │ Access Granted!                │
    │ Return children                │
    │ <DashboardPage />              │
    └────────────────────────────────┘
```

## 5. Store Provider Hierarchy

```
Root
 │
 ├─ ThemeProvider (from Phase 3)
 │  │
 │  ├─ AuthProvider (from Phase 3)
 │  │  │
 │  │  ├─ NotificationProvider (from Phase 3)
 │  │  │  │
 │  │  │  ├─ UIStoreProvider (Phase 5)
 │  │  │  │  │
 │  │  │  │  ├─ UserStoreProvider (Phase 5)
 │  │  │  │  │  │
 │  │  │  │  │  ├─ ImageStoreProvider (Phase 5)
 │  │  │  │  │  │  │
 │  │  │  │  │  │  ├─ AnalysisStoreProvider (Phase 5)
 │  │  │  │  │  │  │  │
 │  │  │  │  │  │  │  ├─ BrowserRouter
 │  │  │  │  │  │  │  │  │
 │  │  │  │  │  │  │  │  ├─ Navigation Component
 │  │  │  │  │  │  │  │  │
 │  │  │  │  │  │  │  │  ├─ Suspense (Loading)
 │  │  │  │  │  │  │  │  │
 │  │  │  │  │  │  │  │  └─ Routes
 │  │  │  │  │  │  │  │     ├─ /login
 │  │  │  │  │  │  │  │     ├─ /dashboard (protected)
 │  │  │  │  │  │  │  │     ├─ /upload (protected)
 │  │  │  │  │  │  │  │     ├─ /analysis/:id (protected)
 │  │  │  │  │  │  │  │     ├─ /admin (admin)
 │  │  │  │  │  │  │  │     └─ * (404)
```

## 6. Navigation Structure

```
┌──────────────────────────────────────────────────────┐
│  Navigation Component                                │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │  AppBar (Sticky)                               │ │
│  │  ┌────────┐         ┌──────────┐  ┌────────┐  │ │
│  │  │ NSIP   │ ........ │ Title.. │  │ User   │  │ │
│  │  │ Logo   │         │ Path    │  │ Menu ▼ │  │ │
│  │  └────────┘         └──────────┘  └────────┘  │ │
│  │                                                  │ │
│  │  ┌────────────────────────────────────────────┐ │
│  │  │ Breadcrumbs (Dashboard > Upload > File)    │ │ │
│  │  └────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  ┌────────────────┐      ┌──────────────────────┐  │
│  │ Sidebar (MD+)  │      │ Drawer (XS-SM)       │  │
│  │ Fixed          │      │ Collapsible          │  │
│  │ Width: 250px   │      │ Overlays Content     │  │
│  │                │      │                      │  │
│  │ • Dashboard    │      │ • Dashboard          │  │
│  │ • Upload       │      │ • Upload             │  │
│  │ • Analysis     │      │ • Analysis           │  │
│  │ • Settings     │      │ • Settings           │  │
│  │ • Admin (if)   │      │ • Admin (if)         │  │
│  │                │      │                      │  │
│  └────────────────┘      └──────────────────────┘  │
│         │                         │                │
│         └─────────┬───────────────┘                │
│                   ↓                                │
│         ┌──────────────────────┐                   │
│         │ Main Content Area    │                   │
│         │ (Page Component)      │                   │
│         │                      │                   │
│         │ Responsive Layout    │                   │
│         │ pl: {xs:0, md:250px} │                   │
│         │                      │                   │
│         └──────────────────────┘                   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## 7. Image Upload Flow (with Store)

```
UploadPage Component
      ↓
User Selects File
      ↓
handleFileUpload(file)
      ↓
const { addImage, updateProgress, updateImage } = useImageStore()
      ↓
Create Image ID
      ↓
addImage({
  id, name, status: 'uploading', progress: 0
})
      ↓
Send to API with XMLHttpRequest
      ↓
On Progress Event:
  updateProgress(id, percentage)
      ↓
Component Re-renders with Progress Bar
      ↓
On Upload Complete:
  updateImage({ ...image, status: 'uploaded', progress: 100 })
      ↓
Component Re-renders
      ↓
Image Appears in Images List
      ↓
Show Success Notification
      ↓
localStorage Updated with New Image
```

## 8. Role-Based Route Access

```
User Has Role: 'user'
      ↓
Navigate to /admin
      ↓
Route.meta.access = 'admin'
      ↓
ProtectedRoute checks:
  requiredRole === 'admin'
      ↓
useAuth().user.role === 'admin'?
      ↓ NO
Navigate to /error
      ↓
ErrorPage Displayed


User Has Role: 'admin'
      ↓
Navigate to /admin
      ↓
Route.meta.access = 'admin'
      ↓
ProtectedRoute checks:
  requiredRole === 'admin'
      ↓
useAuth().user.role === 'admin'?
      ↓ YES
Render AdminPanel
```

## 9. localStorage Persistence

```
App.tsx
   ↓
  <App />
   ↓
useReducer with init function
   ↓
Read localStorage['user-store']
   ↓
Parse JSON to hydrate state
   ↓
   ├─ SUCCESS: Use stored state
   │   ↓
   │   Component renders with persistent data
   │
   └─ ERROR: Use initial state
       ↓
       Component renders with defaults

During User Session:
   ↓
State Change via Dispatch
   ↓
useEffect watches state
   ↓
Write to localStorage
   ↓
   ├─ SUCCESS: Persisted
   │
   └─ ERROR: Silent fail (graceful degradation)

On Logout:
   ↓
logout() action
   ↓
Clear localStorage['user-store']
   ↓
Reset state to initial
   ↓
Redirect to /login
```

## 10. Lazy Loading Routes

```
Production Bundle
      ↓
Main App (index.tsx)
├─ Core libs (React, MUI)
├─ Contexts
└─ Routes container

      ↓
Route Group 1: Public Routes
├─ /login
├─ /register
└─ /404
(Bundled together ~50KB)

      ↓
Route Group 2: Protected Routes
├─ /dashboard
├─ /upload
├─ /analysis/:id
└─ /settings
(Bundled together ~100KB)

      ↓
Route Group 3: Admin Routes
├─ /admin
├─ /admin/users
└─ /admin/stats
(Bundled together ~75KB)

      ↓
User Navigates to /dashboard
      ↓
Route Group 2 Loaded On-Demand
      ↓
Suspense Fallback Shows Loading
      ↓
Component Renders
```

---

## Summary: Data Flow

```
1. User Action (click button, navigate)
        ↓
2. Router matches route
        ↓
3. ProtectedRoute checks auth/role
        ↓
4. Component renders
        ↓
5. Component uses store hook
        ↓
6. Store provides state + dispatch
        ↓
7. Component renders with data
        ↓
8. User interacts with component
        ↓
9. Dispatch action to store
        ↓
10. Reducer updates state
        ↓
11. localStorage synced
        ↓
12. Components re-render
        ↓
13. UI updated
```

These diagrams show the complete architecture, data flows, and interactions throughout Phase 5's routing and state management system.
