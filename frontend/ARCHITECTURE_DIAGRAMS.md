// REACT FRONTEND ARCHITECTURE DIAGRAM
// Comprehensive visual guide to the React application structure

/**
 * HIGH-LEVEL APPLICATION ARCHITECTURE
 * ====================================
 *
 *                           USER BROWSER
 *                              │
 *                    ┌─────────▼─────────┐
 *                    │   React App       │
 *                    │   (Vite Build)    │
 *                    └────────┬──────────┘
 *                             │
 *             ┌───────────────┼───────────────┐
 *             │                               │
 *      ┌──────▼────────┐          ┌──────────▼─────┐
 *      │ Error         │          │ Root Component │
 *      │ Boundary      │          │ (App.tsx)      │
 *      └──────┬────────┘          └────────┬───────┘
 *             │                           │
 *      ┌──────▼────────────────────────┬──▼────┐
 *      │                                │       │
 * ┌────▼──────┐  ┌────────────┐  ┌────▼──┐  ┌─▼──────┐
 * │   Theme   │  │AuthProvider│  │Router │  │Suspense│
 * │ Provider  │  │(Global)    │  │(Lazy  │  │(Code   │
 * │(Context)  │  │Auth State  │  │Routes)│  │Split)  │
 * └───────────┘  └──────┬─────┘  └───────┘  └────────┘
 *                       │
 *          ┌────────────┼──────────────┐
 *          │            │              │
 *    ┌─────▼──────┐  ┌─▼──────────┐  ┌▼──────────────┐
 *    │useAuth     │  │useAuthCont │  │Notification  │
 *    │Hook        │  │ext         │  │Provider      │
 *    │LS Persist  │  │Global      │  │(Toast)       │
 *    │JWT Mgmt    │  │Access      │  │               │
 *    └────────────┘  └────────────┘  └───────────────┘
 *
 *
 * HOOKS LAYER - Data & Logic
 * ===========================
 *
 *   Component Uses Hook
 *        │
 *   ┌────▼────────────────────────────────────────┐
 *   │         Custom Hook (useX)                  │
 *   ├────────────────────────────────────────────┤
 *   │ • State management                         │
 *   │ • API communication                        │
 *   │ • Error handling                           │
 *   │ • Loading states                           │
 *   │ • Caching logic                            │
 *   └────────┬─────────────────────────────────┬─┘
 *            │                                 │
 *    ┌───────▼──────────┐            ┌────────▼────────┐
 *    │   useLocalStorage│            │   useFetch      │
 *    │   (Persistence) │            │   (API+Cache)   │
 *    └──────────────────┘            └────────┬────────┘
 *                                             │
 *                                      ┌──────▼──────────┐
 *                                      │   API Client    │
 *                                      │   (apiClient.ts)│
 *                                      └──────┬──────────┘
 *
 *
 * API CLIENT FLOW
 * ===============
 *
 *   1. Component calls hook
 *         │
 *   ┌─────▼─────────────────────────────────┐
 *   │  Hook (useFetch, useAuth, etc.)       │
 *   └─────┬─────────────────────────────────┘
 *         │
 *   ┌─────▼─────────────────────────────────┐
 *   │  apiClient.get/post/put/delete        │
 *   └─────┬─────────────────────────────────┘
 *         │
 *   ┌─────▼─────────────────────────────────┐
 *   │ REQUEST INTERCEPTOR                   │
 *   │ • Read accessToken from localStorage  │
 *   │ • Inject "Authorization: Bearer..."   │
 *   │ • Add other headers                   │
 *   └─────┬─────────────────────────────────┘
 *         │
 *   ┌─────▼─────────────────────────────────┐
 *   │ HTTP REQUEST (Axios)                  │
 *   │ Send to Backend API                   │
 *   └─────┬─────────────────────────────────┘
 *         │
 *         ├─────────────────────────────────┐
 *         │                                 │
 *    ┌────▼──────────┐         ┌───────────▼──┐
 *    │ Success (200) │         │   Error       │
 *    └────┬──────────┘         └───────────┬───┘
 *         │                                │
 *    ┌────▼──────────────────────────────┐│
 *    │ Response Interceptor - Success    ││
 *    │ • Parse response                  ││
 *    │ • Cache result (if GET)           ││
 *    │ • Return to hook                  ││
 *    └────────────────────────────────────┘│
 *                                          │
 *    ┌─────────────────────────────────────┴──────┐
 *    │ Response Interceptor - Error (401?)         │
 *    │ ┌─────────────────────────────────────────┐ │
 *    │ │ If 401 Unauthorized:                    │ │
 *    │ │ 1. Read refreshToken from localStorage  │ │
 *    │ │ 2. POST /auth/refresh with token        │ │
 *    │ │ 3. Get new accessToken from response    │ │
 *    │ │ 4. Store in localStorage                │ │
 *    │ │ 5. Retry original request with token    │ │
 *    │ │ 6. If refresh fails → redirect login    │ │
 *    │ └─────────────────────────────────────────┘ │
 *    │ ┌─────────────────────────────────────────┐ │
 *    │ │ Other errors (400, 403, 404, 500):     │ │
 *    │ │ 1. Normalize error response             │ │
 *    │ │ 2. Extract status code and message      │ │
 *    │ │ 3. Return ApiError to hook              │ │
 *    │ │ 4. Hook sets error state                │ │
 *    │ │ 5. Component displays error to user     │ │
 *    │ └─────────────────────────────────────────┘ │
 *    └──────────────────────────────────────────────┘
 *
 *
 * COMPONENT HIERARCHY
 * ===================
 *
 *   App.tsx
 *   ├── ErrorBoundary (catch errors)
 *   ├── Theme Provider (light/dark mode)
 *   ├── MUI Theme Provider (Material-UI)
 *   ├── Auth Provider (global auth state)
 *   ├── Notification Provider (toasts)
 *   ├── Router (React Router)
 *   │   └── Suspense (code splitting)
 *   │       └── Routes
 *   │           ├── Login Page
 *   │           ├── Dashboard Page
 *   │           │   ├── Dashboard Layout
 *   │           │   ├── Stats Cards
 *   │           │   ├── Charts
 *   │           │   └── Recent Activity
 *   │           ├── Image Upload Page
 *   │           │   ├── Upload Form
 *   │           │   ├── Progress Bar
 *   │           │   └── Analysis Results
 *   │           ├── Security Monitor
 *   │           │   ├── Threat Dashboard
 *   │           │   ├── Event List
 *   │           │   └── Alerts
 *   │           ├── Audit Logs
 *   │           │   ├── Filters
 *   │           │   └── Data Table
 *   │           └── User Management (admin)
 *   │               ├── User Table
 *   │               ├── Edit Form
 *   │               └── Delete Confirmation
 *   └── (Global Toast Notifications)
 *
 *
 * STATE MANAGEMENT FLOW
 * ====================
 *
 *               Local Component State
 *              (useState in component)
 *                      │
 *                      │ (component-specific)
 *                      │
 *          ┌───────────┴─────────────┐
 *          │                         │
 *    ┌─────▼─────┐              ┌───▼─────┐
 *    │ Form Data │              │ UI State│
 *    │ Toggles   │              │ Modals  │
 *    └───────────┘              └─────────┘
 *
 *                Global Context State
 *             (Context API providers)
 *                      │
 *          ┌───────────┼───────────┐
 *          │           │           │
 *    ┌─────▼──────┐ ┌─▼───────┐ ┌▼────────────┐
 *    │ AuthContext│ │ThemeCtx │ │Notification│
 *    │ • user     │ │• isDark │ │• notifications
 *    │ • tokens   │ │• mode   │ │• addNotif  │
 *    │ • loading  │ │• toggle │ │• remove    │
 *    │ • error    │ └────────┘ └────────────┘
 *    └────────────┘
 *
 *                 API Response Cache
 *            (Global Map in useFetch)
 *                      │
 *          ┌───────────┴────────────┐
 *          │                        │
 *    ┌─────▼─────────────┐    ┌────▼──────────┐
 *    │ GET /api/items    │    │ GET /api/users│
 *    │ TTL: 5 minutes    │    │ TTL: 5 minutes│
 *    └───────────────────┘    └───────────────┘
 *
 *              Persistent Browser Storage
 *                 (localStorage)
 *                      │
 *          ┌───────────┼───────────┐
 *          │           │           │
 *    ┌─────▼──────┐ ┌─▼───────┐ ┌▼────────────┐
 *    │accessToken │ │refresh  │ │theme       │
 *    │(JWT)       │ │Token    │ │preference │
 *    │           │ │(JWT)    │ │           │
 *    └────────────┘ └────────┘ └────────────┘
 *
 *
 * AUTHENTICATION FLOW (DETAILED)
 * ==============================
 *
 *   User enters email/password
 *          │
 *   ┌──────▼──────────────────────┐
 *   │ Click Login Button           │
 *   └──────┬───────────────────────┘
 *          │
 *   ┌──────▼──────────────────────┐
 *   │ Form validation              │
 *   │ (email format, password len) │
 *   └──────┬───────────────────────┘
 *          │
 *   ┌──────▼──────────────────────┐
 *   │ Call useAuth.login()         │
 *   └──────┬───────────────────────┘
 *          │
 *   ┌──────▼──────────────────────────────┐
 *   │ POST /api/auth/login                │
 *   │ {email, password}                   │
 *   └──────┬───────────────────────────────┘
 *          │
 *   ┌──────▼──────────────────────────────┐
 *   │ Backend validates credentials       │
 *   └──────┬───────────────────────────────┘
 *          │
 *    ┌─────┴─────────┐
 *    │               │
 * ┌──▼──────┐   ┌───▼──────────┐
 * │ Success │   │ Failure      │
 * └──┬──────┘   └───┬──────────┘
 *    │              │
 * ┌──▼──────────────────────────┐
 * │Return:                      │
 * │• accessToken (JWT)          │
 * │• refreshToken (JWT)         │
 * │• user profile               │
 * └──┬───────────────────────────┘
 *    │
 * ┌──▼──────────────────────────┐
 * │ useAuth hook:               │
 * │ 1. Save to localStorage     │
 * │ 2. Set user state           │
 * │ 3. Set isAuthenticated=true │
 * │ 4. Clear error              │
 * └──┬───────────────────────────┘
 *    │
 * ┌──▼──────────────────────────┐
 * │AuthContext updates          │
 * │ All components see user!    │
 * └──┬───────────────────────────┘
 *    │
 * ┌──▼──────────────────────────┐
 * │ Navigate to /dashboard      │
 * │ Dashboard page renders      │
 * │ with user data              │
 * └─────────────────────────────┘
 *
 *
 * ERROR HANDLING ARCHITECTURE
 * ===========================
 *
 *     Component Errors (React level)
 *              │
 *    ┌─────────▼──────────┐
 *    │ ErrorBoundary      │
 *    │ catches React      │
 *    │ component errors   │
 *    │ (renders fallback) │
 *    └────────────────────┘
 *              │
 *         ┌────▼─────┐
 *         │ Show     │
 *         │ Error UI │
 *         │ + Try    │
 *         │ Again    │
 *         │ button   │
 *         └──────────┘
 *
 *      API / Async Errors
 *              │
 *    ┌─────────▼────────────────┐
 *    │ Hook catches error       │
 *    │ (try-catch, then handler)│
 *    └────┬────────────────────┘
 *         │
 *    ┌────▼──────────────────────┐
 *    │ Error type?               │
 *    └────┬──────────────┬───┬────┘
 *         │              │   │
 *    ┌────▼──┐  ┌───────▼┐ ┌┴─────────┐
 *    │401    │  │400/422 │ │500/Other │
 *    │(Auth) │  │(Valid) │ │(Server)  │
 *    └────┬──┘  └───┬────┘ └┬────────┘
 *         │         │       │
 *    ┌────▼──┐  ┌───▼───┐ ┌┴────────┐
 *    │Token  │  │Show   │ │Show     │
 *    │Refresh│  │Error  │ │Generic  │
 *    │attempt│  │to user│ │error    │
 *    └────┬──┘  └───────┘ └─────────┘
 *         │
 *    ┌────▼──────────────┐
 *    │Refresh fails?     │
 *    │Logout user        │
 *    │Redirect to /login │
 *    └───────────────────┘
 *
 *
 * PERFORMANCE OPTIMIZATION POINTS
 * ===============================
 *
 *   1. Code Splitting
 *      • Lazy load pages
 *      • Only load when needed
 *      • Reduce initial bundle
 *
 *   2. Response Caching
 *      • Cache GET responses
 *      • 5-minute TTL by default
 *      • Skip unnecessary network calls
 *
 *   3. Debouncing
 *      • Debounce search inputs
 *      • Reduce API requests
 *      • Save bandwidth
 *
 *   4. Memoization
 *      • useMemo for expensive calculations
 *      • useCallback for event handlers
 *      • memo() for components
 *
 *   5. Request Batching (ready to implement)
 *      • Batch multiple requests
 *      • Send as single POST
 *      • Reduce network round trips
 *
 *
 * DEPLOYMENT ARCHITECTURE
 * =======================
 *
 *   Development Build (npm run dev)
 *           │
 *      ┌────▼────┐
 *      │Vite Dev  │
 *      │Server    │
 *      │(:5173)   │
 *      └──────────┘
 *
 *   Production Build (npm run build)
 *           │
 *      ┌────▼──────────┐
 *      │Build Output:  │
 *      │  dist/        │
 *      │  ├─ index.html│
 *      │  ├─ css/      │
 *      │  ├─ js/       │
 *      │  └─ assets/   │
 *      └────┬──────────┘
 *           │
 *      ┌────▼──────────────┐
 *      │Deploy to:         │
 *      │• Vercel           │
 *      │• Netlify          │
 *      │• AWS S3 + CloudFront
 *      │• Docker container │
 *      │• Any web server   │
 *      └───────────────────┘
 *
 */

export default {};
