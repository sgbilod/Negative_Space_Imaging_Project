# Phase 4 Visual Architecture & Flow Diagrams

## 🏗️ COMPONENT ARCHITECTURE

```text
App.tsx (Router & Providers)
│
├── AuthProvider
│   ├── NotificationProvider
│   │   └── ThemeProvider
│   │       └── Router
│   │           └── Routes
│   │
│   ├── [PUBLIC ROUTES]
│   │   ├── /login → LoginPage ✓
│   │   └── /register → RegisterPage ✓
│   │
│   └── [PRIVATE ROUTES - Wrapped by PrivateRoute]
│       ├── /dashboard → Layout
│       │   └── DashboardPage ✓
│       │
│       ├── /upload → Layout
│       │   └── UploadPage ✓
│       │
│       ├── /analysis/:imageId → Layout
│       │   └── AnalysisResultsPage ✓
│       │
│       └── /settings → Layout
│           └── SettingsPage ✓
```

## 📄 PAGE COMPONENT HIERARCHY

```
Page Component
│
├── Container/Wrapper (Box)
│   │
│   ├── Header Section
│   │   ├── Title
│   │   └── Subtitle/Description
│   │
│   ├── Main Content
│   │   ├── Form/Table/Grid
│   │   └── Components
│   │
│   ├── Loading State
│   │   └── CircularProgress/Skeleton
│   │
│   ├── Error State
│   │   └── Alert Component
│   │
│   └── Empty State
│       └── Message + Icon
│
└── Notifications (useNotification)
    ├── Success (green)
    ├── Error (red)
    ├── Warning (orange)
    └── Info (blue)
```

## 🔄 USER AUTHENTICATION FLOW

```
START
  │
  ▼
User Visits /login
  │
  ├─────────────────────────────────┐
  │                                 │
  ▼                                 ▼
LoginPage                     RegistrationFlow
  │                                 │
  ├─ Email Input                    ├─ First Name Input
  ├─ Password Input                 ├─ Last Name Input
  ├─ Remember Me                    ├─ Email Input
  └─ Submit                         ├─ Password Input
      │                             ├─ Confirm Password
      ├─ validateForm()             ├─ Password Strength (0-5)
      ├─ API: POST /login           ├─ Terms Checkbox
      │                             └─ Submit
      │                                 │
      ├─ validateForm()                ├─ validateForm()
      │                                ├─ Password Strength Check
      │  Error                         │
      ├─ success? No                   │  Error
      │  │                             ├─ success? No
      │  └─ show error                 │  │
      │     notification               │  └─ show validation errors
      │                                │
      │     Yes ▼                       │     Yes ▼
      │ ┌──────────────────┐           │ ┌──────────────────┐
      │ │ Save JWT token   │           │ │ Save JWT token   │
      │ │ to localStorage  │           │ │ to localStorage  │
      │ └──────────────────┘           │ └──────────────────┘
      │     │                           │     │
      │     ▼                           │     ▼
      └────────────────────────────────┤ Auto-login
                                       │
                                       ▼
                                 /dashboard
                                   │
                                   ├─ Load User Data
                                   ├─ Show Welcome Message
                                   └─ Display Statistics

END (Authenticated User)
```

## 📤 FILE UPLOAD FLOW

```
START
  │
  ▼
User Opens /upload
  │
  ▼
Display Drop Zone + File Input
  │
  ├─────────────────────────────────────────┐
  │                                         │
  ▼                                         ▼
Drag & Drop File                      Click "Choose File"
  │                                         │
  ├─ handleDrag()                           ├─ File picker opens
  │  └─ Visual feedback (border change)    │
  │                                         ▼
  ├─ handleDrop()                       User Selects Files
  │  │                                      │
  │  ▼                                      ▼
  ├─ handleFiles()                      handleFiles()
  │  │                                      │
  │  ▼                                      ▼
  │  validateFile()                    validateFile()
  │  │                                      │
  │  ├─ Check format (JPEG, PNG, WebP, TIFF)
  │  ├─ Check size (≤50MB)
  │  └─ Check count (≤5 files)
  │      │
  │      Error                         Error
  │      │                              │
  │      ├─ Show validation error       ├─ show toast error
  │      └─ Don't add to queue          └─ Don't add to queue
  │
  │      Success ◄─────────────────────────► Success
  │      │                                      │
  │      ▼                                      ▼
  │   Add to uploadedFiles array
  │   ├─ id (UUID)
  │   ├─ file (File object)
  │   ├─ preview (Blob URL)
  │   ├─ progress (0)
  │   ├─ status ("pending")
  │   └─ error (null)
  │      │
  │      ▼
  └─────────────────────────────┐
                               │
                               ▼
                    Display Upload Queue
                    ├─ File previews (60x60px)
                    ├─ File names
                    ├─ Progress bars (0%)
                    ├─ Status badges
                    └─ Upload buttons
                               │
                               ▼
                    User Clicks "Upload" Button
                               │
                    ┌──────────────────────────┐
                    │  handleUpload()          │
                    │  (single file)           │
                    │   OR                     │
                    │  handleUploadAll()       │
                    │  (all files)             │
                    └──────────────────────────┘
                               │
                               ▼
                    Set status → "uploading"
                    Show progress bar 0%
                               │
                               ▼
                    API: POST /upload
                    └─ With progress callback
                               │
                        ┌──────┴──────┐
                        │             │
                   Progress        Error
                        │             │
                        ▼             ▼
                    0% → 100%    Set error message
                    Update UI    Set status → "failed"
                        │        Show retry button
                        │             │
                        └──────┬──────┘
                               │
                               ▼
                    Status → "completed"
                    Show success toast ✓
                    Enable re-upload
                               │
                               ▼
                    User Views Analysis Results
                               │
                               ▼
                             END
```

## 🖼️ IMAGE ANALYSIS FLOW

```
START
  │
  ▼
User Uploads Image
  │
  ▼
Image → Backend Processing
  │
  ├─ Python CV2 Analysis
  ├─ Detect Negative Space
  ├─ Calculate Confidence
  ├─ Extract Regions
  └─ Generate Analyzed Image
     │
     ▼
  Store in Database
  ├─ original_image_url
  ├─ analyzed_image_url
  ├─ statistics (areas, confidence, etc)
  └─ regions (array of detected areas)
     │
     ▼
User Clicks "View Analysis"
  │
  ▼
Navigate to /analysis/:imageId
  │
  ▼
AnalysisResultsPage Loads
  │
  ├─ Fetch analysis data from API
  └─ useParams(:imageId)
     │
     ▼
Display Dual Image Viewer
  ├─ Original Image (left)
  ├─ Analyzed Image (right)
  │
  └─ Zoom Controls (50% - 300%)
     ├─ Zoom In Button
     ├─ Zoom Out Button
     ├─ Reset Button
     └─ Zoom Level Display
     │
     ├─ Rotate Controls (0°, 90°, 180°, 270°)
     │  ├─ Rotate Button
     │  └─ Reset Button
     │
     └─ Compare Mode Toggle
        ├─ Side-by-side view
        └─ Overlay view (future)
     │
     ▼
Display Statistics Panel
  ├─ Total Areas Found: 5
  ├─ Average Confidence: 85%
  ├─ Contrast Ratio: 4.5:1
  ├─ Processing Time: 2.3s
  └─ Dominant Color: #FF6B6B (red)
     │
     ▼
Display Regions Table
  ├─ ID | Area | Confidence | Location | Description
  ├─ 1  | 245  | 92%       | (120,50) | High confidence
  ├─ 2  | 156  | 87%       | (340,180)| High confidence
  ├─ 3  | 89   | 73%       | (500,220)| Moderate confidence
  ├─ 4  | 342  | 95%       | (50,400) | High confidence
  └─ 5  | 201  | 81%       | (220,380)| Moderate confidence
     │
     ▼
Action Buttons
  ├─ Export
  │  ├─ As PNG (image)
  │  ├─ As CSV (data)
  │  └─ As JSON (full data)
  │
  ├─ Share
  │  ├─ Generate URL
  │  ├─ Copy to Clipboard
  │  └─ Show Confirmation
  │
  └─ Back to Dashboard
     │
     ▼
   END
```

## ⚙️ SETTINGS FLOW

```
START
  │
  ▼
User Navigates to /settings
  │
  ▼
SettingsPage Loads
  │
  ▼
Display 4-Tab Interface
  │
  ├────────────────────────────────────────────────────────────┐
  │                                                            │
  Tab: PROFILE                Tab: PREFERENCES               │
  │    ├─ First Name          │    ├─ Theme                 │
  │    ├─ Last Name           │    │  └─ Light/Dark/Auto    │
  │    ├─ Email (read-only)   │    ├─ Email Notifications   │
  │    └─ Save Button         │    ├─ Analysis Alerts       │
  │                           │    ├─ Weekly Report         │
  │                           │    └─ Save Button           │
  │                           │                             │
  Tab: PRIVACY              Tab: PASSWORD               │
  │    ├─ Public Profile     │    ├─ Current Password      │
  │    ├─ Allow Sharing      │    ├─ New Password          │
  │    ├─ Data Retention     │    ├─ Confirm Password      │
  │    │  └─ 30d/90d/1yr/∞  │    ├─ Change Button         │
  │    └─ Save Button         │    ├─ Danger Zone           │
  │                           │    │  └─ Delete Account     │
  │                           │    │     └─ Confirmation    │
  │                           │    └─ Save Button           │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
     │
     ▼
User Edits a Section
  │
  ├─ Make changes to fields
  ├─ Validation runs in real-time
  ├─ Error messages show if invalid
  └─ Save button becomes active
     │
     ▼
User Clicks Save
  │
  ▼
validateForm()
  │
  ├─ Error Found?
  │  │
  │  ├─ Yes → Show error messages, don't submit
  │  │
  │  └─ No → Continue
  │
  ▼
Set Loading State
Show spinner on button
  │
  ▼
API: PUT /settings
Send updated settings
  │
  ├─ Success
  │  │
  │  ├─ Update local state
  │  ├─ Show success toast
  │  └─ Clear loading
  │
  └─ Error
     │
     ├─ Show error toast
     ├─ Show error message
     └─ Clear loading
     │
     ▼
Return to display state
     │
     ▼
   END
```

## 🛡️ AUTHENTICATION STATE FLOW

```
App Start
  │
  ▼
Check localStorage for JWT token
  │
  ├─ Token exists
  │  │
  │  ├─ Verify token (API call)
  │  │
  │  ├─ Valid
  │  │  │
  │  │  ├─ Load user data
  │  │  ├─ Set isAuthenticated = true
  │  │  └─ Set user data in context
  │  │
  │  └─ Expired/Invalid
  │     │
  │     ├─ Delete token
  │     ├─ Set isAuthenticated = false
  │     └─ Redirect to /login
  │
  └─ No token
     │
     ├─ Set isAuthenticated = false
     ├─ Redirect to /login
     └─ Show login form
        │
        ▼
   User Logs In
        │
        ├─ validateForm()
        ├─ API: POST /login
        │
        ├─ Success
        │  │
        │  ├─ Store JWT token
        │  ├─ Set isAuthenticated = true
        │  ├─ Store user data
        │  └─ Redirect to /dashboard
        │
        └─ Error
           │
           ├─ Show error message
           ├─ Keep on /login
           └─ User can retry
              │
              ▼
   Authenticated User State
              │
              ├─ Can access protected routes
              ├─ Can upload files
              ├─ Can view analysis
              ├─ Can modify settings
              └─ Can see user menu
              │
              ▼
   User Logs Out
              │
              ├─ DELETE JWT token
              ├─ Clear user data
              ├─ Set isAuthenticated = false
              ├─ Show success toast
              └─ Redirect to /login
              │
              ▼
         END (Back to start)
```

## 🎨 RESPONSIVE DESIGN BREAKPOINTS

```
Mobile (xs: 0-600px)
┌─────────────────────┐
│    NAVBAR (full)    │
├─────────────────────┤
│                     │
│   Content 100%      │
│   (single column)   │
│                     │
│                     │
├─────────────────────┤
│    FOOTER (full)    │
└─────────────────────┘

Tablet Portrait (sm: 600-960px)
┌──────────────────────────┐
│      NAVBAR (full)       │
├──────────────────────────┤
│                          │
│   Content 90% width      │
│   (single column)        │
│                          │
│                          │
├──────────────────────────┤
│      FOOTER (full)       │
└──────────────────────────┘

Tablet Landscape (md: 960-1264px)
┌─────────────────────────────────┐
│          NAVBAR (full)          │
├──────────────┬──────────────────┤
│              │                  │
│  SIDEBAR     │   Content        │
│  (250px)     │   (calc 100% -   │
│              │    250px)        │
│              │                  │
├──────────────┴──────────────────┤
│       FOOTER (full)             │
└─────────────────────────────────┘

Desktop (lg: 1264px+)
┌──────────────────────────────────────┐
│            NAVBAR (full)             │
├──────────────┬──────────────────────┤
│              │                      │
│  SIDEBAR     │   Content (grid)     │
│  (270px)     │                      │
│              │   ┌──────┬──────┐    │
│              │   │      │      │    │
│              │   ├──────┼──────┤    │
│              │   │      │      │    │
│              │   └──────┴──────┘    │
│              │                      │
├──────────────┴──────────────────────┤
│         FOOTER (full)               │
└──────────────────────────────────────┘
```

## 📊 STATE MANAGEMENT DIAGRAM

```
App.tsx (Root)
  │
  ├─ AuthContext (Global)
  │  ├─ user: { id, email, firstName, lastName }
  │  ├─ isAuthenticated: boolean
  │  ├─ login(email, password)
  │  ├─ logout()
  │  ├─ loading: boolean
  │  └─ error: string | null
  │
  ├─ NotificationContext (Global)
  │  ├─ notifications: Array
  │  ├─ showNotification(message, severity, duration)
  │  ├─ removeNotification(id)
  │  └─ clearAll()
  │
  ├─ ThemeContext (Global)
  │  ├─ theme: "light" | "dark"
  │  ├─ toggleTheme()
  │  └─ setTheme(theme)
  │
  └─ Component Local States
     │
     ├─ LoginPage
     │  ├─ formData: { email, password, rememberMe }
     │  ├─ errors: { email, password }
     │  ├─ loading: boolean
     │  └─ error: string | null
     │
     ├─ RegisterPage
     │  ├─ formData: { firstName, lastName, email, password, confirmPassword, acceptTerms }
     │  ├─ errors: { firstName, lastName, email, password, confirmPassword }
     │  ├─ passwordStrength: 0-5
     │  ├─ loading: boolean
     │  └─ error: string | null
     │
     ├─ DashboardPage
     │  ├─ stats: { totalImages, completedAnalyses, processingAnalyses, totalAreasFound }
     │  ├─ analyses: Array
     │  ├─ loading: boolean
     │  ├─ error: string | null
     │  └─ menuAnchor: HTMLElement | null
     │
     ├─ UploadPage
     │  ├─ uploadedFiles: Array
     │  ├─ dragActive: boolean
     │  ├─ loading: boolean
     │  └─ successDialog: boolean
     │
     ├─ AnalysisResultsPage
     │  ├─ analysis: AnalysisData
     │  ├─ zoom: number (0.5 - 3.0)
     │  ├─ rotation: 0 | 90 | 180 | 270
     │  ├─ compareMode: boolean
     │  ├─ shareDialog: boolean
     │  ├─ loading: boolean
     │  └─ error: string | null
     │
     └─ SettingsPage
        ├─ activeTab: 0-3
        ├─ settings: SettingsState
        ├─ errors: Object
        ├─ loading: boolean
        ├─ deleteConfirmation: boolean
        └─ deleteConfirmationText: string
```

## 🔌 HOOK DEPENDENCY DIAGRAM

```
useAuth
├─ Provides: user, isAuthenticated, login, logout, loading, error
├─ Used By:
│  ├─ LoginPage (login function)
│  ├─ RegisterPage (register function)
│  ├─ DashboardPage (user name, logout)
│  ├─ SettingsPage (user profile data)
│  └─ PrivateRoute (protection check)
└─ Context: AuthContext

useNotification
├─ Provides: success, error, warning, info, showNotification, removeNotification
├─ Used By:
│  ├─ LoginPage (login/error notifications)
│  ├─ RegisterPage (validation/success notifications)
│  ├─ DashboardPage (logout notification)
│  ├─ UploadPage (upload progress/complete)
│  ├─ AnalysisResultsPage (export/share notifications)
│  └─ SettingsPage (save/delete notifications)
└─ Context: NotificationContext

useImageUpload
├─ Provides: uploadImage, isUploading, error
├─ Used By:
│  └─ UploadPage (file upload)
└─ Dependencies: API client

useNavigate
├─ Provides: navigate function
├─ Used By:
│  ├─ LoginPage (to /register, /dashboard)
│  ├─ RegisterPage (to /login, /dashboard)
│  ├─ DashboardPage (to /upload, /analysis/:id)
│  ├─ AnalysisResultsPage (to /dashboard)
│  └─ SettingsPage (to /dashboard)
└─ Dependency: React Router

useParams
├─ Provides: Dynamic route parameters
├─ Used By:
│  └─ AnalysisResultsPage (imageId)
└─ Dependency: React Router
```

---

**Visual Guide Created:** Phase 4 Documentation
**Purpose:** Help developers understand data flows and architecture
**Audience:** Developers, architects, code reviewers
**Status:** Reference documentation for ongoing development
