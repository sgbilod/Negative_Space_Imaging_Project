# 🚀 PHASE 2 EXECUTION GUIDE
## Negative Space Imaging Project - Full Stack Implementation

**Project Status:** Phase 1 Complete ✅ | Phase 2 Planning Complete ✅ | **Phase 2 Ready to Execute** 🚀

**Current Date:** November 8, 2025
**Target Completion:** Week ending December 13, 2025
**Total Planned Output:** ~10,000 lines of production code + ~2,000 lines of tests

---

## 📋 QUICK START CHECKLIST

Before beginning Phase 2 implementation:

- [ ] Read entire guide (this document)
- [ ] Verify all Phase 1 deliverables are committed to `main`
- [ ] Create `phase-2-implementation` branch from `main`
- [ ] Install all required dependencies (see Dependencies section)
- [ ] Verify environment setup (Node.js, Python, PostgreSQL)
- [ ] Open VS Code with Copilot Chat enabled
- [ ] Have the 4 master prompts ready (included below)

---

## 📊 PHASE 2 OVERVIEW

### What is Phase 2?

Phase 2 transforms the NSIP project from 5-10% implementation to 40-50% implementation by creating the functional MVP core:

| Component | Phase 1 | Phase 2 Goal | Tech Stack |
|-----------|---------|-------------|-----------|
| **Python Core Engine** | 0% | 100% Complete | Python 3.10+, OpenCV, NumPy, scikit-image |
| **Express.js API** | 0% | 100% Complete | TypeScript, Express, Sequelize, PostgreSQL |
| **React Frontend** | 0% | 100% Complete | React 18, TypeScript, Tailwind, React Router |
| **Database** | 0% | 100% Complete | PostgreSQL with migrations |
| **Test Coverage** | 0% | 60%+ Complete | Jest + pytest with integration tests |
| **Docker** | Scaffolded | Working Multi-container | Docker Compose for all services |
| **Total Code** | ~500 lines | ~10,000 lines | Production-ready across all languages |

### Strategic Goals

✅ **By end of Phase 2:**
1. Users can sign up, log in, and manage profiles
2. Image upload with validation and storage working
3. Python engine detects negative space in uploaded images
4. Analysis results stored in PostgreSQL
5. React frontend displays results with visualization
6. All endpoints tested (60%+ code coverage)
7. Complete end-to-end workflow: Upload → Analyze → View Results
8. Docker containers working for all services
9. Error handling, security, and accessibility implemented
10. Ready for Phase 3 (Advanced features)

---

## 🗓️ WEEK-BY-WEEK IMPLEMENTATION PLAN

### WEEK 1: Python Core Engine (Mon Nov 11 - Fri Nov 15)

**Master Prompt:** Master Prompt 01 - Python Core Engine
**Deliverables:** 6 Python files, ~1,100 lines
**Time Estimate:** 3-4 hours coding + 1-2 hours testing

#### Mon: Execution
- [ ] Copy Master Prompt 01 from below
- [ ] Open VS Code → Copilot Chat
- [ ] Paste entire prompt into chat
- [ ] Request generation of all 6 files
- [ ] Save files to `src/python/negative_space/`

#### Files to Generate:
1. `__init__.py` - Package initialization
2. `core/analyzer.py` - Main analysis engine (400 lines)
3. `core/algorithms.py` - Detection algorithms (300 lines)
4. `core/models.py` - Data structures with Pydantic (150 lines)
5. `utils/image_utils.py` - Image handling utilities (200 lines)
6. `exceptions.py` - Custom exceptions (50 lines)

#### Tue-Wed: Testing
- [ ] Verify all imports work: `python -c "from negative_space import NegativeSpaceAnalyzer"`
- [ ] Create test image: `python create_test_image.py`
- [ ] Run basic analysis: `python -c "analyzer = NegativeSpaceAnalyzer(); result = analyzer.analyze('test.jpg')"`
- [ ] Fix any import or dependency issues

#### Thu: Integration
- [ ] Plan FastAPI endpoints needed for Python integration
- [ ] Document expected API request/response format
- [ ] Create simple test to verify analysis output structure

#### Fri: Validation
- [ ] [ ] All imports resolve ✓
- [ ] [ ] Analysis runs without errors ✓
- [ ] [ ] Output structure validated ✓
- [ ] [ ] Documentation complete ✓
- [ ] [ ] Commit to branch: `git commit -m "feat(python): add negative space core engine"`

---

### WEEK 2: Express.js API Layer (Mon Nov 18 - Fri Nov 22)

**Master Prompt:** Master Prompt 02 - Express API Layer
**Deliverables:** 22 TypeScript files, ~3,500 lines
**Time Estimate:** 5-6 hours coding + 2-3 hours testing

#### Mon-Tue: Models & Configuration
- [ ] Copy Master Prompt 02 - PART A from below
- [ ] Generate files 1-9 (configs, models, middleware)
- [ ] Save to `src/`
- [ ] Verify TypeScript compilation: `npm run build`

#### Wed: Services & Routes
- [ ] Copy Master Prompt 02 - PART B from below
- [ ] Generate files 10-21 (services, routes, app setup)
- [ ] Save to `src/`
- [ ] Verify no TypeScript errors

#### Thu: Database Integration
- [ ] Run database migrations (scripts provided)
- [ ] Seed sample data
- [ ] Test database connection

#### Fri: API Testing
- [ ] [ ] Start server: `npm run dev`
- [ ] [ ] Test auth endpoints with Postman/Thunder Client
- [ ] [ ] Test image endpoints
- [ ] [ ] Verify error handling
- [ ] [ ] Commit: `git commit -m "feat(api): add Express.js API layer with all endpoints"`

---

### WEEK 3-4: React Frontend (Mon Dec 2 - Fri Dec 13)

**Master Prompt:** Master Prompt 03 - React Frontend
**Deliverables:** 41 React/TypeScript files, ~5,000 lines
**Time Estimate:** 8-10 hours coding + 2-3 hours testing

#### Week 3 - Pages & Layouts
- [ ] Copy Master Prompt 03 - PART A from below
- [ ] Generate files 1-20 (layouts, pages, components)
- [ ] Save to `src/frontend/`

#### Week 3 - Hooks & Context
- [ ] Copy Master Prompt 03 - PART B from below
- [ ] Generate files 21-34 (hooks, services, utilities)
- [ ] Save to `src/frontend/`

#### Week 4 - App Integration
- [ ] Generate remaining files (35-41)
- [ ] Verify all routes work
- [ ] Test authentication flow
- [ ] Test image upload workflow

#### Week 4 - End-to-End Testing
- [ ] [ ] Complete auth flow works (signup → login → profile)
- [ ] [ ] Image upload triggers analysis
- [ ] [ ] Results display properly
- [ ] [ ] Responsive on mobile/tablet/desktop
- [ ] [ ] Accessibility checks pass
- [ ] [ ] Commit: `git commit -m "feat(frontend): add React.js UI with complete user interface"`

---

### WEEK 5: Database, Tests & Deployment (Mon Dec 9 - Fri Dec 13)

**Master Prompt:** Master Prompt 04 - Database & Testing
**Deliverables:** Migrations, test suites, ~2,000 lines
**Time Estimate:** 4-5 hours

#### Mon-Tue: Database & Tests
- [ ] Copy Master Prompt 04 from below
- [ ] Generate database migrations
- [ ] Generate test suites (unit + integration)
- [ ] Run tests: `npm test` and `pytest`

#### Wed: Coverage Analysis
- [ ] [ ] Run coverage report: `npm run test:coverage`
- [ ] [ ] Verify 60%+ coverage achieved
- [ ] [ ] Add missing tests if needed

#### Thu-Fri: Final Validation
- [ ] [ ] All 10 success criteria met (see below)
- [ ] [ ] Full end-to-end workflow tested
- [ ] [ ] Docker builds successfully
- [ ] [ ] Commit & push: `git push origin phase-2-implementation`
- [ ] [ ] Create pull request on GitHub

---

## 📦 DEPENDENCIES INSTALLATION

Before starting Phase 2, install all required packages:

### Python Dependencies
```bash
cd src/python
pip install -r requirements.txt
# Should include:
# - numpy>=1.24.0
# - opencv-python>=4.8.0
# - scikit-image>=0.22.0
# - pydantic>=2.0.0
# - fastapi>=0.104.0
```

### Node.js Dependencies
```bash
# Run from project root
npm install

# Should already have:
# - typescript, express, sequelize, postgresql
# - tailwind, react, react-router-dom
# - axios, jest, zod, react-hook-form
```

### Database Setup
```bash
# Start PostgreSQL
# On Windows (if using PostgreSQL service)
# On Mac: brew services start postgresql
# On Linux: sudo systemctl start postgresql

# Create development database
createdb negative_space_imaging_dev

# Verify connection
psql -U postgres -d negative_space_imaging_dev -c "SELECT version();"
```

---

## 🎯 MASTER PROMPTS (Ready to Copy & Paste)

### MASTER PROMPT 01: PYTHON CORE ENGINE
**Location in Original Request:** See "🎯 MASTER PROMPT 01: PYTHON CORE ENGINE" section
**Copy:** Everything between triple backticks under "DETAILED MASTER PROMPT FOR COPILOT:"

**Quick Access:**
1. Open this document in VS Code
2. Search for "TASK: Create a comprehensive, production-ready Python module"
3. Copy from there to end of first master prompt section
4. Paste into Copilot Chat

**Execution in Copilot:**
Paste the entire prompt below into Copilot Chat and request: "Generate all 6 Python files"

---

## 📋 MASTER PROMPT 01 CONTENT (Copy Everything Below)

```
TASK: Create a comprehensive, production-ready Python module for negative space detection in images.

REQUIREMENTS:
- Create 6 Python files with type hints throughout
- Implement edge detection algorithms (Canny, Sobel)
- Implement contour analysis and filtering
- Implement confidence scoring system
- Generate JSON reports for analysis results
- Include comprehensive docstrings and error handling
- Use Pydantic for data validation

FILES TO GENERATE:

1. src/python/negative_space/__init__.py
   - Package initialization
   - Export main NegativeSpaceAnalyzer class
   - Version info

2. src/python/negative_space/core/analyzer.py (400 lines)
   - Class: NegativeSpaceAnalyzer
   - Methods:
     * __init__(self, config: dict = None) -> None
     * analyze(self, image_path: str) -> AnalysisResult
     * analyze_bytes(self, image_data: bytes) -> AnalysisResult
     * batch_analyze(self, image_paths: List[str]) -> List[AnalysisResult]
   - Uses algorithms and image_utils internally
   - Returns AnalysisResult objects

3. src/python/negative_space/core/algorithms.py (300 lines)
   - Function: detect_edges(image: np.ndarray, method: str = 'canny') -> np.ndarray
   - Function: find_contours(edges: np.ndarray) -> List[np.ndarray]
   - Function: filter_contours(contours: List[np.ndarray], min_area: int = 100) -> List[np.ndarray]
   - Function: calculate_confidence(contour: np.ndarray, image: np.ndarray) -> float
   - Function: extract_bounding_boxes(contours: List[np.ndarray]) -> List[Dict]
   - All with proper type hints and documentation

4. src/python/negative_space/core/models.py (150 lines)
   - Pydantic models for data validation
   - Model: ContourData
   - Model: AnalysisResult
   - Model: ConfigModel
   - Include JSON serialization methods

5. src/python/negative_space/utils/image_utils.py (200 lines)
   - Function: load_image(path: str) -> np.ndarray
   - Function: load_image_from_bytes(data: bytes) -> np.ndarray
   - Function: resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray
   - Function: convert_to_grayscale(image: np.ndarray) -> np.ndarray
   - Function: enhance_contrast(image: np.ndarray) -> np.ndarray
   - Function: save_visualization(image: np.ndarray, contours: List, path: str) -> None

6. src/python/negative_space/exceptions.py (50 lines)
   - CustomException classes:
     * NegativeSpaceError (base)
     * ImageLoadError
     * AnalysisError
     * ValidationError

IMPLEMENTATION DETAILS:
- Use OpenCV for image processing
- Use NumPy for numerical operations
- Use scikit-image for advanced image processing
- Use Pydantic for data validation
- Include proper error handling and logging
- Add comprehensive docstrings
- Type hint all function parameters and returns
- Include examples in docstrings

TESTING REQUIREMENTS:
- Each file should have a __name__ == '__main__' block with basic tests
- Create a simple test analysis on sample image
- Print success messages for each component

QUALITY STANDARDS:
- All functions documented with docstrings
- All type hints present
- Error handling for edge cases
- Logging statements for debugging
- JSON serializable output
- PEP 8 compliant

STRUCTURE EXAMPLE:
```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import numpy as np
import cv2
from loggers import logger

class NegativeSpaceAnalyzer:
    """Analyzes images to detect negative space regions."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize analyzer with optional config."""
        self.config = config or {}
        logger.info("NegativeSpaceAnalyzer initialized")

    def analyze(self, image_path: str) -> AnalysisResult:
        """Analyze image at path."""
        # Implementation here
        pass
```

Generate all 6 files now.
```

---

### MASTER PROMPT 02: EXPRESS API LAYER
**Location in Original Request:** See "🎯 MASTER PROMPT 02: EXPRESS API LAYER" section
**Copy:** Everything between triple backticks under "DETAILED MASTER PROMPT FOR COPILOT:"

**Quick Access - Part A (Models & Config):**
Request from Copilot after pasting:
```
"Generate files 1-9 (database.ts through requestLogger.ts)"
```

**Quick Access - Part B (Services & Routes):**
Request from Copilot:
```
"Generate files 10-22 (userService.ts through server.ts)"
```

**Note:** Express prompt is large; you may need to break it into 2 Copilot requests

---

## 📋 MASTER PROMPT 02A CONTENT (Copy for Part A)

```
TASK: Create the backend API infrastructure for the Negative Space Imaging Project using Express.js and TypeScript.

PART A: Database Configuration, Models, Middleware

REQUIREMENTS:
- Use Express.js with TypeScript
- Use Sequelize ORM for PostgreSQL
- Implement JWT authentication
- Implement input validation with Joi
- Include proper error handling
- Type-safe throughout

FILES TO GENERATE (Part A - 1-9):

1. src/config/database.ts (120 lines)
   - Sequelize configuration
   - Connection setup for PostgreSQL
   - Environment-based configuration
   - Connection pooling

2. src/models/User.ts (80 lines)
   - Sequelize model for users table
   - Fields: id, email, password_hash, first_name, last_name, created_at, updated_at
   - Add associations

3. src/models/Image.ts (100 lines)
   - Sequelize model for images table
   - Fields: id, user_id, filename, original_filename, file_size, storage_path, uploaded_at
   - Associations with User and AnalysisResult

4. src/models/AnalysisResult.ts (120 lines)
   - Sequelize model for analysis results
   - Fields: id, image_id, negative_space_percentage, regions_count, processing_time_ms, raw_data, created_at
   - Methods to serialize for API response

5. src/types/index.ts (80 lines)
   - TypeScript interfaces
   - Interface: AuthRequest (extends Express Request)
   - Interface: AuthPayload (JWT payload)
   - Interface: ApiResponse (generic)

6. src/middleware/auth.ts (80 lines)
   - JWT verification middleware
   - Function: verifyToken (Express middleware)
   - Function: requireAuth (Express middleware)
   - Error handling for expired tokens

7. src/middleware/validation.ts (100 lines)
   - Input validation middleware using Joi
   - Function: validateRequest (Express middleware factory)
   - Joi schemas for common inputs

8. src/middleware/errorHandler.ts (120 lines)
   - Global error handling middleware
   - Function: errorHandler (Express middleware)
   - Proper error responses with status codes
   - Logging of errors

9. src/middleware/requestLogger.ts (60 lines)
   - HTTP request logging middleware
   - Log: method, path, status, response time
   - Use Morgan or custom implementation

STRUCTURE EXAMPLE:
```typescript
import { DataTypes, Model, Sequelize } from 'sequelize';

export class User extends Model {
  public id!: number;
  public email!: string;
  public password_hash!: string;
  public first_name!: string;
  public last_name!: string;
  public created_at!: Date;
  public updated_at!: Date;
}

export function initUserModel(sequelize: Sequelize): typeof User {
  User.init({
    id: { type: DataTypes.INTEGER, autoIncrement: true, primaryKey: true },
    email: { type: DataTypes.STRING, unique: true, allowNull: false },
    password_hash: { type: DataTypes.STRING, allowNull: false },
    first_name: { type: DataTypes.STRING, allowNull: false },
    last_name: { type: DataTypes.STRING, allowNull: false },
  }, { sequelize, tableName: 'users' });

  return User;
}
```

Generate files 1-9 now.
```

---

## 📋 MASTER PROMPT 02B CONTENT (Copy for Part B - Services & Routes)

```
TASK: Create Express.js services and API routes for Negative Space Imaging Project.

PART B: Services and API Routes (continue from Part A)

FILES TO GENERATE (Part B - 10-22):

10. src/services/userService.ts (150 lines)
    - Class: UserService
    - Methods:
      * registerUser(email, password, first_name, last_name): Promise<User>
      * loginUser(email, password): Promise<{user, token}>
      * getUserById(id): Promise<User>
      * updateUser(id, data): Promise<User>
      * deleteUser(id): Promise<void>
    - Password hashing using bcrypt
    - JWT token generation

11. src/services/imageService.ts (200 lines)
    - Class: ImageService
    - Methods:
      * uploadImage(user_id, file): Promise<Image>
      * getImagesByUserId(user_id, pagination): Promise<Image[]>
      * getImageById(id): Promise<Image>
      * deleteImage(id): Promise<void>
      * getImagePath(id): Promise<string>
    - File system integration
    - Database record creation

12. src/services/analysisService.ts (250 lines)
    - Class: AnalysisService
    - Methods:
      * startAnalysis(image_id, image_path): Promise<AnalysisResult>
      * getAnalysisResult(id): Promise<AnalysisResult>
      * getAnalysesByImageId(image_id): Promise<AnalysisResult[]>
    - Bridge to Python analysis engine
    - Store results in database
    - Error handling for analysis failures

13. src/routes/auth.ts (120 lines)
    - POST /auth/register - User registration
    - POST /auth/login - User login
    - POST /auth/logout - User logout
    - POST /auth/refresh - Refresh JWT token
    - Input validation and error handling

14. src/routes/users.ts (100 lines)
    - GET /users/:id - Get user profile
    - PUT /users/:id - Update user profile
    - DELETE /users/:id - Delete user account
    - GET /users/:id/statistics - Get user analysis statistics
    - Require authentication

15. src/routes/images.ts (200 lines)
    - GET /images - List user's images (paginated)
    - GET /images/:id - Get image details
    - POST /images - Upload new image (multipart/form-data)
    - DELETE /images/:id - Delete image
    - POST /images/:id/analyze - Trigger analysis
    - File upload handling with validation

16. src/routes/analysis.ts (200 lines)
    - GET /analysis/:id - Get specific analysis result
    - GET /images/:id/analysis - Get all analyses for image
    - GET /analysis/status/:id - Check analysis status
    - POST /analysis/batch - Batch analysis
    - WebSocket support for real-time status

17. src/utils/fileUtils.ts (150 lines)
    - Function: saveUploadedFile(file, destination): Promise<string>
    - Function: deleteFile(path): Promise<void>
    - Function: validateFileType(mimetype): boolean
    - Function: validateFileSize(size, maxSize): boolean
    - File validation and storage

18. src/utils/pythonBridge.ts (120 lines)
    - Function: callPythonAnalysis(imagePath): Promise<AnalysisData>
    - Function: spawnPythonProcess(args): Promise<string>
    - Subprocess handling
    - Error handling for Python service

19. src/utils/tokenUtils.ts (80 lines)
    - Function: generateToken(payload): string
    - Function: verifyToken(token): AuthPayload
    - Function: refreshToken(token): string

20. src/utils/index.ts (80 lines)
    - Utility exports
    - Common helper functions

21. src/app.ts (100 lines)
    - Express app configuration
    - Register all middleware
    - Register all routes
    - Error handling setup
    - CORS configuration

22. src/server.ts (50 lines)
    - Server startup
    - Port configuration
    - Database initialization
    - Server listening

STRUCTURE EXAMPLE:
```typescript
import { Request, Response, NextFunction } from 'express';
import { User } from '../models/User';

export class UserService {
  async registerUser(email: string, password: string, first_name: string, last_name: string): Promise<any> {
    // Hash password
    const hash = await bcrypt.hash(password, 10);

    // Create user
    const user = await User.create({
      email,
      password_hash: hash,
      first_name,
      last_name
    });

    // Generate token
    const token = generateToken({ userId: user.id });

    return { user: user.toJSON(), token };
  }
}

// In routes
router.post('/register', async (req: Request, res: Response) => {
  const { email, password, first_name, last_name } = req.body;

  // Validate input
  // Call service
  // Return response
});
```

Generate files 10-22 now.
```

---

### MASTER PROMPT 03: REACT FRONTEND
**Location in Original Request:** See "🎯 MASTER PROMPT 03: REACT FRONTEND" section
**Copy:** Everything between triple backticks under "DETAILED MASTER PROMPT FOR COPILOT:"

**Quick Access - Part A (Pages & Layouts):**
Request from Copilot:
```
"Generate files 1-20 (MainLayout.tsx through AnalysisResultsCard.tsx)"
```

**Quick Access - Part B (Hooks & Services):**
Request from Copilot:
```
"Generate files 21-41 (useAuth.ts through globals.css including app root)"
```

**Note:** React prompt is largest; plan for 3-4 Copilot requests

---

## 📋 MASTER PROMPT 03A CONTENT (Copy for Part A - Pages & Layouts)

```
TASK: Create React 18 + TypeScript frontend for Negative Space Imaging Project.

PART A: Layouts, Pages, and Components (Base Setup)

REQUIREMENTS:
- Use React 18 with TypeScript
- Use Tailwind CSS for styling
- Use React Router v6 for navigation
- Responsive mobile-first design
- Accessibility (WCAG 2.1 AA)
- Component-based architecture

FILES TO GENERATE (Part A - 1-20):

1. src/frontend/layouts/MainLayout.tsx (100 lines)
   - Main app layout component
   - Navbar, sidebar, main content area
   - User menu with logout
   - Navigation links

2. src/frontend/layouts/AuthLayout.tsx (80 lines)
   - Authentication pages layout
   - Centered form layout
   - No navigation bar

3. src/frontend/pages/LoginPage.tsx (100 lines)
   - Email and password inputs
   - Login button with loading state
   - Link to register page
   - Error message display
   - Form validation with Zod

4. src/frontend/pages/RegisterPage.tsx (120 lines)
   - Email, password, confirm password, first name, last name inputs
   - Register button with loading state
   - Link to login page
   - Password strength indicator
   - Form validation

5. src/frontend/pages/DashboardPage.tsx (100 lines)
   - User welcome message
   - Statistics cards (total images, total analyses, etc.)
   - Recent images grid
   - Quick actions (upload, view all)
   - Responsive layout

6. src/frontend/pages/ImagesPage.tsx (150 lines)
   - Gallery of user's uploaded images
   - Pagination support
   - Search/filter functionality
   - Image cards with thumbnails
   - Load more button

7. src/frontend/pages/ImageDetailPage.tsx (120 lines)
   - Full image display
   - Image metadata
   - Analysis results section
   - Delete button
   - Navigate to analysis results

8. src/frontend/pages/UploadPage.tsx (120 lines)
   - Drag-and-drop file upload area
   - File input with validation
   - Progress bar during upload
   - Preview of selected image
   - Upload button with loading state

9. src/frontend/pages/AnalysisPage.tsx (150 lines)
   - Display analysis results
   - Original image
   - Visualization with negative space highlighted
   - Statistics and metrics
   - Download results button

10. src/frontend/pages/ProfilePage.tsx (100 lines)
    - User profile information
    - Edit profile form
    - Password change form
    - Account deletion option
    - Save changes button

11. src/frontend/pages/NotFoundPage.tsx (50 lines)
    - 404 error page
    - Link back to home

12. src/frontend/components/Button.tsx (40 lines)
    - Reusable button component
    - Variants: primary, secondary, danger
    - Sizes: sm, md, lg
    - Loading state
    - Disabled state

13. src/frontend/components/Input.tsx (50 lines)
    - Reusable input component
    - Text, email, password types
    - Label and error message
    - Validation styling
    - Accessibility attributes

14. src/frontend/components/Modal.tsx (80 lines)
    - Reusable modal component
    - Title, body, footer
    - Close button
    - Overlay click to close
    - Keyboard escape to close

15. src/frontend/components/Alert.tsx (60 lines)
    - Alert message component
    - Types: success, error, warning, info
    - Dismissible
    - Icon support

16. src/frontend/components/Spinner.tsx (40 lines)
    - Loading spinner component
    - Sizes: sm, md, lg
    - Used in buttons, pages

17. src/frontend/components/Navbar.tsx (100 lines)
    - Top navigation bar
    - Logo and brand name
    - Navigation links
    - User menu dropdown
    - Mobile hamburger menu

18. src/frontend/components/ImageCard.tsx (80 lines)
    - Card displaying image thumbnail
    - Image name and upload date
    - File size
    - Delete button
    - Click to view details

19. src/frontend/components/AnalysisResultsCard.tsx (100 lines)
    - Card displaying analysis results
    - Key metrics
    - Small visualization preview
    - View full results link

20. src/frontend/components/ProtectedRoute.tsx (60 lines)
    - Route wrapper for authentication
    - Redirect to login if not authenticated
    - Require specific roles if needed

STRUCTURE EXAMPLE:
```typescript
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '@/components/Button';
import Input from '@/components/Input';

export const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Login logic
      navigate('/dashboard');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto">
      <form onSubmit={handleSubmit}>
        <Input
          type="email"
          label="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Input
          type="password"
          label="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <Button type="submit" disabled={loading}>
          {loading ? 'Logging in...' : 'Login'}
        </Button>
      </form>
    </div>
  );
};
```

Generate files 1-20 now.
```

---

## 📋 MASTER PROMPT 03B CONTENT (Copy for Part B - Hooks, Services, Config)

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

### MASTER PROMPT 04: DATABASE & TESTING
**Location in Original Request:** See "🎯 MASTER PROMPT 04: DATABASE & TESTING" section
**Copy:** Everything between triple backticks under "DETAILED MASTER PROMPT FOR COPILOT:"

**Quick Access - Migrations:**
Request from Copilot:
```
"Generate files 1-4 (PostgreSQL migrations and seed script)"
```

**Quick Access - Tests:**
Request from Copilot:
```
"Generate files 5-7 (Jest and pytest test suites with 60%+ coverage)"
```

---

## 📋 MASTER PROMPT 04 CONTENT (Copy Complete)

```
TASK: Create PostgreSQL database migrations, seed data, and comprehensive test suites for Negative Space Imaging Project.

REQUIREMENTS:
- Create PostgreSQL migration files
- Create seed data for development
- Create comprehensive Jest tests for Node/React
- Create pytest test suite for Python
- Achieve 60%+ code coverage
- Include unit and integration tests

FILES TO GENERATE:

1. src/migrations/001_create_users_table.sql (50 lines)
   SQL Migration:
   - CREATE TABLE users
   - Columns: id (PK), email (unique), password_hash, first_name, last_name, created_at, updated_at
   - Indexes: email unique index
   - Constraints: NOT NULL, default timestamps

2. src/migrations/002_create_images_table.sql (60 lines)
   SQL Migration:
   - CREATE TABLE images
   - Columns: id (PK), user_id (FK), filename, original_filename, file_size, storage_path, uploaded_at, created_at
   - Foreign key: user_id → users.id
   - Indexes: user_id index
   - Cascade delete on user delete

3. src/migrations/003_create_analysis_results_table.sql (70 lines)
   SQL Migration:
   - CREATE TABLE analysis_results
   - Columns: id (PK), image_id (FK), negative_space_percentage, regions_count, processing_time_ms, raw_data (JSONB), created_at
   - Foreign key: image_id → images.id
   - Indexes: image_id index
   - Cascade delete on image delete

4. src/seeds/seed_development.ts (100 lines)
   TypeScript seed script:
   - Create 3-5 test users
   - Create 2-3 test images for each user
   - Create analysis results for each image
   - Use factory pattern for consistency
   - Run: npm run seed:dev

5. __tests__/unit/services/userService.test.ts (150 lines)
   Jest test suite:
   - Test: registerUser with valid data
   - Test: registerUser with existing email
   - Test: loginUser with correct password
   - Test: loginUser with wrong password
   - Test: getUserById returns correct user
   - Test: error handling for database failures
   - Mock database calls

6. __tests__/unit/services/imageService.test.ts (150 lines)
   Jest test suite:
   - Test: uploadImage stores file and creates record
   - Test: getImagesByUserId returns user's images with pagination
   - Test: getImageById returns correct image
   - Test: deleteImage removes file and record
   - Test: error handling for missing files
   - Mock file system and database

7. __tests__/integration/auth.integration.test.ts (200 lines)
   Jest integration test:
   - Test: Full registration flow (register → login → access protected route)
   - Test: JWT token validation
   - Test: Token refresh
   - Test: Logout clears token
   - Test: Unauthorized access denied
   - Use test database
   - Setup/teardown database for each test

8. __tests__/integration/images.integration.test.ts (200 lines)
   Jest integration test:
   - Test: Upload image flow (auth → upload → verify storage)
   - Test: List images with pagination
   - Test: Get image details
   - Test: Delete image removes from storage
   - Test: Error handling for oversized files
   - Use test database and temp storage

9. __tests__/integration/analysis.integration.test.ts (200 lines)
   Jest integration test:
   - Test: Start analysis (trigger Python processing)
   - Test: Get analysis results
   - Test: Batch analysis
   - Test: Analysis error handling
   - Mock Python service
   - Use test database

10. src/python/tests/test_analyzer.py (150 lines)
    Pytest unit tests:
    - Test: NegativeSpaceAnalyzer initialization
    - Test: analyze() with valid image
    - Test: analyze_bytes() with image bytes
    - Test: batch_analyze() with multiple images
    - Test: Error handling for missing files
    - Test: Error handling for invalid image formats
    - Fixtures for test images

11. src/python/tests/test_algorithms.py (150 lines)
    Pytest unit tests:
    - Test: detect_edges with Canny method
    - Test: detect_edges with Sobel method
    - Test: find_contours returns correct count
    - Test: filter_contours removes small contours
    - Test: calculate_confidence returns 0-1 value
    - Test: extract_bounding_boxes format validation
    - Fixtures for test images

12. jest.config.js (50 lines)
    Jest configuration:
    - Path aliases matching tsconfig
    - Test environment: node
    - Coverage thresholds: 60% global
    - Setup files
    - Transform configuration

13. pytest.ini (30 lines)
    Pytest configuration:
    - Test discovery patterns
    - Coverage thresholds: 60%
    - Markers for unit/integration tests

COVERAGE TARGETS:
- Services: 90%+
- Routes: 80%+
- Utils: 85%+
- Models: 70%+
- Global minimum: 60%+

STRUCTURE EXAMPLES:

SQL Migration:
```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
```

TypeScript Test:
```typescript
import { UserService } from '@/services/userService';
import { User } from '@/models/User';

describe('UserService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('registerUser', () => {
    it('should register user with valid data', async () => {
      const service = new UserService();
      const result = await service.registerUser(
        'test@test.com',
        'password',
        'John',
        'Doe'
      );

      expect(result.user.email).toBe('test@test.com');
      expect(result.token).toBeDefined();
    });

    it('should throw error on duplicate email', async () => {
      // Test duplicate handling
    });
  });
});
```

Python Test:
```python
import pytest
from negative_space.core.analyzer import NegativeSpaceAnalyzer

@pytest.fixture
def analyzer():
    return NegativeSpaceAnalyzer()

@pytest.fixture
def test_image_path(tmp_path):
    # Create test image
    return str(tmp_path / "test.jpg")

def test_analyze_valid_image(analyzer, test_image_path):
    result = analyzer.analyze(test_image_path)
    assert result is not None
    assert result.negative_space_percentage >= 0
    assert result.negative_space_percentage <= 100
```

Generate all 13 files now.
```

---

## 🔍 INTEGRATION CHECKPOINTS

After each master prompt completion, verify:

### After Master Prompt 01 (Python)
```bash
# Test imports
python -c "from negative_space.core.analyzer import NegativeSpaceAnalyzer"
python -c "from negative_space.core.models import AnalysisResult"

# Test basic analysis
python -c "
from negative_space.core.analyzer import NegativeSpaceAnalyzer
analyzer = NegativeSpaceAnalyzer()
print('✓ Python core engine initialized successfully')
"
```

### After Master Prompt 02 (Express API)
```bash
# Verify TypeScript compilation
npm run build

# Start server (test mode)
npm run dev &
sleep 2

# Test auth endpoint
curl -X POST http://localhost:3001/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"TestPass123","first_name":"Test"}'

# Stop server
pkill -f "npm run dev"
```

### After Master Prompt 03 (React)
```bash
# Verify React builds
npm run build

# Start dev server
npm run dev &

# Check in browser: http://localhost:5173
# Verify: login page loads, navigation works, no console errors

# Stop server
pkill -f "npm run dev"
```

### After Master Prompt 04 (Tests)
```bash
# Run Python tests
cd src/python && pytest tests/ --cov

# Run JavaScript tests
npm test -- --coverage

# Verify coverage > 60%
```

---

## ✅ PHASE 2 SUCCESS CRITERIA

Verify all 10 criteria before considering Phase 2 complete:

### 1. User Authentication ✓
```
- [ ] Users can sign up with email/password
- [ ] Users can log in with credentials
- [ ] JWT token generated on login
- [ ] Token persists across page reload
- [ ] Users can log out
```

### 2. Image Upload ✓
```
- [ ] Upload dialog opens from frontend
- [ ] File validation (type, size) working
- [ ] Progress bar shows during upload
- [ ] Image stored in filesystem
- [ ] Image record created in database
```

### 3. Python Analysis ✓
```
- [ ] Analysis starts automatically after upload
- [ ] Python engine processes image
- [ ] Negative space regions detected
- [ ] Results contain confidence scores and bounding boxes
```

### 4. Results Storage ✓
```
- [ ] Analysis results saved to database
- [ ] Results retrievable by image_id
- [ ] Results include processing time
- [ ] Results include visualization data
```

### 5. Frontend Display ✓
```
- [ ] Results page loads
- [ ] Original image displays
- [ ] Negative space visualization shows
- [ ] Statistics displayed (regions count, percentages, etc.)
```

### 6. End-to-End Workflow ✓
```
- [ ] User logs in
- [ ] User uploads image
- [ ] Python analyzes
- [ ] Results display within 10 seconds
- [ ] All steps complete without errors
```

### 7. API Testing ✓
```
- [ ] All endpoints respond correctly
- [ ] Error handling returns proper status codes
- [ ] Authorization checked on protected routes
- [ ] Input validation works
```

### 8. Database ✓
```
- [ ] PostgreSQL schema created
- [ ] Migrations applied
- [ ] Data persists correctly
- [ ] Indexes on key columns
```

### 9. Test Coverage ✓
```
- [ ] Unit tests written (60%+ coverage)
- [ ] Integration tests written
- [ ] Tests pass: npm test && pytest
- [ ] Coverage report shows >= 60%
```

### 10. Docker ✓
```
- [ ] Docker images build: docker build .
- [ ] Docker Compose works: docker-compose up
- [ ] All services start (Python, Express, React, PostgreSQL)
- [ ] Services communicate correctly
```

---

## 🐛 COMMON ISSUES & SOLUTIONS

### Issue: "Module not found" errors in Python
**Solution:**
```bash
cd src/python
pip install -e .
# Or add src/python to PYTHONPATH
```

### Issue: TypeScript compilation errors
**Solution:**
```bash
# Clear node_modules and rebuild
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Issue: PostgreSQL connection refused
**Solution:**
```bash
# Check if PostgreSQL is running
psql -U postgres

# If not, start it:
# On Windows: net start PostgreSQL14 (or your version)
# On Mac: brew services start postgresql
# On Linux: sudo systemctl start postgresql
```

### Issue: Port already in use
**Solution:**
```bash
# Find and kill process using port 3001
lsof -i :3001
kill -9 <PID>

# Or use different port
PORT=3002 npm run dev
```

### Issue: Tests failing
**Solution:**
```bash
# Run with verbose output
npm test -- --verbose

# Check if database exists for test
createdb negative_space_imaging_test

# Run specific test
npm test -- specific.test.ts
```

---

## 📚 FILE ORGANIZATION REFERENCE

After Phase 2 completion, your directory structure should look like:

```
Negative_Space_Imaging_Project/
├── src/
│   ├── python/
│   │   ├── negative_space/
│   │   │   ├── __init__.py
│   │   │   ├── core/
│   │   │   │   ├── analyzer.py          (400 lines)
│   │   │   │   ├── algorithms.py        (300 lines)
│   │   │   │   └── models.py            (150 lines)
│   │   │   ├── utils/
│   │   │   │   └── image_utils.py       (200 lines)
│   │   │   └── exceptions.py            (50 lines)
│   │   ├── tests/
│   │   │   ├── test_analyzer.py
│   │   │   └── test_algorithms.py
│   │   └── requirements.txt
│   │
│   ├── config/
│   │   ├── database.ts                  (120 lines)
│   │   ├── auth.ts                      (80 lines)
│   │   └── storage.ts                   (100 lines)
│   │
│   ├── models/
│   │   ├── User.ts                      (80 lines)
│   │   ├── Image.ts                     (100 lines)
│   │   └── AnalysisResult.ts            (120 lines)
│   │
│   ├── middleware/
│   │   ├── auth.ts                      (80 lines)
│   │   ├── validation.ts                (100 lines)
│   │   ├── errorHandler.ts              (120 lines)
│   │   └── requestLogger.ts             (60 lines)
│   │
│   ├── services/
│   │   ├── userService.ts               (150 lines)
│   │   ├── imageService.ts              (200 lines)
│   │   └── analysisService.ts           (250 lines)
│   │
│   ├── routes/
│   │   ├── auth.ts                      (120 lines)
│   │   ├── users.ts                     (100 lines)
│   │   ├── images.ts                    (200 lines)
│   │   └── analysis.ts                  (200 lines)
│   │
│   ├── utils/
│   │   ├── fileUtils.ts                 (150 lines)
│   │   ├── pythonBridge.ts              (120 lines)
│   │   └── index.ts                     (80 lines)
│   │
│   ├── types/
│   │   └── index.ts                     (80 lines)
│   │
│   ├── app.ts                           (100 lines)
│   ├── server.ts                        (50 lines)
│   │
│   ├── frontend/
│   │   ├── layouts/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── context/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── styles/
│   │   ├── App.tsx
│   │   └── main.tsx
│   │
│   └── migrations/
│       ├── 001_create_users_table.sql
│       ├── 002_create_images_table.sql
│       └── 003_create_analysis_results_table.sql
│
├── __tests__/
│   ├── unit/
│   │   ├── services/
│   │   │   ├── userService.test.ts
│   │   │   ├── imageService.test.ts
│   │   │   └── analysisService.test.ts
│   │   └── ...
│   │
│   └── integration/
│       ├── auth.integration.test.ts
│       ├── images.integration.test.ts
│       ├── analysis.integration.test.ts
│       └── ...
│
├── public/
├── logs/
├── uploads/
│
├── .env.local                           (Development config)
├── .env.production                      (Production config)
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.python
├── tsconfig.json
├── jest.config.js
├── vite.config.ts
│
├── PHASE_2_EXECUTION_GUIDE.md          (This file)
└── README.md
```

---

## 🚀 GETTING STARTED TODAY

### Action Items (Next 30 minutes):

1. **Read** this entire guide (you're here! ✓)
2. **Create branch:**
   ```bash
   git checkout -b phase-2-implementation
   ```
3. **Verify Python setup:**
   ```bash
   python --version  # Should be 3.10+
   pip --version
   ```
4. **Verify Node.js setup:**
   ```bash
   node --version    # Should be 18+
   npm --version
   ```
5. **Verify PostgreSQL:**
   ```bash
   psql --version
   createdb negative_space_imaging_dev
   ```
6. **Open Copilot Chat:**
   - Open VS Code
   - Click Copilot Chat icon (left sidebar)
   - Verify connection working

### You're Ready to Begin!

Next step: [Copy Master Prompt 01](#master-prompt-01-python-core-engine) and start Week 1!

---

## 📞 QUICK REFERENCE

| Week | Focus | Files | Lines | Status |
|------|-------|-------|-------|--------|
| 1 | Python Core | 6 | 1,100 | 🟢 Ready |
| 2 | Express API | 22 | 3,500 | 🟢 Ready |
| 3-4 | React Frontend | 41 | 5,000 | 🟢 Ready |
| 5 | Database & Tests | - | 2,000 | 🟢 Ready |
| **TOTAL** | **Full Stack MVP** | **70** | **~11,600** | **🟢 READY** |

---

## 🎉 THE BIG PICTURE

By the end of Phase 2, you'll have:

✅ **Full-Stack Application** with Python backend, Express API, and React frontend
✅ **Core Functionality** for image analysis working end-to-end
✅ **Production-Ready Code** with error handling, security, and tests
✅ **MVP** ready for Phase 3 (advanced features, performance optimization)
✅ **Team-Ready** documentation and code structure

**Estimated Total Time:** 35-45 hours across 5 weeks
**Quality Target:** Enterprise-grade, fully typed, comprehensive testing
**Deployment:** Docker containerized, ready for cloud deployment

---

**Let's build something amazing! 🚀**

You're about to transform this project from planning to a fully functional negative space imaging system. The master prompts are crafted, the path is clear, and Copilot is ready to help you code.

Good luck, and enjoy the building! 💪
