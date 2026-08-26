# 🎯 PHASE 2A QUICK REFERENCE - EXPRESS API INFRASTRUCTURE

**Status:** ✅ COMPLETE | **Date:** November 8, 2025 | **Files:** 9 | **Lines:** 860

---

## ⚡ QUICK FACTS

| Item | Value |
|------|-------|
| **Files Generated** | 9 TypeScript files |
| **Total Code Lines** | 860 lines |
| **Architecture** | Layered (Config→Models→Types→Middleware) |
| **Database** | PostgreSQL with Sequelize ORM |
| **Authentication** | JWT with token expiration |
| **Validation** | Joi schemas with factory pattern |
| **Error Handling** | Global middleware with typed errors |
| **Logging** | Winston + Request tracking with IDs |
| **Type Safety** | 100% TypeScript coverage |

---

## 📂 FILES GENERATED

```
src/
├── config/
│   └── database.ts ..................... 120 lines (DB setup & pooling)
├── models/
│   ├── User.ts ........................ 80 lines (User table schema)
│   ├── Image.ts ....................... 100 lines (Image storage schema)
│   └── AnalysisResult.ts .............. 120 lines (Analysis data schema)
├── types/
│   └── index.ts ....................... 80 lines (TypeScript interfaces)
└── middleware/
    ├── auth.ts ........................ 80 lines (JWT verification)
    ├── validation.ts .................. 100 lines (Joi validation)
    ├── errorHandler.ts ................ 120 lines (Error processing)
    └── requestLogger.ts ............... 60 lines (HTTP logging)
```

---

## 🔌 KEY FUNCTIONS AVAILABLE

### Database (config/database.ts)
- `initializeDatabase()` - Setup Sequelize connection
- `testConnection()` - Verify DB connectivity
- `syncDatabase(force)` - Sync models to schema
- `closeConnection()` - Cleanup and close
- `getSequelize()` - Get Sequelize instance

### User Model (models/User.ts)
- `getFullName()` - Return user's full name
- `toJSON()` - Serialize without password (security)

### Image Model (models/Image.ts)
- `getImageInfo()` - Get image metadata
- `getStorageUrl()` - Get CDN/storage path

### Analysis Model (models/AnalysisResult.ts)
- `serialize()` - API response format
- `getSummary()` - Quick stats
- `isQualityAcceptable(threshold)` - Quality check

### Auth Middleware (middleware/auth.ts)
- `verifyToken()` - Verify JWT from header
- `requireAuth()` - Enforce authentication
- `generateToken(payload)` - Create JWT
- `optionalAuth()` - Optional verification
- `requireRole(...roles)` - Role-based access

### Validation Middleware (middleware/validation.ts)
- `validateBody(schema)` - Validate request body
- `validateQuery(schema)` - Validate query params
- `validateParams(schema)` - Validate URL params
- `validate(body, query, params)` - Combined validation
- Pre-built schemas: userRegister, userLogin, imageUpload, analysisResult, pagination

### Error Handler (middleware/errorHandler.ts)
- `errorHandler()` - Global error processor (use as last middleware)
- `asyncHandler(fn)` - Wrap async routes for error catching
- `notFoundHandler()` - Handle 404 routes
- `getSafeErrorMessage(error)` - Sanitize errors for production

### Request Logger (middleware/requestLogger.ts)
- `requestLogger()` - Log all requests with timing
- `verboseRequestLogger()` - Dev mode detailed logging
- `performanceTracker(threshold)` - Identify slow requests

---

## 🛠️ COMMON USAGE PATTERNS

### Setting Up Database
```typescript
import { initializeDatabase, testConnection, getSequelize } from './config/database';

// In your main server file
const sequelize = initializeDatabase();
await testConnection();
await sequelize.sync();
```

### Using Models
```typescript
import { User, initUserModel } from './models/User';
import { Image, initImageModel } from './models/Image';
import { AnalysisResult, initAnalysisResultModel } from './models/AnalysisResult';

const sequelize = getSequelize();
initUserModel(sequelize);
initImageModel(sequelize);
initAnalysisResultModel(sequelize);
```

### Setting Up Middleware Chain
```typescript
import express from 'express';
import { requestLogger, verboseRequestLogger, performanceTracker } from './middleware/requestLogger';
import { optionalAuth, requireAuth } from './middleware/auth';
import { validate, validationSchemas } from './middleware/validation';
import { errorHandler, notFoundHandler, asyncHandler } from './middleware/errorHandler';

const app = express();

// Logging (first)
app.use(requestLogger);
app.use(verboseRequestLogger);
app.use(performanceTracker(1000)); // 1 second threshold

// Parsing
app.use(express.json());

// Validation
app.post('/auth/register', validate(validationSchemas.userRegister), asyncHandler(registerHandler));

// Auth (optional or required)
app.use(optionalAuth); // For routes that might need it

// Routes
app.use('/api', routes);

// 404 (before error handler)
app.use(notFoundHandler);

// Error handling (last)
app.use(errorHandler);
```

### Protected Routes with Validation
```typescript
import { verifyToken, generateToken } from './middleware/auth';
import { validate, validationSchemas } from './middleware/validation';
import { asyncHandler } from './middleware/errorHandler';
import { AuthRequest, ApiResponse } from './types';

router.post(
  '/login',
  validate(validationSchemas.userLogin),
  asyncHandler(async (req: AuthRequest, res) => {
    // Your login logic
    const token = generateToken({ id: user.id, email: user.email });
    const response: ApiResponse = {
      success: true,
      message: 'Login successful',
      data: { token },
      timestamp: new Date(),
    };
    res.json(response);
  })
);

router.get(
  '/profile',
  verifyToken,
  asyncHandler(async (req: AuthRequest, res) => {
    // User authenticated, req.user is available
    res.json({
      success: true,
      data: req.user,
      timestamp: new Date(),
    });
  })
);
```

---

## 📊 DATABASE SCHEMA

### Users Table
```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  first_name VARCHAR(255) NOT NULL,
  last_name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

### Images Table
```sql
CREATE TABLE images (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  filename VARCHAR(255) UNIQUE NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  file_size INTEGER NOT NULL,
  storage_path VARCHAR(255) NOT NULL,
  uploaded_at TIMESTAMP DEFAULT NOW(),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

### Analysis Results Table
```sql
CREATE TABLE analysis_results (
  id SERIAL PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  negative_space_percentage FLOAT NOT NULL CHECK (negative_space_percentage >= 0 AND negative_space_percentage <= 100),
  regions_count INTEGER NOT NULL CHECK (regions_count >= 0),
  processing_time_ms FLOAT NOT NULL,
  raw_data JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## ✅ VERIFICATION CHECKLIST

Before moving to Phase 2B, ensure:

- [ ] All 9 files present in correct directories
- [ ] Run `npm run build` - no TypeScript errors
- [ ] Environment variables configured (.env):
  - `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
  - `JWT_SECRET`, `JWT_EXPIRES_IN`
  - `NODE_ENV` (development/production)
  - `LOG_LEVEL` (debug/info/warn/error)
- [ ] PostgreSQL database created and accessible
- [ ] Dependencies installed: `npm install`
- [ ] Code formatted: `npm run format`
- [ ] Linting passes: `npm run lint`

---

## 🚀 NEXT PHASE (2B)

Ready for Phase 2B which will generate:
1. Route handlers (auth, users, images, analysis)
2. Service layer (business logic)
3. App setup (middleware configuration)
4. Server entry point

**When ready:** Use MASTER_PROMPT_02B_EXPRESS_SERVICES_ROUTES.md

---

## 📈 PROJECT PROGRESS

```
Phase 1 (Python Core):        ✅ COMPLETE (6 files, 1,292 lines)
Phase 2A (Express Backend):   ✅ COMPLETE (9 files, 860 lines)
Phase 2B (Routes & Services): ⏳ PENDING
Phase 3 (React Frontend):     ⏳ PENDING
Phase 4 (Testing):            ⏳ PENDING
Phase 5 (Deployment):         ⏳ PENDING

Total Code Written:           2,152 lines
```

---

**Generated:** November 8, 2025 | **Project:** Negative Space Imaging Project | **Status:** ✅ Phase 2A Complete
