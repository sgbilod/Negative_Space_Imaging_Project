# 🚀 MASTER PROMPT 02B: EXPRESS API - SERVICES & ROUTES
**Week 2 Execution (Part B) | Nov 20, 2025**

**Copy this entire prompt and paste into GitHub Copilot Chat, then request:**
```
"Generate files 10-22 (userService.ts through server.ts)"
```

---

## DETAILED PROMPT FOR COPILOT

```
TASK: Create Express.js services and API routes for Negative Space Imaging Project.

PART B: Services and API Routes (continue from Part A - Master Prompt 02A)

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

## 📋 EXECUTION CHECKLIST

After Copilot generates the files:

- [ ] All 13 files generated successfully
- [ ] Files saved to correct directories
- [ ] No TypeScript compilation errors
- [ ] Services initialize correctly
- [ ] Routes register properly
- [ ] Server starts without errors
- [ ] Files committed to git

## 📍 FILE STRUCTURE

```
src/
├── services/
│   ├── userService.ts
│   ├── imageService.ts
│   └── analysisService.ts
├── routes/
│   ├── auth.ts
│   ├── users.ts
│   ├── images.ts
│   └── analysis.ts
├── utils/
│   ├── fileUtils.ts
│   ├── pythonBridge.ts
│   ├── tokenUtils.ts
│   └── index.ts
├── app.ts
└── server.ts
```

## ✅ TESTING EXPRESS API

After generating all 22 files (Part A + Part B):

```bash
# Install dependencies
npm install

# Compile TypeScript
npm run build

# Start dev server
npm run dev

# Test endpoints (in another terminal)
curl -X POST http://localhost:3001/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"TestPass123","first_name":"Test","last_name":"User"}'
```

## 🔄 NEXT: MASTER PROMPT 03A

After completing Part A & B of Week 2:
1. Review all 22 files for correctness
2. Test all endpoints work
3. Commit changes to git
4. Use **MASTER_PROMPT_03A** for React frontend

---

**Status:** Ready for execution
**Created:** November 8, 2025
**Time to Generate:** 10-15 minutes
**Part:** B of 2 for Week 2
**Prerequisites:** Complete Master Prompt 02A first
