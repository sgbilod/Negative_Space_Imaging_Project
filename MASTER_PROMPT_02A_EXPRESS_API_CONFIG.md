# 🚀 MASTER PROMPT 02A: EXPRESS API - DATABASE & CONFIG
**Week 2 Execution (Part A) | Nov 18-19, 2025**

**Copy this entire prompt and paste into GitHub Copilot Chat, then request:**
```
"Generate files 1-9 (database.ts through requestLogger.ts)"
```

---

## DETAILED PROMPT FOR COPILOT

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

## 📋 EXECUTION CHECKLIST

After Copilot generates the files:

- [ ] All 9 files generated successfully
- [ ] Files saved to correct directories
- [ ] No TypeScript compilation errors
- [ ] Database models initialize correctly
- [ ] Middleware chains properly
- [ ] Files committed to git

## 📍 FILE STRUCTURE

```
src/
├── config/
│   └── database.ts
├── models/
│   ├── User.ts
│   ├── Image.ts
│   └── AnalysisResult.ts
├── types/
│   └── index.ts
└── middleware/
    ├── auth.ts
    ├── validation.ts
    ├── errorHandler.ts
    └── requestLogger.ts
```

## 🔄 NEXT: MASTER PROMPT 02B

After completing this prompt:
1. Review all 9 files for correctness
2. Commit changes to git
3. Use **MASTER_PROMPT_02B** for services & routes

---

**Status:** Ready for execution
**Created:** November 8, 2025
**Time to Generate:** 10-15 minutes
**Part:** A of 2 for Week 2
