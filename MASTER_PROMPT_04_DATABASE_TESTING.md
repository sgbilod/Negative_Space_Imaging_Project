# 🗄️ MASTER PROMPT 04: DATABASE & TESTING
**Week 5 Execution | Dec 9-13, 2025**

**Copy this entire prompt and paste into GitHub Copilot Chat, then request:**
```
"Generate all 13 files now (migrations, seed script, and test suites)"
```

---

## DETAILED PROMPT FOR COPILOT

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

## 📋 EXECUTION CHECKLIST

After Copilot generates the files:

- [ ] All 13 files generated successfully
- [ ] Files saved to correct directories
- [ ] SQL migrations valid and syntactically correct
- [ ] Seed script runs without errors
- [ ] Jest tests compile and run
- [ ] Python tests run with pytest
- [ ] Coverage >= 60%
- [ ] All tests pass
- [ ] Files committed to git

## 📍 FILE STRUCTURE

```
src/
├── migrations/
│   ├── 001_create_users_table.sql
│   ├── 002_create_images_table.sql
│   └── 003_create_analysis_results_table.sql
└── seeds/
    └── seed_development.ts

__tests__/
├── unit/
│   └── services/
│       ├── userService.test.ts
│       └── imageService.test.ts
└── integration/
    ├── auth.integration.test.ts
    ├── images.integration.test.ts
    └── analysis.integration.test.ts

src/python/
└── tests/
    ├── test_analyzer.py
    └── test_algorithms.py

jest.config.js
pytest.ini
```

## ✅ RUNNING TESTS

After generating all 13 files:

```bash
# Run Python tests
cd src/python
pytest tests/ --cov

# Run JavaScript tests
npm test -- --coverage

# Verify coverage > 60%
npm test -- --coverage | grep -E "Statements|Branches|Functions|Lines"

# Verify pytest coverage > 60%
cd src/python && pytest tests/ --cov=negative_space --cov-report=term-missing
```

## 🔄 FINAL INTEGRATION

After completing all 4 Master Prompts:
1. Setup PostgreSQL database
2. Run migrations: `npm run migrate`
3. Seed development data: `npm run seed:dev`
4. Run all tests: `npm test && pytest`
5. Start all services: `docker-compose up`
6. Verify end-to-end workflow

---

## ✅ PHASE 2 SUCCESS CRITERIA

Verify all 10 criteria before considering Phase 2 complete:

### 1. User Authentication ✓
- [ ] Users can sign up with email/password
- [ ] Users can log in with credentials
- [ ] JWT token generated on login
- [ ] Token persists across page reload
- [ ] Users can log out

### 2. Image Upload ✓
- [ ] Upload dialog opens from frontend
- [ ] File validation (type, size) working
- [ ] Progress bar shows during upload
- [ ] Image stored in filesystem
- [ ] Image record created in database

### 3. Python Analysis ✓
- [ ] Analysis starts automatically after upload
- [ ] Python engine processes image
- [ ] Negative space regions detected
- [ ] Results contain confidence scores and bounding boxes

### 4. Results Storage ✓
- [ ] Analysis results saved to database
- [ ] Results retrievable by image_id
- [ ] Results include processing time
- [ ] Results include visualization data

### 5. Frontend Display ✓
- [ ] Results page loads
- [ ] Original image displays
- [ ] Negative space visualization shows
- [ ] Statistics displayed

### 6. End-to-End Workflow ✓
- [ ] User logs in
- [ ] User uploads image
- [ ] Python analyzes
- [ ] Results display within 10 seconds
- [ ] All steps complete without errors

### 7. API Testing ✓
- [ ] All endpoints respond correctly
- [ ] Error handling returns proper status codes
- [ ] Authorization checked on protected routes
- [ ] Input validation works

### 8. Database ✓
- [ ] PostgreSQL schema created
- [ ] Migrations applied
- [ ] Data persists correctly
- [ ] Indexes on key columns

### 9. Test Coverage ✓
- [ ] Unit tests written (60%+ coverage)
- [ ] Integration tests written
- [ ] Tests pass: npm test && pytest
- [ ] Coverage report shows >= 60%

### 10. Docker ✓
- [ ] Docker images build
- [ ] Docker Compose works
- [ ] All services start
- [ ] Services communicate correctly

---

**Status:** Ready for execution
**Created:** November 8, 2025
**Time to Generate:** 15-20 minutes
**Final Week:** Week 5 (Dec 9-13)
**Prerequisites:** Complete Master Prompts 01, 02, and 03 first
