# 📚 MASTER PROMPTS INDEX
**Complete Guide to All Master Prompts - Ready for Copilot Chat**

---

## ⚡ QUICK START

Each master prompt is in its own file. **No searching needed!**

1. **Open** the file for the week you need
2. **Copy** the entire prompt (already formatted)
3. **Paste** into GitHub Copilot Chat
4. **Request** code generation (request is included in each file)

---

## 📋 ALL MASTER PROMPTS

### 🐍 WEEK 1: PYTHON CORE ENGINE

**File:** `MASTER_PROMPT_01_PYTHON_CORE_ENGINE.md`

**Generates:** 6 Python files (~1,100 lines)
- Analyzer with edge detection algorithms
- Contour analysis and confidence scoring
- Image utilities and data models
- Exception handling

**Execution Dates:** November 11-15, 2025

**Time to Generate:** 5-10 minutes

**How to Use:**
```
1. Open: MASTER_PROMPT_01_PYTHON_CORE_ENGINE.md
2. Copy the prompt (already between triple backticks)
3. Paste into Copilot Chat
4. Request: "Generate all 6 Python files"
5. Save to: src/python/negative_space/
```

**Deliverables:**
- ✅ `src/python/negative_space/__init__.py`
- ✅ `src/python/negative_space/core/analyzer.py`
- ✅ `src/python/negative_space/core/algorithms.py`
- ✅ `src/python/negative_space/core/models.py`
- ✅ `src/python/negative_space/utils/image_utils.py`
- ✅ `src/python/negative_space/exceptions.py`

---

### 🚀 WEEK 2: EXPRESS API LAYER (2 Files)

#### 📦 Part A: Database & Configuration

**File:** `MASTER_PROMPT_02A_EXPRESS_API_CONFIG.md`

**Generates:** Files 1-9 (~1,500 lines)
- Database configuration with Sequelize
- User, Image, and AnalysisResult models
- JWT authentication middleware
- Validation and error handling middleware
- Request logging

**Execution Dates:** November 18-19, 2025

**Time to Generate:** 10-15 minutes

**How to Use:**
```
1. Open: MASTER_PROMPT_02A_EXPRESS_API_CONFIG.md
2. Copy the prompt
3. Paste into Copilot Chat
4. Request: "Generate files 1-9"
5. Save to: src/config/ and src/models/ and src/middleware/
```

---

#### 🛣️ Part B: Services & Routes

**File:** `MASTER_PROMPT_02B_EXPRESS_API_ROUTES.md`

**Generates:** Files 10-22 (~2,000 lines)
- User, Image, and Analysis services
- Authentication, user, images, and analysis routes
- File utilities and Python bridge
- Token management and app setup

**Execution Dates:** November 20, 2025

**Time to Generate:** 10-15 minutes

**How to Use:**
```
1. Open: MASTER_PROMPT_02B_EXPRESS_API_ROUTES.md
2. Copy the prompt
3. Paste into Copilot Chat
4. Request: "Generate files 10-22"
5. Save to: src/services/, src/routes/, src/utils/, src/
```

**Prerequisites:** Complete Part A first

---

### ⚛️ WEEK 3-4: REACT FRONTEND (2 Files)

#### 🎨 Part A: Pages & Components

**File:** `MASTER_PROMPT_03A_REACT_FRONTEND_PAGES.md`

**Generates:** Files 1-20 (~2,500 lines)
- Main and Auth layouts
- 11 pages (Login, Register, Dashboard, Images, Upload, Analysis, Profile, etc.)
- 8 reusable components (Button, Input, Modal, Alert, Spinner, Navbar, Cards)

**Execution Dates:** December 2-6, 2025

**Time to Generate:** 15-20 minutes

**How to Use:**
```
1. Open: MASTER_PROMPT_03A_REACT_FRONTEND_PAGES.md
2. Copy the prompt
3. Paste into Copilot Chat
4. Request: "Generate files 1-20"
5. Save to: src/frontend/layouts/, src/frontend/pages/, src/frontend/components/
```

---

#### 🪝 Part B: Hooks & Services

**File:** `MASTER_PROMPT_03B_REACT_FRONTEND_HOOKS.md`

**Generates:** Files 21-41 (~2,500 lines)
- 6 custom hooks (useAuth, useApi, useForm, usePagination, useLocalStorage, useNotification)
- 2 context providers (Auth, Notification)
- 4 service files (auth, image, analysis, user)
- Utilities, styling, and app entry point

**Execution Dates:** December 6-13, 2025

**Time to Generate:** 15-20 minutes

**How to Use:**
```
1. Open: MASTER_PROMPT_03B_REACT_FRONTEND_HOOKS.md
2. Copy the prompt
3. Paste into Copilot Chat
4. Request: "Generate files 21-41"
5. Save to: src/frontend/hooks/, src/frontend/context/, src/frontend/services/, etc.
```

**Prerequisites:** Complete Part A first

---

### 🗄️ WEEK 5: DATABASE & TESTING

**File:** `MASTER_PROMPT_04_DATABASE_TESTING.md`

**Generates:** 13 files (~2,000 lines)
- 3 PostgreSQL migrations (users, images, analysis_results)
- 1 TypeScript seed script with sample data
- 3 Jest test suites (unit & integration)
- 3+ pytest test suites (Python unit tests)
- Configuration files (jest.config.js, pytest.ini)

**Execution Dates:** December 9-13, 2025

**Time to Generate:** 15-20 minutes

**How to Use:**
```
1. Open: MASTER_PROMPT_04_DATABASE_TESTING.md
2. Copy the prompt
3. Paste into Copilot Chat
4. Request: "Generate all 13 files now"
5. Save to: src/migrations/, src/seeds/, __tests__/, src/python/tests/
```

**Deliverables:**
- ✅ 3 SQL migration files
- ✅ 1 TypeScript seed script
- ✅ 3 Jest test files
- ✅ 3+ pytest test files
- ✅ Configuration files

---

## 🎯 EXECUTION TIMELINE

```
Week 1 (Nov 11-15)
├── Monday 9 AM: Open MASTER_PROMPT_01
├── Generate: 6 Python files
├── Test: Import and basic functionality
└── Commit: Phase 2 Week 1 complete

Week 2 (Nov 18-22)
├── Monday-Tuesday: Open MASTER_PROMPT_02A
├── Generate: Files 1-9 (Config & Models)
├── Wednesday: Open MASTER_PROMPT_02B
├── Generate: Files 10-22 (Services & Routes)
├── Test: Express API endpoints
└── Commit: Phase 2 Week 2 complete

Weeks 3-4 (Dec 2-13)
├── Week 3: Open MASTER_PROMPT_03A
├── Generate: Files 1-20 (Pages & Components)
├── Week 3-4: Open MASTER_PROMPT_03B
├── Generate: Files 21-41 (Hooks & Services)
├── Test: React frontend loads and navigation works
└── Commit: Phase 2 Weeks 3-4 complete

Week 5 (Dec 9-13)
├── Monday-Tuesday: Open MASTER_PROMPT_04
├── Generate: 13 files (Migrations & Tests)
├── Run migrations and seed data
├── Run all tests: npm test && pytest
├── Test coverage: >= 60%
└── Commit: Phase 2 Week 5 complete
```

---

## 💾 FILE LOCATIONS

All files are in the project root directory. No nested folders needed:

```
Project Root/
├── MASTER_PROMPT_01_PYTHON_CORE_ENGINE.md
├── MASTER_PROMPT_02A_EXPRESS_API_CONFIG.md
├── MASTER_PROMPT_02B_EXPRESS_API_ROUTES.md
├── MASTER_PROMPT_03A_REACT_FRONTEND_PAGES.md
├── MASTER_PROMPT_03B_REACT_FRONTEND_HOOKS.md
├── MASTER_PROMPT_04_DATABASE_TESTING.md
└── (This file)
```

**Backup Locations:**
- All prompts also embedded in: `PHASE_2_EXECUTION_GUIDE.md`
- Quick reference: `MASTER_PROMPTS_QUICK_ACCESS.md`

---

## ✅ SUCCESS CHECKLIST

### Per Week:

**Week 1:**
- [ ] All 6 Python files generated
- [ ] Imports resolve: `from negative_space import NegativeSpaceAnalyzer`
- [ ] Basic test runs successfully
- [ ] Committed and pushed

**Week 2 (Part A):**
- [ ] Files 1-9 generated
- [ ] TypeScript compiles: `npm run build`
- [ ] No errors
- [ ] Committed

**Week 2 (Part B):**
- [ ] Files 10-22 generated
- [ ] All 22 files compile together
- [ ] Test endpoints work
- [ ] Committed and pushed

**Weeks 3-4 (Part A):**
- [ ] Files 1-20 generated
- [ ] React renders
- [ ] No console errors
- [ ] Committed

**Weeks 3-4 (Part B):**
- [ ] Files 21-41 generated
- [ ] Complete app builds: `npm run build`
- [ ] App loads in browser
- [ ] All navigation works
- [ ] Committed and pushed

**Week 5:**
- [ ] All 13 files generated
- [ ] Migrations apply: `npm run migrate`
- [ ] Seed runs: `npm run seed:dev`
- [ ] Tests pass: `npm test && pytest`
- [ ] Coverage >= 60%
- [ ] Committed and pushed

---

## 🚀 PHASE 2 COMPLETION CRITERIA

All 10 criteria must be met:

1. **✅ User Authentication** - Login/register working with JWT
2. **✅ Image Upload** - Upload, validation, storage working
3. **✅ Python Analysis** - Negative space detection complete
4. **✅ Results Storage** - Analysis saved to database
5. **✅ Frontend Display** - Results visualized on UI
6. **✅ End-to-End Workflow** - Complete flow works (login → upload → analyze → display)
7. **✅ API Testing** - All endpoints tested and working
8. **✅ Database** - Schema, migrations, indexes correct
9. **✅ Test Coverage** - 60%+ coverage, all tests pass
10. **✅ Docker** - Services build and run in containers

---

## 🎓 USAGE TIPS

### Copy Entire Prompt:
```
1. Open the .md file
2. Look for triple backticks: ```
3. Select everything between the backticks
4. Copy: Ctrl+C
5. Paste into Copilot Chat: Ctrl+V
```

### Generate Code:
```
1. Paste the prompt
2. Let Copilot understand it
3. Type the request (already shown in each file)
4. Wait for generation
5. Copy generated code
6. Save to correct directory
```

### Verify Generation:
```
1. Check file count matches prompt
2. Check line counts are reasonable
3. Verify imports work
4. Check for syntax errors
5. Commit and push
```

### If Generation Fails:
```
1. Try again - sometimes Copilot needs a second attempt
2. Ask for files in smaller groups
3. Review the file structure examples in the prompt
4. Make sure you copied the ENTIRE prompt
5. Check Copilot is connected to GitHub account
```

---

## 📞 REFERENCE DOCUMENTS

**Also Available:**
- `PHASE_2_EXECUTION_GUIDE.md` - Full guide with all prompts embedded
- `PHASE_2_LAUNCH_DASHBOARD.md` - Week 1 launch dashboard
- `COMPLETE_DOCUMENTATION_INDEX.md` - Full documentation index

---

## 🎉 YOU'RE READY!

All 6 master prompts are ready to use. Each one is self-contained, properly formatted, and ready to copy-paste into Copilot Chat.

**No more searching. No more hunting through documents.**

Just:
1. **Open** the file for your week
2. **Copy** the prompt
3. **Paste** into Copilot
4. **Generate** code
5. **Done!**

---

**Status:** ✅ **ALL MASTER PROMPTS READY**

**Created:** November 8, 2025

**Total Prompts:** 6 files (Python, Express A, Express B, React A, React B, Database & Tests)

**Total Code to Generate:** ~10,600 lines across 70+ files

**Execution Timeline:** November 11 - December 13, 2025

**Next Action:** Open `MASTER_PROMPT_01_PYTHON_CORE_ENGINE.md` on Monday, Nov 11 at 9 AM

🚀 **Let's build this MVP!**
