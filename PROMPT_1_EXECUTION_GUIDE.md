# 🎯 PROMPT 1 EXECUTION GUIDE - DOCKER HEALTH CHECK SYSTEM

**Status:** ✅ Ready to Execute  
**Estimated Duration:** 60 minutes total (45 min Copilot + 15 min testing)  
**Parallel Tasks:** 6 optimization tasks included  

---

## 📋 WHAT PROMPT 1 GENERATES

### Deliverables
- Docker health check system (Bash & PowerShell)
- Service monitoring (Node.js module)
- Python health monitoring module
- Interactive HTML dashboard
- Automated alerting configuration
- Complete documentation

### Services Monitored
- ✅ PostgreSQL database (port 5432)
- ✅ Redis cache (port 6379)
- ✅ Node.js backend (port 3000)
- ✅ Python AI/ML service (port 5000)
- ✅ React frontend (port 3001)

---

## 🚀 QUICK START (5 Steps)

### Step 1: Open Copilot Chat
```
1. Open VS Code
2. Press: Ctrl+Shift+I
3. Copilot Chat panel opens
```

### Step 2: Copy the Prompt
Copy this exact prompt into Copilot:

```
Create a comprehensive Docker health check system for a production multi-service application.

Requirements:
1. Health check scripts (PowerShell and Bash)
2. Monitor all 5 services (PostgreSQL, Redis, Node.js, Python, React)
3. Database connectivity verification
4. API endpoint validation
5. Resource usage monitoring (CPU, memory)
6. Container restart detection
7. Automated alerting
8. HTML dashboard generation
9. JSON report export
10. Graceful degradation

Deliverables:
- docker-health-check.sh (Bash)
- docker-health-check.ps1 (PowerShell)
- health-check.js (Node.js monitoring)
- health_monitor.py (Python monitoring)
- health-dashboard.html (Visual dashboard)
- Complete README

Ensure:
- Full error handling
- Type safety
- Production-ready code
- Well-commented
- RESTful endpoints
```

### Step 3: Paste into Copilot and Execute
- Click in Copilot Chat input
- Paste prompt
- Press Enter
- Wait 10-15 minutes

### Step 4: Save Generated Files
Create these files from Copilot output:
- `scripts/docker-health-check.sh`
- `scripts/docker-health-check.ps1`
- `src/health-checks/health-check.js`
- `src/health-checks/health_monitor.py`
- `public/health-dashboard.html`

### Step 5: Test & Commit
```bash
# Make executable
chmod +x scripts/docker-health-check.sh

# Test
bash scripts/docker-health-check.sh

# Commit
git add scripts/ src/health-checks/ public/
git commit -m "feat(monitoring): add Docker health check system"
git push origin main
```

---

## 🔄 PARALLEL TASKS (While Copilot Generates)

Execute these 6 tasks simultaneously while Copilot is generating code:

### Task A: GitHub Projects Board (15 min)
Create visual tracking dashboard:
1. Go to: https://github.com/sgbilod/Negative_Space_Imaging_Project/projects
2. Click "New project"
3. Create table with columns: Prompt #, Status, Duration, Files, Tests, Committed
4. Add rows for all 17 prompts
5. Mark Prompt 1 as "In Progress"

**Benefit:** Track progress across all prompts

### Task B: Environment Variables (10 min)
Create `.env.local` with:
```bash
DB_USER=nsi_admin
DB_PASSWORD=secure_password_change_me
DATABASE_URL=postgresql://nsi_admin:secure_password_change_me@localhost:5432/negative_space
REDIS_URL=redis://localhost:6379
NODE_ENV=development
JWT_SECRET=dev_secret_change_me
PYTHON_SERVICE_URL=http://localhost:5000
```

**Benefit:** Ready to run docker-compose immediately

### Task C: Docker Service Test (20 min)
Verify all services start:
```bash
docker-compose up -d
docker-compose ps
curl http://localhost:3000/health
curl http://localhost:5000/health
docker-compose down
```

**Benefit:** Validate infrastructure works

### Task D: Execution Logging (10 min)
Create `PROMPT_EXECUTION_LOG.md` documenting:
- Prompt number and name
- Date and duration
- Generated files
- Test status
- Commit hash

**Benefit:** Audit trail of all prompt executions

### Task E: Prepare Prompt 2 (10 min)
Create directories for next prompt:
```bash
mkdir -p migrations
mkdir -p src/database/schemas
mkdir -p src/database/seeds
```
Review what Prompt 2 generates

**Benefit:** Seamless transition to Prompt 2

### Task F: Git Workflow Setup (5 min)
Configure git hooks for efficient commits:
```bash
mkdir -p .githooks
# Create prepare-commit-msg hook
git config core.hooksPath .githooks
```

**Benefit:** Faster, more organized commits

---

## ⏱️ TIMELINE

```
Minute 0-2:   Copy Prompt 1 into Copilot
Minute 2-15:  Copilot generates code (MEANWHILE: Execute Tasks A-F)
  - Task A: Projects board (0-15 min)
  - Task B: Environment vars (0-10 min)
  - Task C: Docker test (0-20 min)
  - Task D: Logging (0-10 min)
  - Task E: Prep Prompt 2 (0-10 min)
  - Task F: Git setup (0-5 min)
Minute 15-20: Save generated files
Minute 20-30: Test generated code
Minute 30-35: Commit to GitHub
Minute 35-45: Verify CI/CD pipeline
Minute 45-60: Document results

TOTAL: 60 minutes for Prompt 1 PLUS all parallel setup tasks
```

---

## ✅ SUCCESS CRITERIA

### Code Generation
- [ ] Bash script generates without errors
- [ ] PowerShell script generates without errors
- [ ] Node.js module is valid JavaScript
- [ ] Python module has correct syntax
- [ ] HTML dashboard loads correctly

### Functionality
- [ ] All 5 services detected when running
- [ ] Health endpoints respond correctly
- [ ] Resource metrics collected
- [ ] Dashboard displays all data
- [ ] Alerts configured properly

### Integration
- [ ] Files committed to GitHub
- [ ] CI/CD pipeline passes all checks
- [ ] No merge conflicts
- [ ] All parallel tasks complete

---

## 📚 REFERENCE CODES

Use these for focused discussions:
- `[REF:PROMPT-1-EXEC]` - This execution guide
- `[REF:TASK-A]` through `[REF:TASK-F]` - Individual parallel tasks
- `[REF:PARALLEL-EXEC]` - Parallel execution strategy

---

## 🎉 WHAT'S NEXT

After Prompt 1 completes:
1. **Prompt 2:** Database Schema (same day)
2. **Prompt 3:** Environment Configuration (same day)
3. **Continue:** One prompt every 45 minutes
4. **Goal:** All 17 prompts by end of week

---

**Ready to execute? Open Copilot Chat and paste the prompt above! 🚀**

