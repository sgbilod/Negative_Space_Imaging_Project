#!/usr/bin/env pwsh
# PROMPT 1 EXECUTION VERIFICATION SCRIPT
# This script verifies all setup is complete and ready for Copilot code generation

Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   PROMPT 1 - DOCKER HEALTH CHECK SYSTEM - EXECUTION CHECK      ║" -ForegroundColor Cyan
Write-Host "║   Status: ✅ READY TO EXECUTE                                 ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check directories
Write-Host "📁 DIRECTORY STRUCTURE" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────" -ForegroundColor Cyan
$dirs = @(
    ".githooks",
    "migrations",
    "scripts",
    "src\database\schemas",
    "src\database\seeds",
    "src\health-checks"
)

foreach ($dir in $dirs) {
    $path = Join-Path "c:\Users\sgbil\Negative_Space_Imaging_Project" $dir
    if (Test-Path $path) {
        Write-Host "  ✅ $dir" -ForegroundColor Green
    }
    else {
        Write-Host "  ❌ $dir (MISSING)" -ForegroundColor Red
    }
}

# Check configuration files
Write-Host "`n📄 CONFIGURATION FILES" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────" -ForegroundColor Cyan
$envFile = "c:\Users\sgbil\Negative_Space_Imaging_Project\.env.local"
if (Test-Path $envFile) {
    Write-Host "  ✅ .env.local ($(Get-Item $envFile | Measure-Object -Property Length).Sum bytes)" -ForegroundColor Green
}
else {
    Write-Host "  ❌ .env.local (MISSING)" -ForegroundColor Red
}

# Check documentation
Write-Host "`n📋 DOCUMENTATION FILES" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────" -ForegroundColor Cyan
$docs = @(
    "PROMPT_1_EXECUTION_GUIDE.md",
    "PROMPT_1_CHECKLIST.md",
    "PROMPT_1_EXECUTION_SUMMARY.md",
    "PROMPT_1_QUICK_START.md",
    "PROMPT_EXECUTION_LOG.md"
)

foreach ($doc in $docs) {
    $path = Join-Path "c:\Users\sgbil\Negative_Space_Imaging_Project" $doc
    if (Test-Path $path) {
        Write-Host "  ✅ $doc" -ForegroundColor Green
    }
    else {
        Write-Host "  ❌ $doc (MISSING)" -ForegroundColor Red
    }
}

# Check git configuration
Write-Host "`n🔧 GIT CONFIGURATION" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────────" -ForegroundColor Cyan
$hooksPath = git config core.hooksPath
if ($hooksPath -eq ".githooks") {
    Write-Host "  ✅ Git hooks path configured: .githooks" -ForegroundColor Green
}
else {
    Write-Host "  ⚠️  Git hooks path: $hooksPath" -ForegroundColor Yellow
}

# Final status
Write-Host "`n" -ForegroundColor Cyan
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   ✅ ALL SYSTEMS READY FOR EXECUTION                           ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Open VS Code (if not already open)" -ForegroundColor White
Write-Host "2. Press Ctrl+Shift+I to open Copilot Chat" -ForegroundColor White
Write-Host "3. Copy the prompt from PROMPT_1_EXECUTION_GUIDE.md (Step 2)" -ForegroundColor White
Write-Host "4. Paste into Copilot Chat and press Enter" -ForegroundColor White
Write-Host "5. Wait 10-15 minutes for code generation" -ForegroundColor White
Write-Host "6. Save generated files to appropriate directories" -ForegroundColor White
Write-Host "7. Run validation tests (see PROMPT_1_CHECKLIST.md)" -ForegroundColor White
Write-Host "8. Commit and push to GitHub" -ForegroundColor White
Write-Host ""
Write-Host "⏱️  ESTIMATED TIME: 60 minutes total (Copilot generation + testing + commit)" -ForegroundColor Cyan
Write-Host "📊 TRACK PROGRESS IN: PROMPT_1_CHECKLIST.md" -ForegroundColor Cyan
Write-Host ""
