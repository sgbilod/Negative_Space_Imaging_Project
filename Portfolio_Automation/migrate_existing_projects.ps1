param(
    [string]$SourceFolder = ".."
)

$projects = Get-ChildItem -Path $SourceFolder -Directory
foreach ($project in $projects) {
    $src = Join-Path $project.FullName "src"
    $tests = Join-Path $project.FullName "tests"
    $docs = Join-Path $project.FullName "docs"
    if (-not (Test-Path $src)) { New-Item -ItemType Directory -Path $src }
    if (-not (Test-Path $tests)) { New-Item -ItemType Directory -Path $tests }
    if (-not (Test-Path $docs)) { New-Item -ItemType Directory -Path $docs }
    # Move files to correct folders (customize as needed)
    # Initialize git if missing
    if (-not (Test-Path (Join-Path $project.FullName ".git"))) {
        git init $project.FullName
    }
    # Add author info, update dashboard, push to GitHub (manual steps)
}
Write-Host "Migration complete. Review each project for manual updates."
