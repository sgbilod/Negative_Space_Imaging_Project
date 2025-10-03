param(
    [string]$ProjectName,
    [string]$TemplateType
)

$basePath = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)"
$projectPath = Join-Path $basePath $ProjectName
New-Item -ItemType Directory -Path $projectPath -Force | Out-Null
Copy-Item -Path "$basePath\Templates\$TemplateType\*" -Destination $projectPath -Recurse -Force
Write-Host "Project '$ProjectName' created from template '$TemplateType'."
Start-Process code $projectPath
