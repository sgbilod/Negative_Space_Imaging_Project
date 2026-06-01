#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Docker Health Check and Initialization Wrapper for Windows/PowerShell

.DESCRIPTION
    PowerShell wrapper script for Docker health checking on Windows systems.
    Provides convenient access to bash and Node.js scripts with automatic
    WSL2 handling, logging, and error reporting.

.PARAMETER Verbose
    Enable verbose output

.PARAMETER ScriptType
    Type of script to run: 'health' or 'init' (default: 'health')

.PARAMETER LogFile
    Custom log file path

.PARAMETER Repair
    Attempt automatic repair of failed services (health check only)

.PARAMETER Timeout
    Connection timeout in seconds (default: 30)

.PARAMETER RetryAttempts
    Number of retry attempts (default: 3)

.EXAMPLE
    .\docker-health-check.ps1 -ScriptType health -Verbose
    .\docker-health-check.ps1 -ScriptType init -LogFile "C:\logs\docker-init.log"
    .\docker-health-check.ps1 -ScriptType health -Repair

.NOTES
    Requires Docker Desktop or Docker installed with WSL2 backend
    PowerShell 5.1 or higher recommended
    Author: DevOps Team
    Date: 2025-10-17
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('health', 'init')]
    [string]$ScriptType = 'health',

    [Parameter(Mandatory = $false)]
    [string]$LogFile,

    [Parameter(Mandatory = $false)]
    [switch]$Repair,

    [Parameter(Mandatory = $false)]
    [int]$Timeout = 30,

    [Parameter(Mandatory = $false)]
    [int]$RetryAttempts = 3,

    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

# ============================================================================
# CONSTANTS
# ============================================================================

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$BashScriptPath = Join-Path $ScriptRoot "docker-health-check.sh"
$NodeScriptPath = Join-Path $ScriptRoot "docker-init.js"
$LogDir = Join-Path $ProjectRoot "logs"

# Colors for output
$Colors = @{
    Red    = 'Red'
    Green  = 'Green'
    Yellow = 'Yellow'
    Cyan   = 'Cyan'
    Blue   = 'Blue'
    Gray   = 'Gray'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

<#
.SYNOPSIS
    Write colored console message
#>
function Write-ColorOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Color
}

<#
.SYNOPSIS
    Write log message to file and console
#>
function Write-LogMessage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [ValidateSet('INFO', 'WARN', 'ERROR', 'OK', 'DEBUG')]
        [string]$Level = 'INFO'
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"

    # Map level to color
    $colorMap = @{
        'INFO'  = 'Blue'
        'WARN'  = 'Yellow'
        'ERROR' = 'Red'
        'OK'    = 'Green'
        'DEBUG' = 'Cyan'
    }

    $color = $colorMap[$Level] ?? 'White'
    Write-ColorOutput $logMessage -Color $color

    # Write to log file if specified
    if ($LogFile) {
        Add-Content -Path $LogFile -Value $logMessage -ErrorAction SilentlyContinue
    }
}

<#
.SYNOPSIS
    Check if Docker Desktop is running
#>
function Test-DockerRunning {
    try {
        $dockerInfo = docker info 2>$null
        return $?
    }
    catch {
        return $false
    }
}

<#
.SYNOPSIS
    Check if bash/WSL is available
#>
function Test-BashAvailable {
    $bashPath = Get-Command bash -ErrorAction SilentlyContinue
    return $null -ne $bashPath
}

<#
.SYNOPSIS
    Check if Node.js is available
#>
function Test-NodeAvailable {
    $nodePath = Get-Command node -ErrorAction SilentlyContinue
    return $null -ne $nodePath
}

<#
.SYNOPSIS
    Ensure log directory exists
#>
function Ensure-LogDirectory {
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        Write-LogMessage "Created log directory: $LogDir" -Level 'INFO'
    }
}

# ============================================================================
# DOCKER HEALTH CHECK EXECUTION
# ============================================================================

<#
.SYNOPSIS
    Run Docker health check script
#>
function Invoke-HealthCheck {
    Write-ColorOutput "`n========================================" -Color Cyan
    Write-ColorOutput "Docker Health Check Script" -Color Cyan
    Write-ColorOutput "========================================`n" -Color Cyan

    # Verify Docker is running
    if (-not (Test-DockerRunning)) {
        Write-LogMessage "Docker daemon is not running" -Level 'ERROR'
        Write-LogMessage "Please start Docker Desktop and try again" -Level 'INFO'
        return 1
    }

    Write-LogMessage "Docker daemon is running" -Level 'OK'

    # Check if bash script exists
    if (-not (Test-Path $BashScriptPath)) {
        Write-LogMessage "Bash script not found: $BashScriptPath" -Level 'ERROR'
        return 1
    }

    # Check if bash is available
    if (-not (Test-BashAvailable)) {
        Write-LogMessage "bash not found. Please install WSL2 or Git Bash" -Level 'ERROR'
        return 1
    }

    # Build command arguments
    $arguments = @()
    if ($Verbose) {
        $arguments += "--verbose"
    }
    if ($Repair) {
        $arguments += "--repair"
    }
    if ($LogFile) {
        $arguments += "--log-file", $LogFile
    }

    # Execute health check
    Write-LogMessage "Executing health check script..." -Level 'INFO'

    try {
        & bash $BashScriptPath @arguments
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-LogMessage "Health check completed successfully" -Level 'OK'
        }
        else {
            Write-LogMessage "Health check completed with issues (exit code: $exitCode)" -Level 'WARN'
        }

        return $exitCode
    }
    catch {
        Write-LogMessage "Failed to execute health check: $_" -Level 'ERROR'
        return 1
    }
}

# ============================================================================
# DOCKER INITIALIZATION EXECUTION
# ============================================================================

<#
.SYNOPSIS
    Run Docker initialization script
#>
function Invoke-Initialization {
    Write-ColorOutput "`n========================================" -Color Cyan
    Write-ColorOutput "Docker Initialization Script" -Color Cyan
    Write-ColorOutput "========================================`n" -Color Cyan

    # Verify Docker is running
    if (-not (Test-DockerRunning)) {
        Write-LogMessage "Docker daemon is not running" -Level 'ERROR'
        Write-LogMessage "Please start Docker Desktop and try again" -Level 'INFO'
        return 1
    }

    Write-LogMessage "Docker daemon is running" -Level 'OK'

    # Check if Node.js script exists
    if (-not (Test-Path $NodeScriptPath)) {
        Write-LogMessage "Node.js script not found: $NodeScriptPath" -Level 'ERROR'
        return 1
    }

    # Check if Node.js is available
    if (-not (Test-NodeAvailable)) {
        Write-LogMessage "Node.js not found. Please install Node.js" -Level 'ERROR'
        return 1
    }

    # Build command arguments
    $arguments = @()
    if ($Verbose) {
        $arguments += "--verbose"
    }
    if ($LogFile) {
        $arguments += "--log-file", $LogFile
    }
    $arguments += "--timeout", $Timeout.ToString()
    $arguments += "--retry-attempts", $RetryAttempts.ToString()

    # Execute initialization
    Write-LogMessage "Executing initialization script..." -Level 'INFO'

    try {
        & node $NodeScriptPath @arguments
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-LogMessage "Initialization completed successfully" -Level 'OK'
        }
        else {
            Write-LogMessage "Initialization completed with issues (exit code: $exitCode)" -Level 'WARN'
        }

        return $exitCode
    }
    catch {
        Write-LogMessage "Failed to execute initialization: $_" -Level 'ERROR'
        return 1
    }
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function Main {
    # Ensure log directory exists
    Ensure-LogDirectory

    # Initialize log file if specified
    if ($LogFile) {
        if (-not (Test-Path (Split-Path $LogFile))) {
            New-Item -ItemType Directory -Path (Split-Path $LogFile) -Force | Out-Null
        }
        Write-LogMessage "Log file: $LogFile" -Level 'INFO'
    }

    # Write startup information
    Write-LogMessage "Docker Health Check & Init Wrapper Started" -Level 'INFO'
    Write-LogMessage "Script Type: $ScriptType" -Level 'DEBUG'
    Write-LogMessage "Verbose: $Verbose" -Level 'DEBUG'

    # Execute appropriate script
    $exitCode = 0

    switch ($ScriptType) {
        'health' {
            $exitCode = Invoke-HealthCheck
        }
        'init' {
            $exitCode = Invoke-Initialization
        }
        default {
            Write-LogMessage "Unknown script type: $ScriptType" -Level 'ERROR'
            $exitCode = 1
        }
    }

    Write-LogMessage "Execution completed with exit code: $exitCode" -Level 'INFO'
    exit $exitCode
}

# Execute main function
Main
