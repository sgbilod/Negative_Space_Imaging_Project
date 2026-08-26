# Create-Project.ps1
# Project Creation Helper PowerShell Script
# Author: Stephen Bilodeau
# Date: August 2025

# Path to the actual project management scripts
$SCRIPTS_DIR = "C:\Users\sgbil\OneDrive\Desktop\Negative_Space_Imaging_Project\scripts"

function Create-Project {
    param (
        [Parameter(Mandatory = $false)]
        [string]$Name,

        [Parameter(Mandatory = $false)]
        [string]$Template = "default",

        [Parameter(Mandatory = $false)]
        [string]$Description = "",

        [Parameter(Mandatory = $false)]
        [switch]$NoVSCode
    )

    if ([string]::IsNullOrEmpty($Name)) {
        # Launch GUI
        & python "$SCRIPTS_DIR\project_generator.py" gui
    }
    else {
        # Create project with parameters
        $args = @("$SCRIPTS_DIR\project_generator.py", "create", "--name", $Name, "--template", $Template)

        if ($Description) {
            $args += @("--desc", $Description)
        }

        if ($NoVSCode) {
            $args += "--no-vscode"
        }

        & python $args
    }
}

# If the script is being run directly (not imported as a module)
if ($MyInvocation.InvocationName -ne '.') {
    # Process any command-line parameters
    $params = @{}

    if ($args.Length -gt 0) {
        $params["Name"] = $args[0]
    }

    if ($args.Length -gt 1) {
        $params["Template"] = $args[1]
    }

    if ($args.Length -gt 2) {
        $params["Description"] = $args[2]
    }

    # Call the function
    Create-Project @params
}

# Export the function so it can be used in other scripts
Export-ModuleMember -Function Create-Project
