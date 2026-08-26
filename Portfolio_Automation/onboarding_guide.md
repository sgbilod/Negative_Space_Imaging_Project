# How to Start a New Project

1. Run `new_project.ps1` with your desired project name and template type:
   ```powershell
   .\new_project.ps1 -ProjectName "MyProject" -TemplateType "nextjs"
   ```
2. The script will scaffold your project, initialize git, and open it in VS Code.
3. Add your author info to README.md and LICENSE.
4. Push to GitHub and set up remote.
5. Update the master dashboard (`portfolio_index.md`).

---

## Template Types
- Place reusable templates in `Templates/` (e.g., nextjs, fastapi, nodejs).

## Automation Scripts
- Scripts for migration, CI/CD, and documentation are in `Portfolio_Automation/`.

## Ownership
- Add yourself to `.github/CODEOWNERS` for every repo.

## CI/CD
- Use provided workflow examples in `.github/workflows/` for instant setup.
