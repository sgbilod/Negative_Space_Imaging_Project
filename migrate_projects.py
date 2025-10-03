import os
import shutil

TEMPLATE = "project-template"
PROJECTS = ["project1", "project2"]  # Add your existing project names here

for project in PROJECTS:
    new_path = f"{project}_migrated"
    shutil.copytree(TEMPLATE, new_path)
    # Move code, tests, docs, etc. as needed
    # Example: shutil.move(f"{project}/old_code.py", f"{new_path}/src/")
    print(f"Migrated {project} to {new_path}")
