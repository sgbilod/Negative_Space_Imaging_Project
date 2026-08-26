# Documentation for documentation_generator.py

```python
"""
Copyright (c) 2025 Negative Space Imaging Project
All rights reserved.

This source code is protected by international copyright laws and treaties.
Unauthorized reproduction or distribution of this code, or any portion of it,
may result in severe civil and criminal penalties, and will be prosecuted
to the maximum extent possible under law.

File: documentation_generator.py
Last Modified: 2025-08-06T02:06:31.706915
"""
"""
Automated documentation generator.
This script generates documentation from code comments.
"""
import os
import subprocess

def generate_documentation(root_dir):
    output_dir = os.path.join(root_dir, "docs")
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file.replace('.py', '.md'))
                
                # Try UTF-8 first, fall back to latin-1 if needed
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Documentation for {file}\n\n")
                    f.write("```python\n")
                    f.write(content)
                    f.write("\n```")

    print(f"Documentation generated in {output_dir}")

if __name__ == "__main__":
    generate_documentation("c:\\Users\\sgbil\\OneDrive\\Desktop\\Negative_Space_Imaging_Project")

```