#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Dashboard Generator
Author: Stephen Bilodeau
Date: August 2025

Generates an HTML dashboard for visualizing and managing projects
created with the project automation system.
"""

import os
import json
import glob
import datetime
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate project dashboard HTML"
    )
    parser.add_argument(
        "--projects-dir",
        default=os.path.expanduser("~/Projects"),
        help="Root directory containing projects"
    )
    parser.add_argument(
        "--output",
        default="project_dashboard.html",
        help="Output HTML file path"
    )
    return parser.parse_args()


def collect_project_data(projects_dir):
    """Collect metadata from all projects in the directory."""
    projects = []

    # Find all .project-info.json files
    info_files = glob.glob(
        os.path.join(projects_dir, "**/.project-info.json"),
        recursive=True
    )

    for info_file in info_files:
        try:
            # Try different encodings to handle BOM
            try:
                with open(info_file, 'r', encoding='utf-8-sig') as f:
                    project_info = json.load(f)
            except:
                with open(info_file, 'r', encoding='utf-8') as f:
                    project_info = json.load(f)

            project_dir = os.path.dirname(info_file)

            # Get git information if available
            git_info = {
                'has_git': os.path.exists(os.path.join(project_dir, '.git')),
                'last_commit': None,
                'commit_count': 0
            }

            if git_info['has_git']:
                try:
                    import subprocess
                    os.chdir(project_dir)
                    # Get last commit date
                    result = subprocess.run(
                        ['git', 'log', '-1', '--format=%cd', '--date=short'],
                        capture_output=True, text=True, check=True
                    )
                    git_info['last_commit'] = result.stdout.strip()

                    # Get commit count
                    result = subprocess.run(
                        ['git', 'rev-list', '--count', 'HEAD'],
                        capture_output=True, text=True, check=True
                    )
                    git_info['commit_count'] = int(result.stdout.strip())
                except (subprocess.SubprocessError, ValueError):
                    pass  # Ignore git errors

            # Check for README.md
            readme_path = os.path.join(project_dir, 'README.md')
            has_readme = os.path.exists(readme_path)

            # Get file count
            file_count = sum(len(files) for _, _, files in os.walk(project_dir))

            # Get total size
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(project_dir)
                for filename in filenames
            )

            # Add to projects list
            projects.append({
                'info': project_info,
                'git': git_info,
                'has_readme': has_readme,
                'file_count': file_count,
                'size_mb': total_size / (1024 * 1024),  # Convert to MB
                'path': project_dir
            })

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing {info_file}: {e}")

    # Sort projects by creation date (newest first)
    projects.sort(
        key=lambda p: datetime.datetime.strptime(
            p['info'].get('created', '2000-01-01 00:00:00'),
            '%Y-%m-%d %H:%M:%S'
        ),
        reverse=True
    )

    return projects


def generate_html(projects, output_file):
    """Generate HTML dashboard from project data."""
    # Count statistics
    total_projects = len(projects)
    projects_by_template = {}
    for project in projects:
        template = project['info'].get('template', 'unknown')
        projects_by_template[template] = projects_by_template.get(template, 0) + 1

    # Start HTML content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .dashboard-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .stats-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            flex: 1;
            min-width: 200px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin-top: 0;
            font-size: 16px;
            color: #6c757d;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            margin: 10px 0;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .template-tag {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            background-color: #6c757d;
        }}
        .template-default {{ background-color: #6c757d; }}
        .template-python {{ background-color: #007bff; }}
        .template-web {{ background-color: #28a745; }}
        .template-research {{ background-color: #dc3545; }}
        .template-data {{ background-color: #fd7e14; }}
        .search-bar {{
            margin: 20px 0;
            width: 100%;
        }}
        .search-bar input {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }}
        .timestamp {{
            text-align: right;
            color: #6c757d;
            font-size: 14px;
            margin-top: 30px;
        }}
        .action-button {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
            display: inline-block;
        }}
        .action-button:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>Project Dashboard</h1>
        <div>
            <a href="PROJECT_ORGANIZATION_GUIDE.md" class="action-button">View Guide</a>
            <button class="action-button" onclick="window.location.reload()">Refresh</button>
        </div>
    </div>

    <div class="stats-container">
        <div class="stat-card">
            <h3>Total Projects</h3>
            <div class="stat-value">{total_projects}</div>
        </div>
"""

    # Add template stats
    for template, count in projects_by_template.items():
        percentage = (count / total_projects) * 100 if total_projects > 0 else 0
        html += f"""
        <div class="stat-card">
            <h3>{template.capitalize()} Projects</h3>
            <div class="stat-value">{count}</div>
            <div>{percentage:.1f}%</div>
        </div>"""

    # Add search bar
    html += """
    </div>

    <div class="search-bar">
        <input type="text" id="projectSearch" onkeyup="searchProjects()"
               placeholder="Search for projects...">
    </div>

    <table id="projectsTable">
        <thead>
            <tr>
                <th>Project Name</th>
                <th>Description</th>
                <th>Template</th>
                <th>Created</th>
                <th>Last Commit</th>
                <th>Files</th>
                <th>Size</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
"""

    # Add project rows
    for project in projects:
        info = project['info']
        git = project['git']
        template = info.get('template', 'default')
        name = info.get('name', 'Unknown')
        description = info.get('description', '')
        created = info.get('created', 'Unknown')
        last_commit = git.get('last_commit', 'N/A')
        file_count = project.get('file_count', 0)
        size_mb = project.get('size_mb', 0)
        path = project.get('path', '')

        # Convert path to VS Code workspace file if it exists
        workspace_file = os.path.join(path, f"{name}.code-workspace")
        if os.path.exists(workspace_file):
            code_path = workspace_file
        else:
            code_path = path

        html += f"""
        <tr>
            <td>{name}</td>
            <td>{description}</td>
            <td><span class="template-tag template-{template}">{template}</span></td>
            <td>{created}</td>
            <td>{last_commit}</td>
            <td>{file_count}</td>
            <td>{size_mb:.1f} MB</td>
            <td>
                <a href="vscode://file/{code_path}" class="action-button">Open</a>
            </td>
        </tr>"""

    # Complete the HTML
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""
        </tbody>
    </table>

    <div class="timestamp">
        Generated on {timestamp}
    </div>

    <script>
    function searchProjects() {{
        const input = document.getElementById('projectSearch');
        const filter = input.value.toUpperCase();
        const table = document.getElementById('projectsTable');
        const tr = table.getElementsByTagName('tr');

        for (let i = 1; i < tr.length; i++) {{
            let txtValue = '';
            const td = tr[i].getElementsByTagName('td');

            // Combine text from first 3 columns (name, description, template)
            for (let j = 0; j < 3; j++) {{
                txtValue += td[j].textContent || td[j].innerText;
            }}

            if (txtValue.toUpperCase().indexOf(filter) > -1) {{
                tr[i].style.display = '';
            }} else {{
                tr[i].style.display = 'none';
            }}
        }}
    }}
    </script>
</body>
</html>"""

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Dashboard generated at {output_file}")
    return output_file


def main():
    """Main function to run the dashboard generator."""
    args = parse_args()
    projects = collect_project_data(args.projects_dir)
    dashboard_file = generate_html(projects, args.output)

    # Try to open the dashboard in the default browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(dashboard_file))
        print(f"Dashboard opened in your browser at {dashboard_file}")
    except Exception as e:
        print(f"Could not open dashboard in browser: {e}")


if __name__ == "__main__":
    main()
