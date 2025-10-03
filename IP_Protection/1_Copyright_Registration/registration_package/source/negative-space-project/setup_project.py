"""
Project installation script
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Using Python {sys.version}")

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("Virtual environment already exists")
        return
    
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print("Virtual environment created successfully")

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")
    
    # Determine the pip executable based on the platform
    if sys.platform == "win32":
        pip_path = Path("venv") / "Scripts" / "pip"
    else:
        pip_path = Path("venv") / "bin" / "pip"
    
    # Upgrade pip
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    
    print("Dependencies installed successfully")

def setup_project_structure():
    """Ensure all required directories exist."""
    dirs = [
        "data",
        "data/raw",
        "data/processed",
        "output",
        "output/models",
        "output/visualizations",
        "output/demos"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    print("Project directory structure set up successfully")

def create_open3d_environment_check():
    """Create a script to check Open3D environment."""
    script_content = """
import sys
try:
    import open3d as o3d
    print(f"Open3D version: {o3d.__version__}")
    print("Open3D is working correctly")
except ImportError:
    print("Open3D is not installed")
    sys.exit(1)
except Exception as e:
    print(f"Error importing Open3D: {e}")
    sys.exit(1)

# Test visualization capability
try:
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]))
    print("Successfully created test point cloud")
    # Uncomment to test visualization
    # o3d.visualization.draw_geometries([pcd])
except Exception as e:
    print(f"Error testing Open3D: {e}")
"""
    
    with open("check_open3d.py", "w") as f:
        f.write(script_content)
    
    print("Created Open3D environment check script")

def main():
    """Main setup function."""
    print("Setting up Negative Space Imaging Project...")
    
    check_python_version()
    create_virtual_environment()
    install_dependencies()
    setup_project_structure()
    create_open3d_environment_check()
    
    print("\nSetup completed successfully!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("    venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    
    print("\nTo verify Open3D installation:")
    if sys.platform == "win32":
        print("    venv\\Scripts\\python check_open3d.py")
    else:
        print("    venv/bin/python check_open3d.py")

if __name__ == "__main__":
    main()
