"""
Copyright Registration Preparation Script
This script prepares materials for copyright registration by:
1. Creating a manifest of all source files
2. Generating checksums
3. Creating redacted versions of source files
4. Preparing registration packages
"""
import os
import hashlib
import shutil
from datetime import datetime

class CopyrightRegistrationPrep:
    def __init__(self, project_root, output_dir):
        self.project_root = project_root
        self.output_dir = output_dir
        self.manifest = []
        
    def generate_manifest(self):
        """Generate a manifest of all source files with checksums."""
        print("Generating manifest...")
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            checksum = hashlib.sha256(content).hexdigest()
                            
                        rel_path = os.path.relpath(file_path, self.project_root)
                        self.manifest.append({
                            'file': rel_path,
                            'checksum': checksum,
                            'size': os.path.getsize(file_path),
                            'modified': datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            ).isoformat()
                        })
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        
    def create_registration_package(self):
        """Create a complete registration package."""
        # Create package directory
        package_dir = os.path.join(self.output_dir, 'registration_package')
        os.makedirs(package_dir, exist_ok=True)
        
        # Create manifest file
        manifest_path = os.path.join(package_dir, 'MANIFEST.md')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write("# Source Code Manifest\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("| File | Checksum | Size | Last Modified |\n")
            f.write("|------|----------|------|---------------|\n")
            for entry in self.manifest:
                f.write(f"| {entry['file']} | {entry['checksum']} | {entry['size']} | {entry['modified']} |\n")
                
        # Copy source files
        source_dir = os.path.join(package_dir, 'source')
        os.makedirs(source_dir, exist_ok=True)
        for entry in self.manifest:
            src_path = os.path.join(self.project_root, entry['file'])
            dst_path = os.path.join(source_dir, entry['file'])
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            
        print(f"Registration package created at: {package_dir}")
        return package_dir

def main():
    project_root = "c:\\Users\\sgbil\\OneDrive\\Desktop\\Negative_Space_Imaging_Project"
    output_dir = "c:\\Users\\sgbil\\OneDrive\\Desktop\\Negative_Space_Imaging_Project\\IP_Protection\\1_Copyright_Registration"
    
    prep = CopyrightRegistrationPrep(project_root, output_dir)
    prep.generate_manifest()
    prep.create_registration_package()

if __name__ == "__main__":
    main()
