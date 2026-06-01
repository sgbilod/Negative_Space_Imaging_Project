#!/usr/bin/env python
"""
Hoag's Object Demo - Simplified

A minimal script demonstrating Hoag's object image processing
in the Negative Space Imaging system.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime


def main():
    """Process and display Hoag's object."""
    parser = argparse.ArgumentParser(description="Hoag's Object Demo")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--view", action="store_true", help="View image after processing")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print("\n=== HOAG'S OBJECT - NEGATIVE SPACE IMAGING DEMO ===")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Path to Hoag's object image
    image_path = os.path.join(os.path.dirname(__file__), "Hoag's_object.jpg")
    
    # Process the image using demo_acquisition
    try:
        print("\nProcessing image...")
        result = subprocess.run(
            ["python", "hoag_demo.py"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("Image processed successfully!")
        
        # Find the output image path from the output
        output_lines = result.stdout.splitlines()
        saved_line = next((line for line in output_lines if "Saved to:" in line), None)
        
        if saved_line:
            output_path = saved_line.split("Saved to:", 1)[1].strip()
            print(f"Processed image: {output_path}")
            
            # Open image in default viewer if requested
            if args.view and os.path.exists(output_path):
                print("\nOpening image...")
                if sys.platform == 'win32':
                    os.startfile(output_path)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', output_path])
                else:  # Linux
                    subprocess.run(['xdg-open', output_path])
        
        # Get image dimensions from log
        dimensions_line = next(
            (line for line in output_lines if "Image dimensions:" in line), 
            None
        )
        if dimensions_line:
            dimensions = dimensions_line.split("dimensions:", 1)[1].strip()
            print(f"Image dimensions: {dimensions}")
        
        print("\nHoag's Object is a perfect example of negative space in astronomy:")
        print("- The bright outer ring represents positive space (visible matter)")
        print("- The dark gap between the ring and center represents negative space")
        print("- This negative space reveals valuable information about the galaxy structure")
        
        print("\nNegative Space Imaging helps identify and analyze these important")
        print("regions that are often overlooked in traditional image processing.")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing image: {e}")
        print(f"Error output: {e.stderr}")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
