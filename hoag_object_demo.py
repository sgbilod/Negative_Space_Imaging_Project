#!/usr/bin/env python
"""
Hoag's Object Demo - Simple

A simplified demo using the existing demo_acquisition.py functionality to process
the Hoag's object image, which is a perfect example of negative space in astronomy.
"""

import os
import sys
from demo_acquisition import ImageAcquisition

def main():
    """Process Hoag's object using the demo acquisition pipeline."""
    # Path to Hoag's object image
    image_path = os.path.join(os.path.dirname(__file__), "Hoag's_object.jpg")
    output_dir = "./output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Hoag's Object Image Processing Demo ===")
    print(f"Processing image: {image_path}")
    
    try:
        # Initialize acquisition with the file source
        acquisition = ImageAcquisition(
            source_type="file",
            source_path=image_path,
            output_dir=output_dir
        )
        
        # Acquire the image
        image = acquisition.acquire()
        
        # Preprocess the image
        processed = acquisition.preprocess(
            resize=True,
            enhance=True,
            denoise=True
        )
        
        # Save the processed image
        output_path = acquisition.save(format="jpg")
        
        print("\nImage processing completed successfully!")
        print(f"Image ID: {acquisition.image_id}")
        print(f"Saved to: {output_path}")
        print(f"Metadata saved to: {os.path.join(output_dir, f'{acquisition.image_id}_metadata.json')}")
        
        return 0
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Hoag's Object Image Processing Demo ===")
    print(f"Processing image: {image_path}")
    
    try:
        # Initialize acquisition with the file source
        acquisition = ImageAcquisition(
            source_type="file",
            source_path=image_path,
            output_dir=output_dir
        )
        
        # Acquire the image
        _ = acquisition.acquire()
        
        # Preprocess the image
        _ = acquisition.preprocess(
            resize=True,
            enhance=True,
            denoise=True
        )
        
        # Save the processed image
        output_path = acquisition.save(format="jpg")
        
        print("\nImage processing completed successfully!")
        print(f"Image ID: {acquisition.image_id}")
        print(f"Saved to: {output_path}")
        
        # Create a more readable metadata path
        metadata_path = os.path.join(output_dir, 
                                    f'{acquisition.image_id}_metadata.json')
        print(f"Metadata saved to: {metadata_path}")
        
        return 0
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
