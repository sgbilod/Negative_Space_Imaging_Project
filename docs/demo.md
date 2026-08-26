# Documentation for demo.py

```python
"""
Negative Space Imaging Project - Demo Script

This script demonstrates the basic functionality of the project:
1. Capturing images from a camera
2. Preprocessing with negative space focus
3. Detecting features optimized for negative space boundaries
4. Visualizing the results

Usage:
    python demo.py [--camera_id N] [--mode MODE] [--feature_type TYPE]

Example:
    python demo.py --camera_id 0 --mode negative_space_focus --feature_type void_edge
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.acquisition.camera_interface import CameraInterface
from src.acquisition.image_preprocessor import ImagePreprocessor, PreprocessingMode
from src.acquisition.metadata_extractor import MetadataExtractor
from src.reconstruction.feature_detector import FeatureDetector, FeatureType

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Negative Space Imaging Demo")
    
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera ID to use (default: 0)')
    
    parser.add_argument('--mode', type=str, default='negative_space_focus',
                       choices=['standard', 'negative_space_focus', 'feature_enhancement',
                                'low_light', 'high_contrast'],
                       help='Preprocessing mode (default: negative_space_focus)')
    
    parser.add_argument('--feature_type', type=str, default='void_edge',
                       choices=['sift', 'orb', 'boundary', 'void_edge', 
                                'object_silhouette', 'multi_scale'],
                       help='Feature detection method (default: void_edge)')
    
    parser.add_argument('--save_dir', type=str, default='output/demo',
                       help='Directory to save results (default: output/demo)')
    
    parser.add_argument('--use_sample', action='store_true',
                       help='Use a sample image instead of camera')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def capture_or_load_image(args):
    """Capture an image from camera or load a sample image"""
    if args.use_sample:
        # Load sample image if camera not available
        sample_path = os.path.join('data', 'images', 'sample.jpg')
        
        # If sample doesn't exist, create a test pattern
        if not os.path.exists(sample_path):
            print("Sample image not found, creating test pattern...")
            ensure_directory(os.path.dirname(sample_path))
            
            # Create a test pattern with objects and negative space
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some geometric shapes
            cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)  # Red square
            cv2.circle(img, (400, 150), 70, (0, 255, 0), -1)  # Green circle
            cv2.rectangle(img, (150, 300), (300, 400), (255, 0, 0), -1)  # Blue rectangle
            cv2.circle(img, (450, 350), 50, (255, 255, 0), -1)  # Yellow circle
            
            cv2.imwrite(sample_path, img)
            print(f"Test pattern saved to {sample_path}")
        
        # Load the image
        image = cv2.imread(sample_path)
        success = image is not None
        
        if success:
            print(f"Loaded sample image from {sample_path}")
        else:
            print(f"Failed to load sample image from {sample_path}")
            sys.exit(1)
        
        return success, image
    else:
        # Capture from camera
        print(f"Initializing camera (ID: {args.camera_id})...")
        camera = CameraInterface(camera_type="webcam", camera_id=args.camera_id)
        
        if not camera.connect():
            print("Failed to connect to camera, falling back to sample image...")
            return capture_or_load_image(argparse.Namespace(**{**vars(args), 'use_sample': True}))
        
        print("Camera connected. Press Enter to capture an image...")
        input()
        
        success, image = camera.capture_image()
        
        if success:
            print("Image captured successfully")
        else:
            print("Failed to capture image, falling back to sample image...")
            camera.disconnect()
            return capture_or_load_image(argparse.Namespace(**{**vars(args), 'use_sample': True}))
        
        camera.disconnect()
        return success, image

def process_image(image, mode_str):
    """Preprocess the image with the specified mode"""
    print(f"Preprocessing image with mode: {mode_str}...")
    
    try:
        mode = PreprocessingMode(mode_str)
    except ValueError:
        print(f"Invalid mode: {mode_str}, falling back to STANDARD")
        mode = PreprocessingMode.STANDARD
    
    preprocessor = ImagePreprocessor(mode=mode)
    processed_image, metadata = preprocessor.preprocess(image)
    
    print(f"Preprocessing complete in {metadata['processing_time']:.2f} seconds")
    
    return processed_image, metadata

def detect_features(image, feature_type_str):
    """Detect features in the image with the specified method"""
    print(f"Detecting features with method: {feature_type_str}...")
    
    try:
        feature_type = FeatureType(feature_type_str)
    except ValueError:
        print(f"Invalid feature type: {feature_type_str}, falling back to SIFT")
        feature_type = FeatureType.SIFT
    
    detector = FeatureDetector(feature_type=feature_type)
    keypoints, descriptors = detector.detect(image)
    
    print(f"Detected {len(keypoints)} features")
    
    return keypoints, descriptors, detector

def visualize_results(original, processed, keypoints, detector, save_dir):
    """Visualize the results"""
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Draw features on the processed image
    features_img = detector.draw_features(processed, keypoints)
    features_rgb = cv2.cvtColor(features_img, cv2.COLOR_BGR2RGB)
    
    # Plot the images
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Processed Image")
    plt.imshow(processed_rgb)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Detected Features ({len(keypoints)})")
    plt.imshow(features_rgb)
    plt.axis('off')
    
    # Save the figure
    ensure_directory(save_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, f"results_{timestamp}.png"), dpi=300, bbox_inches='tight')
    
    print(f"Results saved to {save_dir}/results_{timestamp}.png")
    
    # Also save individual images
    cv2.imwrite(os.path.join(save_dir, f"original_{timestamp}.jpg"), original)
    cv2.imwrite(os.path.join(save_dir, f"processed_{timestamp}.jpg"), processed)
    cv2.imwrite(os.path.join(save_dir, f"features_{timestamp}.jpg"), features_img)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Capture or load image
    success, image = capture_or_load_image(args)
    
    # Process image
    processed_image, preprocessing_metadata = process_image(image, args.mode)
    
    # Detect features
    keypoints, descriptors, detector = detect_features(processed_image, args.feature_type)
    
    # Visualize results
    visualize_results(image, processed_image, keypoints, detector, args.save_dir)
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()

```