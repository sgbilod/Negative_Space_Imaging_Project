"""
Fallback Demo

This script demonstrates how to use the centralized fallback system
to create code that works with or without optional dependencies.

Usage:
    python fallback_demo.py
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import from centralized fallbacks
try:
    from src.utils.fallbacks import (
        np, plt, cv2, o3d, web3, pd, hashlib,
        NUMPY_AVAILABLE, MATPLOTLIB_AVAILABLE, OPENCV_AVAILABLE,
        OPEN3D_AVAILABLE, WEB3_AVAILABLE, PANDAS_AVAILABLE,
        HASHLIB_AVAILABLE
    )
except ImportError:
    logger.error("Could not import fallbacks module. Make sure you're running from project root.")
    sys.exit(1)

def ensure_directory(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def display_availability():
    """Display which modules are available"""
    availability = {
        "NumPy": NUMPY_AVAILABLE,
        "Matplotlib": MATPLOTLIB_AVAILABLE,
        "OpenCV": OPENCV_AVAILABLE,
        "Open3D": OPEN3D_AVAILABLE,
        "Web3": WEB3_AVAILABLE,
        "Pandas": PANDAS_AVAILABLE,
        "Hashlib": HASHLIB_AVAILABLE
    }
    
    logger.info("=== Module Availability ===")
    for module, available in availability.items():
        status = "Available ✅" if available else "Not Available ❌"
        logger.info(f"{module}: {status}")
    
    return availability

def create_sample_data():
    """Create sample data for testing"""
    # Create a sample array with NumPy or fallback
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    logger.info(f"Sample array: {array}")
    
    return array

def visualize_sample_data(array, output_path="output/fallback_demo/visualization.png"):
    """Visualize sample data with Matplotlib or fallback"""
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot the data
    plt.bar(range(len(array.flatten())), array.flatten())
    plt.title("Sample Data Visualization")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    ensure_directory(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")

def process_sample_image(image_path="output/fallback_demo/test_image.png"):
    """Process a sample image with OpenCV or fallback"""
    # Create a simple test image if OpenCV is available
    if OPENCV_AVAILABLE:
        # Create a gradient image
        img = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            for j in range(200):
                img[i, j] = (i + j) // 2
        
        # Save the image
        ensure_directory(os.path.dirname(image_path))
        cv2.imwrite(image_path, img)
        logger.info(f"Test image created and saved to {image_path}")
        
        # Read and process the image
        processed = cv2.GaussianBlur(img, (5, 5), 0)
        
        return processed
    else:
        logger.warning("Image processing skipped (OpenCV not available)")
        return None

def create_sample_point_cloud(output_path="output/fallback_demo/point_cloud.ply"):
    """Create a sample point cloud with Open3D or fallback"""
    if OPEN3D_AVAILABLE:
        # Create a simple point cloud
        pcd = o3d.geometry.PointCloud()
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        colors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0.5, 0.5, 0.5],
            [1, 1, 1]
        ])
        
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save the point cloud
        ensure_directory(os.path.dirname(output_path))
        o3d.io.write_point_cloud(output_path, pcd)
        logger.info(f"Point cloud saved to {output_path}")
        
        return pcd
    else:
        logger.warning("Point cloud creation skipped (Open3D not available)")
        return None

def create_sample_blockchain_hash(data="test data"):
    """Create a sample blockchain hash with Web3 or fallback"""
    # Use hashlib for cryptographic operations regardless of Web3 availability
    hash_value = hashlib.sha256(data.encode()).hexdigest()
    logger.info(f"Generated hash: {hash_value}")
    
    if WEB3_AVAILABLE:
        # In a real implementation, we would register this on a blockchain
        logger.info("Web3 is available for blockchain operations")
    else:
        logger.info("Using simulated blockchain operations")
    
    return hash_value

def create_sample_time_series(output_path="output/fallback_demo/time_series.csv"):
    """Create a sample time series with Pandas or fallback"""
    if PANDAS_AVAILABLE:
        # Create a simple time series
        dates = pd.date_range(start='2025-01-01', periods=10)
        values = np.random.randn(10)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Save to CSV
        ensure_directory(os.path.dirname(output_path))
        df.to_csv(output_path, index=False)
        logger.info(f"Time series saved to {output_path}")
        
        return df
    else:
        logger.warning("Time series creation skipped (Pandas not available)")
        
        # Create a simple structure instead
        dates = [f"2025-01-{i+1}" for i in range(10)]
        values = [i / 10.0 for i in range(10)]
        
        logger.info(f"Simple time series: {list(zip(dates, values))}")
        
        return list(zip(dates, values))

def main():
    """Main function"""
    logger.info("=== Fallback Demo ===")
    
    # Check which modules are available
    availability = display_availability()
    
    # Create output directory
    ensure_directory("output/fallback_demo")
    
    # Run demos for each module
    sample_data = create_sample_data()
    visualize_sample_data(sample_data)
    process_sample_image()
    create_sample_point_cloud()
    create_sample_blockchain_hash()
    create_sample_time_series()
    
    logger.info("Fallback demo completed successfully!")
    
    # Provide summary
    logger.info("\n=== Summary ===")
    logger.info("This demo showed how the centralized fallback system allows code to:")
    logger.info("1. Work with or without optional dependencies")
    logger.info("2. Provide graceful degradation when packages are missing")
    logger.info("3. Use a consistent interface regardless of package availability")
    
    missing_packages = [name for name, available in availability.items() if not available]
    if missing_packages:
        logger.info(f"\nMissing packages: {', '.join(missing_packages)}")
        logger.info("To install missing packages, use:")
        logger.info("pip install " + " ".join(name.lower() for name in missing_packages 
                                             if name not in ["Hashlib"]))
    else:
        logger.info("\nAll packages are available!")

if __name__ == "__main__":
    main()
