"""
Basic tests for the Negative Space Imaging Project core modules.
This ensures that the main modules can be imported and initialized correctly.
"""

import os
import sys
import unittest
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TestBasicImports(unittest.TestCase):
    """Test that all core modules can be imported"""
    
    def test_acquisition_imports(self):
        """Test imports from the acquisition module"""
        try:
            from src.acquisition.camera_interface import CameraInterface
            from src.acquisition.image_preprocessor import ImagePreprocessor, PreprocessingMode
            from src.acquisition.metadata_extractor import MetadataExtractor, SpatialMetadata
            
            # Verify classes can be instantiated
            camera = CameraInterface(camera_type="webcam", camera_id=0)
            preprocessor = ImagePreprocessor(mode=PreprocessingMode.STANDARD)
            extractor = MetadataExtractor()
            
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Import error: {str(e)}")
    
    def test_reconstruction_imports(self):
        """Test imports from the reconstruction module"""
        try:
            from src.reconstruction.feature_detector import FeatureDetector, FeatureType
            
            # Verify classes can be instantiated
            detector = FeatureDetector(feature_type=FeatureType.SIFT)
            
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Import error: {str(e)}")

class TestCameraInterface(unittest.TestCase):
    """Test the camera interface functionality"""
    
    def test_camera_initialization(self):
        """Test that the camera interface can be initialized"""
        from src.acquisition.camera_interface import CameraInterface
        
        camera = CameraInterface(camera_type="webcam", camera_id=0)
        self.assertEqual(camera.camera_type, "webcam")
        self.assertEqual(camera.camera_id, 0)
        self.assertFalse(camera.connected)

class TestImagePreprocessor(unittest.TestCase):
    """Test the image preprocessor functionality"""
    
    def test_preprocessor_initialization(self):
        """Test that the image preprocessor can be initialized"""
        from src.acquisition.image_preprocessor import ImagePreprocessor, PreprocessingMode
        
        preprocessor = ImagePreprocessor(mode=PreprocessingMode.STANDARD)
        self.assertEqual(preprocessor.mode, PreprocessingMode.STANDARD)
        
        # Test mode switching
        preprocessor.set_mode(PreprocessingMode.NEGATIVE_SPACE_FOCUS)
        self.assertEqual(preprocessor.mode, PreprocessingMode.NEGATIVE_SPACE_FOCUS)

class TestFeatureDetector(unittest.TestCase):
    """Test the feature detector functionality"""
    
    def test_detector_initialization(self):
        """Test that the feature detector can be initialized"""
        from src.reconstruction.feature_detector import FeatureDetector, FeatureType
        
        detector = FeatureDetector(feature_type=FeatureType.SIFT)
        self.assertEqual(detector.feature_type, FeatureType.SIFT)
        
        # Test type switching
        detector.set_feature_type(FeatureType.BOUNDARY)
        self.assertEqual(detector.feature_type, FeatureType.BOUNDARY)

if __name__ == '__main__':
    unittest.main()
