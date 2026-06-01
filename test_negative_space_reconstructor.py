import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'negative-space-project', 'src'))
from negative_space_reconstructor import NegativeSpaceReconstructor

class TestNegativeSpaceReconstructor(unittest.TestCase):
    def setUp(self):
        self.reconstructor = NegativeSpaceReconstructor()

    def test_add_image(self):
        self.assertIsNone(self.reconstructor.add_image('sample.png'))

    def test_extract_features(self):
        self.assertIsNone(self.reconstructor.extract_features())

    def test_reconstruct_3d_model(self):
        self.assertIsNone(self.reconstructor.reconstruct_3d_model())

    def test_map_negative_space(self):
        self.assertIsNone(self.reconstructor.map_negative_space())

    def test_integrate_blockchain(self):
        self.assertIsNone(self.reconstructor.integrate_blockchain())

if __name__ == '__main__':
    unittest.main()
