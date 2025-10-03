#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for the project.
"""

import unittest
import sys
import os

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import example_function


class TestMain(unittest.TestCase):
    """Test case for the main module."""

    def test_example_function(self):
        """Test the example function returns the expected greeting."""
        self.assertEqual(example_function(), "Hello from the project!")


if __name__ == "__main__":
    unittest.main()
