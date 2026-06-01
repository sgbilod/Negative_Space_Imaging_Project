#!/usr/bin/env python
"""
Image Dimension Checker for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script provides functionality to check and validate image dimensions
for the Negative Space Imaging System.
"""

from PIL import Image

# Open the image
img = Image.open("Hoag's_object.jpg")

# Print the dimensions
print(f"Image dimensions: {img.width}x{img.height}")
