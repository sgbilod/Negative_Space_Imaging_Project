"""
Enhanced Proof-of-Concept Demo for Negative Space Imaging Project
Loads real image data, performs negative space reconstruction, and visualizes results.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.negative_space_reconstructor import NegativeSpaceReconstructor

# Load a real image (replace with your own path)
image_path = "Hoag's_object.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Show original image
plt.figure(figsize=(8, 6))
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

# Feature extraction (ORB)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(image, None)
image_with_kp = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=0)

plt.figure(figsize=(8, 6))
plt.title("ORB Keypoints")
plt.imshow(image_with_kp)
plt.axis('off')
plt.show()

# Negative space reconstruction (demo logic)
reconstructor = NegativeSpaceReconstructor()
reconstructor.image_collection = [image_path]
reconstructor.feature_points = {image_path: keypoints}
reconstructor.negative_space_map = {image_path: [kp.pt for kp in keypoints]}

# Visualize negative space points
neg_points = np.array([kp.pt for kp in keypoints])
plt.figure(figsize=(8, 6))
plt.title("Negative Space Points")
plt.imshow(image, cmap='gray')
if len(neg_points) > 0:
    plt.scatter(neg_points[:,0], neg_points[:,1], s=10, c='red', label='Negative Space')
plt.axis('off')
plt.legend()
plt.show()

# Blockchain tokenization (prints only)
tokens = reconstructor.tokenize_negative_space_stub()
print("Blockchain tokens for negative space regions:", tokens)

# Quantum encryption (prints only)
key = b'0'*32  # Demo key
enc = reconstructor.quantum_encrypt("demo_data", key)
print("Quantum-encrypted demo data:", enc)
