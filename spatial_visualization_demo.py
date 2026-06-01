"""
Proof-of-Concept: Spatial Visualization Demo
"""
import matplotlib.pyplot as plt
from negative_space_reconstructor import NegativeSpaceReconstructor

def visualize_negative_space(image_path):
    recon = NegativeSpaceReconstructor()
    recon.add_image(image_path)
    recon.extract_features()
    recon.reconstruct_3d_model()
    recon.map_negative_space()
    for img, hull in recon.negative_space_map.items():
        plt.figure()
        plt.title(f"Negative Space Visualization for {image_path}")
        if hull is not None:
            plt.plot(hull[:, 0, 0], hull[:, 0, 1], 'ro-')
        plt.show()

if __name__ == "__main__":
    # Example usage
    visualize_negative_space("sample_image.png")
