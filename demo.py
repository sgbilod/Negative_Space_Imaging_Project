
import sys
try:
    from src.processing.negative_space_detection import detect_negative_space
    from src.utils.image_loader import load_image
except ImportError:
    # Fallback for direct execution in root
    from processing.negative_space_detection import detect_negative_space
    from utils.image_loader import load_image


def run_demo(image_path):
    image = load_image(image_path)
    result = detect_negative_space(image)
    print(f"Negative space detection result: {result}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo.py <image_path>")
        sys.exit(1)
    run_demo(sys.argv[1])
    # ...expand with visualization and multiple algorithms
