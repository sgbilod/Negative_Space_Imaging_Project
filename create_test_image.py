#!/usr/bin/env python
"""
Test Image Generator for Negative Space Imaging
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script creates test images with controlled negative space patterns
for testing the negative space detection and analysis algorithms.

Usage:
    python create_test_image.py --type [pattern_type] --output [output_path]
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def create_geometric_pattern(width, height, pattern_type, bg_color=(240, 240, 240),
                             fg_color=(30, 30, 30), complexity=0.5, seed=None):
    """
    Create an image with geometric patterns and controlled negative space.

    Args:
        width: Image width
        height: Image height
        pattern_type: Type of pattern ('circles', 'rectangles', 'mixed', etc.)
        bg_color: Background color (RGB tuple)
        fg_color: Foreground color (RGB tuple)
        complexity: Complexity of the pattern (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        PIL Image with the generated pattern
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create a blank image with background color
    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Determine number of shapes based on complexity
    max_shapes = int(50 * complexity)
    num_shapes = random.randint(max(5, max_shapes // 2), max_shapes)

    # Generate shapes based on pattern type
    if pattern_type == 'circles':
        _draw_circle_pattern(draw, width, height, num_shapes, fg_color)
    elif pattern_type == 'rectangles':
        _draw_rectangle_pattern(draw, width, height, num_shapes, fg_color)
    elif pattern_type == 'lines':
        _draw_line_pattern(draw, width, height, num_shapes, fg_color)
    elif pattern_type == 'mixed':
        _draw_mixed_pattern(draw, width, height, num_shapes, fg_color)
    elif pattern_type == 'negative_letters':
        _draw_negative_letters(draw, width, height, fg_color)
    elif pattern_type == 'negative_logo':
        _draw_negative_logo(draw, width, height, fg_color)
    elif pattern_type == 'random':
        # Choose a random pattern type
        pattern_funcs = [
            _draw_circle_pattern,
            _draw_rectangle_pattern,
            _draw_line_pattern,
            _draw_mixed_pattern
        ]
        random.choice(pattern_funcs)(draw, width, height, num_shapes, fg_color)
    else:
        # Default to a mixed pattern
        _draw_mixed_pattern(draw, width, height, num_shapes, fg_color)

    return image


def _draw_circle_pattern(draw, width, height, num_shapes, color):
    """Draw a pattern of circles with varying sizes."""
    for _ in range(num_shapes):
        # Randomly determine circle size
        radius = random.randint(10, min(width, height) // 4)

        # Randomly determine circle position
        x = random.randint(0, width)
        y = random.randint(0, height)

        # Draw the circle
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color
        )


def _draw_rectangle_pattern(draw, width, height, num_shapes, color):
    """Draw a pattern of rectangles with varying sizes."""
    for _ in range(num_shapes):
        # Randomly determine rectangle size
        rect_width = random.randint(20, width // 3)
        rect_height = random.randint(20, height // 3)

        # Randomly determine rectangle position
        x = random.randint(0, width - rect_width)
        y = random.randint(0, height - rect_height)

        # Draw the rectangle
        draw.rectangle(
            [(x, y), (x + rect_width, y + rect_height)],
            fill=color
        )


def _draw_line_pattern(draw, width, height, num_shapes, color):
    """Draw a pattern of lines with varying thickness."""
    for _ in range(num_shapes):
        # Randomly determine line endpoints
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)

        # Randomly determine line thickness
        thickness = random.randint(1, 10)

        # Draw the line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)


def _draw_mixed_pattern(draw, width, height, num_shapes, color):
    """Draw a mixed pattern of different shapes."""
    # Distribute shapes among different types
    circles = num_shapes // 3
    rectangles = num_shapes // 3
    lines = num_shapes - circles - rectangles

    _draw_circle_pattern(draw, width, height, circles, color)
    _draw_rectangle_pattern(draw, width, height, rectangles, color)
    _draw_line_pattern(draw, width, height, lines, color)


def _draw_negative_letters(draw, width, height, color):
    """Draw text with intentional negative space between letters."""
    # Background rectangle covering most of the image
    margin = width // 10
    draw.rectangle(
        [(margin, margin), (width - margin, height - margin)],
        fill=color
    )

    # Draw text in background color to create negative space
    font_size = height // 4
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Use default font if Arial is not available
        font = ImageFont.load_default()

    text = "NSI"  # Negative Space Imaging
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position text in center
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2

    # Draw text in background color to create negative space
    draw.text((text_x, text_y), text, font=font, fill=(240, 240, 240))


def _draw_negative_logo(draw, width, height, color):
    """Draw a simplified logo with intentional negative space."""
    # Background circle covering most of the image
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3

    # Draw outer circle
    draw.ellipse(
        [(center_x - radius, center_y - radius),
         (center_x + radius, center_y + radius)],
        fill=color
    )

    # Draw inner circle for negative space
    inner_radius = radius // 2
    draw.ellipse(
        [(center_x - inner_radius, center_y - inner_radius),
         (center_x + inner_radius, center_y + inner_radius)],
        fill=(240, 240, 240)
    )

    # Draw spokes for more complex negative space
    for angle in range(0, 360, 45):
        angle_rad = np.radians(angle)
        spoke_outer_x = center_x + int(radius * 0.9 * np.cos(angle_rad))
        spoke_outer_y = center_y + int(radius * 0.9 * np.sin(angle_rad))
        spoke_inner_x = center_x + int(inner_radius * 1.2 * np.cos(angle_rad))
        spoke_inner_y = center_y + int(inner_radius * 1.2 * np.sin(angle_rad))

        # Draw spoke
        spoke_width = max(3, radius // 20)
        draw.line(
            [(spoke_inner_x, spoke_inner_y), (spoke_outer_x, spoke_outer_y)],
            fill=(240, 240, 240),
            width=spoke_width
        )


def add_metadata(image, pattern_type, width, height, complexity, seed):
    """Add metadata to the image as text in a corner."""
    draw = ImageDraw.Draw(image)

    # Prepare metadata text
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_text = [
        f"Negative Space Imaging - Test Pattern",
        f"Type: {pattern_type}",
        f"Size: {width}x{height}",
        f"Complexity: {complexity:.2f}",
        f"Seed: {seed}",
        f"Created: {timestamp}"
    ]

    # Use a small font
    font_size = max(10, min(width, height) // 40)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Position text in the bottom-left corner
    text_y = height - (len(metadata_text) * font_size * 1.2)

    # Draw text with contrasting color and background
    bg_color = (200, 200, 200, 180)  # Light gray with some transparency
    for i, line in enumerate(metadata_text):
        text_y_pos = text_y + (i * font_size * 1.2)
        # Get text dimensions for background
        text_bbox = draw.textbbox((10, text_y_pos), line, font=font)

        # Draw semi-transparent background
        draw.rectangle(
            [text_bbox[0] - 5, text_bbox[1] - 2, text_bbox[2] + 5, text_bbox[3] + 2],
            fill=bg_color
        )

        # Draw text
        draw.text((10, text_y_pos), line, fill=(0, 0, 0), font=font)

    return image


def main():
    """Main function to parse arguments and generate the test image."""
    parser = argparse.ArgumentParser(description="Generate test images with controlled negative space")
    parser.add_argument("--type", choices=[
        "circles", "rectangles", "lines", "mixed", "negative_letters", "negative_logo", "random"
    ], default="mixed", help="Type of pattern to generate")
    parser.add_argument("--width", type=int, default=800, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=600, help="Image height in pixels")
    parser.add_argument("--complexity", type=float, default=0.5,
                        help="Pattern complexity (0.0-1.0)")
    parser.add_argument("--bg", default="240,240,240", help="Background color (R,G,B)")
    parser.add_argument("--fg", default="30,30,30", help="Foreground color (R,G,B)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", default="./test_images",
                        help="Output directory for generated images")
    parser.add_argument("--metadata", action="store_true",
                        help="Add metadata to the image")
    args = parser.parse_args()

    # Parse colors
    try:
        bg_color = tuple(map(int, args.bg.split(',')))
        fg_color = tuple(map(int, args.fg.split(',')))
    except ValueError:
        print("Error: Colors must be in the format R,G,B (e.g., 240,240,240)")
        return 1

    # Use current time as seed if not provided
    seed = args.seed if args.seed is not None else int(datetime.now().timestamp())

    # Ensure output directory exists
    create_directory(args.output)

    # Generate the test image
    print(f"Generating {args.type} pattern with complexity {args.complexity}...")
    image = create_geometric_pattern(
        args.width, args.height, args.type,
        bg_color=bg_color, fg_color=fg_color,
        complexity=args.complexity, seed=seed
    )

    # Add metadata if requested
    if args.metadata:
        image = add_metadata(
            image, args.type, args.width, args.height, args.complexity, seed
        )

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_image_{args.type}_{timestamp}_{seed}.png"
    output_path = os.path.join(args.output, filename)

    # Save the image
    image.save(output_path, format="PNG")
    print(f"Image saved to: {output_path}")

    # Also save metadata to a separate JSON file
    if args.metadata:
        import json
        metadata = {
            "filename": filename,
            "pattern_type": args.type,
            "width": args.width,
            "height": args.height,
            "complexity": args.complexity,
            "bg_color": bg_color,
            "fg_color": fg_color,
            "seed": seed,
            "created": datetime.now().isoformat(),
        }

        metadata_path = os.path.join(args.output, f"{os.path.splitext(filename)[0]}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
