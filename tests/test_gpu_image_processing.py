"""
Test suite for GPU Image Processing
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import pytest
import torch
import numpy as np
import os
from pathlib import Path
from gpu.image_processing import GPUImageProcessor


@pytest.fixture
def processor():
    """Fixture to create image processor instance."""
    proc = GPUImageProcessor()
    yield proc
    proc.cleanup()


@pytest.fixture
def test_image():
    """Fixture to create test image."""
    return np.random.randint(
        0,
        255,
        (100, 100, 3),
        dtype=np.uint8
    )


def test_image_loading(processor, test_image):
    """Test image loading to GPU."""
    # Test numpy array
    tensor = processor.load_image(test_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == 'cuda'

    # Test saving and loading from file
    test_path = "test_save.jpg"
    try:
        processor.save_image(tensor, test_path)
        loaded = processor.load_image(test_path)
        assert isinstance(loaded, torch.Tensor)
        assert loaded.device.type == 'cuda'

    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def test_preprocessing(processor, test_image):
    """Test image preprocessing."""
    tensor = processor.load_image(test_image)

    # Test normalization
    normalized = processor.preprocess(tensor, normalize=True)
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0

    # Test resizing
    resized = processor.preprocess(
        tensor,
        normalize=False,
        resize=(50, 50)
    )
    assert resized.shape[-2:] == (50, 50)


def test_filters(processor, test_image):
    """Test filter application."""
    tensor = processor.load_image(test_image)

    # Test multiple filters
    filtered = processor.apply_filters(tensor, [
        {
            'type': 'gaussian_blur',
            'params': {'kernel_size': 3, 'sigma': 1.0}
        },
        {
            'type': 'sharpen',
            'params': {'strength': 1.0}
        },
        {
            'type': 'contrast',
            'params': {'factor': 1.2}
        }
    ])

    assert isinstance(filtered, torch.Tensor)
    assert filtered.shape == tensor.shape


def test_feature_detection(processor, test_image):
    """Test feature detection."""
    tensor = processor.load_image(test_image)
    tensor = processor.preprocess(tensor, normalize=True)

    # Test Sobel edge detection
    edges = processor.detect_features(
        tensor,
        method='sobel',
        threshold=0.1
    )

    assert isinstance(edges, torch.Tensor)
    assert edges.dtype == torch.bool


def test_image_enhancement(processor, test_image):
    """Test image enhancement."""
    tensor = processor.load_image(test_image)
    tensor = processor.preprocess(tensor, normalize=True)

    # Test super-resolution
    enhanced = processor.enhance_image(
        tensor,
        method='super_resolution',
        scale_factor=2
    )

    assert isinstance(enhanced, torch.Tensor)
    assert enhanced.shape[-2:] == (
        tensor.shape[-2] * 2,
        tensor.shape[-1] * 2
    )


def test_image_combination(processor, test_image):
    """Test image combination."""
    tensors = [
        processor.load_image(test_image) for _ in range(3)
    ]

    # Test different combination methods
    for method in ['average', 'maximum', 'minimum']:
        combined = processor.combine_images(tensors, method=method)
        assert isinstance(combined, torch.Tensor)
        assert combined.shape == tensors[0].shape


def test_batch_processing(processor, test_image):
    """Test batch image processing."""
    images = [test_image] * 3

    operations = [
        {
            'type': 'preprocess',
            'params': {
                'normalize': True,
                'resize': (50, 50)
            }
        },
        {
            'type': 'filters',
            'params': {
                'filters': [
                    {
                        'type': 'gaussian_blur',
                        'params': {
                            'kernel_size': 3,
                            'sigma': 1.0
                        }
                    }
                ]
            }
        },
        {
            'type': 'detect_features',
            'params': {
                'method': 'sobel',
                'threshold': 0.1
            }
        }
    ]

    results = processor.process_batch(images, operations)
    assert len(results) == len(images)
    for result in results:
        assert isinstance(result, torch.Tensor)
        assert result.device.type == 'cuda'


def test_error_handling(processor):
    """Test error handling."""
    # Test invalid image path
    with pytest.raises(Exception):
        processor.load_image("nonexistent.jpg")

    # Test invalid filter type
    tensor = processor.load_image(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )
    with pytest.raises(Exception):
        processor.apply_filters(tensor, [
            {'type': 'invalid_filter'}
        ])

    # Test invalid feature detection method
    with pytest.raises(ValueError):
        processor.detect_features(
            tensor,
            method='invalid_method'
        )

    # Test invalid enhancement method
    with pytest.raises(ValueError):
        processor.enhance_image(
            tensor,
            method='invalid_method'
        )

    # Test invalid combination method
    with pytest.raises(ValueError):
        processor.combine_images(
            [tensor],
            method='invalid_method'
        )


@pytest.mark.stress
def test_stress_batch_processing(processor):
    """Stress test batch processing."""
    # Create large batch of images
    images = [
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        for _ in range(10)
    ]

    operations = [
        {
            'type': 'preprocess',
            'params': {
                'normalize': True,
                'resize': (256, 256)
            }
        },
        {
            'type': 'filters',
            'params': {
                'filters': [
                    {
                        'type': 'gaussian_blur',
                        'params': {
                            'kernel_size': 5,
                            'sigma': 1.0
                        }
                    },
                    {
                        'type': 'sharpen',
                        'params': {
                            'strength': 1.5
                        }
                    },
                    {
                        'type': 'contrast',
                        'params': {
                            'factor': 1.2
                        }
                    }
                ]
            }
        },
        {
            'type': 'enhance',
            'params': {
                'method': 'super_resolution',
                'scale_factor': 2
            }
        }
    ]

    results = processor.process_batch(images, operations)
    assert len(results) == len(images)

    # Verify results
    for result in results:
        assert isinstance(result, torch.Tensor)
        assert result.shape[-2:] == (512, 512)
        assert not torch.isnan(result).any()


@pytest.mark.stress
def test_stress_memory_management(processor):
    """Stress test memory management."""
    initial_memory = torch.cuda.memory_allocated()

    # Process multiple large images
    for _ in range(5):
        image = np.random.randint(
            0,
            255,
            (1024, 1024, 3),
            dtype=np.uint8
        )
        tensor = processor.load_image(image)

        # Apply memory-intensive operations
        tensor = processor.preprocess(
            tensor,
            normalize=True,
            resize=(2048, 2048)
        )

        tensor = processor.apply_filters(tensor, [
            {
                'type': 'gaussian_blur',
                'params': {
                    'kernel_size': 7,
                    'sigma': 1.5
                }
            }
        ])

        tensor = processor.enhance_image(
            tensor,
            scale_factor=2
        )

        del tensor
        torch.cuda.empty_cache()

    # Check memory cleanup
    final_memory = torch.cuda.memory_allocated()
    assert final_memory <= initial_memory * 1.1  # Allow for small overhead
