"""
Pytest fixtures for AugmentLens test suite.

These fixtures generate synthetic test data to avoid external dependencies
and ensure reproducible test runs.
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_rgb_frame() -> np.ndarray:
    """
    Generate a 256x256 RGB test image with a centered white square.
    
    We use a simple geometric pattern rather than random noise because
    it makes visual debugging of failed tests trivial—you can immediately
    see if a transform broke the expected structure.
    """
    canvas_height, canvas_width = 256, 256
    frame_buffer = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Draw a centered 100x100 white square as our "object"
    square_size = 100
    top_left_y = (canvas_height - square_size) // 2
    top_left_x = (canvas_width - square_size) // 2
    frame_buffer[
        top_left_y : top_left_y + square_size,
        top_left_x : top_left_x + square_size,
    ] = 255
    
    return frame_buffer


@pytest.fixture
def synthetic_binary_mask() -> np.ndarray:
    """
    Generate a 256x256 binary mask matching the synthetic_rgb_frame's square.
    
    The mask uses 0/255 values (not 0/1) because that's what most CV pipelines
    expect when they treat masks as images.
    """
    canvas_height, canvas_width = 256, 256
    mask_buffer = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    square_size = 100
    top_left_y = (canvas_height - square_size) // 2
    top_left_x = (canvas_width - square_size) // 2
    mask_buffer[
        top_left_y : top_left_y + square_size,
        top_left_x : top_left_x + square_size,
    ] = 255
    
    return mask_buffer


@pytest.fixture
def synthetic_bounding_box() -> tuple[int, int, int, int]:
    """
    Return bounding box (x_min, y_min, x_max, y_max) for the centered square.
    
    This corresponds to the square drawn in synthetic_rgb_frame.
    """
    canvas_size = 256
    square_size = 100
    top_left = (canvas_size - square_size) // 2
    bottom_right = top_left + square_size
    return (top_left, top_left, bottom_right, bottom_right)
