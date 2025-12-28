"""
Unit tests for BaseRenderer.

These tests verify the image normalization utilities handle all
common CV formats correctly.
"""

import numpy as np
import pytest

from augmentlens.renderers.base_renderer import BaseRenderer
from augmentlens.core.snapshot_state import SnapshotState


class ConcreteRenderer(BaseRenderer):
    """Minimal concrete implementation for testing the ABC."""
    
    def render(self, snapshot_to_render: SnapshotState, **render_options):
        # Just return normalized frames for testing
        return {
            "raw": self._normalize_for_display(snapshot_to_render.raw_frame),
            "augmented": self._normalize_for_display(snapshot_to_render.augmented_frame),
        }


class TestNormalizeForDisplayDtype:
    """Tests for dtype conversion."""

    def test_uint8_passes_through(self) -> None:
        """uint8 input should remain uint8."""
        renderer = ConcreteRenderer()
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        result = renderer._normalize_for_display(image)
        
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, image)

    def test_float_0_to_1_scales_to_255(self) -> None:
        """float [0, 1] should scale to [0, 255] uint8."""
        renderer = ConcreteRenderer()
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        
        result = renderer._normalize_for_display(image)
        
        assert result.dtype == np.uint8
        assert result.max() == 127  # 0.5 * 255 = 127.5 → 127

    def test_float_0_to_255_casts_directly(self) -> None:
        """float [0, 255] should cast to uint8 without scaling."""
        renderer = ConcreteRenderer()
        image = np.ones((100, 100, 3), dtype=np.float32) * 128.0
        
        result = renderer._normalize_for_display(image)
        
        assert result.dtype == np.uint8
        assert result.max() == 128

    def test_clips_out_of_range_values(self) -> None:
        """Values outside [0, 255] should be clipped."""
        renderer = ConcreteRenderer()
        image = np.array([[[-10, 300, 128]]], dtype=np.float32)
        
        result = renderer._normalize_for_display(image)
        
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 0  # -10 clipped to 0
        assert result[0, 0, 1] == 255  # 300 clipped to 255
        assert result[0, 0, 2] == 128


class TestNormalizeForDisplayChannelOrder:
    """Tests for CHW/HWC conversion."""

    def test_hwc_unchanged(self) -> None:
        """HWC format should pass through unchanged."""
        renderer = ConcreteRenderer()
        # (H=100, W=80, C=3) - clearly HWC
        image = np.zeros((100, 80, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel
        
        result = renderer._normalize_for_display(image)
        
        assert result.shape == (100, 80, 3)
        assert result[0, 0, 0] == 255  # Red still in first channel

    def test_chw_transposed_to_hwc(self) -> None:
        """CHW format should be transposed to HWC."""
        renderer = ConcreteRenderer()
        # (C=3, H=100, W=80) - clearly CHW
        image = np.zeros((3, 100, 80), dtype=np.uint8)
        image[0, :, :] = 255  # Red channel
        
        result = renderer._normalize_for_display(image)
        
        assert result.shape == (100, 80, 3)
        assert result[0, 0, 0] == 255  # Red moved to last dim

    def test_grayscale_2d_gets_channel_dim(self) -> None:
        """2D grayscale should get channel dimension added."""
        renderer = ConcreteRenderer()
        image = np.zeros((100, 80), dtype=np.uint8)
        
        result = renderer._normalize_for_display(image)
        
        assert result.shape == (100, 80, 1)

    def test_single_channel_chw(self) -> None:
        """Single channel CHW (1, H, W) should transpose."""
        renderer = ConcreteRenderer()
        image = np.zeros((1, 100, 80), dtype=np.uint8)
        
        result = renderer._normalize_for_display(image)
        
        assert result.shape == (100, 80, 1)


class TestNormalizeForDisplayTensor:
    """Tests for PyTorch tensor handling."""

    def test_torch_tensor_converted(self) -> None:
        """PyTorch tensor should convert to numpy."""
        # Skip if torch not installed
        torch = pytest.importorskip("torch")
        
        renderer = ConcreteRenderer()
        tensor = torch.zeros((3, 100, 80), dtype=torch.float32)
        tensor[0, :, :] = 1.0  # Red channel, [0, 1] range
        
        result = renderer._normalize_for_display(tensor)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (100, 80, 3)
        assert result[0, 0, 0] == 255  # Scaled and transposed

    def test_torch_tensor_with_grad_detached(self) -> None:
        """Tensor with gradients should be detached safely."""
        torch = pytest.importorskip("torch")
        
        renderer = ConcreteRenderer()
        tensor = torch.zeros((3, 50, 50), requires_grad=True)
        
        # Should not raise
        result = renderer._normalize_for_display(tensor)
        
        assert isinstance(result, np.ndarray)


class TestNormalizeForDisplayEdgeCases:
    """Tests for edge cases and error handling."""

    def test_rgba_4_channel_preserved(self) -> None:
        """4-channel RGBA should be preserved."""
        renderer = ConcreteRenderer()
        image = np.zeros((100, 80, 4), dtype=np.uint8)
        
        result = renderer._normalize_for_display(image)
        
        assert result.shape == (100, 80, 4)

    def test_invalid_ndim_raises(self) -> None:
        """4D array should raise ValueError."""
        renderer = ConcreteRenderer()
        image = np.zeros((1, 3, 100, 80), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            renderer._normalize_for_display(image)

    def test_list_input_converted(self) -> None:
        """Nested list should convert to array."""
        renderer = ConcreteRenderer()
        image_list = [[[128, 128, 128]]]
        
        result = renderer._normalize_for_display(image_list)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
