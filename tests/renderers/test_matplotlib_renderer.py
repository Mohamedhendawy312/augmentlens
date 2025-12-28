"""
Unit tests for MatplotlibRenderer.

These tests verify the renderer produces correct visualizations
and handles all SnapshotState variants.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from augmentlens.core.snapshot_state import BoundingBoxAnnotation, SnapshotState
from augmentlens.renderers.matplotlib_renderer import MatplotlibRenderer


# Use non-interactive backend for testing
import matplotlib
matplotlib.use("Agg")


class TestMatplotlibRendererBasic:
    """Basic functionality tests."""

    def test_returns_figure_object(self, synthetic_rgb_frame: np.ndarray) -> None:
        """render() should return a matplotlib Figure."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        fig = renderer.render(snapshot, show=False)
        
        assert isinstance(fig, Figure)
        # Cleanup
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_creates_two_subplots(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Figure should have exactly 2 axes (original + augmented)."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        fig = renderer.render(snapshot, show=False)
        
        assert len(fig.axes) == 2
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_axes_are_turned_off(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Both axes should have axis('off') for clean display."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        fig = renderer.render(snapshot, show=False)
        
        for ax in fig.axes:
            # Check that axis visibility is off
            assert not ax.axison
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestMatplotlibRendererTitles:
    """Tests for title behavior."""

    def test_left_title_is_original(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Left panel should be titled 'Original'."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        fig = renderer.render(snapshot, show=False)
        
        left_title = fig.axes[0].get_title()
        assert left_title == "Original"
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_right_title_includes_transform_name(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Right panel should show transform name if available."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            transform_name="Compose([HorizontalFlip])",
        )
        
        fig = renderer.render(snapshot, show=False)
        
        right_title = fig.axes[1].get_title()
        assert "Compose([HorizontalFlip])" in right_title
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_right_title_truncates_long_names(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Long transform names should be truncated."""
        renderer = MatplotlibRenderer()
        long_name = "Compose([HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast])"
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            transform_name=long_name,
        )
        
        fig = renderer.render(snapshot, show=False)
        
        right_title = fig.axes[1].get_title()
        assert len(right_title) <= 55  # "Augmented: " + 40 chars + "..."
        assert "..." in right_title
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestMatplotlibRendererBboxes:
    """Tests for bounding box rendering."""

    def test_bboxes_create_patches(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Bounding boxes should add Rectangle patches to axes."""
        renderer = MatplotlibRenderer()
        bbox = BoundingBoxAnnotation(x_min=10, y_min=20, x_max=100, y_max=150, label="test")
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_bounding_boxes=(bbox,),
            augmented_bounding_boxes=(bbox,),
        )
        
        fig = renderer.render(snapshot, show=False)
        
        # Each axis should have patches
        left_patches = fig.axes[0].patches
        right_patches = fig.axes[1].patches
        
        assert len(left_patches) >= 1
        assert len(right_patches) >= 1
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_bboxes_can_be_disabled(self, synthetic_rgb_frame: np.ndarray) -> None:
        """show_bboxes=False should suppress bbox rendering."""
        renderer = MatplotlibRenderer()
        bbox = BoundingBoxAnnotation(x_min=10, y_min=20, x_max=100, y_max=150)
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_bounding_boxes=(bbox,),
        )
        
        fig = renderer.render(snapshot, show=False, show_bboxes=False)
        
        # No patches should be added
        assert len(fig.axes[0].patches) == 0
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestMatplotlibRendererMasks:
    """Tests for mask overlay rendering."""

    def test_mask_overlay_added(
        self, synthetic_rgb_frame: np.ndarray, synthetic_binary_mask: np.ndarray
    ) -> None:
        """Mask should add an additional imshow layer."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_segmentation_mask=synthetic_binary_mask,
            augmented_segmentation_mask=synthetic_binary_mask.copy(),
        )
        
        fig = renderer.render(snapshot, show=False)
        
        # Each axis should have 2 images (base + mask overlay)
        left_images = fig.axes[0].images
        assert len(left_images) >= 2
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_masks_can_be_disabled(
        self, synthetic_rgb_frame: np.ndarray, synthetic_binary_mask: np.ndarray
    ) -> None:
        """show_masks=False should suppress mask overlay."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_segmentation_mask=synthetic_binary_mask,
        )
        
        fig = renderer.render(snapshot, show=False, show_masks=False)
        
        # Should only have 1 image (base only)
        left_images = fig.axes[0].images
        assert len(left_images) == 1
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestMatplotlibRendererSave:
    """Tests for file saving."""

    def test_saves_to_path(
        self, synthetic_rgb_frame: np.ndarray, tmp_path
    ) -> None:
        """save_path should create a file."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        save_file = tmp_path / "test_output.png"
        fig = renderer.render(snapshot, show=False, save_path=save_file)
        
        assert save_file.exists()
        assert save_file.stat().st_size > 0  # Not empty
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_creates_parent_directories(
        self, synthetic_rgb_frame: np.ndarray, tmp_path
    ) -> None:
        """Should create parent directories if they don't exist."""
        renderer = MatplotlibRenderer()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        save_file = tmp_path / "nested" / "dirs" / "output.png"
        fig = renderer.render(snapshot, show=False, save_path=save_file)
        
        assert save_file.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
