"""
Unit tests for SnapshotState and BoundingBoxAnnotation.

These tests verify the defensive validation catches common mistakes
and that property accessors return correct values.
"""

import numpy as np
import pytest

from augmentlens.core.snapshot_state import (
    BoundingBoxAnnotation,
    SnapshotState,
    SnapshotValidationError,
)


class TestBoundingBoxAnnotation:
    """Tests for the BoundingBoxAnnotation dataclass."""

    def test_valid_box_creation(self) -> None:
        """Standard box should construct without error."""
        annotation = BoundingBoxAnnotation(x_min=10, y_min=20, x_max=100, y_max=200, label="car")
        assert annotation.area_pixels == 90 * 180
        assert not annotation.is_degenerate

    def test_degenerate_box_zero_width(self) -> None:
        """Box with zero width should be flagged as degenerate."""
        annotation = BoundingBoxAnnotation(x_min=50, y_min=20, x_max=50, y_max=100)
        assert annotation.is_degenerate
        assert annotation.area_pixels == 0

    def test_swapped_x_coordinates_raises(self) -> None:
        """Swapped x coords should raise with helpful message."""
        with pytest.raises(SnapshotValidationError, match="x_min.*cannot exceed.*x_max"):
            BoundingBoxAnnotation(x_min=100, y_min=20, x_max=10, y_max=200)

    def test_swapped_y_coordinates_raises(self) -> None:
        """Swapped y coords should raise with helpful message."""
        with pytest.raises(SnapshotValidationError, match="y_min.*cannot exceed.*y_max"):
            BoundingBoxAnnotation(x_min=10, y_min=200, x_max=100, y_max=20)


class TestSnapshotStateValidation:
    """Tests for SnapshotState input validation."""

    def test_valid_snapshot_minimal(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Minimal valid snapshot with just frames should work."""
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        assert snapshot.raw_frame_dimensions == (256, 256, 3)
        assert not snapshot.has_masks
        assert not snapshot.has_bounding_boxes

    def test_non_array_input_raises(self) -> None:
        """Passing a list instead of ndarray should raise."""
        fake_frame = [[1, 2, 3], [4, 5, 6]]
        real_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        with pytest.raises(SnapshotValidationError, match="must be a numpy ndarray"):
            SnapshotState(raw_frame=fake_frame, augmented_frame=real_frame)  # type: ignore[arg-type]

    def test_empty_array_raises(self) -> None:
        """Empty array should raise with data loader hint."""
        empty_frame = np.array([])
        real_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        with pytest.raises(SnapshotValidationError, match="data loader"):
            SnapshotState(raw_frame=empty_frame, augmented_frame=real_frame)

    def test_zero_dimension_raises(self) -> None:
        """Array with zero height produces size=0, caught by empty check."""
        zero_height_frame = np.zeros((0, 100, 3), dtype=np.uint8)
        real_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Zero height means size=0, so the empty check fires first
        with pytest.raises(SnapshotValidationError, match="is empty"):
            SnapshotState(raw_frame=zero_height_frame, augmented_frame=real_frame)

    def test_mask_dimension_mismatch_raises(
        self, synthetic_rgb_frame: np.ndarray
    ) -> None:
        """Mask with wrong dimensions should raise."""
        wrong_size_mask = np.zeros((100, 100), dtype=np.uint8)
        
        with pytest.raises(SnapshotValidationError, match="do not match"):
            SnapshotState(
                raw_frame=synthetic_rgb_frame,
                augmented_frame=synthetic_rgb_frame.copy(),
                raw_segmentation_mask=wrong_size_mask,
            )


class TestSnapshotStateProperties:
    """Tests for SnapshotState computed properties."""

    def test_dimensions_changed_same_size(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Same-size frames should report no dimension change."""
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        assert not snapshot.dimensions_changed

    def test_dimensions_changed_different_size(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Cropped frame should report dimension change."""
        cropped_frame = synthetic_rgb_frame[:128, :128, :]
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=cropped_frame,
        )
        assert snapshot.dimensions_changed

    def test_mask_survival_ratio_full_survival(
        self, synthetic_rgb_frame: np.ndarray, synthetic_binary_mask: np.ndarray
    ) -> None:
        """Identical masks should give 100% survival."""
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_segmentation_mask=synthetic_binary_mask,
            augmented_segmentation_mask=synthetic_binary_mask.copy(),
        )
        assert snapshot.quantify_mask_integrity() == 1.0

    def test_mask_survival_ratio_partial(
        self, synthetic_rgb_frame: np.ndarray, synthetic_binary_mask: np.ndarray
    ) -> None:
        """Half the mask zeroed should give 50% survival."""
        half_mask = synthetic_binary_mask.copy()
        half_mask[:128, :] = 0  # Zero out top half
        
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_segmentation_mask=synthetic_binary_mask,
            augmented_segmentation_mask=half_mask,
        )
        survival = snapshot.quantify_mask_integrity()
        assert survival is not None
        # The square is centered, so zeroing top half removes ~half the mask
        assert 0.4 < survival < 0.6

    def test_mask_survival_ratio_no_masks(self, synthetic_rgb_frame: np.ndarray) -> None:
        """No masks should return None."""
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        assert snapshot.quantify_mask_integrity() is None

    def test_has_bounding_boxes(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Snapshot with boxes should report has_bounding_boxes=True."""
        box = BoundingBoxAnnotation(x_min=10, y_min=10, x_max=50, y_max=50)
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_bounding_boxes=(box,),
        )
        assert snapshot.has_bounding_boxes


class TestSnapshotStateImmutability:
    """Verify that SnapshotState is truly immutable."""

    def test_cannot_modify_attribute(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Attempting to modify a field should raise."""
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        with pytest.raises(AttributeError):
            snapshot.transform_name = "hacked"  # type: ignore[misc]
