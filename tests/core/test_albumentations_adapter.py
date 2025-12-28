"""
Unit tests for AlbumentationsAdapter.

These tests verify the adapter correctly wraps Albumentations pipelines,
handles edge cases, and provides helpful error messages.
"""

import numpy as np
import pytest

# Skip all tests if Albumentations is not installed
albumentations = pytest.importorskip("albumentations")

from augmentlens.core.adapters.albumentations_adapter import AlbumentationsAdapter
from augmentlens.core.adapters.base_adapter import AdapterCompatibilityError
from augmentlens.core.snapshot_state import BoundingBoxAnnotation, SnapshotState


class TestAlbumentationsAdapterConstruction:
    """Tests for adapter initialization."""

    def test_accepts_compose_pipeline(self) -> None:
        """Compose pipeline should be accepted."""
        pipeline = albumentations.Compose([albumentations.HorizontalFlip(p=0.5)])
        adapter = AlbumentationsAdapter(pipeline)
        assert adapter.target_library_name == "albumentations"

    def test_accepts_single_transform(self) -> None:
        """Single transform (not Compose) should be accepted."""
        transform = albumentations.HorizontalFlip(p=1.0)
        adapter = AlbumentationsAdapter(transform)
        assert adapter.target_library_name == "albumentations"

    def test_rejects_non_albumentations_object(self) -> None:
        """Non-Albumentations object should raise TypeError."""
        with pytest.raises(TypeError, match="Expected an Albumentations pipeline"):
            AlbumentationsAdapter(lambda x: x)

    def test_rejects_none(self) -> None:
        """None should raise TypeError."""
        with pytest.raises(TypeError, match="Expected an Albumentations pipeline"):
            AlbumentationsAdapter(None)

    def test_repr_includes_pipeline_info(self) -> None:
        """Repr should show useful debugging info."""
        pipeline = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
        ])
        adapter = AlbumentationsAdapter(pipeline)
        repr_str = repr(adapter)
        
        assert "AlbumentationsAdapter" in repr_str
        assert "backend=albumentations" in repr_str


class TestAlbumentationsAdapterApply:
    """Tests for the apply() method."""

    def test_basic_transform_applies(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Basic transform should produce augmented output."""
        # Use a deterministic transform for testing
        pipeline = albumentations.HorizontalFlip(p=1.0)
        adapter = AlbumentationsAdapter(pipeline)
        
        input_snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),  # Placeholder
        )
        
        result = adapter.apply(input_snapshot)
        
        # HorizontalFlip should produce a different image
        assert isinstance(result, SnapshotState)
        assert result.augmented_frame is not None
        # Verify the flip happened - compare with manually flipped
        expected_flipped = np.fliplr(synthetic_rgb_frame)
        np.testing.assert_array_equal(result.augmented_frame, expected_flipped)

    def test_mask_is_transformed(
        self, synthetic_rgb_frame: np.ndarray, synthetic_binary_mask: np.ndarray
    ) -> None:
        """Mask should be transformed alongside image."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        adapter = AlbumentationsAdapter(pipeline)
        
        input_snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_segmentation_mask=synthetic_binary_mask,
        )
        
        result = adapter.apply(input_snapshot)
        
        # Mask should also be flipped
        assert result.augmented_segmentation_mask is not None
        expected_flipped_mask = np.fliplr(synthetic_binary_mask)
        np.testing.assert_array_equal(result.augmented_segmentation_mask, expected_flipped_mask)

    def test_mask_output_is_numpy_array(
        self, synthetic_rgb_frame: np.ndarray, synthetic_binary_mask: np.ndarray
    ) -> None:
        """Output mask should always be numpy array, not list."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        adapter = AlbumentationsAdapter(pipeline)
        
        input_snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_segmentation_mask=synthetic_binary_mask,
        )
        
        result = adapter.apply(input_snapshot)
        
        assert isinstance(result.augmented_segmentation_mask, np.ndarray)

    def test_transform_name_is_captured(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Transform name should be recorded in output snapshot."""
        pipeline = albumentations.Compose([albumentations.HorizontalFlip(p=1.0)])
        adapter = AlbumentationsAdapter(pipeline)
        
        input_snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        result = adapter.apply(input_snapshot)
        
        assert "Compose" in result.transform_name
        assert "HorizontalFlip" in result.transform_name


class TestAlbumentationsAdapterBoundingBoxes:
    """Tests for bounding box handling."""

    def test_bboxes_are_transformed(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Bounding boxes should be transformed with the image."""
        # Use a pipeline with bbox support - labels via label_fields
        pipeline = albumentations.Compose(
            [albumentations.HorizontalFlip(p=1.0)],
            bbox_params=albumentations.BboxParams(
                format="pascal_voc",
                label_fields=["bbox_labels"],
            ),
        )
        adapter = AlbumentationsAdapter(pipeline)
        
        # Create a box on the left side of the image
        original_box = BoundingBoxAnnotation(x_min=10, y_min=50, x_max=50, y_max=100, label="test")
        
        input_snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_bounding_boxes=(original_box,),
        )
        
        result = adapter.apply(input_snapshot)
        
        # After horizontal flip, box should move to the right side
        assert len(result.augmented_bounding_boxes) == 1
        flipped_box = result.augmented_bounding_boxes[0]
        
        # Original x_min=10, x_max=50 on 256-wide image
        # After flip: new_x_min = width - old_x_max = 256 - 50 = 206
        #             new_x_max = width - old_x_min = 256 - 10 = 246
        assert flipped_box.x_min == 206
        assert flipped_box.x_max == 246
        # Label should be preserved
        assert flipped_box.label == "test"


class TestBboxCoordinateClamping:
    """Tests for bbox sanitization handling floating-point edge cases."""

    def test_clamp_fixes_negative_coordinates(self) -> None:
        """Negative coords from rotation should be clamped to 0."""
        from augmentlens.core.adapters.albumentations_adapter import _clamp_bbox_coordinates
        
        # Simulate floating-point error from rotation: x_min slightly negative
        raw_bboxes = [(-1e-9, 10, 50, 100)]
        
        clamped = _clamp_bbox_coordinates(raw_bboxes, frame_height=256, frame_width=256)
        
        assert clamped[0][0] == 0.0  # x_min clamped to 0

    def test_clamp_fixes_overflow_coordinates(self) -> None:
        """Coords exceeding image bounds should be clamped."""
        from augmentlens.core.adapters.albumentations_adapter import _clamp_bbox_coordinates
        
        # Simulate x_max slightly exceeding image width
        raw_bboxes = [(10, 20, 256.001, 100)]
        
        clamped = _clamp_bbox_coordinates(raw_bboxes, frame_height=256, frame_width=256)
        
        assert clamped[0][2] == 256.0  # x_max clamped to width

    def test_clamp_handles_normalized_coordinates(self) -> None:
        """Normalized coords (0-1) should be clamped to [0, 1]."""
        from augmentlens.core.adapters.albumentations_adapter import _clamp_bbox_coordinates
        
        # Normalized coords with slight overflow
        raw_bboxes = [(-0.001, 0.1, 1.002, 0.9)]
        
        clamped = _clamp_bbox_coordinates(raw_bboxes, frame_height=256, frame_width=256)
        
        assert clamped[0][0] == 0.0  # x_min clamped
        assert clamped[0][2] == 1.0  # x_max clamped

    def test_clamp_preserves_valid_coordinates(self) -> None:
        """Valid coordinates should pass through unchanged."""
        from augmentlens.core.adapters.albumentations_adapter import _clamp_bbox_coordinates
        
        raw_bboxes = [(10, 20, 100, 200)]
        
        clamped = _clamp_bbox_coordinates(raw_bboxes, frame_height=256, frame_width=256)
        
        assert clamped[0] == (10, 20, 100, 200)


class TestAlbumentationsAdapterValidation:
    """Tests for validate_compatibility()."""

    def test_rejects_invalid_dtype(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Non-standard dtype should raise AdapterCompatibilityError."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        adapter = AlbumentationsAdapter(pipeline)
        
        # Convert to float64, which Albumentations doesn't like
        float64_frame = synthetic_rgb_frame.astype(np.float64)
        
        snapshot = SnapshotState(
            raw_frame=float64_frame,
            augmented_frame=float64_frame.copy(),
        )
        
        with pytest.raises(AdapterCompatibilityError, match="uint8 or float32"):
            adapter.validate_compatibility(snapshot)

    def test_accepts_uint8(self, synthetic_rgb_frame: np.ndarray) -> None:
        """uint8 should pass validation."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        adapter = AlbumentationsAdapter(pipeline)
        
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,  # Already uint8
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        # Should not raise
        adapter.validate_compatibility(snapshot)


class TestAlbumentationsAdapterErrorHandling:
    """Tests for error wrapping."""

    def test_pipeline_crash_wrapped_with_context(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Pipeline crashes should be wrapped with helpful context."""
        # Create a pipeline that expects bboxes but we pass bboxes without proper config
        # This will crash because the pipeline has no bbox_params but receives bboxes
        pipeline = albumentations.Compose(
            [albumentations.HorizontalFlip(p=1.0)],
            # No bbox_params - will crash when bboxes are passed
        )
        adapter = AlbumentationsAdapter(pipeline)
        
        # Provide bboxes to a pipeline that doesn't expect them
        box = BoundingBoxAnnotation(x_min=10, y_min=50, x_max=50, y_max=100)
        
        input_snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_bounding_boxes=(box,),
        )
        
        with pytest.raises(AdapterCompatibilityError) as exc_info:
            adapter.apply(input_snapshot)
        
        error_message = str(exc_info.value)
        # Error should include helpful context
        assert "crashed during execution" in error_message
        assert "frame shape" in error_message.lower()
