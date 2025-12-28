"""
Unit tests for PipelineHook.

These tests verify the main user-facing class correctly wraps pipelines,
auto-detects adapters, and maintains API compatibility.
"""

import numpy as np
import pytest

# Skip all tests if Albumentations is not installed
albumentations = pytest.importorskip("albumentations")

from augmentlens.core.pipeline_hook import PipelineHook, UnsupportedPipelineError
from augmentlens.core.snapshot_state import SnapshotState


class TestPipelineHookConstruction:
    """Tests for PipelineHook initialization."""

    def test_auto_detects_albumentations(self) -> None:
        """Should auto-detect Albumentations pipelines."""
        pipeline = albumentations.Compose([albumentations.HorizontalFlip(p=0.5)])
        hook = PipelineHook(pipeline)
        
        assert hook._adapter is not None
        assert hook._adapter.target_library_name == "albumentations"

    def test_accepts_explicit_type(self) -> None:
        """Should accept explicit pipeline_type parameter."""
        pipeline = albumentations.HorizontalFlip(p=0.5)
        hook = PipelineHook(pipeline, pipeline_type="albumentations")
        
        assert hook._adapter is not None
        assert hook._adapter.target_library_name == "albumentations"

    def test_rejects_none_pipeline(self) -> None:
        """None pipeline should raise UnsupportedPipelineError."""
        with pytest.raises(UnsupportedPipelineError, match="cannot be None"):
            PipelineHook(None)

    def test_rejects_unknown_pipeline(self) -> None:
        """Unknown pipeline type should raise UnsupportedPipelineError."""
        with pytest.raises(UnsupportedPipelineError, match="Cannot auto-detect"):
            PipelineHook(lambda x: x)

    def test_passthrough_mode_skips_adapter(self) -> None:
        """Passthrough mode should not create adapter."""
        pipeline = albumentations.HorizontalFlip(p=0.5)
        hook = PipelineHook(pipeline, passthrough=True)
        
        assert hook._adapter is None
        assert hook.passthrough_enabled is True

    def test_repr_shows_mode(self) -> None:
        """Repr should indicate capture/passthrough mode."""
        pipeline = albumentations.HorizontalFlip(p=0.5)
        
        hook_capture = PipelineHook(pipeline, passthrough=False)
        hook_passthrough = PipelineHook(pipeline, passthrough=True)
        
        assert "capture" in repr(hook_capture)
        assert "passthrough" in repr(hook_passthrough)


class TestPipelineHookCallable:
    """Tests for the __call__ method."""

    def test_returns_same_format_as_original(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Output format should match original Albumentations format."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline)
        
        result = hook(image=synthetic_rgb_frame)
        
        # Should return dict with 'image' key
        assert isinstance(result, dict)
        assert "image" in result
        assert isinstance(result["image"], np.ndarray)

    def test_captures_snapshot(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Should capture SnapshotState after call."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline)
        
        assert hook.latest_capture is None
        
        hook(image=synthetic_rgb_frame)
        
        assert hook.latest_capture is not None
        assert isinstance(hook.latest_capture, SnapshotState)

    def test_passthrough_skips_capture(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Passthrough mode should not capture snapshots."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline, passthrough=True)
        
        result = hook(image=synthetic_rgb_frame)
        
        # Should still work
        assert isinstance(result, dict)
        assert "image" in result
        
        # But no capture
        assert hook.latest_capture is None

    def test_transform_is_applied(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Transform should actually be applied to the image."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline)
        
        result = hook(image=synthetic_rgb_frame)
        
        # HorizontalFlip should flip the image
        expected = np.fliplr(synthetic_rgb_frame)
        np.testing.assert_array_equal(result["image"], expected)

    def test_mask_is_transformed(
        self, synthetic_rgb_frame: np.ndarray, synthetic_binary_mask: np.ndarray
    ) -> None:
        """Mask should be transformed along with image."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline)
        
        result = hook(image=synthetic_rgb_frame, mask=synthetic_binary_mask)
        
        assert "mask" in result
        expected_mask = np.fliplr(synthetic_binary_mask)
        np.testing.assert_array_equal(result["mask"], expected_mask)

    def test_snapshot_contains_raw_and_augmented(
        self, synthetic_rgb_frame: np.ndarray
    ) -> None:
        """Captured snapshot should have both raw and augmented frames."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline)
        
        hook(image=synthetic_rgb_frame)
        
        snapshot = hook.latest_capture
        assert snapshot is not None
        
        # Raw should be original
        np.testing.assert_array_equal(snapshot.raw_frame, synthetic_rgb_frame)
        
        # Augmented should be flipped
        expected_augmented = np.fliplr(synthetic_rgb_frame)
        np.testing.assert_array_equal(snapshot.augmented_frame, expected_augmented)


class TestPipelineHookDropInCompatibility:
    """Tests verifying PipelineHook is a true drop-in replacement."""

    def test_identical_output_to_raw_pipeline(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Output should be byte-for-byte identical to raw pipeline."""
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline)
        
        # Run both
        raw_result = pipeline(image=synthetic_rgb_frame)
        hook_result = hook(image=synthetic_rgb_frame)
        
        # Should be identical
        np.testing.assert_array_equal(raw_result["image"], hook_result["image"])

    def test_passthrough_has_zero_capture_overhead(
        self, synthetic_rgb_frame: np.ndarray
    ) -> None:
        """Passthrough mode should call original pipeline directly."""
        call_count = 0
        original_call = albumentations.HorizontalFlip.__call__
        
        def counting_call(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_call(self, *args, **kwargs)
        
        pipeline = albumentations.HorizontalFlip(p=1.0)
        hook = PipelineHook(pipeline, passthrough=True)
        
        # Monkey-patch for this test
        albumentations.HorizontalFlip.__call__ = counting_call
        try:
            hook(image=synthetic_rgb_frame)
            # Should have called the original pipeline
            assert call_count == 1
        finally:
            # Restore
            albumentations.HorizontalFlip.__call__ = original_call


class TestPipelineHookWithBboxes:
    """Tests for bounding box handling through the hook."""

    def test_bboxes_flow_through(self, synthetic_rgb_frame: np.ndarray) -> None:
        """Bounding boxes should be captured and transformed."""
        pipeline = albumentations.Compose(
            [albumentations.HorizontalFlip(p=1.0)],
            bbox_params=albumentations.BboxParams(
                format="pascal_voc",
                label_fields=["bbox_labels"],
            ),
        )
        hook = PipelineHook(pipeline)
        
        bboxes = [(10, 50, 50, 100)]
        labels = ["test"]
        
        result = hook(image=synthetic_rgb_frame, bboxes=bboxes, bbox_labels=labels)
        
        # Bboxes should be in result
        assert "bboxes" in result
        assert len(result["bboxes"]) == 1
        
        # Should also be in snapshot
        snapshot = hook.latest_capture
        assert snapshot is not None
        assert len(snapshot.augmented_bounding_boxes) == 1
        
        # Should be flipped
        flipped_box = snapshot.augmented_bounding_boxes[0]
        assert flipped_box.x_min == 206  # 256 - 50
        assert flipped_box.x_max == 246  # 256 - 10
