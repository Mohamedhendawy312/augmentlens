"""
Unit tests for BaseAdapter abstract base class.

These tests verify that the ABC contract is enforced correctly
and that incomplete implementations fail appropriately.
"""

import numpy as np
import pytest

from augmentlens.core.adapters.base_adapter import (
    AdapterCompatibilityError,
    BaseAdapter,
)
from augmentlens.core.snapshot_state import SnapshotState


class IncompleteAdapter(BaseAdapter):
    """Adapter that doesn't implement required methods - should fail."""
    pass


class MinimalCompleteAdapter(BaseAdapter):
    """Minimal adapter that implements only the required abstract members."""
    
    @property
    def target_library_name(self) -> str:
        return "test_library"
    
    def apply(self, input_snapshot: SnapshotState, **runtime_context) -> SnapshotState:
        # Just return the input unchanged for testing
        return input_snapshot
    
    def to_native_format(self, output_snapshot: SnapshotState) -> dict:
        return {"image": output_snapshot.augmented_frame}


class FullyImplementedAdapter(BaseAdapter):
    """Adapter that also overrides validate_compatibility()."""
    
    @property
    def target_library_name(self) -> str:
        return "full_test_library"
    
    def apply(self, input_snapshot: SnapshotState, **runtime_context) -> SnapshotState:
        return input_snapshot
    
    def to_native_format(self, output_snapshot: SnapshotState) -> dict:
        return {"image": output_snapshot.augmented_frame}
    
    def validate_compatibility(self, snapshot_to_validate: SnapshotState) -> None:
        # Example validation: reject empty bounding boxes
        if snapshot_to_validate.has_bounding_boxes:
            for bbox in snapshot_to_validate.raw_bounding_boxes:
                if bbox.is_degenerate:
                    raise AdapterCompatibilityError(
                        f"Degenerate bounding box not supported by {self.target_library_name}"
                    )


class TestBaseAdapterContract:
    """Verify the ABC enforces its contract correctly."""

    def test_incomplete_adapter_cannot_instantiate(self) -> None:
        """Adapter without required methods should fail on instantiation."""
        with pytest.raises(TypeError, match="abstract method"):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_minimal_adapter_can_instantiate(self) -> None:
        """Adapter with all abstract members implemented should work."""
        adapter = MinimalCompleteAdapter()
        assert adapter.target_library_name == "test_library"

    def test_repr_includes_library_name(self) -> None:
        """String representation should be informative with backend= format."""
        adapter = MinimalCompleteAdapter()
        repr_str = repr(adapter)
        assert "MinimalCompleteAdapter" in repr_str
        assert "backend=test_library" in repr_str


class TestValidateCompatibility:
    """Test the validate_compatibility() behavior."""

    def test_default_raises_not_implemented(
        self, synthetic_rgb_frame: np.ndarray
    ) -> None:
        """Base validate_compatibility() should raise NotImplementedError."""
        adapter = MinimalCompleteAdapter()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        with pytest.raises(NotImplementedError, match="has not implemented"):
            adapter.validate_compatibility(snapshot)

    def test_overridden_validation_passes(
        self, synthetic_rgb_frame: np.ndarray
    ) -> None:
        """Subclass with override should not raise for valid input."""
        adapter = FullyImplementedAdapter()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        # Should not raise - no bboxes means nothing to validate
        adapter.validate_compatibility(snapshot)

    def test_overridden_validation_catches_bad_input(
        self, synthetic_rgb_frame: np.ndarray
    ) -> None:
        """Subclass validation should catch incompatible data."""
        from augmentlens.core.snapshot_state import BoundingBoxAnnotation
        
        adapter = FullyImplementedAdapter()
        degenerate_box = BoundingBoxAnnotation(x_min=50, y_min=50, x_max=50, y_max=100)
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
            raw_bounding_boxes=(degenerate_box,),
        )
        
        with pytest.raises(AdapterCompatibilityError, match="Degenerate"):
            adapter.validate_compatibility(snapshot)


class TestApplyMethod:
    """Test the apply() abstract method."""

    def test_apply_returns_snapshot(self, synthetic_rgb_frame: np.ndarray) -> None:
        """apply() should return a SnapshotState."""
        adapter = MinimalCompleteAdapter()
        snapshot = SnapshotState(
            raw_frame=synthetic_rgb_frame,
            augmented_frame=synthetic_rgb_frame.copy(),
        )
        
        result = adapter.apply(snapshot)
        
        assert isinstance(result, SnapshotState)
