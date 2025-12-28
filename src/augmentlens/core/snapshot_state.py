"""
Immutable data container for before/after augmentation snapshots.

This module defines the core data structure that flows through the entire
AugmentLens pipeline. Every hook, analyzer, and renderer consumes SnapshotState.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


class SnapshotValidationError(ValueError):
    """
    Raised when SnapshotState receives invalid or inconsistent data.
    
    We use a custom exception so callers can catch specifically this error
    without swallowing unrelated ValueErrors from numpy or other libraries.
    """
    pass


@dataclass(frozen=True, slots=True)
class BoundingBoxAnnotation:
    """
    Axis-aligned bounding box in pixel coordinates.
    
    We store as (x_min, y_min, x_max, y_max) because that's what most CV
    frameworks expect. COCO uses (x, y, width, height) but conversion is
    trivial and this format avoids off-by-one errors in area calculations.
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str = ""
    
    def __post_init__(self) -> None:
        if self.x_min > self.x_max:
            raise SnapshotValidationError(
                f"Bounding box x_min ({self.x_min}) cannot exceed x_max ({self.x_max}). "
                "Did you swap the coordinates?"
            )
        if self.y_min > self.y_max:
            raise SnapshotValidationError(
                f"Bounding box y_min ({self.y_min}) cannot exceed y_max ({self.y_max}). "
                "Did you swap the coordinates?"
            )
    
    @property
    def area_pixels(self) -> int:
        """Box area in pixels. Returns 0 for degenerate boxes."""
        return max(0, (self.x_max - self.x_min) * (self.y_max - self.y_min))
    
    @property
    def is_degenerate(self) -> bool:
        """True if box has zero area (collapsed to line or point)."""
        return self.area_pixels == 0


def _validate_image_array(
    array_to_check: np.ndarray,
    array_name: str,
    expected_ndim: int | None = None,
) -> None:
    """
    Defensive validation for image/mask arrays.
    
    We centralize this logic because the same checks apply to both
    raw_frame and augmented_frame, and copy-pasting validation is
    how bugs sneak into production code.
    """
    if not isinstance(array_to_check, np.ndarray):
        raise SnapshotValidationError(
            f"{array_name} must be a numpy ndarray, got {type(array_to_check).__name__}. "
            "If you're passing a PIL Image, convert it with np.array() first."
        )
    
    if array_to_check.size == 0:
        raise SnapshotValidationError(
            f"{array_name} is empty (size=0). This usually means your data loader "
            "returned a broken sample or your crop removed the entire image."
        )
    
    # Catch the classic "height is 0" bug from bad resizing
    if any(dim == 0 for dim in array_to_check.shape):
        raise SnapshotValidationError(
            f"{array_name} has a zero-dimension: shape={array_to_check.shape}. "
            "This typically happens when a resize operation receives an invalid target size."
        )
    
    if expected_ndim is not None and array_to_check.ndim != expected_ndim:
        raise SnapshotValidationError(
            f"{array_name} expected {expected_ndim}D array, got {array_to_check.ndim}D "
            f"with shape {array_to_check.shape}."
        )


def _validate_mask_matches_image(
    mask_array: np.ndarray,
    image_array: np.ndarray,
    mask_name: str,
    image_name: str,
) -> None:
    """
    Ensure mask spatial dimensions match the corresponding image.
    
    This catches the subtle bug where you forget to apply the same
    resize to your mask as to your image.
    """
    # Masks are 2D (H, W), images are 3D (H, W, C)
    mask_height, mask_width = mask_array.shape[:2]
    image_height, image_width = image_array.shape[:2]
    
    if (mask_height, mask_width) != (image_height, image_width):
        raise SnapshotValidationError(
            f"{mask_name} spatial dimensions ({mask_height}x{mask_width}) do not match "
            f"{image_name} dimensions ({image_height}x{image_width}). "
            "Ensure your augmentation pipeline transforms masks alongside images."
        )


@dataclass(frozen=True, slots=True)
class SnapshotState:
    """
    Immutable container holding before/after state of an augmentation step.
    
    This is the core data structure that flows through AugmentLens:
    - Hooks capture raw_frame and produce augmented_frame
    - Analyzers read both to compute diagnostics
    - Renderers visualize the comparison
    
    We use frozen=True because snapshots should never be mutated after creation.
    If you need to modify data, create a new SnapshotState instead.
    
    Slots are enabled for memory efficiency when processing large batches.
    """
    
    # Required: the actual image data
    raw_frame: np.ndarray
    augmented_frame: np.ndarray
    
    # Optional: segmentation masks (2D arrays, same H/W as frames)
    raw_segmentation_mask: np.ndarray | None = None
    augmented_segmentation_mask: np.ndarray | None = None
    
    # Optional: bounding boxes
    raw_bounding_boxes: tuple[BoundingBoxAnnotation, ...] = field(default_factory=tuple)
    augmented_bounding_boxes: tuple[BoundingBoxAnnotation, ...] = field(default_factory=tuple)
    
    # Metadata about the transform that was applied
    transform_name: str = "unknown"
    transform_parameters: dict[str, Any] = field(default_factory=dict)
    
    # Unique identifier for this snapshot (useful for caching/logging)
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    
    # Timestamp for ordering and debugging
    captured_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> None:
        """
        Validate all inputs on construction.
        
        We validate eagerly rather than lazily because catching bad data
        at the point of creation makes debugging 10x easier than discovering
        corruption three stages later in the pipeline.
        """
        # Validate required frames
        _validate_image_array(self.raw_frame, "raw_frame")
        _validate_image_array(self.augmented_frame, "augmented_frame")
        
        # TODO: Handle grayscale (2D) images - currently we assume 3D (H, W, C)
        # TODO: Handle 4-channel RGBA images correctly
        
        # Validate masks if provided
        if self.raw_segmentation_mask is not None:
            _validate_image_array(self.raw_segmentation_mask, "raw_segmentation_mask")
            _validate_mask_matches_image(
                self.raw_segmentation_mask,
                self.raw_frame,
                "raw_segmentation_mask",
                "raw_frame",
            )
        
        if self.augmented_segmentation_mask is not None:
            _validate_image_array(self.augmented_segmentation_mask, "augmented_segmentation_mask")
            _validate_mask_matches_image(
                self.augmented_segmentation_mask,
                self.augmented_frame,
                "augmented_segmentation_mask",
                "augmented_frame",
            )
    
    @property
    def raw_frame_dimensions(self) -> tuple[int, int, int]:
        """Returns (height, width, channels) of raw frame."""
        return self.raw_frame.shape  # type: ignore[return-value]
    
    @property
    def augmented_frame_dimensions(self) -> tuple[int, int, int]:
        """Returns (height, width, channels) of augmented frame."""
        return self.augmented_frame.shape  # type: ignore[return-value]
    
    @property
    def dimensions_changed(self) -> bool:
        """True if augmentation changed the image dimensions."""
        return self.raw_frame.shape != self.augmented_frame.shape
    
    @property
    def has_masks(self) -> bool:
        """True if this snapshot includes segmentation masks."""
        return (
            self.raw_segmentation_mask is not None
            or self.augmented_segmentation_mask is not None
        )
    
    @property
    def has_bounding_boxes(self) -> bool:
        """True if this snapshot includes bounding box annotations."""
        return bool(self.raw_bounding_boxes or self.augmented_bounding_boxes)
    
    def quantify_mask_integrity(self) -> float | None:
        """
        Calculate what fraction of the original mask survived augmentation.
        
        Returns None if masks aren't available. Returns 0.0 if the original
        mask was empty (to avoid division by zero).
        
        This is a key diagnostic: if this returns < 0.5, the augmentation
        probably destroyed your annotation.
        """
        if self.raw_segmentation_mask is None or self.augmented_segmentation_mask is None:
            return None
        
        raw_nonzero_count = np.count_nonzero(self.raw_segmentation_mask)
        if raw_nonzero_count == 0:
            # Original mask was empty, survival ratio is meaningless
            return 0.0
        
        augmented_nonzero_count = np.count_nonzero(self.augmented_segmentation_mask)
        return augmented_nonzero_count / raw_nonzero_count
