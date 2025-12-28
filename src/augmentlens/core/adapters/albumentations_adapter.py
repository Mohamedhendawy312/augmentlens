"""
Concrete adapter for Albumentations augmentation library.

This adapter wraps Albumentations Compose or BasicTransform objects,
handling the translation between our SnapshotState format and the
dictionary-based format Albumentations expects.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from augmentlens.core.adapters.base_adapter import (
    AdapterCompatibilityError,
    BaseAdapter,
)
from augmentlens.core.snapshot_state import (
    BoundingBoxAnnotation,
    SnapshotState,
)
from augmentlens.utils.type_guards import is_albumentations_pipeline


def _convert_bboxes_to_albumentations_format(
    annotation_boxes: tuple[BoundingBoxAnnotation, ...],
) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """
    Convert our BoundingBoxAnnotation objects to Albumentations format.
    
    Modern Albumentations expects bboxes as (x_min, y_min, x_max, y_max)
    with labels passed separately via label_fields parameter.
    
    Returns tuple of (bboxes_list, labels_list).
    """
    bboxes_only = [
        (box.x_min, box.y_min, box.x_max, box.y_max)
        for box in annotation_boxes
    ]
    labels_only = [box.label for box in annotation_boxes]
    return bboxes_only, labels_only


def _convert_bboxes_from_albumentations_result(
    albumentations_bboxes: list,
    bbox_labels: list[str] | None = None,
) -> tuple[BoundingBoxAnnotation, ...]:
    """
    Convert Albumentations output back to our BoundingBoxAnnotation format.
    
    Albumentations returns bboxes as lists of (x_min, y_min, x_max, y_max).
    Labels come from separate list if provided.
    We round to int because our BoundingBoxAnnotation uses pixel coordinates.
    """
    if bbox_labels is None:
        bbox_labels = [""] * len(albumentations_bboxes)
    
    converted_boxes = []
    for bbox_data, label in zip(albumentations_bboxes, bbox_labels):
        x_min, y_min, x_max, y_max = bbox_data[:4]
        
        converted_boxes.append(
            BoundingBoxAnnotation(
                x_min=int(round(x_min)),
                y_min=int(round(y_min)),
                x_max=int(round(x_max)),
                y_max=int(round(y_max)),
                label=str(label),
            )
        )
    return tuple(converted_boxes)


def _ensure_numpy_array(possibly_list_mask: Any) -> np.ndarray:
    """
    Force mask output to be a numpy array.
    
    Albumentations sometimes returns lists instead of arrays depending on
    the version and transform combination. We normalize to ndarray here
    to avoid downstream type errors.
    """
    if isinstance(possibly_list_mask, np.ndarray):
        return possibly_list_mask
    return np.array(possibly_list_mask)


def _clamp_bbox_coordinates(
    raw_bboxes: list,
    frame_height: int,
    frame_width: int,
) -> list:
    """
    Clamp bbox coordinates to valid image bounds.
    
    Rotation and affine transforms can produce floating-point errors like
    x_min=-1e-9 or x_max=width+0.001 which crash downstream training loops.
    We pull these back to valid bounds to be 'safe by default'.
    
    We assume pixel coordinates (pascal_voc format). If coordinates appear
    normalized (all values between 0 and 1), we clamp to [0, 1] instead.
    """
    if not raw_bboxes:
        return raw_bboxes
    
    # Detect if coordinates are normalized by checking if max value <= 1
    # (with small tolerance for floating point)
    all_coords = [coord for bbox in raw_bboxes for coord in bbox[:4]]
    is_normalized = max(all_coords) <= 1.5  # Generous threshold
    
    clamped_bboxes = []
    for bbox in raw_bboxes:
        x_min, y_min, x_max, y_max = bbox[:4]
        
        if is_normalized:
            # Clamp to [0, 1] for normalized coords
            x_min = max(0.0, min(1.0, x_min))
            y_min = max(0.0, min(1.0, y_min))
            x_max = max(0.0, min(1.0, x_max))
            y_max = max(0.0, min(1.0, y_max))
        else:
            # Clamp to [0, dimension] for pixel coords
            x_min = max(0.0, min(float(frame_width), x_min))
            y_min = max(0.0, min(float(frame_height), y_min))
            x_max = max(0.0, min(float(frame_width), x_max))
            y_max = max(0.0, min(float(frame_height), y_max))
        
        # Preserve any additional fields (like label index) beyond the coords
        clamped_bbox = (x_min, y_min, x_max, y_max) + tuple(bbox[4:])
        clamped_bboxes.append(clamped_bbox)
    
    return clamped_bboxes


class AlbumentationsAdapter(BaseAdapter):
    """
    Adapter for Albumentations augmentation pipelines.
    
    Wraps an Albumentations Compose or BasicTransform object, translating
    between our SnapshotState format and the dictionary format that
    Albumentations expects.
    
    Example
    -------
    >>> import albumentations as A
    >>> pipeline = A.Compose([A.HorizontalFlip(p=1.0)])
    >>> adapter = AlbumentationsAdapter(pipeline)
    >>> result = adapter.apply(input_snapshot)
    """
    
    def __init__(self, augmentation_pipeline: Any) -> None:
        """
        Initialize the adapter with an Albumentations pipeline.
        
        Parameters
        ----------
        augmentation_pipeline : albumentations.Compose or albumentations.BasicTransform
            The Albumentations transform or composition to wrap.
        
        Raises
        ------
        TypeError
            If the pipeline is not a valid Albumentations transform.
        """
        if not is_albumentations_pipeline(augmentation_pipeline):
            received_type = type(augmentation_pipeline).__name__
            raise TypeError(
                f"Expected an Albumentations pipeline (Compose or BasicTransform), "
                f"got {received_type}. Make sure you're passing an Albumentations "
                "transform, not a raw function or a transform from another library."
            )
        
        self._augmentation_pipeline = augmentation_pipeline
    
    @property
    def target_library_name(self) -> str:
        return "albumentations"
    
    @property
    def pipeline_repr(self) -> str:
        """Human-readable representation of the wrapped pipeline."""
        return repr(self._augmentation_pipeline)
    
    def validate_compatibility(self, snapshot_to_validate: SnapshotState) -> None:
        """
        Check if the snapshot is compatible with Albumentations.
        
        Albumentations requires:
        - Image as numpy array with shape (H, W, C) or (H, W) for grayscale
        - uint8 dtype for most transforms (some accept float32)
        """
        frame_to_check = snapshot_to_validate.raw_frame
        
        # Check dtype - most Albumentations transforms expect uint8
        if frame_to_check.dtype not in (np.uint8, np.float32):
            raise AdapterCompatibilityError(
                f"Albumentations typically expects uint8 or float32 images, "
                f"got dtype={frame_to_check.dtype}. Consider converting with "
                ".astype(np.uint8) before passing to the pipeline."
            )
        
        # TODO: Check bbox format compatibility with pipeline's bbox_params
    
    def apply(
        self,
        input_snapshot: SnapshotState,
        **runtime_context,
    ) -> SnapshotState:
        """
        Apply the Albumentations pipeline to the input snapshot.
        
        Unpacks SnapshotState into Albumentations dict format, runs the
        pipeline, and repacks results into a new SnapshotState.
        """
        albumentations_input_dict: dict[str, Any] = {
            "image": input_snapshot.raw_frame,
        }
        
        if input_snapshot.raw_segmentation_mask is not None:
            albumentations_input_dict["mask"] = input_snapshot.raw_segmentation_mask
        
        # Labels passed separately per modern Albumentations API
        if input_snapshot.raw_bounding_boxes:
            bboxes_coords, bbox_labels = _convert_bboxes_to_albumentations_format(
                input_snapshot.raw_bounding_boxes,
            )
            albumentations_input_dict["bboxes"] = bboxes_coords
            albumentations_input_dict["bbox_labels"] = bbox_labels
        
        # Execute the pipeline with error handling
        try:
            augmentation_result = self._augmentation_pipeline(**albumentations_input_dict)
        except Exception as pipeline_exception:
            # Wrap the exception with context about what input caused the crash
            raise AdapterCompatibilityError(
                f"Albumentations pipeline crashed during execution. "
                f"Input frame shape: {input_snapshot.raw_frame.shape}, "
                f"dtype: {input_snapshot.raw_frame.dtype}. "
                f"Has mask: {input_snapshot.raw_segmentation_mask is not None}. "
                f"Num bboxes: {len(input_snapshot.raw_bounding_boxes)}. "
                f"Original error: {pipeline_exception}"
            ) from pipeline_exception
        
        # Extract and normalize outputs
        augmented_frame = augmentation_result["image"]
        
        # Handle mask output - force to numpy array
        augmented_mask = None
        if "mask" in augmentation_result and augmentation_result["mask"] is not None:
            augmented_mask = _ensure_numpy_array(augmentation_result["mask"])
        
        # Handle bbox output (labels from separate field)
        # Clamp coordinates first to handle floating-point edge cases from rotation
        augmented_bboxes: tuple[BoundingBoxAnnotation, ...] = ()
        if "bboxes" in augmentation_result:
            output_frame_height, output_frame_width = augmented_frame.shape[:2]
            clamped_raw_bboxes = _clamp_bbox_coordinates(
                augmentation_result["bboxes"],
                output_frame_height,
                output_frame_width,
            )
            output_labels = augmentation_result.get("bbox_labels")
            augmented_bboxes = _convert_bboxes_from_albumentations_result(
                clamped_raw_bboxes,
                output_labels,
            )
        
        # Build new SnapshotState with augmented data
        # We use dataclasses.replace to create a new frozen instance
        return replace(
            input_snapshot,
            augmented_frame=augmented_frame,
            augmented_segmentation_mask=augmented_mask,
            augmented_bounding_boxes=augmented_bboxes,
            transform_name=self._get_transform_name(),
            transform_parameters=self._extract_transform_params(),
        )
    
    def _get_transform_name(self) -> str:
        """Extract a readable name for the transform."""
        pipeline_class_name = type(self._augmentation_pipeline).__name__
        
        # For Compose, try to list the child transforms
        if hasattr(self._augmentation_pipeline, "transforms"):
            child_names = [
                type(t).__name__ 
                for t in self._augmentation_pipeline.transforms[:3]  # Limit to first 3
            ]
            if len(self._augmentation_pipeline.transforms) > 3:
                child_names.append("...")
            return f"{pipeline_class_name}([{', '.join(child_names)}])"
        
        return pipeline_class_name
    
    def _extract_transform_params(self) -> dict[str, Any]:
        """Extract serializable parameters from the transform for logging."""
        params: dict[str, Any] = {}
        
        # Try to get probability if it exists
        if hasattr(self._augmentation_pipeline, "p"):
            params["probability"] = self._augmentation_pipeline.p
        
        # TODO: Extract more detailed params for common transforms
        
        return params
    
    def to_native_format(self, output_snapshot: SnapshotState) -> dict[str, Any]:
        """
        Convert SnapshotState back to Albumentations' native dict format.
        
        This enables PipelineHook to be a drop-in replacement — users get
        back the same {'image': ..., 'mask': ..., 'bboxes': ...} dict.
        """
        native_output: dict[str, Any] = {
            "image": output_snapshot.augmented_frame,
        }
        
        if output_snapshot.augmented_segmentation_mask is not None:
            native_output["mask"] = output_snapshot.augmented_segmentation_mask
        
        # Include bboxes if present (convert back to list of tuples)
        if output_snapshot.augmented_bounding_boxes:
            native_output["bboxes"] = [
                (box.x_min, box.y_min, box.x_max, box.y_max)
                for box in output_snapshot.augmented_bounding_boxes
            ]
            native_output["bbox_labels"] = [
                box.label for box in output_snapshot.augmented_bounding_boxes
            ]
        
        return native_output
    
    def __repr__(self) -> str:
        return f"<AlbumentationsAdapter (backend={self.target_library_name}) pipeline={self._get_transform_name()}>"
