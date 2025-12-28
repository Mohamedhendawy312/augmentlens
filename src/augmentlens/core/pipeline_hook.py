"""
PipelineHook: The main user-facing class for augmentation debugging.

This class wraps any supported augmentation pipeline (Albumentations, Torchvision)
and provides a drop-in replacement that captures before/after snapshots while
maintaining full API compatibility with the original pipeline.
"""

from __future__ import annotations

from typing import Any

from augmentlens.core.adapters.base_adapter import BaseAdapter
from augmentlens.core.snapshot_state import SnapshotState
from augmentlens.utils.type_guards import (
    is_albumentations_pipeline,
    is_torchvision_pipeline,
)


class UnsupportedPipelineError(TypeError):
    """
    Raised when PipelineHook receives a pipeline it cannot handle.
    
    This typically means the pipeline is from an unsupported library or
    is not a valid augmentation transform object.
    """
    pass


class PipelineHook:
    """
    Drop-in replacement wrapper for augmentation pipelines with debug capture.
    
    Wraps Albumentations or Torchvision pipelines, mirroring their callable
    interface exactly while capturing before/after snapshots for visualization.
    
    Example
    -------
    >>> import albumentations as A
    >>> from augmentlens import PipelineHook
    >>> 
    >>> # Original code:
    >>> # aug = A.Compose([A.HorizontalFlip(p=0.5)])
    >>> # result = aug(image=img, mask=mask)
    >>> 
    >>> # With AugmentLens:
    >>> aug = PipelineHook(A.Compose([A.HorizontalFlip(p=0.5)]))
    >>> result = aug(image=img, mask=mask)  # Same API!
    >>> aug.latest_capture  # Access the captured SnapshotState
    
    Parameters
    ----------
    pipeline : Any
        The augmentation pipeline to wrap (Albumentations Compose, etc.)
    pipeline_type : str, optional
        Force adapter selection: 'albumentations' or 'torchvision'.
        If None, auto-detects from pipeline class hierarchy.
    passthrough : bool, default=False
        If True, skip all capture logic and delegate directly to the
        original pipeline. Use this in production training for zero overhead.
    """
    
    def __init__(
        self,
        pipeline: Any,
        pipeline_type: str | None = None,
        passthrough: bool = False,
    ) -> None:
        if pipeline is None:
            raise UnsupportedPipelineError(
                "Pipeline cannot be None. Pass an Albumentations or Torchvision "
                "transform object."
            )
        
        self._original_pipeline = pipeline
        self._passthrough_enabled = passthrough
        self._latest_capture: SnapshotState | None = None
        
        # Skip adapter resolution if passthrough mode - we won't use it
        if passthrough:
            self._adapter: BaseAdapter | None = None
        else:
            self._adapter = self._resolve_adapter(pipeline, pipeline_type)
    
    def _resolve_adapter(
        self,
        pipeline: Any,
        forced_type: str | None,
    ) -> BaseAdapter:
        """
        Detect pipeline type and return appropriate adapter instance.
        
        Uses duck-typing via class hierarchy inspection to avoid hard
        dependencies on augmentation libraries.
        """
        # If user explicitly specified type, trust them
        if forced_type is not None:
            return self._create_adapter_for_type(forced_type, pipeline)
        
        # Auto-detect based on class hierarchy
        if is_albumentations_pipeline(pipeline):
            return self._create_adapter_for_type("albumentations", pipeline)
        
        if is_torchvision_pipeline(pipeline):
            return self._create_adapter_for_type("torchvision", pipeline)
        
        # Unknown pipeline type
        pipeline_class = type(pipeline).__name__
        pipeline_module = type(pipeline).__module__
        raise UnsupportedPipelineError(
            f"Cannot auto-detect pipeline type for {pipeline_class} from {pipeline_module}. "
            "Supported libraries: Albumentations, Torchvision. "
            "If this is a supported library, try passing pipeline_type='albumentations' explicitly."
        )
    
    def _create_adapter_for_type(self, adapter_type: str, pipeline: Any) -> BaseAdapter:
        """Instantiate the correct adapter class for given type."""
        adapter_type_lower = adapter_type.lower()
        
        if adapter_type_lower == "albumentations":
            # Import here to avoid hard dependency
            from augmentlens.core.adapters.albumentations_adapter import AlbumentationsAdapter
            return AlbumentationsAdapter(pipeline)
        
        if adapter_type_lower == "torchvision":
            # TODO: Implement TorchvisionAdapter
            raise NotImplementedError(
                "Torchvision adapter is not yet implemented. Coming soon!"
            )
        
        raise UnsupportedPipelineError(
            f"Unknown pipeline type '{adapter_type}'. "
            "Supported types: 'albumentations', 'torchvision'."
        )
    
    @property
    def latest_capture(self) -> SnapshotState | None:
        """
        The most recent SnapshotState captured during __call__.
        
        Returns None if no augmentation has been run yet or if
        passthrough mode is enabled.
        """
        return self._latest_capture
    
    @property
    def capture_history(self) -> list[SnapshotState]:
        """
        All captured snapshots from this session.
        
        TODO: Implement history buffer with configurable max size.
        For now, only latest_capture is available.
        """
        if self._latest_capture is not None:
            return [self._latest_capture]
        return []
    
    @property
    def passthrough_enabled(self) -> bool:
        """True if capture logic is bypassed for production performance."""
        return self._passthrough_enabled
    
    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the augmentation pipeline, capturing before/after state.
        
        This method mimics the original pipeline's __call__ signature exactly,
        so PipelineHook can be a drop-in replacement.
        
        Parameters
        ----------
        **kwargs
            Arguments to pass to the underlying pipeline. For Albumentations:
            image=, mask=, bboxes=, etc.
        
        Returns
        -------
        dict
            Result in the same format as the original pipeline would return.
        """
        # Production mode: zero overhead passthrough
        if self._passthrough_enabled:
            return self._original_pipeline(**kwargs)
        
        assert self._adapter is not None, "Adapter should be set when not in passthrough mode"
        
        input_snapshot = self._build_snapshot_from_kwargs(kwargs)
        
        output_snapshot = self._adapter.apply(input_snapshot)
        
        self._latest_capture = output_snapshot
        
        return self._adapter.to_native_format(output_snapshot)
    
    def _build_snapshot_from_kwargs(self, input_kwargs: dict[str, Any]) -> SnapshotState:
        """
        Convert pipeline input kwargs to a SnapshotState.
        
        Maps common kwargs from different libraries to our unified format.
        """
        from augmentlens.core.snapshot_state import BoundingBoxAnnotation
        
        raw_image = input_kwargs.get("image")
        if raw_image is None:
            raise ValueError(
                "Pipeline input must include 'image' key. "
                "Got keys: " + ", ".join(input_kwargs.keys())
            )
        
        raw_mask = input_kwargs.get("mask")
        
        # Extract bboxes (optional) - convert from Albumentations format
        raw_bboxes: tuple[BoundingBoxAnnotation, ...] = ()
        if "bboxes" in input_kwargs and input_kwargs["bboxes"]:
            bbox_labels = input_kwargs.get("bbox_labels", [])
            if not bbox_labels:
                bbox_labels = [""] * len(input_kwargs["bboxes"])
            
            converted_boxes = []
            for bbox_data, label in zip(input_kwargs["bboxes"], bbox_labels):
                x_min, y_min, x_max, y_max = bbox_data[:4]
                converted_boxes.append(
                    BoundingBoxAnnotation(
                        x_min=int(x_min),
                        y_min=int(y_min),
                        x_max=int(x_max),
                        y_max=int(y_max),
                        label=str(label),
                    )
                )
            raw_bboxes = tuple(converted_boxes)
        
        return SnapshotState(
            raw_frame=raw_image,
            augmented_frame=raw_image,  # Placeholder until transform runs
            raw_segmentation_mask=raw_mask,
            raw_bounding_boxes=raw_bboxes,
        )
    
    def __repr__(self) -> str:
        mode_str = "passthrough" if self._passthrough_enabled else "capture"
        adapter_info = repr(self._adapter) if self._adapter else "None"
        return f"<PipelineHook mode={mode_str} adapter={adapter_info}>"
