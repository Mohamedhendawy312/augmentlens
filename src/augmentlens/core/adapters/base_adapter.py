"""
Abstract base class for library-specific augmentation adapters.

Each adapter normalizes a library's transform API into a common interface.
This allows the PipelineHook to work with Albumentations, Torchvision,
or any future library without knowing the details of each one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import - SnapshotState is only needed for type hints
    from augmentlens.core.snapshot_state import SnapshotState


class AdapterCompatibilityError(ValueError):
    """
    Raised when an adapter detects incompatible input data.
    
    Examples: bounding box format mismatch, unsupported image dtype,
    or missing required fields for a specific library.
    """
    pass


class BaseAdapter(ABC):
    """
    Abstract base class for library-specific augmentation adapters.
    
    Adapters serve two purposes:
    1. Normalize different library APIs into a common `apply()` interface
    2. Handle library-specific quirks (bbox formats, mask channels, etc.)
    
    Subclasses must implement:
    - `target_library_name`: Identifies which library this adapter wraps
    - `apply()`: Executes the transform and returns updated SnapshotState
    
    The optional `validate_compatibility()` method lets adapters check
    if input data is in the correct format before attempting transforms.
    """
    
    @property
    @abstractmethod
    def target_library_name(self) -> str:
        """
        Return the name of the library this adapter targets.
        
        Used for logging, error messages, and adapter selection logic.
        Examples: 'albumentations', 'torchvision', 'kornia'
        """
        ...
    
    @abstractmethod
    def apply(
        self,
        input_snapshot: SnapshotState,
        **runtime_context,
    ) -> SnapshotState:
        """
        Apply the wrapped transform to the input snapshot.
        
        This method receives a SnapshotState with raw_frame populated
        and must return a new SnapshotState with augmented_frame set.
        The implementation should also transform masks and bboxes if present.
        
        Parameters
        ----------
        input_snapshot : SnapshotState
            Snapshot with raw_frame (and optionally masks/bboxes) populated.
            The augmented_* fields may or may not be populated depending
            on whether this is the first transform in a pipeline.
        **runtime_context
            Future-proofing for flags like `force_deterministic=True` or
            `render_speed='fast'`. Subclasses can consume what they need
            and ignore the rest.
        
        Returns
        -------
        SnapshotState
            New snapshot with augmented_frame (and augmented masks/bboxes)
            populated based on the transform result.
        
        Raises
        ------
        AdapterCompatibilityError
            If the input data is incompatible with the wrapped library.
        """
        ...
    
    @abstractmethod
    def to_native_format(
        self,
        output_snapshot: SnapshotState,
    ) -> dict:
        """
        Convert SnapshotState back to the library's native output format.
        
        This enables PipelineHook to be a drop-in replacement for the original
        pipeline — users get back the same dictionary/tuple format they expect.
        
        Parameters
        ----------
        output_snapshot : SnapshotState
            The snapshot containing augmented data to repackage.
        
        Returns
        -------
        dict
            Data in the library's native output format (e.g., Albumentations
            returns {'image': ..., 'mask': ..., 'bboxes': ...}).
        """
        ...
    
    def validate_compatibility(
        self,
        snapshot_to_validate: SnapshotState,
    ) -> None:
        """
        Check if the snapshot's data is compatible with this adapter's library.
        
        Override this in subclasses to add library-specific validation.
        For example, Albumentations expects bboxes in a specific format,
        and Torchvision may require tensors instead of numpy arrays.
        
        Parameters
        ----------
        snapshot_to_validate : SnapshotState
            The snapshot whose data should be validated.
        
        Raises
        ------
        NotImplementedError
            Default behavior - subclasses should override with real checks.
        AdapterCompatibilityError
            When subclass detects incompatible data format.
        """
        # We raise NotImplementedError rather than pass silently because
        # forgetting to implement this in a subclass should be obvious,
        # not a silent source of runtime bugs.
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented validate_compatibility(). "
            "Either implement it or explicitly override with 'pass' if no validation is needed."
        )
    
    def __repr__(self) -> str:
        """Readable representation for debugging and log inspection."""
        return f"<{self.__class__.__name__} (backend={self.target_library_name})>"
