"""
Core pipeline interception logic.

This module contains the main hook mechanism and state capture classes.
"""

from augmentlens.core.snapshot_state import (
    BoundingBoxAnnotation,
    SnapshotState,
    SnapshotValidationError,
)
from augmentlens.core.pipeline_hook import (
    PipelineHook,
    UnsupportedPipelineError,
)

__all__ = [
    "BoundingBoxAnnotation",
    "PipelineHook",
    "SnapshotState",
    "SnapshotValidationError",
    "UnsupportedPipelineError",
]
