"""
AugmentLens: Debug and visualize Computer Vision augmentation pipelines.

This package provides tools to hook into Albumentations and Torchvision
pipelines, capturing before/after snapshots for visual debugging.

Example
-------
>>> import albumentations as A
>>> from augmentlens import PipelineHook, MatplotlibRenderer
>>>
>>> aug = PipelineHook(A.Compose([A.HorizontalFlip(p=0.5)]))
>>> result = aug(image=my_image, mask=my_mask)
>>>
>>> renderer = MatplotlibRenderer()
>>> renderer.render(aug.latest_capture, save_path="debug.png")
"""

__version__ = "0.1.0"

# Core classes - the main user-facing API
from augmentlens.core.pipeline_hook import PipelineHook, UnsupportedPipelineError
from augmentlens.core.snapshot_state import (
    BoundingBoxAnnotation,
    SnapshotState,
    SnapshotValidationError,
)

# Visualization
from augmentlens.renderers.matplotlib_renderer import MatplotlibRenderer
from augmentlens.renderers.base_renderer import BaseRenderer

# Explicit public API - prevents namespace pollution from 'import *'
__all__ = [
    # Version
    "__version__",
    # Core
    "PipelineHook",
    "SnapshotState",
    "BoundingBoxAnnotation",
    "UnsupportedPipelineError",
    "SnapshotValidationError",
    # Renderers
    "MatplotlibRenderer",
    "BaseRenderer",
]
