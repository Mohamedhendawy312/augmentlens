"""
Library-specific adapters for different augmentation frameworks.

Each adapter normalizes the library's API into a common interface
that the PipelineHook can consume without library-specific code.
"""

from augmentlens.core.adapters.base_adapter import (
    AdapterCompatibilityError,
    BaseAdapter,
)

# Albumentations adapter is conditionally available
try:
    from augmentlens.core.adapters.albumentations_adapter import AlbumentationsAdapter
    _HAS_ALBUMENTATIONS_ADAPTER = True
except ImportError:
    _HAS_ALBUMENTATIONS_ADAPTER = False

__all__ = [
    "AdapterCompatibilityError",
    "BaseAdapter",
]

if _HAS_ALBUMENTATIONS_ADAPTER:
    __all__.append("AlbumentationsAdapter")
