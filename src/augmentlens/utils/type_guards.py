"""
Type guard utilities for pipeline detection.

These functions detect library types without importing them directly,
avoiding hard dependencies and import-time costs for users who don't
need specific libraries.
"""

from __future__ import annotations

from typing import Any


def is_albumentations_pipeline(candidate: Any) -> bool:
    """
    Check if the object is a valid Albumentations transform.
    
    Uses duck-typing via class hierarchy inspection to avoid importing
    Albumentations. Users who don't use Albumentations shouldn't pay
    the import cost.
    
    Parameters
    ----------
    candidate : Any
        Object to check.
    
    Returns
    -------
    bool
        True if the object inherits from Albumentations transform base classes.
    """
    class_hierarchy_names = [cls.__name__ for cls in type(candidate).__mro__]
    albumentations_base_classes = {"BasicTransform", "Compose", "BaseCompose"}
    return bool(albumentations_base_classes & set(class_hierarchy_names))


def is_torchvision_pipeline(candidate: Any) -> bool:
    """
    Check if the object is a Torchvision transform.
    
    Parameters
    ----------
    candidate : Any
        Object to check.
    
    Returns
    -------
    bool
        True if the object is a Torchvision transform.
    """
    class_hierarchy_names = [cls.__name__ for cls in type(candidate).__mro__]
    # Torchvision transforms have "Transform" or "Compose" in hierarchy
    # AND come from the torchvision module
    has_transform_base = bool({"Transform", "Compose"} & set(class_hierarchy_names))
    is_from_torchvision = "torchvision" in str(type(candidate).__module__)
    return has_transform_base and is_from_torchvision
