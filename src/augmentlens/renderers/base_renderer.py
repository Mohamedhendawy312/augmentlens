"""
Abstract base class for visualization renderers.

Renderers consume SnapshotState objects and produce visual output
(matplotlib figures, HTML, images, etc.). This base class provides
common image normalization utilities so subclasses never deal with
shape/dtype conversion directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from augmentlens.core.snapshot_state import SnapshotState


class BaseRenderer(ABC):
    """
    Abstract base class for visualization backends.
    
    Subclasses implement `render()` to produce specific output formats
    (matplotlib figures, HTML reports, etc.). The base class provides
    `_normalize_for_display()` to handle the chaos of CV image formats.
    
    Design Philosophy
    -----------------
    Renderers should receive clean, standardized data. All the messy
    format conversion (CHW→HWC, float→uint8, Tensor→ndarray) happens
    in the base class so subclasses can focus on visualization logic.
    """
    
    @abstractmethod
    def render(
        self,
        snapshot_to_render: SnapshotState,
        **render_options,
    ) -> Any:
        """
        Render visuals from a captured snapshot.
        
        Parameters
        ----------
        snapshot_to_render : SnapshotState
            The snapshot containing before/after data to visualize.
        **render_options
            Renderer-specific options (figure size, colormap, etc.)
        
        Returns
        -------
        Any
            Renderer-specific output (matplotlib Figure, path to file, etc.)
        """
        ...
    
    def _normalize_for_display(
        self,
        raw_image_data: Any,
    ) -> np.ndarray:
        """
        Convert any standard CV image format to display-ready (H, W, C) uint8.
        
        This method handles the 'shape hell' of computer vision:
        - PyTorch Tensors → numpy arrays
        - CHW layout → HWC layout
        - float [0, 1] → uint8 [0, 255]
        - Detaches gradients and moves to CPU if needed
        
        Parameters
        ----------
        raw_image_data : ndarray or Tensor
            Image in any common format:
            - numpy (H, W, C) or (C, H, W)
            - torch.Tensor (C, H, W) or (H, W, C)
            - dtype: uint8, float32, float64
            - range: [0, 255] or [0.0, 1.0]
        
        Returns
        -------
        np.ndarray
            Standardized (H, W, C) uint8 array ready for display.
        
        Raises
        ------
        ValueError
            If the input format cannot be interpreted.
        """
        image_array = self._to_numpy(raw_image_data)
        image_array = self._fix_channel_order(image_array)
        image_array = self._normalize_dtype(image_array)
        return image_array
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """
        Convert Tensor or array-like to numpy ndarray.
        
        Handles PyTorch tensors by detaching from graph and moving to CPU.
        """
        # Check for PyTorch tensor by class name to avoid hard dependency
        data_class_name = type(data).__name__
        if data_class_name == "Tensor":
            # It's a PyTorch tensor - detach, CPU, numpy
            # We access methods directly to avoid importing torch
            if hasattr(data, "detach"):
                data = data.detach()
            if hasattr(data, "cpu"):
                data = data.cpu()
            if hasattr(data, "numpy"):
                data = data.numpy()
            else:
                data = np.array(data)
        
        if not isinstance(data, np.ndarray):
            # Last resort: try to convert
            try:
                data = np.array(data)
            except Exception as conversion_error:
                raise ValueError(
                    f"Cannot convert {type(data).__name__} to numpy array. "
                    f"Supported types: numpy.ndarray, torch.Tensor. "
                    f"Error: {conversion_error}"
                ) from conversion_error
        
        return data
    
    def _fix_channel_order(self, image: np.ndarray) -> np.ndarray:
        """
        Transpose CHW to HWC if needed.
        
        Detection heuristic: if shape is (C, H, W) where C in {1, 3, 4}
        and H, W are larger, assume channels-first and transpose.
        """
        if image.ndim == 2:
            # Grayscale without channel dim - add it
            return image[:, :, np.newaxis]
        
        if image.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D image array, got shape {image.shape}. "
                "Cannot determine channel layout."
            )
        
        dim0, dim1, dim2 = image.shape
        
        # Heuristic: if first dim is small (1, 3, or 4) and others are larger,
        # it's probably CHW format
        likely_channels_first = (
            dim0 in (1, 3, 4)
            and dim1 > 4
            and dim2 > 4
        )
        
        # Additional check: if last dim is 1/3/4 and first is large, it's HWC
        likely_channels_last = (
            dim2 in (1, 3, 4)
            and dim0 > 4
            and dim1 > 4
        )
        
        if likely_channels_first and not likely_channels_last:
            # CHW → HWC
            return np.transpose(image, (1, 2, 0))
        
        # Already HWC or ambiguous - leave as is
        return image
    
    def _normalize_dtype(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to uint8 [0, 255] for display.
        
        If max value <= 1.0 and dtype is float, assume [0, 1] range and scale.
        Otherwise, clip to [0, 255] and cast.
        """
        if image.dtype == np.uint8:
            return image
        
        # Check if it looks like [0, 1] float data
        is_float = np.issubdtype(image.dtype, np.floating)
        max_value = image.max()
        
        if is_float and max_value <= 1.0:
            # Scale from [0, 1] to [0, 255]
            scaled = image * 255.0
        else:
            # Already in [0, 255] range (or close to it)
            scaled = image
        
        # Clip to valid range and cast
        clipped = np.clip(scaled, 0, 255)
        return clipped.astype(np.uint8)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
