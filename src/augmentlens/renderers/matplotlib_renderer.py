"""
Matplotlib-based renderer for side-by-side augmentation visualization.

Produces a 1x2 grid showing Original vs. Augmented images with
optional bounding box and mask overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from augmentlens.core.snapshot_state import SnapshotState
from augmentlens.renderers.base_renderer import BaseRenderer


# Standard diff colors - easily distinguishable
ORIGINAL_BBOX_COLOR = "limegreen"
AUGMENTED_BBOX_COLOR = "red"
MASK_COLORMAP = "viridis"
MASK_OVERLAY_ALPHA = 0.3


class MatplotlibRenderer(BaseRenderer):
    """
    Render before/after comparison using Matplotlib.
    
    Produces a 1x2 grid with:
    - Left: Original image with original bboxes (lime green)
    - Right: Augmented image with augmented bboxes (red)
    
    Optionally overlays segmentation masks with transparency.
    
    Example
    -------
    >>> from augmentlens.renderers import MatplotlibRenderer
    >>> renderer = MatplotlibRenderer()
    >>> fig = renderer.render(snapshot, show=True, save_path="debug.png")
    """
    
    def __init__(
        self,
        figsize: tuple[float, float] = (12, 6),
        dpi: int = 100,
    ) -> None:
        """
        Initialize the renderer with default figure settings.
        
        Parameters
        ----------
        figsize : tuple[float, float], default=(12, 6)
            Figure size in inches (width, height).
        dpi : int, default=100
            Dots per inch for saved figures.
        """
        self._default_figsize = figsize
        self._default_dpi = dpi
    
    def render(
        self,
        snapshot_to_render: SnapshotState,
        show: bool = True,
        save_path: str | Path | None = None,
        **render_options,
    ) -> Figure:
        """
        Render a side-by-side comparison of original vs. augmented.
        
        Parameters
        ----------
        snapshot_to_render : SnapshotState
            The captured snapshot to visualize.
        show : bool, default=True
            If True, display the figure with plt.show().
        save_path : str or Path, optional
            If provided, save the figure to this path.
        **render_options
            Additional options:
            - figsize: Override default figure size
            - dpi: Override default DPI
            - show_bboxes: bool, default=True
            - show_masks: bool, default=True
            - mask_alpha: float, default=0.3
        
        Returns
        -------
        Figure
            The matplotlib Figure object (for embedding in Jupyter/TensorBoard).
        """
        # Extract render options with defaults
        figsize = render_options.get("figsize", self._default_figsize)
        dpi = render_options.get("dpi", self._default_dpi)
        show_bboxes = render_options.get("show_bboxes", True)
        show_masks = render_options.get("show_masks", True)
        mask_alpha = render_options.get("mask_alpha", MASK_OVERLAY_ALPHA)
        
        # Normalize images for display using inherited helper
        original_display = self._normalize_for_display(snapshot_to_render.raw_frame)
        augmented_display = self._normalize_for_display(snapshot_to_render.augmented_frame)
        
        # Create the 1x2 figure
        fig, (ax_original, ax_augmented) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # === Left panel: Original ===
        ax_original.imshow(original_display)
        ax_original.set_title("Original", fontsize=12, fontweight="bold")
        ax_original.axis("off")
        
        # Overlay original mask if present
        if show_masks and snapshot_to_render.raw_segmentation_mask is not None:
            self._overlay_mask(
                ax_original,
                snapshot_to_render.raw_segmentation_mask,
                alpha=mask_alpha,
            )
        
        # Draw original bboxes
        if show_bboxes and snapshot_to_render.raw_bounding_boxes:
            for bbox in snapshot_to_render.raw_bounding_boxes:
                self._draw_bbox(ax_original, bbox, color=ORIGINAL_BBOX_COLOR)
        
        # === Right panel: Augmented ===
        ax_augmented.imshow(augmented_display)
        
        # Build title from transform metadata
        augmented_title = self._build_augmented_title(snapshot_to_render)
        ax_augmented.set_title(augmented_title, fontsize=12, fontweight="bold")
        ax_augmented.axis("off")
        
        # Overlay augmented mask if present
        if show_masks and snapshot_to_render.augmented_segmentation_mask is not None:
            self._overlay_mask(
                ax_augmented,
                snapshot_to_render.augmented_segmentation_mask,
                alpha=mask_alpha,
            )
        
        # Draw augmented bboxes
        if show_bboxes and snapshot_to_render.augmented_bounding_boxes:
            for bbox in snapshot_to_render.augmented_bounding_boxes:
                self._draw_bbox(ax_augmented, bbox, color=AUGMENTED_BBOX_COLOR)
        
        # UX polish: tight layout prevents title overlap
        plt.tight_layout()
        
        # Handle output
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def _build_augmented_title(self, snapshot: SnapshotState) -> str:
        """
        Build a descriptive title for the augmented panel.
        
        Shows the transform name if available, otherwise just 'Augmented'.
        """
        if snapshot.transform_name and snapshot.transform_name != "unknown":
            # Truncate long transform names
            transform_str = snapshot.transform_name
            if len(transform_str) > 40:
                transform_str = transform_str[:37] + "..."
            return f"Augmented: {transform_str}"
        return "Augmented"
    
    def _draw_bbox(
        self,
        ax: plt.Axes,
        bbox: Any,
        color: str,
        linewidth: float = 2.0,
    ) -> None:
        """
        Draw a single bounding box on the axes.
        
        Uses matplotlib.patches.Rectangle for clean rendering.
        """
        # Extract coordinates from BoundingBoxAnnotation
        x_min, y_min = bbox.x_min, bbox.y_min
        width = bbox.x_max - bbox.x_min
        height = bbox.y_max - bbox.y_min
        
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        
        # Add label if present
        if bbox.label:
            ax.text(
                x_min,
                y_min - 5,
                bbox.label,
                fontsize=9,
                color=color,
                fontweight="bold",
                verticalalignment="bottom",
            )
    
    def _overlay_mask(
        self,
        ax: plt.Axes,
        mask_data: np.ndarray,
        alpha: float = 0.3,
    ) -> None:
        """
        Overlay a segmentation mask with transparency.
        
        Non-zero regions of the mask are shown with the colormap,
        zero regions are fully transparent.
        """
        # Normalize mask for display
        mask_display = self._normalize_for_display(mask_data)
        
        # Squeeze to 2D if needed (grayscale mask)
        if mask_display.ndim == 3 and mask_display.shape[2] == 1:
            mask_display = mask_display[:, :, 0]
        elif mask_display.ndim == 3:
            # Multi-channel mask - convert to grayscale
            mask_display = mask_display.mean(axis=2).astype(np.uint8)
        
        # Create masked array where zeros are transparent
        masked = np.ma.masked_where(mask_display == 0, mask_display)
        
        ax.imshow(masked, cmap=MASK_COLORMAP, alpha=alpha)
    
    def __repr__(self) -> str:
        return f"<MatplotlibRenderer figsize={self._default_figsize} dpi={self._default_dpi}>"
