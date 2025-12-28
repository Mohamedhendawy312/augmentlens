"""
Visualization backends for rendering debug reports.

Renderers are decoupled from core logic - the hook captures data,
renderers visualize it. This separation allows swapping Matplotlib
for HTML, Plotly, or other backends without touching capture logic.
"""

from augmentlens.renderers.base_renderer import BaseRenderer
from augmentlens.renderers.matplotlib_renderer import MatplotlibRenderer

__all__ = [
    "BaseRenderer",
    "MatplotlibRenderer",
]
