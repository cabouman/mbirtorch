"""The viewers: the slice viewer and the geometry viewer.

``slice_figure`` holds the slice viewer: ``VolumeStack``, the pure-numpy data
model, and ``SliceViewer`` with its entry point ``slice_viewer``.  ``geometry_scene``
holds the geometry viewer's model layer, ``GeometryScene``, and
``geometry_figure`` holds its matplotlib figure, ``GeometryFigure``, with its entry
point ``geometry_viewer``.  Each module imports numpy and the matplotlib base
package at most, and none of them imports torch, so the geometry scene can be
tested without a display and the whole subpackage loads without a GUI toolkit.

The package-level functions ``mbirtorch.slice_viewer`` and
``mbirtorch.geometry_viewer`` come from ``mbirtorch.view_utils``, which wraps
the slice viewer's entry point with the tensor and data-dict conversions that
the models' outputs need and re-exports the geometry viewer's as it is.
"""

from .slice_figure import SliceViewer, VolumeStack, slice_viewer
from .geometry_scene import GeometryScene
from .geometry_figure import GeometryFigure, geometry_viewer

__all__ = ['SliceViewer', 'VolumeStack', 'slice_viewer', 'GeometryScene',
           'GeometryFigure', 'geometry_viewer']
