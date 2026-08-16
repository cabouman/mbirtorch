"""Interactive multi-volume slice viewer.

This module is package-independent by design: it imports numpy, matplotlib,
and (lazily, only for HDF5 file access) h5py, and nothing else.  A host
package wraps the public entry point to supply its own conversions -- a
tensor-to-numpy shim, serialization of rich data dicts to plain-string dicts,
and an optional HDF5 save function -- so the identical file can serve any
host package.

The module has two layers.  ``VolumeStack`` is the pure-numpy data model:
volumes, axis permutations, slice positions, intensity range, difference
images, ROI statistics, and file loading.  The matplotlib view/controller
(``SliceViewer`` and the ``slice_viewer`` entry point) sits on top of it and
holds no data logic of its own.
"""

import enum
import os
import sys
import time
import warnings

import numpy as np
import matplotlib  # base package only; no GUI toolkit is touched at import

__all__ = ['SliceViewer', 'VolumeStack', 'slice_viewer']

# Shape of the placeholder volume shown for a dataset passed as None.
PLACEHOLDER_SHAPE = (20, 20, 20)

# --- appearance constants ---
TOOLTIP_FONT_SIZE = 9
TOOLTIP_BOX_ALPHA = 0.9
TOOLTIP_OFFSET = (10, 10)
TOOLTIP_TEXT = "Click and drag to move\nClick edge to resize\nPress Esc to remove"

CIRCLE_COLOR = 'red'
CIRCLE_LINEWIDTH = 2
CIRCLE_ALPHA = 1.0
CIRCLE_FILL = False

SLICE_AXIS_FONT_SIZE = 9
SLICE_AXIS_LABEL_FONT_SIZE = 8
SLICE_AXIS_RADIO_SIZE = 30

STRIP_FONT_SIZE = 8
STATS_FONT_SIZE = 12
DIALOG_FONT_SIZE = 9
DIALOG_BODY_FONT_SIZE = 8
DIALOG_ZORDER = 50
DIALOG_PAGE_LINES = 30

ROI_STATS_THROTTLE_S = 0.3

# Backends with no interactive window; show() warns and returns under these.
NONINTERACTIVE_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template', 'cairo'}

# Backends where the partial-redraw (blit) fast path is verified: Agg for
# headless tests, TkAgg for the remote-X11 sessions the fast path exists for.
# Everywhere else the viewer uses plain full redraws.  macosx in particular
# reports blit support but repaints the whole window per blit, and its
# Retina buffers are larger than the logical canvas size.
BLIT_BACKENDS = {'agg', 'tkagg'}

# Extensions offered by the file-load browser (save always writes .h5).
FILE_EXTENSIONS = ('.npy', '.npz', '.h5', '.hdf5')
FILE_DIALOG_ROWS = 11

# Returned by the native-dialog chain when no native file dialog can run;
# the caller then falls back to the in-figure browser.
_NATIVE_UNAVAILABLE = object()


def multiline(*lines):
    return '\n'.join(lines)


class Mode(enum.Enum):
    """Interaction mode for the controller; exactly one is active at a time."""
    IDLE = enum.auto()
    DRAW_ROI = enum.auto()
    MOVE_ROI = enum.auto()
    RESIZE_ROI = enum.auto()
    SELECT_COMPARISON = enum.auto()


# Populated by _load_pyplot on first viewer construction, so that importing
# this module never resolves a matplotlib backend or touches a GUI toolkit.
plt = None
gridspec = None
make_axes_locatable = None
Slider = RangeSlider = RadioButtons = Button = TextBox = None
Bbox = None
Rectangle = None
IdentityTransform = None


def _load_pyplot():
    """Import pyplot and the widget classes on first use."""
    global plt, gridspec, make_axes_locatable, Slider, RangeSlider, \
        RadioButtons, Button, TextBox, Bbox, Rectangle, IdentityTransform
    if plt is not None:
        return
    import matplotlib.pyplot as _plt
    from matplotlib import gridspec as _gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable as _mal
    from matplotlib.widgets import (Slider as _Slider, RangeSlider as _RangeSlider,
                                    RadioButtons as _RadioButtons,
                                    Button as _Button, TextBox as _TextBox)
    from matplotlib.transforms import (Bbox as _Bbox,
                                       IdentityTransform as _IdentityTransform)
    from matplotlib.patches import Rectangle as _Rectangle
    plt = _plt
    gridspec = _gridspec
    make_axes_locatable = _mal
    Slider, RangeSlider = _Slider, _RangeSlider
    RadioButtons = _RadioButtons
    Button, TextBox = _Button, _TextBox
    Bbox, Rectangle, IdentityTransform = _Bbox, _Rectangle, _IdentityTransform


class VolumeStack:
    """Pure-numpy data model for the slice viewer.

    Holds one or more 3D volumes together with the state the viewer displays:
    per-volume axis permutations, a shared master slice position with
    proportional per-volume mapping, a shared display intensity range,
    difference-image state, and per-volume labels and data dicts.  The class
    imports no GUI code, so every behavior is unit-testable headlessly.

    Contract notes:
        - ``datasets`` entries are numpy-convertible 2D or 3D arrays, or None
          (None becomes a placeholder zero volume).  Conversion from framework
          tensors happens in the host package's wrapper, not here.
        - ``data_dicts`` entries are dicts whose values are strings.  Any
          serialization from richer structures happens in the wrapper before
          construction.
        - Default ``vmin``/``vmax`` scan every voxel of every volume once at
          construction; pass explicit values to skip the scan for very large
          volumes.

    Args:
        datasets (sequence of ndarray or None): One or more 2D or 3D arrays.
            2D arrays are promoted to 3D via a trailing singleton axis.
        data_dicts (None, dict, or list of dict/None, optional): String-valued
            dict(s) associated with the volumes.  A bare dict is accepted only
            for a single volume; otherwise the list length must match.
        vmin (float, optional): Display minimum.  Defaults to the global
            minimum across all volumes.
        vmax (float, optional): Display maximum.  Defaults to the global
            maximum across all volumes.
        slice_label (str or list of str, optional): Label(s) for the slice
            title.  Defaults to "Slice".
        slice_axis (int or list of int, optional): Axis to slice along (0, 1,
            or 2) for all volumes or per volume.  Defaults to 2.
    """

    def __init__(self, datasets, data_dicts=None, vmin=None, vmax=None,
                 slice_label=None, slice_axis=None):
        datasets = list(datasets)
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")
        self.n_volumes = len(datasets)

        self.axes_perms = self._normalize_slice_axes(slice_axis)
        self.labels = self._normalize_labels(slice_label)
        self.data_dicts = self._normalize_data_dicts(data_dicts)

        # original_data holds each volume in its as-passed axis order; data
        # holds the permuted view actually displayed (slice axis last).
        self.original_data = []
        self.data = []
        for i, dataset in enumerate(datasets):
            if dataset is None:
                dataset = np.zeros(PLACEHOLDER_SHAPE)
            dataset = np.asarray(dataset)
            if dataset.ndim == 2:
                dataset = dataset[..., np.newaxis]
            elif dataset.ndim != 3:
                raise ValueError("Each input data must be a 2D or 3D array")
            self.original_data.append(dataset)
            self.data.append(np.transpose(dataset, self.axes_perms[i]))

        # Difference-image state: None, or a dict with keys
        # 'comparison_index', 'use_abs', 'prev_label'.
        self._difference_info = [None] * self.n_volumes

        # Each volume opens at its own midpoint; the master index (the shared
        # slider position) starts at volume 0's midpoint.  The first
        # set_master_index call snaps all volumes onto the proportional map.
        self.cur_slices = [d.shape[2] // 2 for d in self.data]
        self.master_index = self.cur_slices[0]

        self.vmin, self.vmax = self.resolve_range(vmin, vmax)

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    @staticmethod
    def perm_from_slice_axis(slice_axis):
        """Return the display permutation for a slice axis.

        The permutation lists the two in-plane axes in ascending order
        followed by the slice axis, e.g. axis 0 -> [1, 2, 0].
        """
        slice_axis = int(slice_axis)
        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")
        return sorted({0, 1, 2} - {slice_axis}) + [slice_axis]

    def _normalize_slice_axes(self, slice_axis):
        if slice_axis is None or isinstance(slice_axis, (int, np.integer)):
            axis = 2 if slice_axis is None else int(slice_axis)
            slice_axes = [axis] * self.n_volumes
        else:
            slice_axes = [int(s) for s in slice_axis]
            if len(slice_axes) != self.n_volumes:
                raise ValueError(
                    "slice_axis must be a single int or a list of ints of the "
                    "same length as the number of datasets")
        return [self.perm_from_slice_axis(s) for s in slice_axes]

    def _normalize_labels(self, slice_label):
        if slice_label is None or isinstance(slice_label, str):
            return ["Slice" if slice_label is None else slice_label] * self.n_volumes
        labels = [str(label) for label in slice_label]
        if len(labels) != self.n_volumes:
            raise ValueError(
                "slice_label must be a single string or a list of strings of "
                "the same length as the number of datasets")
        return labels

    def _normalize_data_dicts(self, data_dicts):
        if data_dicts is None:
            return [None] * self.n_volumes
        if isinstance(data_dicts, dict):
            data_dicts = [data_dicts]
        data_dicts = list(data_dicts)
        if len(data_dicts) != self.n_volumes or \
                not all(isinstance(d, dict) or d is None for d in data_dicts):
            raise ValueError(
                "data_dicts must be single dict or a list of dicts of the "
                "same length as the number of datasets")
        return data_dicts

    # ------------------------------------------------------------------
    # Slice position: master index with proportional per-volume mapping
    # ------------------------------------------------------------------

    @property
    def max_slices(self):
        """Depth of the deepest volume in display orientation."""
        return max(d.shape[2] for d in self.data)

    @property
    def slice_counts(self):
        """Per-volume displayed depth."""
        return [d.shape[2] for d in self.data]

    @property
    def slice_axes(self):
        """Per-volume slice axis (the last entry of each permutation)."""
        return [perm[-1] for perm in self.axes_perms]

    @property
    def master_fraction(self):
        """Master position as a fraction of the deepest volume, in [0, 1]."""
        if self.max_slices > 1:
            return self.master_index / (self.max_slices - 1)
        return 0.0

    def set_master_index(self, index):
        """Set the shared slice position; return the volumes whose slice changed.

        The master index runs over the deepest volume, 0 to max_slices - 1
        (values outside that range are clipped).  Each volume's slice is the
        proportional position ``round(fraction * (depth - 1))``, so the
        deepest volume follows the master exactly and shallower volumes track
        proportionally rather than clipping at their last slice.
        """
        index = int(np.clip(int(np.round(index)), 0, self.max_slices - 1))
        self.master_index = index
        return self._recompute_slices()

    def _set_master_fraction(self, fraction):
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if self.max_slices > 1:
            self.master_index = int(np.round(fraction * (self.max_slices - 1)))
        else:
            self.master_index = 0
        return self._recompute_slices()

    def _recompute_slices(self):
        fraction = self.master_fraction
        changed = []
        for i, d in enumerate(self.data):
            new_slice = int(np.round(fraction * (d.shape[2] - 1)))
            if new_slice != self.cur_slices[i]:
                self.cur_slices[i] = new_slice
                changed.append(i)
        return changed

    def slice_image(self, i):
        """Return the 2D array currently displayed for volume ``i``."""
        return self.data[i][:, :, self.cur_slices[i]]

    # ------------------------------------------------------------------
    # Axis permutations
    # ------------------------------------------------------------------

    def set_perm(self, i, new_perm):
        """Re-orient volume ``i``; return True if the permutation changed.

        ``new_perm`` is either a slice axis (int) or a full permutation of
        (0, 1, 2).  The master fraction is preserved, so every volume keeps
        its proportional position under the new depths.
        """
        if isinstance(new_perm, (int, np.integer)):
            new_perm = self.perm_from_slice_axis(new_perm)
        new_perm = [int(p) for p in new_perm]
        if sorted(new_perm) != [0, 1, 2]:
            raise ValueError("Permutation must be a permutation of (0, 1, 2)")
        if new_perm == self.axes_perms[i]:
            return False

        fraction = self.master_fraction
        inverse_perm = np.argsort(self.axes_perms[i])
        unpermuted = np.transpose(self.data[i], inverse_perm)
        self.axes_perms[i] = new_perm
        self.data[i] = np.transpose(unpermuted, new_perm)
        self._set_master_fraction(fraction)
        return True

    def transpose(self, i):
        """Swap the two in-plane axes of volume ``i``."""
        perm = list(self.axes_perms[i])
        perm[0], perm[1] = perm[1], perm[0]
        self.set_perm(i, perm)

    # ------------------------------------------------------------------
    # Intensity range
    # ------------------------------------------------------------------

    def data_range(self):
        """Return (min, max) over all volumes' current data."""
        lo = min(float(np.min(d)) for d in self.data)
        hi = max(float(np.max(d)) for d in self.data)
        return lo, hi

    def resolve_range(self, vmin=None, vmax=None):
        """Fill missing bounds from the data and validate.

        None bounds are replaced by the data minimum/maximum.  Equal bounds
        are split by a small scale-aware epsilon so downstream widgets always
        see a nonempty range.  vmin > vmax raises ValueError.
        """
        if vmin is None or vmax is None:
            data_lo, data_hi = self.data_range()
            vmin = data_lo if vmin is None else vmin
            vmax = data_hi if vmax is None else vmax
        vmin, vmax = float(vmin), float(vmax)
        if vmin > vmax:
            raise ValueError("Minimum must be less than maximum")
        if vmin == vmax:
            eps = 1e-6
            scale = float(np.clip(eps * np.abs(vmax), a_min=eps, a_max=None))
            vmin, vmax = vmin - scale, vmax + scale
        return vmin, vmax

    def set_range(self, vmin=None, vmax=None):
        """Set the display range; None bounds are filled from the data."""
        self.vmin, self.vmax = self.resolve_range(vmin, vmax)
        return self.vmin, self.vmax

    # ------------------------------------------------------------------
    # Difference images
    # ------------------------------------------------------------------

    def is_difference(self, i):
        """Return True if volume ``i`` currently shows a difference image."""
        return self._difference_info[i] is not None

    def can_difference(self, baseline_index, comparison_index):
        """Return True if the pair is a valid difference: distinct indices
        with equal original shapes.  Orientations may differ; the comparison
        is re-oriented into the baseline's frame."""
        return (baseline_index != comparison_index
                and self.original_data[baseline_index].shape
                == self.original_data[comparison_index].shape)

    def apply_difference(self, baseline_index, comparison_index, use_abs=False):
        """Replace volume ``baseline_index`` with (comparison - baseline).

        The comparison uses the volumes' current data, so a difference against
        an already-differenced volume compares against what is displayed.  If
        the two volumes are viewed in different orientations (transposed, or
        sliced along different axes), the comparison is re-oriented into the
        baseline's frame first.  With ``use_abs`` the absolute difference is
        shown.  The previous label is saved and restored by :meth:`restore`.
        """
        if not self.can_difference(baseline_index, comparison_index):
            raise ValueError(
                "Difference requires two distinct volumes with the same shape")
        comparison = self.data[comparison_index]
        if self.axes_perms[comparison_index] != self.axes_perms[baseline_index]:
            inverse_perm = np.argsort(self.axes_perms[comparison_index])
            comparison = np.transpose(np.transpose(comparison, inverse_perm),
                                      self.axes_perms[baseline_index])
        difference = comparison - self.data[baseline_index]
        if use_abs:
            difference = np.abs(difference)
        self._difference_info[baseline_index] = {
            'comparison_index': comparison_index,
            'use_abs': use_abs,
            'prev_label': self.labels[baseline_index],
        }
        self.data[baseline_index] = difference
        if use_abs:
            label_prepend = 'abs(Image {} minus current): '.format(comparison_index)
        else:
            label_prepend = 'Image {} minus current: '.format(comparison_index)
        self.labels[baseline_index] = label_prepend + self.labels[baseline_index]

    def restore(self, i):
        """Restore volume ``i`` from its original data, ending any difference."""
        self.data[i] = np.transpose(self.original_data[i], self.axes_perms[i])
        info = self._difference_info[i]
        if info is not None:
            self.labels[i] = info['prev_label']
        self._difference_info[i] = None

    # ------------------------------------------------------------------
    # ROI statistics
    # ------------------------------------------------------------------

    def roi_stats(self, i, x, y, radius):
        """Statistics of the current slice of volume ``i`` inside a circle.

        ``x`` and ``y`` are data coordinates (column, row) of the circle
        center.  Returns a dict with keys 'mean', 'std', 'min', 'max', or
        None when no pixel center falls inside the circle.
        """
        slice_2d = self.slice_image(i)
        ny, nx = slice_2d.shape[:2]
        yv, xv = np.ogrid[:ny, :nx]
        mask = (xv - x) ** 2 + (yv - y) ** 2 <= radius ** 2
        values = slice_2d[mask]
        if values.size == 0:
            return None
        return {'mean': float(np.mean(values)), 'std': float(np.std(values)),
                'min': float(np.min(values)), 'max': float(np.max(values))}

    # ------------------------------------------------------------------
    # File load
    # ------------------------------------------------------------------

    @staticmethod
    def list_file_arrays(file_path):
        """List the arrays available in a file, for a chooser dialog.

        Returns (names, shapes) for .npz and .h5/.hdf5 files, or None for
        .npy files (a single unnamed array).  Unsupported extensions raise
        ValueError.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.npy':
            return None
        if ext == '.npz':
            with np.load(file_path) as array_dict:
                names = list(array_dict.files)
                shapes = [array_dict[name].shape for name in names]
            return names, shapes
        if ext in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                names = list(f.keys())
                shapes = [f[name].shape for name in names]
            return names, shapes
        raise ValueError("Unsupported file type: {}".format(ext))

    @staticmethod
    def read_file_array(file_path, name=None):
        """Read one array from a file; return (array, data_dict).

        ``name`` selects the array for .npz and .h5/.hdf5 files (default:
        the first).  The data dict comes from HDF5 dataset attributes,
        coerced to strings; it is None for the other formats.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.npy':
            return np.load(file_path), None
        if ext == '.npz':
            with np.load(file_path) as array_dict:
                key = array_dict.files[0] if name is None else name
                return array_dict[key], None
        if ext in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                key = list(f.keys())[0] if name is None else name
                dataset = f[key]
                data_dict = {
                    str(k): v.decode() if isinstance(v, bytes) else str(v)
                    for k, v in dataset.attrs.items()
                } or None
                return dataset[()], data_dict
        raise ValueError("Unsupported file type: {}".format(ext))

    def load_array(self, image_index, new_array, data_dict=None):
        """Replace volume(s) with a loaded array; return the affected indices.

        2D arrays are promoted to 3D.  A 3D array replaces volume
        ``image_index``.  A 4D array replaces volumes 0..k-1 with its slabs
        along the last axis, where k = min(last dim, n_volumes); the primary
        index (which receives ``data_dict`` and a permutation reset) becomes
        min(image_index, k - 1).  Anything else raises ValueError.

        Replaced volumes leave any difference state, since the data they were
        differenced against is gone.  The master position moves to the primary
        volume's middle slice.
        """
        new_array = np.asarray(new_array)
        if new_array.ndim == 2:
            new_array = new_array[..., np.newaxis]
        if new_array.ndim == 3:
            replaced = [image_index]
            self.original_data[image_index] = new_array
        elif new_array.ndim == 4:
            if new_array.shape[-1] == 0:
                raise ValueError("Loaded 4D array must contain at least one volume")
            num_volumes_to_load = min(new_array.shape[-1], self.n_volumes)
            replaced = list(range(num_volumes_to_load))
            for j in replaced:
                self.original_data[j] = new_array[..., j]
            image_index = min(image_index, num_volumes_to_load - 1)
        else:
            raise ValueError("Loaded array must be 2D, 3D, or 4D")

        self.data_dicts[image_index] = data_dict
        # The primary volume returns to the canonical display order for its
        # slice axis; other replaced volumes keep their permutations.
        self.axes_perms[image_index] = self.perm_from_slice_axis(
            self.axes_perms[image_index][-1])
        for j in replaced:
            if self._difference_info[j] is not None:
                self.labels[j] = self._difference_info[j]['prev_label']
                self._difference_info[j] = None
            self.data[j] = np.transpose(self.original_data[j], self.axes_perms[j])

        depth = self.data[image_index].shape[2]
        middle_fraction = (depth // 2) / (depth - 1) if depth > 1 else 0.0
        self._set_master_fraction(middle_fraction)
        return replaced


def _save_data_hdf5(file_path, array, array_name='volume', attributes_dict=None):
    """Default save function: one HDF5 dataset with string attributes.

    The layout is a single named dataset whose attributes hold the data dict,
    so files round-trip through :meth:`VolumeStack.read_file_array`.
    Host packages can inject a richer
    writer through ``slice_viewer(..., save_fn=...)``.
    """
    import h5py
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        dataset = f.create_dataset(array_name, data=np.asarray(array))
        for key, value in (attributes_dict or {}).items():
            dataset.attrs[str(key)] = str(value)


# ---------------------------------------------------------------------------
# Native (Tk) dialogs
# ---------------------------------------------------------------------------
# Module-level so they can be exercised standalone.  Each creates a hidden Tk
# root at call time, runs a short modal loop, and destroys the root before
# returning -- the same lazy pattern as the native file dialogs, so importing
# this module never touches a GUI toolkit.  Callers catch exceptions and fall
# back to the in-figure dialogs.

def _tk_menu(labels):
    """Show a menu-like popup at the mouse pointer; return the chosen label.

    Returns None when dismissed (Escape, click-away focus loss, or Cancel).
    """
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    result = {'label': None}
    try:
        top = tkinter.Toplevel(root)
        top.overrideredirect(True)
        listbox = tkinter.Listbox(
            top, height=len(labels), activestyle='none', relief='solid',
            borderwidth=1, highlightthickness=0, exportselection=False)
        for label in labels:
            listbox.insert('end', label)
        listbox.configure(width=max(len(label) for label in labels) + 2)
        listbox.pack()

        def choose(_event=None):
            selection = listbox.curselection()
            if selection:
                result['label'] = labels[selection[0]]
            top.destroy()

        def highlight(event):
            listbox.selection_clear(0, 'end')
            listbox.selection_set(listbox.nearest(event.y))

        listbox.bind('<ButtonRelease-1>', choose)
        listbox.bind('<Return>', choose)
        listbox.bind('<Motion>', highlight)
        top.bind('<Escape>', lambda _event: top.destroy())
        top.geometry(f'+{root.winfo_pointerx()}+{root.winfo_pointery()}')
        top.attributes('-topmost', True)
        top.lift()
        listbox.focus_force()
        # Dismiss on click-away (focus loss); bound after focus has settled
        # so the binding cannot fire during the popup's own creation.
        top.after(200, lambda: top.winfo_exists() and top.bind(
            '<FocusOut>', lambda _event: top.destroy()))
        root.wait_window(top)
    finally:
        root.destroy()
    return result['label']


def _tk_range_dialog(current_vmin, current_vmax):
    """Modal min/max entry dialog.

    Returns ('apply', min_text, max_text), ('data',), or None if cancelled.
    Blank entries mean "keep the current bound"; parsing stays with the
    caller so this and the in-figure dialog share one code path.
    """
    import tkinter
    from tkinter import ttk

    root = tkinter.Tk()
    root.withdraw()
    result = {'value': None}
    try:
        top = tkinter.Toplevel(root)
        top.title('Set intensity range')
        top.resizable(False, False)
        frame = ttk.Frame(top, padding=12)
        frame.grid()
        hint = (f'Blank keeps the current bound '
                f'({current_vmin:.6g}, {current_vmax:.6g}).')
        ttk.Label(frame, text=hint).grid(column=0, row=0, columnspan=3,
                                         sticky='w', pady=(0, 8))
        ttk.Label(frame, text='Min').grid(column=0, row=1, sticky='w')
        min_entry = ttk.Entry(frame, width=18)
        min_entry.grid(column=1, row=1, columnspan=2, sticky='we', pady=2)
        ttk.Label(frame, text='Max').grid(column=0, row=2, sticky='w')
        max_entry = ttk.Entry(frame, width=18)
        max_entry.grid(column=1, row=2, columnspan=2, sticky='we', pady=2)

        def apply(_event=None):
            result['value'] = ('apply', min_entry.get(), max_entry.get())
            top.destroy()

        def data_range():
            result['value'] = ('data',)
            top.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(column=0, row=3, columnspan=3, pady=(10, 0))
        ttk.Button(buttons, text='Apply',
                   command=apply).grid(column=0, row=0, padx=3)
        ttk.Button(buttons, text='Data range',
                   command=data_range).grid(column=1, row=0, padx=3)
        ttk.Button(buttons, text='Cancel',
                   command=top.destroy).grid(column=2, row=0, padx=3)
        top.bind('<Return>', apply)
        top.bind('<Escape>', lambda _event: top.destroy())
        top.attributes('-topmost', True)
        top.lift()
        min_entry.focus_force()
        root.wait_window(top)
    finally:
        root.destroy()
    return result['value']


# Keys always offered by the save-time dict editor.
DICT_EDITOR_KEYS = ('model_params', 'notes', 'recon_log', 'recon_params')


def _tk_dict_editor(data_dict):
    """Modal editor for the data dict written by Save.

    Shows the union of DICT_EDITOR_KEYS and any existing keys, with a key
    list on the left and a multi-line editor on the right.  Returns
    ('save', edited_dict) (blank entries dropped), ('as-is',) to save the
    dict unchanged, or None to cancel the save.
    """
    import tkinter
    from tkinter import ttk

    existing = {str(k): str(v) for k, v in (data_dict or {}).items()}
    keys = sorted(set(DICT_EDITOR_KEYS) | set(existing))
    values = {key: existing.get(key, '') for key in keys}

    root = tkinter.Tk()
    root.withdraw()
    result = {'value': None}
    try:
        top = tkinter.Toplevel(root)
        top.title('Edit data dict before saving')
        frame = ttk.Frame(top, padding=10)
        frame.grid(sticky='nsew')
        listbox = tkinter.Listbox(frame, height=max(len(keys), 4),
                                  exportselection=False)
        for key in keys:
            listbox.insert('end', key)
        listbox.grid(column=0, row=0, sticky='ns', padx=(0, 8))
        text = tkinter.Text(frame, width=64, height=18, wrap='none')
        text.grid(column=1, row=0, sticky='nsew')
        state = {'key': keys[0]}
        listbox.selection_set(0)
        text.insert('1.0', values[state['key']])

        def store_current():
            values[state['key']] = text.get('1.0', 'end-1c')

        def on_select(_event=None):
            selection = listbox.curselection()
            if not selection or keys[selection[0]] == state['key']:
                return
            store_current()
            state['key'] = keys[selection[0]]
            text.delete('1.0', 'end')
            text.insert('1.0', values[state['key']])

        listbox.bind('<<ListboxSelect>>', on_select)

        def save():
            store_current()
            edited = {key: value for key, value in values.items() if value}
            result['value'] = ('save', edited)
            top.destroy()

        def save_as_is():
            result['value'] = ('as-is',)
            top.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(column=0, row=1, columnspan=2, pady=(8, 0))
        ttk.Button(buttons, text='Save',
                   command=save).grid(column=0, row=0, padx=4)
        ttk.Button(buttons, text='Save without editing',
                   command=save_as_is).grid(column=1, row=0, padx=4)
        ttk.Button(buttons, text='Cancel save',
                   command=top.destroy).grid(column=2, row=0, padx=4)
        top.bind('<Escape>', lambda _event: top.destroy())
        top.attributes('-topmost', True)
        top.lift()
        root.wait_window(top)
    finally:
        root.destroy()
    return result['value']


class SliceViewer:
    """Interactive multi-volume slice viewer for 2D and 3D arrays (matplotlib).

    The viewer displays one panel per volume with a shared slice slider,
    a shared intensity-range slider, and slice-axis radio buttons.
    Right-click an image for the context menu of per-image actions
    (difference/error images, data-dict display, transpose, file load/save,
    view reset) and the couple/decouple toggles.  Left-click and drag on an
    image draws an ROI circle with live statistics; press 'h' for help and
    Esc to clear overlays.  File dialogs and (under TkAgg) the menu and
    range dialogs use native windows; other backends use in-figure dialogs.

    Construction builds the figure but does not display it; call
    :meth:`show` to display.  All data logic lives in :class:`VolumeStack`
    (available as ``self.stack``).

    Args:
        *datasets (ndarray or None): One or more 2D or 3D arrays to display.
            2D arrays are promoted to 3D via a trailing singleton axis, and
            None values are replaced with placeholder zero arrays.
        data_dicts (None, dict, or list of dict/None, optional): String-valued
            dict(s) associated with the volumes, viewable in the viewer.
        title (str, optional): Figure title.  Defaults to ''.
        vmin (float, optional): Minimum display intensity.  Defaults to the
            global minimum across all datasets.
        vmax (float, optional): Maximum display intensity.  Defaults to the
            global maximum across all datasets.
        slice_label (str or list of str, optional): Label(s) shown before the
            slice number in each panel title.  Defaults to "Slice".
        slice_axis (int or list of int, optional): Axis along which to slice
            (0, 1, or 2).  Defaults to the last axis (2).
        cmap (str, optional): Colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): Show the "Press h for help" hint.
            Defaults to True.
        save_fn (callable, optional): Replacement for the built-in HDF5
            writer, called as ``save_fn(file_path, array, array_name,
            attributes_dict)``.  Defaults to a minimal h5py writer.
    """

    def __init__(self, *datasets, data_dicts=None, title='', vmin=None,
                 vmax=None, slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, save_fn=None):
        _load_pyplot()
        self.stack = VolumeStack(datasets, data_dicts=data_dicts, vmin=vmin,
                                 vmax=vmax, slice_label=slice_label,
                                 slice_axis=slice_axis)
        self.title = title
        self.cmap = cmap
        self.show_instructions = show_instructions
        self.save_fn = save_fn if save_fn is not None else _save_data_hdf5

        # Every piece of interaction state is initialized before any widget
        # or callback is created, so no callback can ever observe a
        # partially constructed viewer.
        self._init_interaction_state()
        self._build_figure()
        self._connect_events()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _init_interaction_state(self):
        n = self.stack.n_volumes
        self.mode = Mode.IDLE
        self._mode_data = {}
        # User-facing sync settings, separated from the reentrancy latch.
        self.sync_limits = True
        self.sync_axes = len(set(self.stack.slice_axes)) == 1
        self._in_sync_callback = False
        # ROI graphics: one circle and one stats text per volume, or None.
        self.circles = [None] * n
        self.stats_texts = [None] * n
        self.tooltips = []
        # Trailing-edge throttle state for ROI statistics.
        self._last_stats_time = 0.0
        self._stats_timer = None
        # Overlays.
        self._dialog = None
        self._message_artist = None
        # Blitting support; flips on once the first full draw has happened.
        self.enable_blit = True
        self._renderer_ready = False
        self._clear_rect = None
        self._last_blit_regions = {}
        # File dialog memory.
        self._last_dir = os.getcwd()

    def _build_figure(self):
        n = self.stack.n_volumes
        self.fig = plt.figure(figsize=(6 * n, 8))
        self.fig.suptitle(self.title)
        self.gs = gridspec.GridSpec(nrows=4, ncols=n,
                                    height_ratios=[15, 1.7, 1, 1],
                                    left=0.12, right=0.95, top=0.90,
                                    bottom=0.05, hspace=0.5, figure=self.fig)
        self.axes = [None] * n
        self.caxes = [None] * n
        self.images = [None] * n
        self._create_panels()
        self._create_tooltips()
        self._create_axis_row()
        self._create_slice_slider()
        self._create_intensity_slider()

        # The opaque rectangle drawn under partial redraws; animated=True
        # keeps it out of ordinary full draws.
        self._clear_rect = Rectangle((0, 0), 1, 1,
                                     facecolor=self.fig.get_facecolor(),
                                     edgecolor='none', animated=True,
                                     transform=IdentityTransform())
        self.fig.add_artist(self._clear_rect)

        if self.show_instructions:
            self.fig.text(0.01, 0.25, multiline('Press h', 'for help'),
                          fontdict={'color': 'red'})

    def _title_text(self, i):
        stack = self.stack
        return multiline(
            f"{stack.labels[i]} {stack.cur_slices[i]}",
            f"Shape: {stack.original_data[i].shape}, "
            f"Axes: {np.array(stack.axes_perms[i])}")

    def _create_panels(self):
        stack = self.stack
        for i in range(stack.n_volumes):
            ax = self.fig.add_subplot(self.gs[0, i])
            img = ax.imshow(stack.slice_image(i), cmap=self.cmap,
                            aspect='equal', vmin=stack.vmin, vmax=stack.vmax)
            # Limits are managed explicitly (_reset_view); autoscale must not
            # re-derive them from set_extent during refresh, which would leak
            # one volume's extent onto the others through the zoom sync.
            ax.set_autoscale_on(False)
            ax.set_title(self._title_text(i), fontsize=10)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            self.fig.colorbar(img, cax=cax, orientation='vertical')
            ax.zorder = 2
            cax.zorder = 1
            self.axes[i] = ax
            self.caxes[i] = cax
            self.images[i] = img

    def _create_tooltips(self):
        # Single owner: the tooltips are created exactly once, here.
        self.tooltips = [
            ax.annotate(TOOLTIP_TEXT, xy=(0, 0), xytext=TOOLTIP_OFFSET,
                        textcoords='offset points', ha='left',
                        fontsize=TOOLTIP_FONT_SIZE,
                        bbox=dict(boxstyle='round', fc='w',
                                  alpha=TOOLTIP_BOX_ALPHA),
                        arrowprops=dict(arrowstyle='->'), visible=False)
            for ax in self.axes
        ]

    # --- slice-axis radios and range button -------------------------------

    def _create_axis_row(self):
        n = self.stack.n_volumes
        # Each cell splits into [radio | spare]; the last spare cell holds
        # the Set-range button.  Global toggles (couple axes/zoom) live in
        # the right-click context menu.
        self._radio_slots = []
        spare_slots = []
        for i in range(n):
            sub = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=self.gs[1, i], wspace=0.3)
            self._radio_slots.append(sub[0, 0])
            spare_slots.append(sub[0, 1])

        self.axis_radios = []
        self._radio_axes = []
        self._rebuild_axis_radios()

        range_ax = self.fig.add_subplot(spare_slots[-1])
        self.range_button = Button(range_ax, 'Set range')
        self.range_button.label.set_fontsize(STRIP_FONT_SIZE)
        self.range_button.on_clicked(lambda _event: self._open_range_dialog())

    def _rebuild_axis_radios(self):
        for ax in self._radio_axes:
            ax.remove()
        self._radio_axes = []
        self.axis_radios = []

        def make_radio(slot, active):
            ax = self.fig.add_subplot(slot)
            ax.set_title("Slice axis", loc='left',
                         fontsize=SLICE_AXIS_FONT_SIZE)
            radio = RadioButtons(ax, labels=["0", "1", "2"], active=active,
                                 radio_props={'s': [SLICE_AXIS_RADIO_SIZE]})
            for label in radio.labels:
                label.set_fontsize(SLICE_AXIS_LABEL_FONT_SIZE)
            self._radio_axes.append(ax)
            self.axis_radios.append(radio)
            return radio

        if self.sync_axes or self.stack.n_volumes == 1:
            radio = make_radio(self._radio_slots[0],
                               self.stack.slice_axes[0])
            radio.on_clicked(
                lambda label: self._on_axis_selected(None, int(label)))
        else:
            for i in range(self.stack.n_volumes):
                radio = make_radio(self._radio_slots[i],
                                   self.stack.slice_axes[i])
                radio.on_clicked(
                    lambda label, i=i: self._on_axis_selected(i, int(label)))

    # --- sliders ---------------------------------------------------------

    def _slider_slot(self, row):
        # Inset the slider inside its row so the left label and right value
        # text render inside the figure instead of clipping at its edges.
        sub = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=self.gs[row, :], width_ratios=[1.7, 8.0, 2.0])
        return sub[0, 1]

    def _create_slice_slider(self):
        self._slice_slider_ax = self.fig.add_subplot(self._slider_slot(2))
        self.slice_slider = None
        if self.stack.max_slices > 1:
            self._make_slice_slider()
        else:
            self._slice_slider_ax.set_visible(False)

    def _make_slice_slider(self):
        self.slice_slider = Slider(self._slice_slider_ax, label="Slice",
                                   valmin=0,
                                   valmax=self.stack.max_slices - 1,
                                   valinit=self.stack.master_index,
                                   valstep=1, valfmt='%0.0f')
        self.slice_slider.drawon = False
        self.slice_slider.on_changed(self._on_slice_slider)
        self._slice_slider_ax.set_visible(True)

    @staticmethod
    def _intensity_valfmt(vmin, vmax):
        log_range = np.log10(vmax - vmin)
        digits = max(-int(np.round(log_range)) + 2, 0)
        return '%0.' + str(digits) + 'f'

    def _create_intensity_slider(self):
        ax = self.fig.add_subplot(self._slider_slot(3))
        stack = self.stack
        self.intensity_slider = RangeSlider(
            ax=ax, label="Intensity range", valmin=stack.vmin,
            valmax=stack.vmax, valinit=(stack.vmin, stack.vmax),
            valfmt=self._intensity_valfmt(stack.vmin, stack.vmax))
        self.intensity_slider.drawon = False
        self.intensity_slider.on_changed(self._on_intensity_slider)

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _connect_events(self):
        canvas = self.fig.canvas
        canvas.mpl_connect('button_press_event', self._on_button_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)
        canvas.mpl_connect('key_press_event', self._on_key)
        canvas.mpl_connect('draw_event', self._on_draw_event)
        for ax in self.axes:
            ax.callbacks.connect(
                'xlim_changed', lambda a: self._sync_limits_from(a, 'x'))
            ax.callbacks.connect(
                'ylim_changed', lambda a: self._sync_limits_from(a, 'y'))
        self._patch_toolbar()

    def _patch_toolbar(self):
        """Adapt the navigation toolbar to this viewer (instance patches).

        Home/Back/Forward restore every panel's own saved view through
        ``_update_view``; each restore fires the zoom-sync callback, so
        without the latch the last panel restored would stamp its view onto
        all the others (visibly cropping them when displayed shapes differ).
        The restore therefore runs with the sync latch held.

        On macosx only, ``drag_zoom`` also needs help: that backend fills
        motion events' ``buttons`` from the LIVE hardware state
        (NSEvent.pressedMouseButtons), so motion events still queued when a
        fast drag releases are processed with an empty buttons set, and
        matplotlib >= 3.10 treats that as a missed release and cancels the
        zoom.  During an active zoom, missing button state counts as
        still-held; a real release still completes the gesture normally.
        """
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is None:
            return
        import types

        stock_update_view = type(toolbar)._update_view

        def _update_view(tb):
            already_syncing = self._in_sync_callback
            self._in_sync_callback = True
            try:
                stock_update_view(tb)
            finally:
                self._in_sync_callback = already_syncing

        toolbar._update_view = types.MethodType(_update_view, toolbar)

        # Home with an empty navigation stack is a silent no-op in stock
        # matplotlib; in this viewer "home" is unambiguous, so fall back to
        # the explicit full-extent reset.
        stock_home = type(toolbar).home

        def home(tb, *args):
            if tb._nav_stack() is None:
                self._reset_view()
                self.fig.canvas.draw_idle()
                return
            stock_home(tb, *args)

        toolbar.home = types.MethodType(home, toolbar)

        if matplotlib.get_backend().lower() == 'macosx':
            stock_drag_zoom = type(toolbar).drag_zoom

            def drag_zoom(tb, event):
                if tb._zoom_info is not None and not event.buttons:
                    event.buttons = frozenset({tb._zoom_info.button})
                stock_drag_zoom(tb, event)

            toolbar.drag_zoom = types.MethodType(drag_zoom, toolbar)

    def _on_draw_event(self, _event):
        self._renderer_ready = True

    def _toolbar_active(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return bool(getattr(toolbar, 'mode', ''))

    def _sync_limits_from(self, src_ax, which):
        # One implementation for both directions; sync_limits is the user
        # setting, _in_sync_callback the reentrancy latch.
        if not self.sync_limits or self._in_sync_callback:
            return
        if src_ax not in self.axes:
            return
        self._in_sync_callback = True
        try:
            lim = src_ax.get_xlim() if which == 'x' else src_ax.get_ylim()
            for ax in self.axes:
                if ax is not src_ax:
                    (ax.set_xlim if which == 'x' else ax.set_ylim)(lim)
        finally:
            self._in_sync_callback = False
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # The one button-press dispatcher
    # ------------------------------------------------------------------

    def _volume_index_of(self, ax):
        try:
            return self.axes.index(ax)
        except ValueError:
            return None

    def _on_button_press(self, event):
        if self._dialog is not None:
            # A click outside an open context menu dismisses it; all other
            # dialogs' widgets handle their own events.
            if (self._dialog['kind'] == 'menu'
                    and event.inaxes not in self._dialog.get('menu_axes', ())):
                self._close_dialog()
            return
        if event.button == 3:
            # Right-click works even while the toolbar's pan/zoom tool is
            # active, as in the reference viewer.
            if event.inaxes is self.intensity_slider.ax:
                self._open_range_dialog()
                return
            i = self._volume_index_of(event.inaxes)
            if i is not None and self.mode is not Mode.SELECT_COMPARISON:
                self._open_context_menu(i, event)
            return
        if event.button != 1:
            return
        # Comparison selection outranks the toolbar tools: while the
        # instruction overlay is up, a left-click on an image always means
        # "this one", even if a pan/zoom mode is somehow armed.
        if self.mode is Mode.SELECT_COMPARISON:
            i = self._volume_index_of(event.inaxes)
            if i is not None:
                self._complete_difference(i)
            return
        if self._toolbar_active():
            return
        i = self._volume_index_of(event.inaxes)
        if i is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Hit-test the ROI circles: edge -> resize, interior -> move,
        # anywhere else -> start a new ROI.
        for circle in self.circles:
            if circle is None:
                continue
            x, y = circle.center
            radius = circle.get_radius()
            dist = np.hypot(event.xdata - x, event.ydata - y)
            if abs(dist - radius) <= 0.1 * radius:
                self._set_mode(Mode.RESIZE_ROI, anchor=(x, y))
                return
            if dist <= radius:
                self._set_mode(Mode.MOVE_ROI,
                               offset=(event.xdata - x, event.ydata - y))
                return
        self._remove_roi_graphics()
        self._set_mode(Mode.DRAW_ROI, start=(event.xdata, event.ydata))
        for j, ax in enumerate(self.axes):
            circle = plt.Circle((event.xdata, event.ydata), 0,
                                color=CIRCLE_COLOR, lw=CIRCLE_LINEWIDTH,
                                fill=CIRCLE_FILL, alpha=CIRCLE_ALPHA)
            self.circles[j] = circle
            ax.add_patch(circle)
        self._hide_tooltips()
        self._partial_redraw()

    def _set_mode(self, mode, **data):
        self.mode = mode
        self._mode_data = data

    def _on_motion(self, event):
        if self._dialog is not None or self._toolbar_active():
            return
        if event.inaxes not in self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.mode is Mode.IDLE or self.mode is Mode.SELECT_COMPARISON:
            self._update_tooltips(event)
            return
        if self.mode is Mode.DRAW_ROI:
            x0, y0 = self._mode_data['start']
            radius = np.hypot(event.xdata - x0, event.ydata - y0)
            for circle in self.circles:
                circle.set_radius(radius)
        elif self.mode is Mode.MOVE_ROI:
            dx, dy = self._mode_data['offset']
            center = (event.xdata - dx, event.ydata - dy)
            for circle in self.circles:
                circle.center = center
        elif self.mode is Mode.RESIZE_ROI:
            x0, y0 = self._mode_data['anchor']
            radius = np.hypot(event.xdata - x0, event.ydata - y0)
            for circle in self.circles:
                circle.set_radius(radius)
        self._display_roi_stats()
        self._partial_redraw()

    def _on_release(self, _event):
        if self.mode in (Mode.DRAW_ROI, Mode.MOVE_ROI, Mode.RESIZE_ROI):
            self._set_mode(Mode.IDLE)
            # Force the final update so a fast release never leaves stats
            # showing the pre-release position.
            self._display_roi_stats(force=True)
            self._partial_redraw()

    def _on_key(self, event):
        if self._dialog is not None:
            if event.key == 'escape':
                self._close_dialog()
            return
        handlers = {'h': lambda: self._show_message(True, message_type='help'),
                    'escape': self._on_escape}
        handler = handlers.get(event.key)
        if handler is not None:
            handler()

    def _on_escape(self):
        self._show_message(False)
        self._set_mode(Mode.IDLE)
        self._remove_roi_graphics()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # ROI graphics, tooltips, and statistics
    # ------------------------------------------------------------------

    def _remove_roi_graphics(self):
        for artist in self.circles + self.stats_texts:
            if artist is not None:
                artist.remove()
        n = self.stack.n_volumes
        self.circles = [None] * n
        self.stats_texts = [None] * n
        self._hide_tooltips()

    def _hide_tooltips(self):
        for tooltip in self.tooltips:
            tooltip.set_visible(False)

    def _update_tooltips(self, event):
        changed = []
        for i, circle in enumerate(self.circles):
            if circle is None:
                continue
            x, y = circle.center
            radius = circle.get_radius()
            dist = np.hypot(event.xdata - x, event.ydata - y)
            visible = dist <= 1.1 * radius and event.inaxes is self.axes[i]
            if visible:
                self.tooltips[i].xy = circle.center
            if visible != self.tooltips[i].get_visible():
                self.tooltips[i].set_visible(visible)
                changed.append(i)
        if changed:
            self._partial_redraw(changed)

    def _display_roi_stats(self, force=False):
        if all(circle is None for circle in self.circles):
            return
        now = time.monotonic()
        if not force and now - self._last_stats_time < ROI_STATS_THROTTLE_S:
            # Trailing edge: guarantee one final update after the burst.
            self._schedule_trailing_stats()
            return
        self._last_stats_time = now
        for i, circle in enumerate(self.circles):
            if circle is None:
                continue
            x, y = circle.center
            stats = self.stack.roi_stats(i, x, y, circle.get_radius())
            if stats is None:
                continue
            text = multiline(
                f"(µ, σ)=({stats['mean']:.3g}, {stats['std']:.3g})",
                f"(min, max)=({stats['min']:.3g}, {stats['max']:.3g})")
            if self.stats_texts[i] is not None:
                self.stats_texts[i].set_text(text)
            else:
                self.stats_texts[i] = self.axes[i].text(
                    0.05, 0.95, text, transform=self.axes[i].transAxes,
                    fontsize=STATS_FONT_SIZE, va='top',
                    bbox=dict(facecolor='white', alpha=1.0))

    def _schedule_trailing_stats(self):
        if self._stats_timer is not None:
            return
        timer = self.fig.canvas.new_timer(
            interval=int(ROI_STATS_THROTTLE_S * 1000) + 50)
        timer.single_shot = True
        timer.add_callback(self._trailing_stats_fire)
        timer.start()
        self._stats_timer = timer

    def _trailing_stats_fire(self):
        self._stats_timer = None
        self._display_roi_stats(force=True)
        self._partial_redraw()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def _show_message(self, show, message_type=None, message=None):
        if self._message_artist is not None:
            self._message_artist.remove()
            self._message_artist = None
        if show:
            if message_type == 'help':
                message = multiline(
                    'Left-click and drag on an image for an ROI',
                    'Right-click an image for the menu',
                    'Right-click the intensity slider or press Set range '
                    'for exact bounds',
                    'Press [esc] to remove ROI/messages/dialogs',
                    'Close the window to quit')
            elif message_type == 'difference':
                message = multiline(
                    'Select another image of the same shape',
                    'or press [esc] to exit')
            if message is None:
                return
            self._message_artist = self.fig.text(
                0.25, 0.5, message, ha='left', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9), zorder=20)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # In-place refresh (there is no teardown/rebuild path)
    # ------------------------------------------------------------------

    def refresh(self, volume_index=None):
        """Push model state into the panels in place: data, extent, title."""
        indices = (range(self.stack.n_volumes) if volume_index is None
                   else [volume_index])
        for i in indices:
            slice_2d = self.stack.slice_image(i)
            self.images[i].set_data(slice_2d)
            h, w = slice_2d.shape
            self.images[i].set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
            self.axes[i].set_title(self._title_text(i), fontsize=10)

    def _reset_view(self, volume_index=None):
        # Hold the sync latch: a programmatic reset gives each volume its
        # own full extent, and must not propagate one volume's extent onto
        # panels with different displayed shapes (e.g. after decoupled axis
        # changes).
        indices = (range(self.stack.n_volumes) if volume_index is None
                   else [volume_index])
        already_syncing = self._in_sync_callback
        self._in_sync_callback = True
        try:
            for i in indices:
                h, w = self.stack.slice_image(i).shape
                self.axes[i].set_xlim(-0.5, w - 0.5)
                self.axes[i].set_ylim(h - 0.5, -0.5)
        finally:
            self._in_sync_callback = already_syncing

    def _update_slice_slider(self):
        """Track model depth changes: bounds, value, and visibility."""
        max_slices = self.stack.max_slices
        if max_slices <= 1:
            if self.slice_slider is not None:
                self._slice_slider_ax.set_visible(False)
            return
        if self.slice_slider is None:
            self._make_slice_slider()
            return
        self._slice_slider_ax.set_visible(True)
        self.slice_slider.valmax = max_slices - 1
        self.slice_slider.ax.set_xlim(0, max_slices - 1)
        self.slice_slider.set_val(self.stack.master_index)

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------

    def _on_slice_slider(self, value):
        changed = self.stack.set_master_index(value)
        for i in changed:
            self.refresh(i)
        if changed:
            self._display_roi_stats()
        self._partial_redraw(changed, widgets=('slice',))

    def _on_intensity_slider(self, value):
        for img in self.images:
            img.set_clim(value[0], value[1])
        self._partial_redraw(widgets=('intensity',))

    # ------------------------------------------------------------------
    # Slice-axis selection and coupling
    # ------------------------------------------------------------------

    def _reset_navigation(self):
        """Clear the toolbar's saved views; they are stale after structural
        changes (axis change, transpose, load), and Home falls back to the
        explicit full-extent reset when the stack is empty."""
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is not None:
            try:
                toolbar.update()
            except Exception:
                pass

    def _on_axis_selected(self, volume_index, axis):
        indices = (range(self.stack.n_volumes) if volume_index is None
                   else [volume_index])
        any_changed = False
        for i in indices:
            if self.stack.set_perm(i, axis):
                any_changed = True
                self._reset_view(i)
        if not any_changed:
            return
        self.refresh()  # fraction-preserving remap can move any volume
        self._update_slice_slider()
        self._reset_navigation()
        self.fig.canvas.draw_idle()

    def _toggle_couple_axes(self):
        self.sync_axes = not self.sync_axes
        self._rebuild_axis_radios()
        if self.sync_axes:
            # Harmonize every volume to volume 0's slice axis.
            self._on_axis_selected(None, self.stack.slice_axes[0])
        self.fig.canvas.draw_idle()

    def _toggle_couple_zoom(self):
        self.sync_limits = not self.sync_limits

    # ------------------------------------------------------------------
    # The context menu
    # ------------------------------------------------------------------

    def _in_process_tk_ok(self):
        """True when opening an in-process Tk window is safe.

        Only under TkAgg, where Tk already runs the GUI: creating a Tk root
        inside another toolkit's event loop crashes at the native level on
        the macosx backend (SIGBUS), so everywhere else the viewer uses its
        in-figure dialogs or out-of-process (osascript) ones.
        """
        return matplotlib.get_backend().lower() == 'tkagg'

    def _menu_items(self, i):
        """(label, callback) pairs for volume ``i``'s context menu.

        A None callback (Cancel) just closes the menu.
        """
        stack = self.stack
        items = []
        if stack.n_volumes > 1:
            items.append((('Decouple' if self.sync_axes else 'Couple')
                          + ' slice axes', self._toggle_couple_axes))
            items.append((('Decouple' if self.sync_limits else 'Couple')
                          + ' pan/zoom', self._toggle_couple_zoom))
            if stack.is_difference(i):
                items.append(('Restore original image',
                              lambda i=i: self._on_restore(i)))
            else:
                items.append(('Replace with difference image',
                              lambda i=i: self._on_difference_button(i)))
                items.append(('Replace with error image',
                              lambda i=i: self._on_error_button(i)))
        items += [
            ('Show data dict', lambda i=i: self._on_dict_button(i)),
            ('Transpose image', lambda i=i: self._on_transpose_button(i)),
            ('Load', lambda i=i: self._on_load_button(i)),
            ('Save data to h5', lambda i=i: self._on_save_button(i)),
            ('Reset', lambda i=i: self._on_reset_button(i)),
            ('Cancel', None),
        ]
        return items

    def _open_context_menu(self, i, event):
        items = self._menu_items(i)
        if self._in_process_tk_ok():
            def run():
                try:
                    chosen = _tk_menu([label for label, _cb in items])
                except Exception:
                    self._open_menu_dialog(items, event)
                    return
                for label, callback in items:
                    if label == chosen and callback is not None:
                        callback()
            # Launch from Tk's scheduler, not from inside this callback.
            try:
                self.fig.canvas.manager.window.after(10, run)
                return
            except Exception:
                pass
            run()
            return
        self._open_menu_dialog(items, event)

    def _open_menu_dialog(self, items, event):
        """In-figure popup menu at the click position (non-Tk backends)."""
        dialog = self._open_dialog('menu', dim=False)
        canvas_w, canvas_h = self.fig.canvas.get_width_height()
        fx, fy = event.x / canvas_w, event.y / canvas_h
        fig_w, fig_h = self.fig.get_size_inches()
        longest = max(len(label) for label, _cb in items)
        w = min(max(1.9, 0.078 * longest) / fig_w, 0.6)
        row_h = 0.26 / fig_h
        h = row_h * len(items) + 0.012
        x0 = float(np.clip(fx, 0.005, 1 - w - 0.005))
        y0 = float(np.clip(fy - h, 0.005, 1 - h - 0.005))
        panel = self.fig.add_axes((x0, y0, w, h), zorder=DIALOG_ZORDER + 1,
                                  facecolor='white')
        panel.set_xticks([])
        panel.set_yticks([])
        dialog['axes'].append(panel)
        dialog['panel_ax'] = panel
        menu_axes = []
        for j, (label, callback) in enumerate(items):
            button = self._dialog_button(
                label, (x0 + 0.004, y0 + h - (j + 1) * row_h + 0.004,
                        w - 0.008, row_h - 0.004),
                lambda callback=callback: self._menu_dispatch(callback),
                key=f'item{j}')
            button.label.set_fontsize(DIALOG_FONT_SIZE)
            menu_axes.append(button.ax)
        dialog['menu_axes'] = menu_axes
        self.fig.canvas.draw_idle()

    def _menu_dispatch(self, callback):
        self._close_dialog(draw=False)
        if callback is not None:
            callback()
        else:
            self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Strip actions
    # ------------------------------------------------------------------

    def _on_difference_button(self, i, use_abs=False):
        # NOTE: no toolbar state is touched here.  Toggling pan/zoom
        # programmatically desynchronizes the macosx backend's NATIVE
        # buttons from matplotlib's internal mode (the visual inverts);
        # selection clicks work with a tool armed because the dispatcher
        # checks SELECT_COMPARISON before the toolbar guard.
        if self.stack.is_difference(i):
            self._on_restore(i)
            return
        if self.stack.n_volumes == 2:
            other = 1 - i
            if self.stack.can_difference(i, other):
                self._apply_difference(i, other, use_abs)
                return
        self._set_mode(Mode.SELECT_COMPARISON, baseline_index=i,
                       use_abs=use_abs)
        self._show_message(True, message_type='difference')

    def _on_error_button(self, i):
        if self.stack.is_difference(i):
            return  # the strip shows Restore in this state; Err is inert
        self._on_difference_button(i, use_abs=True)

    def _complete_difference(self, comparison_index):
        baseline_index = self._mode_data['baseline_index']
        use_abs = self._mode_data['use_abs']
        if not self.stack.can_difference(baseline_index, comparison_index):
            self._show_message(True, message_type='difference')
            return  # stay in selection mode, as the reference does
        self._set_mode(Mode.IDLE)
        self._show_message(False)
        self._apply_difference(baseline_index, comparison_index, use_abs)

    def _apply_difference(self, baseline_index, comparison_index, use_abs):
        self.stack.apply_difference(baseline_index, comparison_index, use_abs)
        self.refresh(baseline_index)
        self.fig.canvas.draw_idle()

    def _on_restore(self, i):
        self.stack.restore(i)
        self.refresh(i)
        self.fig.canvas.draw_idle()

    def _on_transpose_button(self, i):
        self.stack.transpose(i)
        self._reset_view(i)
        self.refresh()
        self._update_slice_slider()
        self._reset_navigation()
        self.fig.canvas.draw_idle()

    def _on_reset_button(self, i):
        # View-level reset: zoom, ROI, overlays, and any pending selection.
        self._reset_view(None if self.sync_limits else i)
        self._remove_roi_graphics()
        self._set_mode(Mode.IDLE)
        self._show_message(False)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Dialog infrastructure: modal in-figure panels
    # ------------------------------------------------------------------
    # A dialog is a set of high-zorder axes: a translucent full-figure
    # backdrop plus a white panel holding widgets.  The backdrop's zorder
    # makes event.inaxes resolve to dialog axes only, so the widgets and
    # dispatcher underneath are inert while a dialog is open.

    def _main_widgets(self):
        widgets = list(self.axis_radios)
        widgets.append(self.range_button)
        if self.slice_slider is not None:
            widgets.append(self.slice_slider)
        widgets.append(self.intensity_slider)
        return widgets

    def _open_dialog(self, kind, dim=True):
        self._close_dialog(draw=False)
        backdrop = self.fig.add_axes((0, 0, 1, 1), zorder=DIALOG_ZORDER,
                                     facecolor=(1, 1, 1, 0.6) if dim
                                     else 'none')
        backdrop.set_xticks([])
        backdrop.set_yticks([])
        self._dialog = {'kind': kind, 'axes': [backdrop], 'widgets': {},
                        'texts': {}, 'state': {}}
        # Matplotlib widgets hit-test geometrically (Axes.contains), not by
        # stacking order, so widgets underneath the dialog must be disabled
        # explicitly.  (.active is the flag every widget's ignore() checks;
        # set_active cannot be used because RadioButtons/CheckButtons
        # repurpose that name for option selection.)
        for widget in self._main_widgets():
            widget.active = False
        return self._dialog

    def _close_dialog(self, draw=True):
        if self._dialog is None:
            return
        for ax in self._dialog['axes']:
            ax.remove()
        self._dialog = None
        for widget in self._main_widgets():
            widget.active = True
        if draw:
            self.fig.canvas.draw_idle()

    def _dialog_panel(self, width_in, height_in):
        """Add the centered white panel; return its (x0, y0, w, h) in figure
        fractions for laying out the widgets inside."""
        fig_w, fig_h = self.fig.get_size_inches()
        w = min(width_in / fig_w, 0.92)
        h = min(height_in / fig_h, 0.88)
        x0, y0 = (1 - w) / 2, (1 - h) / 2
        panel = self.fig.add_axes((x0, y0, w, h), zorder=DIALOG_ZORDER + 1,
                                  facecolor='white')
        panel.set_xticks([])
        panel.set_yticks([])
        self._dialog['axes'].append(panel)
        self._dialog['panel_ax'] = panel
        return x0, y0, w, h

    def _dialog_button(self, label, rect, callback, key=None):
        ax = self.fig.add_axes(rect, zorder=DIALOG_ZORDER + 2)
        button = Button(ax, label)
        button.label.set_fontsize(DIALOG_FONT_SIZE)
        button.on_clicked(lambda _event: callback())
        self._dialog['axes'].append(ax)
        self._dialog['widgets'][key if key is not None else label] = button
        return button

    def _dialog_textbox(self, name, label, rect, initial):
        ax = self.fig.add_axes(rect, zorder=DIALOG_ZORDER + 2)
        textbox = TextBox(ax, label, initial=initial, textalignment='left')
        textbox.label.set_fontsize(DIALOG_FONT_SIZE)
        textbox.text_disp.set_fontsize(DIALOG_FONT_SIZE)
        textbox.text_disp.set_clip_on(True)  # long paths clip at the box edge
        self._dialog['axes'].append(ax)
        self._dialog['widgets'][name] = textbox
        return textbox

    def _dialog_text(self, name, xy, text, **kwargs):
        # Texts are children of the panel axes so they draw above the panel's
        # opaque face; the transform keeps their coordinates in figure
        # fractions, and panel removal removes them.
        kwargs.setdefault('fontsize', DIALOG_FONT_SIZE)
        panel = self._dialog['panel_ax']
        artist = panel.text(xy[0], xy[1], text,
                            transform=self.fig.transFigure, **kwargs)
        self._dialog['texts'][name] = artist
        return artist

    # --- intensity-range dialog ------------------------------------------

    def _open_range_dialog(self):
        if self._in_process_tk_ok():
            def run():
                try:
                    result = _tk_range_dialog(self.stack.vmin,
                                              self.stack.vmax)
                except Exception:
                    self._open_range_dialog_infigure()
                    return
                self._handle_range_result(result)
            try:
                self.fig.canvas.manager.window.after(10, run)
                return
            except Exception:
                pass
            run()
            return
        self._open_range_dialog_infigure()

    def _handle_range_result(self, result):
        """Apply a Tk range-dialog result; errors go to a figure overlay."""
        if result is None:
            return
        try:
            if result[0] == 'data':
                self.stack.set_range(None, None)
            else:
                self._parse_and_set_range(result[1], result[2])
        except ValueError as e:
            self._show_message(
                True, message=f"{e}. Press Esc to dismiss.")
            return
        self._set_intensity_bounds()
        self.fig.canvas.draw_idle()

    def _parse_and_set_range(self, min_text, max_text):
        """Shared range parsing: blank keeps the current bound."""
        try:
            vmin = (self.stack.vmin if not min_text.strip()
                    else float(min_text))
            vmax = (self.stack.vmax if not max_text.strip()
                    else float(max_text))
        except ValueError:
            raise ValueError(
                'Enter numbers, or leave blank to keep a bound')
        self.stack.set_range(vmin, vmax)

    def _open_range_dialog_infigure(self):
        stack = self.stack
        dialog = self._open_dialog('range')
        x0, y0, w, h = self._dialog_panel(4.6, 2.6)
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.06),
                          'Set intensity range', fontweight='bold')
        # Fields start empty so new bounds can be typed without fighting the
        # cursor; a blank field keeps its current bound.
        self._dialog_text('hint', (x0 + 0.02 * w, y0 + h - 0.11),
                          f'Blank keeps the current bound '
                          f'({stack.vmin:.6g}, {stack.vmax:.6g}).')
        box_w, box_h = 0.45 * w, 0.14 * h
        self._dialog_textbox('min', 'Min ',
                             (x0 + 0.14 * w, y0 + 0.52 * h, box_w, box_h), '')
        self._dialog_textbox('max', 'Max ',
                             (x0 + 0.14 * w, y0 + 0.32 * h, box_w, box_h), '')
        self._dialog_text('error', (x0 + 0.02 * w, y0 + 0.22 * h), '',
                          color='red')
        btn_h = 0.14 * h
        self._dialog_button('Apply',
                            (x0 + 0.30 * w, y0 + 0.05 * h, 0.18 * w, btn_h),
                            self._apply_range_dialog)
        self._dialog_button('Data range',
                            (x0 + 0.52 * w, y0 + 0.05 * h, 0.24 * w, btn_h),
                            self._apply_data_range)
        self._dialog_button('Cancel',
                            (x0 + 0.80 * w, y0 + 0.05 * h, 0.18 * w, btn_h),
                            self._close_dialog)
        self.fig.canvas.draw_idle()
        return dialog

    def _apply_range_dialog(self):
        widgets = self._dialog['widgets']
        try:
            self._parse_and_set_range(widgets['min'].text,
                                      widgets['max'].text)
        except ValueError as e:
            self._dialog['texts']['error'].set_text(str(e))
            self.fig.canvas.draw_idle()
            return
        self._close_dialog(draw=False)
        self._set_intensity_bounds()
        self.fig.canvas.draw_idle()

    def _apply_data_range(self):
        self.stack.set_range(None, None)
        self._close_dialog(draw=False)
        self._set_intensity_bounds()
        self.fig.canvas.draw_idle()

    def _set_intensity_bounds(self):
        """Push stack.vmin/vmax into the range slider and the images."""
        stack = self.stack
        slider = self.intensity_slider
        slider.valmin = stack.vmin
        slider.valmax = stack.vmax
        slider.valfmt = self._intensity_valfmt(stack.vmin, stack.vmax)
        slider.ax.set_xlim(stack.vmin, stack.vmax)
        slider.set_val((stack.vmin, stack.vmax))

    # --- data-dict dialogs -------------------------------------------------

    def _on_dict_button(self, i):
        data_dict = self.stack.data_dicts[i]
        if not data_dict:
            self._open_text_dialog('No data dict',
                                   'No data dict included with this image')
            return
        names = list(data_dict.keys())
        if len(names) == 1:
            self._open_text_dialog(names[0], str(data_dict[names[0]]))
            return
        self._open_dict_chooser(i)

    def _open_dict_chooser(self, i):
        names = list(self.stack.data_dicts[i].keys())
        self._open_choice_dialog(
            'Choose an entry to display', names,
            lambda name, i=i: self._open_text_dialog(
                name, str(self.stack.data_dicts[i][name]),
                back=lambda i=i: self._open_dict_chooser(i)))

    def _open_choice_dialog(self, title, labels, on_choice, sublabels=None):
        self._open_dialog('choice')
        rows = len(labels) + 1
        row_h_in = 0.42
        panel_h = min(rows * row_h_in + 0.7, 7.0)
        x0, y0, w, h = self._dialog_panel(4.6, panel_h)
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.05), title,
                          fontweight='bold')
        btn_h = (h - 0.12) / rows * 0.8
        for j, label in enumerate(labels):
            text = label if sublabels is None else f"{label}: {sublabels[j]}"
            top = y0 + h - 0.10 - (j + 1) * (h - 0.12) / rows
            button = self._dialog_button(
                text, (x0 + 0.05 * w, top, 0.9 * w, btn_h),
                lambda label=label: self._dialog_choice_made(on_choice, label))
            button.label.set_fontsize(DIALOG_FONT_SIZE)
        top = y0 + h - 0.10 - rows * (h - 0.12) / rows
        self._dialog_button('Cancel', (x0 + 0.65 * w, top, 0.3 * w, btn_h),
                            self._close_dialog)
        self.fig.canvas.draw_idle()

    def _dialog_choice_made(self, on_choice, label):
        self._close_dialog(draw=False)
        on_choice(label)

    def _open_text_dialog(self, title, body, back=None):
        self._open_dialog('text')
        x0, y0, w, h = self._dialog_panel(5.6, 6.0)
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.045), title,
                          fontweight='bold')
        lines = str(body).splitlines() or ['']
        pages = [lines[k:k + DIALOG_PAGE_LINES]
                 for k in range(0, len(lines), DIALOG_PAGE_LINES)]
        state = self._dialog['state']
        state['pages'] = pages
        state['page'] = 0
        body_artist = self._dialog_text(
            'body', (x0 + 0.03 * w, y0 + h - 0.09), '',
            fontsize=DIALOG_BODY_FONT_SIZE, family='monospace', va='top')
        body_artist.set_wrap(True)
        btn_w, btn_h = 0.16 * w, 0.05
        if len(pages) > 1:
            self._dialog_button('Prev',
                                (x0 + 0.03 * w, y0 + 0.02, btn_w, btn_h),
                                lambda: self._flip_text_page(-1))
            self._dialog_button('Next',
                                (x0 + 0.22 * w, y0 + 0.02, btn_w, btn_h),
                                lambda: self._flip_text_page(+1))
        if back is not None:
            self._dialog_button('Back',
                                (x0 + 0.55 * w, y0 + 0.02, btn_w, btn_h),
                                lambda: self._dialog_choice_made(
                                    lambda _label: back(), None))
        self._dialog_button('Close',
                            (x0 + 0.78 * w, y0 + 0.02, btn_w, btn_h),
                            self._close_dialog)
        self._render_text_page()
        self.fig.canvas.draw_idle()

    def _render_text_page(self):
        state = self._dialog['state']
        pages, page = state['pages'], state['page']
        text = '\n'.join(pages[page])
        if len(pages) > 1:
            text += f"\n\n[page {page + 1} of {len(pages)}]"
        self._dialog['texts']['body'].set_text(text)

    def _flip_text_page(self, step):
        state = self._dialog['state']
        state['page'] = int(np.clip(state['page'] + step, 0,
                                    len(state['pages']) - 1))
        self._render_text_page()
        self.fig.canvas.draw_idle()

    # --- file load/save dialogs --------------------------------------------
    # File selection tries the platform's native dialog first -- Qt's on Qt
    # backends, the macOS panel via osascript (a separate process, so it
    # cannot conflict with the GUI event loop), then tkinter's -- and falls
    # back to the in-figure browser when none is available.  Everything is
    # resolved lazily at click time; nothing here adds an import-time
    # dependency.

    def _native_choose_file(self, mode, directory, initial_file):
        """Return a chosen path, None if cancelled, or _NATIVE_UNAVAILABLE."""
        backend = matplotlib.get_backend().lower()
        if backend in NONINTERACTIVE_BACKENDS:
            return _NATIVE_UNAVAILABLE
        if backend.startswith('qt'):
            try:
                return self._choose_file_qt(mode, directory, initial_file)
            except Exception:
                pass
        if sys.platform == 'darwin':
            try:
                return self._choose_file_osascript(mode, directory,
                                                   initial_file)
            except Exception:
                pass
        # In-process Tk is safe only under TkAgg; under other toolkits'
        # event loops it can crash the process outright, so fall back to
        # the in-figure browser instead of risking it.
        if not self._in_process_tk_ok():
            return _NATIVE_UNAVAILABLE
        try:
            return self._choose_file_tkinter(mode, directory, initial_file)
        except Exception:
            return _NATIVE_UNAVAILABLE

    def _choose_file_qt(self, mode, directory, initial_file):
        from matplotlib.backends.qt_compat import QtWidgets
        parent = getattr(self.fig.canvas.manager, 'window', None)
        if mode == 'load':
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent, 'Load a volume', directory,
                'Volumes (*.npy *.npz *.h5 *.hdf5);;All files (*)')
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                parent, 'Save volume to HDF5',
                os.path.join(directory, initial_file), 'HDF5 (*.h5)')
        return path or None

    def _choose_file_osascript(self, mode, directory, initial_file):
        import subprocess

        def quoted(text):
            return text.replace('\\', '\\\\').replace('"', '\\"')

        if mode == 'load':
            script = ('POSIX path of (choose file with prompt '
                      '"Load a volume" of type {"npy", "npz", "h5", "hdf5"} '
                      f'default location POSIX file "{quoted(directory)}")')
        else:
            script = ('POSIX path of (choose file name with prompt '
                      '"Save volume to HDF5" '
                      f'default name "{quoted(initial_file)}" '
                      f'default location POSIX file "{quoted(directory)}")')
        result = subprocess.run(['osascript', '-e', script],
                                capture_output=True, text=True)
        if result.returncode != 0:
            if 'canc' in (result.stderr or '').lower():
                return None  # the user cancelled the panel
            raise RuntimeError(result.stderr.strip())
        path = result.stdout.strip()
        return path or None

    def _choose_file_tkinter(self, mode, directory, initial_file):
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        try:
            if mode == 'load':
                path = filedialog.askopenfilename(
                    parent=root, initialdir=directory, title='Load a volume',
                    filetypes=[('Volumes', '*.npy *.npz *.h5 *.hdf5'),
                               ('All files', '*.*')])
            else:
                path = filedialog.asksaveasfilename(
                    parent=root, initialdir=directory,
                    initialfile=initial_file, title='Save volume to HDF5',
                    defaultextension='.h5', filetypes=[('HDF5', '*.h5')])
        finally:
            root.destroy()
        return path or None

    def _file_action(self, i, mode):
        def run():
            chosen = self._native_choose_file(mode, self._last_dir,
                                              'volume.h5')
            if chosen is _NATIVE_UNAVAILABLE:
                self._open_file_dialog(i, mode)
                return
            if chosen is None:
                return
            self._last_dir = os.path.dirname(chosen) or self._last_dir
            if mode == 'load':
                self._start_load(i, chosen)
            else:
                self._finish_save(i, chosen)

        if matplotlib.get_backend().lower() == 'tkagg':
            # Launch from Tk's scheduler rather than from inside this
            # callback; launching a second Tk root re-entrantly misbehaves.
            try:
                self.fig.canvas.manager.window.after(10, run)
                return
            except Exception:
                pass
        run()

    def _open_file_dialog(self, i, mode, directory=None, page=0,
                          filename=None, select_name=None):
        """Load/save dialog with a directory listing.

        Clicking a directory navigates into it; clicking a file selects it
        (and, for load, applies immediately).  The path box stays editable
        for direct entry.
        """
        directory = os.path.abspath(directory or self._last_dir)
        self._last_dir = directory
        is_load = mode == 'load'
        self._open_dialog('file')
        x0, y0, w, h = self._dialog_panel(6.0, 6.4)
        title = ('Load a volume (.npy, .npz, .h5, .hdf5)' if is_load
                 else 'Save volume and data dict to HDF5')
        self._dialog_text('title', (x0 + 0.02 * w, y0 + h - 0.04), title,
                          fontweight='bold')
        if filename is None:
            filename = '' if is_load else 'volume.h5'
        self._dialog_textbox('path', 'Path ',
                             (x0 + 0.09 * w, y0 + h - 0.105, 0.88 * w, 0.045),
                             os.path.join(directory, filename))
        self._dialog_text('error', (x0 + 0.02 * w, y0 + 0.115), '',
                          color='red')

        state = self._dialog['state']
        state.update(volume_index=i, mode=mode, directory=directory)
        try:
            names = sorted(os.listdir(directory))
        except OSError as e:
            names = []
            self._dialog['texts']['error'].set_text(str(e))
        entries = []
        parent = os.path.dirname(directory)
        if parent != directory:
            entries.append(('[..]', parent, True))
        visible = [n for n in names if not n.startswith('.')]
        for name in visible:
            full = os.path.join(directory, name)
            if os.path.isdir(full):
                entries.append((name + os.sep, full, True))
        for name in visible:
            full = os.path.join(directory, name)
            if os.path.isfile(full) and \
                    name.lower().endswith(FILE_EXTENSIONS):
                entries.append((name, full, False))
        # Navigating up re-selects the directory we came from: land on the
        # page that contains it rather than restarting at page one.
        if select_name is not None:
            for index, (label, _full, _is_dir) in enumerate(entries):
                if label.rstrip(os.sep) == select_name:
                    page = index // FILE_DIALOG_ROWS
                    break
        n_pages = max(1, -(-len(entries) // FILE_DIALOG_ROWS))
        page = int(np.clip(page, 0, n_pages - 1))
        state['page'] = page
        shown = entries[page * FILE_DIALOG_ROWS:
                        (page + 1) * FILE_DIALOG_ROWS]
        state['entries_shown'] = shown

        list_top = y0 + h - 0.135
        row_h = 0.042
        for j, (label, full, is_dir) in enumerate(shown):
            going_up = os.path.basename(directory) if label == '[..]' else None
            button = self._dialog_button(
                label, (x0 + 0.04 * w, list_top - (j + 1) * (row_h + 0.008),
                        0.92 * w, row_h),
                lambda full=full, is_dir=is_dir, going_up=going_up:
                    self._file_dialog_entry(full, is_dir, going_up),
                key=f'entry{j}')
            button.label.set_fontsize(DIALOG_BODY_FONT_SIZE)
        if n_pages > 1:
            self._dialog_text('page', (x0 + 0.04 * w, y0 + 0.075),
                              f'page {page + 1} of {n_pages}')
            self._dialog_button('Prev', (x0 + 0.24 * w, y0 + 0.06,
                                         0.13 * w, 0.045),
                                lambda: self._file_dialog_page(-1))
            self._dialog_button('Next', (x0 + 0.40 * w, y0 + 0.06,
                                         0.13 * w, 0.045),
                                lambda: self._file_dialog_page(+1))
        accept_label = 'Load' if is_load else 'Save'
        accept = (self._load_dialog_accept if is_load
                  else self._save_dialog_accept)
        self._dialog_button(accept_label,
                            (x0 + 0.58 * w, y0 + 0.015, 0.18 * w, 0.045),
                            lambda i=i: accept(i))
        self._dialog_button('Cancel',
                            (x0 + 0.79 * w, y0 + 0.015, 0.17 * w, 0.045),
                            self._close_dialog)
        self.fig.canvas.draw_idle()

    def _file_dialog_entry(self, full_path, is_dir, select_name=None):
        state = self._dialog['state']
        if is_dir:
            filename = None
            if state['mode'] == 'save':
                filename = os.path.basename(
                    self._dialog['widgets']['path'].text) or 'volume.h5'
            self._open_file_dialog(state['volume_index'], state['mode'],
                                   directory=full_path, filename=filename,
                                   select_name=select_name)
            return
        self._dialog['widgets']['path'].set_val(full_path)
        if state['mode'] == 'load':
            self._load_dialog_accept(state['volume_index'])
        else:
            self.fig.canvas.draw_idle()

    def _file_dialog_page(self, step):
        state = self._dialog['state']
        filename = None
        if state['mode'] == 'save':
            filename = os.path.basename(
                self._dialog['widgets']['path'].text) or 'volume.h5'
        self._open_file_dialog(state['volume_index'], state['mode'],
                               directory=state['directory'],
                               page=state['page'] + step, filename=filename)

    def _dialog_error(self, message):
        self._dialog['texts']['error'].set_text(message)
        self.fig.canvas.draw_idle()

    def _on_load_button(self, i):
        self._file_action(i, 'load')

    def _load_dialog_accept(self, i):
        path = os.path.expanduser(self._dialog['widgets']['path'].text.strip())
        if os.path.isdir(path):
            self._open_file_dialog(i, 'load', directory=path)
            return
        if not os.path.isfile(path):
            self._dialog_error(f"File not found: {path or '(empty)'}")
            return
        self._last_dir = os.path.dirname(path) or self._last_dir
        self._start_load(i, path)

    def _file_error(self, message):
        """Route an error to the open dialog, or to a figure overlay."""
        if self._dialog is not None:
            self._dialog_error(message)
        else:
            self._show_message(True, message=f"{message}. Press Esc to exit.")

    def _start_load(self, i, path):
        """Load ``path`` into volume ``i``, via the array chooser if needed."""
        try:
            listing = VolumeStack.list_file_arrays(path)
        except Exception as e:
            self._file_error(f"Failed to load file: {e}")
            return
        if listing is None or len(listing[0]) == 1:
            self._close_dialog(draw=False)
            self._finish_load(i, path, None)
            return
        names, shapes = listing
        self._close_dialog(draw=False)
        self._open_choice_dialog(
            f"Arrays in {os.path.basename(path)}", names,
            lambda name, i=i, path=path: self._finish_load(i, path, name),
            sublabels=[f"shape={tuple(s)}" for s in shapes])

    def _finish_load(self, i, path, name):
        try:
            array, data_dict = VolumeStack.read_file_array(path, name)
            replaced = self.stack.load_array(i, array, data_dict)
        except Exception as e:
            self._show_message(
                True, message=f"Failed to load file: {e}. Press Esc to exit.")
            return
        for j in replaced:
            self._reset_view(j)
        self.refresh()
        self._update_slice_slider()
        self._reset_navigation()
        self.fig.canvas.draw_idle()

    def _on_save_button(self, i):
        self._file_action(i, 'save')

    def _save_dialog_accept(self, i):
        path = os.path.expanduser(self._dialog['widgets']['path'].text.strip())
        if not path:
            self._dialog_error('Enter a file path')
            return
        if os.path.isdir(path):
            self._dialog_error('Path is a directory; add a file name')
            return
        self._finish_save(i, path)

    def _finish_save(self, i, path):
        if not path.lower().endswith('.h5'):
            path += '.h5'
        # Under TkAgg, offer the data dict for editing before it is written
        # (the reference's easygui flow, as one editor window).  Elsewhere
        # the dict is saved as-is.
        if self._in_process_tk_ok():
            try:
                result = _tk_dict_editor(self.stack.data_dicts[i])
            except Exception:
                result = ('as-is',)
            if result is None:
                return  # the user cancelled the save
            if result[0] == 'save':
                self.stack.data_dicts[i] = result[1] or None
        try:
            self.save_fn(path, self.stack.original_data[i], 'volume',
                         self.stack.data_dicts[i])
        except Exception as e:
            self._file_error(f"Failed to save: {e}")
            return
        self._last_dir = os.path.dirname(path) or self._last_dir
        self._close_dialog(draw=False)
        self._show_message(True,
                           message=f"Saved to {path}. Press Esc to dismiss.")

    # ------------------------------------------------------------------
    # Partial redraws (blitting)
    # ------------------------------------------------------------------
    # A partial redraw repaints only the affected panels or slider rows:
    # paint an opaque rectangle over the region, redraw those axes into the
    # canvas buffer, and blit the region.  No artist is marked animated, so
    # full draws and savefig need no special casing, and backends without
    # blit support (e.g. macosx) simply fall back to draw_idle.

    def _partial_redraw(self, volume_indices=None, widgets=()):
        canvas = self.fig.canvas
        overlay_active = (self._dialog is not None
                          or self._message_artist is not None)
        if (not self.enable_blit or not self._renderer_ready
                or overlay_active
                or matplotlib.get_backend().lower() not in BLIT_BACKENDS
                or not getattr(canvas, 'supports_blit', False)):
            canvas.draw_idle()
            return
        renderer = canvas.get_renderer()
        # Clip to the renderer buffer, not the logical canvas size: on HiDPI
        # displays the two differ by the device pixel ratio.
        buffer_w = getattr(renderer, 'width', 0) or canvas.get_width_height()[0]
        buffer_h = getattr(renderer, 'height', 0) or canvas.get_width_height()[1]
        canvas_box = Bbox([[0, 0], [buffer_w, buffer_h]])
        groups = []
        indices = (range(self.stack.n_volumes) if volume_indices is None
                   else volume_indices)
        for i in indices:
            groups.append((('volume', i), [self.axes[i], self.caxes[i]]))
        widget_axes = {'slice': self._slice_slider_ax,
                       'intensity': self.intensity_slider.ax}
        for name in widgets:
            ax = widget_axes[name]
            if ax.get_visible():
                groups.append((('widget', name), [ax]))
        for key, group in groups:
            boxes = []
            for ax in group:
                try:
                    boxes.append(ax.get_tightbbox(renderer))
                except Exception:
                    boxes.append(ax.bbox)
            current = Bbox.union(boxes).padded(6)
            # Clear the union with the group's previous region so text that
            # shrank or moved since the last redraw leaves no ghost.
            previous = self._last_blit_regions.get(key)
            region = (current if previous is None
                      else Bbox.union([current, previous]))
            self._last_blit_regions[key] = current
            region = Bbox.intersection(region, canvas_box)
            if region is None:
                continue
            self._clear_rect.set_bounds(region.x0, region.y0,
                                        region.width, region.height)
            self._clear_rect.draw(renderer)
            for ax in group:
                ax.draw(renderer)
            canvas.blit(region)
        try:
            canvas.flush_events()
        except NotImplementedError:
            pass

    # ------------------------------------------------------------------
    # Showing
    # ------------------------------------------------------------------

    def show(self, block=True):
        """Display the viewer window.

        With ``block=True`` this runs the GUI event loop until every open
        matplotlib window is closed, then closes this figure by object so
        its widgets are torn down deterministically.  With ``block=False``
        the window is drawn and left open; it becomes fully interactive when
        a later blocking show (or ``plt.show()``) runs the event loop, and
        the caller must keep the viewer alive (see ``slice_viewer``).

        On a non-interactive backend (e.g. Agg) this warns and returns,
        leaving the figure available for ``savefig``.
        """
        if matplotlib.get_backend().lower() in NONINTERACTIVE_BACKENDS:
            warnings.warn(
                'slice_viewer: matplotlib backend '
                f'{matplotlib.get_backend()!r} is non-interactive; the '
                'viewer window cannot be shown.')
            return
        if block:
            plt.show()
            plt.close(self.fig)
        else:
            self.fig.show()
            self.fig.canvas.draw_idle()


# Keeps block=False viewers (and their toolkit widgets) alive for callers
# who drop the returned viewer; the next blocking call adopts and closes them.
_NONBLOCKING_VIEWERS = []


def slice_viewer(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
                 slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, block=True, save_fn=None):
    """Launch an interactive viewer for one or more 2D or 3D image arrays.

    This function builds a :class:`SliceViewer`, shows it, and returns it.
    Features include synchronized slice navigation with proportional mapping
    across volumes of unequal depth, ROI statistics, difference images, axis
    transposition, file load/save, dynamic intensity range adjustment, and a
    right-click context menu of per-image actions.

    Args:
        *datasets (ndarray or None): One or more 2D or 3D arrays to display.
            2D arrays are promoted to 3D via a trailing singleton axis, and
            None values are replaced with placeholder zero arrays.
        data_dicts (None, dict, or list of dict/None, optional): String-valued
            dict(s) associated with the volumes, viewable in the viewer.
        title (str, optional): Figure title.  Defaults to ''.
        vmin (float, optional): Minimum display intensity.  Defaults to the
            global minimum across all datasets.
        vmax (float, optional): Maximum display intensity.  Defaults to the
            global maximum across all datasets.
        slice_label (str or list of str, optional): Label(s) shown before the
            slice number in each panel title.  Defaults to "Slice".
        slice_axis (int or list of int, optional): Axis along which to slice
            (0, 1, or 2).  Defaults to the last axis (2).
        cmap (str, optional): Colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): Show the "Press h for help" hint.
            Defaults to True.
        block (bool, optional): If True (default), block until the window is
            closed.  If False, leave the window open and return immediately;
            the window becomes fully interactive when the next blocking
            slice_viewer runs, and that blocking call returns when ALL open
            windows are closed.
        save_fn (callable, optional): Replacement for the built-in HDF5
            writer, called as ``save_fn(file_path, array, array_name,
            attributes_dict)``.

    Returns:
        SliceViewer: the viewer object.  Nonblocking callers may keep it to
        interact programmatically; a module-level registry also keeps it
        alive if the return value is dropped.
    """
    viewer = SliceViewer(*datasets, data_dicts=data_dicts, title=title,
                         vmin=vmin, vmax=vmax, slice_label=slice_label,
                         slice_axis=slice_axis, cmap=cmap,
                         show_instructions=show_instructions, save_fn=save_fn)
    viewer.show(block=block)
    if not block:
        _NONBLOCKING_VIEWERS.append(viewer)
        return viewer
    # The blocking show returned, so every open window has been closed.
    # Close any earlier nonblocking viewers and, under TkAgg, collect now on
    # the main thread: matplotlib's TkAgg backend leaves orphaned tkinter
    # objects that a later background-thread GC would finalize with no Tk
    # mainloop running ("main thread is not in main loop").
    for nonblocking_viewer in _NONBLOCKING_VIEWERS:
        plt.close(nonblocking_viewer.fig)
    _NONBLOCKING_VIEWERS.clear()
    if matplotlib.get_backend() == 'TkAgg':
        import gc
        gc.collect()
    return viewer
