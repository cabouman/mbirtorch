"""mbirtorch-side wrapper for the slice viewer.

The viewer module itself (mbirtorch/viewer.py) is package-independent: it
imports only numpy, matplotlib, and (lazily) h5py, so the identical file can
later serve mbirjax.  This wrapper supplies the mbirtorch-specific
conversions on the way in: torch tensors (including CUDA and MPS tensors)
become numpy arrays, and rich data dicts -- e.g. the recon_dict returned by
:meth:`TomographyModel.recon` -- are serialized to dicts of display strings.
"""

import pprint

import numpy as np

from .viewer import SliceViewer, VolumeStack
from .viewer import slice_viewer as _slice_viewer

__all__ = ['SliceViewer', 'VolumeStack', 'convert_subdicts_to_strings',
           'slice_viewer']


def _to_numpy(dataset):
    """Convert one dataset to a numpy array; None passes through."""
    if dataset is None:
        return None
    if hasattr(dataset, 'detach'):
        # A torch tensor, possibly on a CUDA or MPS device; np.asarray alone
        # cannot convert device tensors, and this avoids importing torch.
        return dataset.detach().cpu().numpy()
    return np.asarray(dataset)


def convert_subdicts_to_strings(data_dict):
    """Serialize the entries of a data dict to strings for display.

    Dict-valued entries (e.g. 'recon_params' and 'model_params' from
    :meth:`TomographyModel.recon`) are rendered as readable multi-line
    strings; other non-string values are converted with str.  Non-dict
    inputs are returned unchanged.
    """
    if not isinstance(data_dict, dict):
        return data_dict
    string_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            string_dict[key] = pprint.pformat(value, width=100,
                                              sort_dicts=False)
        elif isinstance(value, str):
            string_dict[key] = value
        else:
            string_dict[key] = str(value)
    return string_dict


def slice_viewer(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
                 slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, block=True, save_fn=None):
    """
    Launch an interactive viewer for inspecting one or more 2D or 3D arrays.

    This function provides a graphical interface for exploring one or more 3D
    volumes or 2D slices.  Features include synchronized slice navigation
    with proportional mapping across volumes of unequal depth, ROI statistics,
    difference images, axis transposition, file loading and saving, dynamic
    intensity range adjustment, and a right-click context menu of per-image
    actions.

    Each image can have an associated data dict, typically the recon_dict
    from :meth:`TomographyModel.recon <mbirtorch.TomographyModel.recon>`,
    which can be viewed as text within the viewer.

    Designed primarily for inspecting CT or other volumetric reconstructions
    in research workflows.

    Args:
        *datasets (ndarray, tensor, or None): One or more 2D or 3D arrays to
            display.  Torch tensors (including CUDA and MPS tensors) are
            converted to numpy automatically.
            - 2D arrays are promoted to 3D via a trailing singleton axis.
            - None values are replaced with placeholder zero arrays.

        data_dicts (None or dict or list of None or dicts, optional):
            Dictionary of entries associated with the data (e.g., the
            recon_dict from :meth:`TomographyModel.recon <mbirtorch.TomographyModel.recon>`).
            Nested dicts are serialized to display strings automatically.
        title (str, optional): Figure title.  Defaults to an empty string.
        vmin (float, optional): Minimum intensity value for display.
            Defaults to the global minimum across all datasets.
        vmax (float, optional): Maximum intensity value for display.
            Defaults to the global maximum across all datasets.
        slice_label (str or list of str, optional): Label(s) for the current
            slice.  Defaults to "Slice".
        slice_axis (int or list of int, optional): Axis along which to slice
            (0, 1, or 2).  Defaults to the last axis (2).
        cmap (str, optional): Colormap to use.  Defaults to "gray".
        show_instructions (bool, optional): Whether to display the help hint
            in the figure.  Defaults to True.
        block (bool, optional): If True (default), block until the window is
            closed.  If False, leave the window open and return immediately
            -- useful for showing, e.g., a sinogram next to a reconstruction.
            A nonblocking window becomes fully interactive when the next
            blocking slice_viewer runs, and that blocking call returns when
            ALL open windows are closed.
        save_fn (callable, optional): Replacement for the built-in HDF5
            writer used by the viewer's Save action, called as
            ``save_fn(file_path, array, array_name, attributes_dict)``.

    Returns:
        SliceViewer: the viewer object.  Nonblocking callers may keep it to
        interact programmatically; a module-level registry also keeps the
        window alive if the return value is dropped.

    Notes:
        - Right-click an image for a menu with options such as difference
          images, axis transposition, and file load/save.
        - Right-click the intensity slider (or press Set range) to enter
          exact display bounds.
        - Press 'h' for a help overlay.  Press Esc to clear overlays or
          remove the ROI.

    Example:
        >>> import mbirtorch
        >>> recon, recon_dict = ct_model.recon(sinogram, weights=weights)
        >>> mbirtorch.slice_viewer(phantom, recon, data_dicts=[None, recon_dict],
        ...                        title='ground truth phantom and recon')
    """
    datasets = [_to_numpy(dataset) for dataset in datasets]
    if isinstance(data_dicts, dict):
        data_dicts = convert_subdicts_to_strings(data_dicts)
    elif isinstance(data_dicts, (list, tuple)):
        data_dicts = [convert_subdicts_to_strings(d) for d in data_dicts]
    return _slice_viewer(*datasets, data_dicts=data_dicts, title=title,
                         vmin=vmin, vmax=vmax, slice_label=slice_label,
                         slice_axis=slice_axis, cmap=cmap,
                         show_instructions=show_instructions, block=block,
                         save_fn=save_fn)
