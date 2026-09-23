"""4D reconstruction of one continuous scan of a moving object.

The scan is divided into overlapping angular windows, one per time frame,
and one volume is reconstructed per frame.  The volumes are held in one
array of shape ``(frames, x, y, z)``.  :class:`MACE4DModel` computes them as
the consensus of four agents.  The first agent is the proximal map of each
frame's data-fit term.  The other three are qGGMRF denoisers, one for each
pair of spatial axes, on the volumes that hold the frame axis and that pair.
:func:`temporal_filter_matrix` builds the frame-axis filter that removes the
modulation the overlapping windows produce, and :func:`apply_temporal_filter`
applies it.
"""

import concurrent.futures
import csv
import datetime
import functools
import os
import threading
import time
import warnings

import numpy as np
import torch

from .denoising import QGGMRFDenoiser
from .mace import MACE, ForwardProxAgent, HyperplaneAgent, Task, resolve_device_pool
from .parameter_handler import ParameterHandler
from .utilities import construct_time_frame_models

# Iterations of the per-frame reconstruction that initializes the 4D image.
_INIT_ITERATIONS = 15
# Iterations and stop threshold of each denoiser sweep.  The threshold is tighter
# than the 0.2 percent a standalone denoise uses.
_DENOISE_MAX_ITERATIONS = 15
_DENOISE_STOP_THRESHOLD_PCT = 0.05
# The filter is applied to the data-fit outputs in slabs of about this size.
_FILTER_SLAB_BYTES = 64 * 2 ** 20

# Each entry is the name of a hyperplane orientation and the spatial axis it
# fixes, numbered in the (frames, x, y, z) order of the 4D array.
_ORIENTATIONS = [('XY-t', 3), ('YZ-t', 1), ('XZ-t', 2)]

_TIMING_FIELDS = ['iteration', 'prox_total_sec', 'denoise_total_sec', 'makespan_sec',
                  'iteration_total_sec', 'consensus_change_pct', 'denoise_mean_iterations']
_TASK_FIELDS = ['iteration', 'kind', 'index', 'part', 'worker', 'start_sec', 'end_sec']


def temporal_filter_matrix(num_frames, period, harmonics=True, band_width=1):
    """
    The matrix of the frame-axis filter that removes a periodic modulation.

    A scan divided into overlapping angular windows produces a modulation
    along the frame axis with a period equal to the number of frames per
    rotation.  The filter zeroes the type-I discrete cosine modes at that
    period and its harmonics, with ``band_width`` modes on each side.  It
    acts along the frame axis alone, so it is one matrix of shape
    ``(num_frames, num_frames)``.  Apply it with :func:`apply_temporal_filter`.

    At a period of 6 and three frames the matrix is zero, so the filtered
    array is zero.  The number of frames must be several times the period
    before the filter leaves much variation along the frame axis.

    Args:
        num_frames (int): number of frames, at least 2.
        period (float): the main period of the modulation, in frames.
        harmonics (bool or list of int, optional): True removes the main
            period and every harmonic with a period of at least two frames;
            False removes the main period only; a list names the harmonic
            indices to remove.  Defaults to True.
        band_width (int, optional): modes zeroed on each side of a removed
            mode.  Defaults to 1.

    Returns:
        torch.Tensor: the float32 matrix, on the CPU.
    """
    from scipy.fft import dct, idct

    num_frames = int(num_frames)
    if num_frames < 2:
        raise ValueError(f'the frame axis needs at least 2 frames to filter; got {num_frames}.')
    if harmonics is False:
        harmonic_list = [1]
    elif harmonics is True:
        harmonic_list = list(range(1, int(np.floor(period / 2)) + 1))
    else:
        harmonic_list = list(harmonics)

    # The filter is linear along the frame axis, so filtering the identity
    # matrix column by column gives the matrix of the filter.
    coefficients = dct(np.eye(num_frames, dtype=np.float32), type=1, norm='ortho', axis=0)
    for harmonic in harmonic_list:
        removed_period = period / harmonic
        center = int(round(2 * (num_frames - 1) / removed_period))
        low = max(0, center - band_width)
        high = min(num_frames, center + band_width + 1)
        if low < high:
            coefficients[low:high, :] = 0
    matrix = idct(coefficients, type=1, norm='ortho', axis=0).astype(np.float32, copy=False)
    return torch.as_tensor(np.ascontiguousarray(matrix))


def apply_temporal_filter(x, matrix, axis=0):
    """
    Apply a frame-axis filter matrix along one axis of a tensor.

    Args:
        x (torch.Tensor): the array to filter.
        matrix (torch.Tensor): a square matrix whose size is the length of
            ``axis``, from :func:`temporal_filter_matrix`.  It is moved to the
            device and dtype of ``x``.
        axis (int, optional): the axis the matrix acts along.  Defaults to 0.

    Returns:
        torch.Tensor: a new tensor of the shape of ``x``.
    """
    matrix = matrix.to(device=x.device, dtype=x.dtype)
    moved = x.movedim(axis, 0)
    filtered = torch.tensordot(matrix, moved, dims=1)
    return filtered.movedim(0, axis)


def _normalize_prior_weights(prior_weight):
    """Return the four agent weights ``[forward, xyt, yzt, xzt]``.  A scalar
    prior weight ``w`` gives ``[1 - w, w/3, w/3, w/3]``.  A list of three
    gives ``[1 - sum, w1, w2, w3]``."""
    if isinstance(prior_weight, (list, tuple, np.ndarray)):
        prior = [float(w) for w in prior_weight]
        if len(prior) != 3:
            raise ValueError('mace_prior_weight as a list must have 3 entries [xyt, yzt, xzt].')
    else:
        w = float(prior_weight) / 3.0
        prior = [w, w, w]
    if any(w < 0 for w in prior) or sum(prior) > 1.0:
        raise ValueError('mace_prior_weight must be nonnegative and sum to at most 1.')
    return [1.0 - sum(prior)] + prior


def _write_run_info(path, run_settings):
    """Write each run setting to ``path`` as one ``key = value`` line."""
    width = max(len(key) for key in run_settings)
    lines = ['# MACE4DModel run settings']
    lines += ['{:<{width}} = {}'.format(key, value, width=width)
              for key, value in run_settings.items()]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _describe_filter(matrix, num_frames, period):
    """Return one line saying which periods the filter removes and how many
    modes of the frame axis it removes and keeps.  The filter is a
    projection, so the number of modes it keeps is its trace."""
    harmonics = list(range(1, int(np.floor(period / 2)) + 1))
    periods = [period / h for h in harmonics]
    kept = int(round(float(matrix.trace())))
    removed = num_frames - kept
    words = ', '.join(f'{p:g}' for p in periods)
    return (f'removes periods of {words} frames; {removed} of {num_frames} modes removed, '
            f'{kept} kept')


def _permutation(axis):
    """Return the axis order that puts ``axis`` first and the frame axis
    second."""
    return (axis,) + tuple(d for d in range(4) if d != axis)


def _slab_batch_size(num_planes, volume_shape, slab_gb):
    """Return the number of hyperplanes per denoiser task.

    A task sweeps one slab, which is a stack of hyperplane volumes, on one
    device.  It holds a few arrays of the slab's size while it does so.  The
    slab is sized by the byte budget ``slab_gb``, in units of 2**30 bytes.
    The budget is fixed rather than read from the device, because the sweep
    saturates a GPU well below this size and smaller slabs spread more evenly
    over the pool.

    The count is the largest that fits the budget, at most ``num_planes``.
    The planes are then divided into equal slabs so that the padded last slab
    is nearly full.

    Args:
        num_planes (int): the hyperplanes of the orientation.
        volume_shape (tuple of int): the shape of one hyperplane volume.
        slab_gb (float): the budget, in units of ``2 ** 30`` bytes.

    Returns:
        int: the hyperplanes per slab, between 1 and ``num_planes``.
    """
    num_planes = int(num_planes)
    volume_bytes = 4 * int(np.prod(volume_shape))
    fit = max(1, int(float(slab_gb) * 2 ** 30) // volume_bytes)
    num_slabs = -(-num_planes // min(num_planes, fit))
    return -(-num_planes // num_slabs)


def _centered_block(shape, num_frames, point_budget):
    """Return the slices of a block centered in each axis of ``shape``.  The
    block holds at most ``point_budget`` entries over ``num_frames``
    frames."""
    total = int(num_frames) * int(np.prod(shape))
    scale = 1.0 if total <= point_budget else (point_budget / total) ** (1.0 / len(shape))
    block = []
    for extent in shape:
        width = int(min(int(extent), max(1, round(int(extent) * scale))))
        start = (int(extent) - width) // 2
        block.append(slice(start, start + width))
    return tuple(block)


def _rms_neighbor_difference(values, support, axes):
    """Return the root mean square of the difference between neighboring
    entries along ``axes``, over the entries that ``support`` holds.  Returns
    None when the support holds no such pair."""
    total, count = 0.0, 0
    for axis in axes:
        if values.shape[axis] < 2:
            continue
        head, tail = [slice(None)] * values.ndim, [slice(None)] * values.ndim
        head[axis], tail[axis] = slice(1, None), slice(None, -1)
        mask = support[tuple(head)]
        difference = (values[tuple(head)] - values[tuple(tail)])[mask]
        total += float(np.sum(np.square(difference, dtype=np.float64)))
        count += int(difference.size)
    return None if count == 0 else float(np.sqrt(total / count))


def _spread_along_time(image, chosen, sigma_noise, point_budget=5_000_000):
    """Return the spread of the image along the frame axis against its spread
    along the three spatial axes.

    Each spread is the root mean square of a neighbor difference over the
    voxels that hold the object.  Their ratio says how much a voxel changes
    between frames compared with how much it changes between adjacent voxels.
    That is the quantity ``nbr_weight_time`` balances.

    The ratio is read from the frames in ``chosen`` over a block centered in
    the three spatial axes.  Those frames are evenly spaced, so when they lie
    a stride apart the temporal difference is taken across that stride rather
    than between adjacent frames.

    Args:
        image (tensor): the 4D image, of shape ``(frames, x, y, z)``.
        chosen (numpy): the indices of the frames to read.
        sigma_noise (float): the noise level, which sets the threshold that
            separates the object from the background.
        point_budget (int, optional): the most voxels to read per frame set.

    Returns:
        dict: the ratio, the two spreads, the number of frames read, the
        stride between them, and the shape of the block.  The ratio is None
        when fewer than two frames are chosen or the object is not found.
    """
    chosen = np.asarray(chosen, dtype=int).ravel()
    stride = int(chosen[1] - chosen[0]) if chosen.size > 1 else 1
    block = _centered_block(tuple(int(n) for n in image.shape[1:]), chosen.size, point_budget)
    frames = np.stack([np.asarray(image[int(t)][block], dtype=np.float32) for t in chosen])
    threshold = 0.05 * float(np.mean(np.abs(frames))) + float(sigma_noise)
    threshold = min(threshold, float(np.amax(frames)))
    support = frames >= threshold
    temporal = _rms_neighbor_difference(frames, support, (0,))
    spatial = _rms_neighbor_difference(frames, support, (1, 2, 3))
    ratio = None if temporal is None or not spatial else temporal / spatial
    return {'ratio': ratio, 'temporal': temporal, 'spatial': spatial,
            'frames read': int(chosen.size), 'stride': stride,
            'block': tuple(int(n) for n in frames.shape[1:])}


class _DataFitAgent:
    """The proximal maps of every frame's data-fit term, as one agent of the
    4D array.

    One :class:`ForwardProxAgent` per frame runs on the frame's device, and
    each frame starts from its own previous output.  When the filter is on,
    the agent's contribution to the consensus is the filter applied along the
    frame axis to the stack of frame outputs.
    """

    def __init__(self, frame_agents, frame_devices, init_stack, filter_matrix, keep_stack,
                 adopt_stack=False):
        self.agents = list(frame_agents)
        self.devices = list(frame_devices)
        self.filter_matrix = filter_matrix
        self.fold_after_all = filter_matrix is not None
        if not keep_stack:
            self._stack = None
        elif adopt_stack:
            # The array is written in place from here on, so it is taken over
            # only when no one else holds it.
            self._stack = init_stack
        else:
            self._stack = init_stack.clone()

    def tasks(self, w, iteration=0):
        tasks = []
        for t in range(len(self.agents)):
            region = (slice(t, t + 1), slice(None), slice(None), slice(None))
            tasks.append(Task(functools.partial(self._run_frame, w, t, iteration),
                              device=self.devices[t], region=region))
        return tasks

    def _run_frame(self, w, t, iteration, device):
        agent = self.agents[t]
        if self._stack is not None and agent.use_warm_start:
            agent.load_state_dict({'previous_output': self._stack[t]})
        output = agent(w[t], iteration)
        if self._stack is not None:
            self._stack[t].copy_(output)
            # The host stack holds the warm start, so the frame agent's
            # device copy is released.
            agent._previous_output = None
        if self.fold_after_all:
            return None
        return output.unsqueeze(0)

    def pieces(self):
        stack = self._stack
        num_slices = int(stack.shape[-1])
        slice_bytes = stack[..., :1].numel() * stack.element_size()
        slab = max(1, _FILTER_SLAB_BYTES // slice_bytes)
        for z0 in range(0, num_slices, slab):
            z1 = min(z0 + slab, num_slices)
            region = (slice(None), slice(None), slice(None), slice(z0, z1))
            yield region, apply_temporal_filter(stack[..., z0:z1], self.filter_matrix, axis=0)

    def state_dict(self):
        return {'stack': None if self._stack is None else self._stack.clone(),
                'agents': [agent.state_dict() for agent in self.agents]}

    def load_state_dict(self, state):
        if state.get('stack') is not None and self._stack is not None:
            self._stack.copy_(state['stack'])
        for agent, saved in zip(self.agents, state.get('agents', [])):
            if saved is not None:
                agent.load_state_dict(saved)


class MACE4DModel(ParameterHandler):
    """
    Space-time reconstruction of one continuous CT scan.

    The views are taken to be recorded in time order.  The scan is divided
    into overlapping time frames, each a window of consecutive views, and
    :meth:`recon` returns one volume per frame.  The constructor fixes the
    frames; the reconstruction parameters are set with :meth:`set_params`.
    The sinogram is passed to :meth:`recon`.  By default the reconstruction
    applies a filter along the frame axis, which removes the periodic
    modulation that the overlapping frames produce; when the frames are too
    few for the filter it is turned off with a warning.

    Args:
        ct_model (TomographyModel): the model of the full scan, a
            ConeBeamModel or a ParallelBeamModel with one angle per view.
        frames_per_rotation (int, optional): frames per full rotation.  This
            is also the period of the temporal filter.  Defaults to 6.
        frame_overlap_factor (float, optional): the number of frames that
            cover any one view.  Each frame spans
            ``frame_overlap_factor * 360 / frames_per_rotation`` degrees.
            Defaults to 2.0.
        num_frames (int, optional): reconstruct only this many frames,
            counted from the start of the scan.  Defaults to None, every
            frame.

    Example:
        >>> mace = mbirtorch.MACE4DModel(ct_model, frames_per_rotation=6)
        >>> mace.set_params(mace_prior_weight=0.5, rho_mann=0.5)
        >>> weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')
        >>> recon_4d, recon_dict = mace.recon(sinogram, weights=weights, max_iterations=10)
    """

    def __init__(self, ct_model, frames_per_rotation=6, frame_overlap_factor=2.0, num_frames=None):
        super().__init__()
        self.ct_model = ct_model
        self.frames_per_rotation = frames_per_rotation
        self.frame_overlap_factor = frame_overlap_factor
        self.sinogram_shape = tuple(int(n) for n in ct_model.get_params('sinogram_shape'))

        if num_frames is not None and num_frames < 1:
            raise ValueError(f'num_frames must be at least 1; got {num_frames}.')
        self.model_list, self.view_slices = construct_time_frame_models(
            ct_model, frames_per_rotation=frames_per_rotation,
            frame_overlap_factor=frame_overlap_factor)
        if num_frames is not None and num_frames < len(self.model_list):
            self.model_list = self.model_list[:num_frames]
            self.view_slices = self.view_slices[:num_frames]
        self.num_frames = len(self.model_list)
        self.recon_shape = tuple(int(n) for n in self.model_list[0].get_params('recon_shape'))

        self.set_params(no_warning=True, no_compile=True,
                        mace_prior_weight=0.5, rho_mann=0.5, prox_num_iterations=3,
                        prox_stop_threshold=0.02, prox_partition_advance=1.0,
                        prox_warm_start=True, denoiser_warm_start=False, sigma_prox=None,
                        dejitter=True, dejitter_verbose=0,
                        sigma_noise=None, nbr_weight_time=1.0, sharpness=0.0,
                        denoise_slab_gb=2.0)
        self._devices = None

    def refresh_device_bindings(self):
        """Do nothing.  This model holds no projectors, and each frame model
        refreshes its own."""

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Set reconstruction parameters by keyword.

        Parameters of the geometry and of the regularization belong to
        ``ct_model``; set them there before constructing this model.

        Args:
            no_warning (bool, optional): disable the parameter-name check and
                warnings.  Defaults to False.
            no_compile (bool, optional): kept for the base class interface.
            mace_prior_weight (float or list of 3 floats): the total weight of
                the three priors, or one weight each for XY-t, YZ-t, and
                XZ-t.  The data-fit agent gets the rest of 1.  Checked when
                set.  Defaults to 0.5.
            rho_mann (float): the step of the consensus iteration.  Defaults
                to 0.5.
            prox_num_iterations (int): iterations of each data-fit call.
                Defaults to 3.
            prox_stop_threshold (float): the stop threshold, in percent, of
                the per-frame reconstruction that initializes the run.
                Defaults to 0.02.
            prox_partition_advance (float): how many entries of the partition
                sequence each data-fit call moves forward per iteration.
                Defaults to 1.0.
            prox_warm_start (bool): start each data-fit call from the frame's
                previous output.  Defaults to True.
            denoiser_warm_start (bool): start each denoiser sweep from its
                previous output, at the cost of one full-size array per
                orientation.  Defaults to False.
            denoise_slab_gb (float): the size, in GB of ``2**30`` bytes, of
                the stack of hyperplane volumes one denoiser task sweeps.
                Each orientation is divided into slabs of at most this size,
                and a task holds about three arrays of the slab's size on its
                device.  Defaults to 2.0.
            sigma_prox (float or None): the strength of every data-fit call.
                None sets it from the data.  Defaults to None.
            dejitter (bool): apply the temporal filter along the frame axis.
                Defaults to True.
            dejitter_verbose (int): log the periods the filter removes and
                the modes it keeps.  Defaults to 0.
            sigma_noise (float or None): the noise level of all three
                denoisers.  None estimates it from the initial image.
                Defaults to None.
            sigma_x (float): the prior strength of all three denoisers.
                Setting it disables the automatic estimate, as it does for
                every model.  Defaults to the estimate.
            sharpness (float): scales the automatic ``sigma_x``.  It scales
                the denoisers alone; each frame's own prior takes its
                sharpness from ``ct_model``.  Defaults to 0.
            nbr_weight_time (float): the weight of a frame neighbor against a
                spatial neighbor in the denoisers' priors.  The frame
                direction appears in all three hyperplane volumes and each
                spatial direction in two of them, so the default of 1.0 gives
                a frame neighbor 1.5 times the weight of a spatial one, and
                2/3 makes the two equal.  Defaults to 1.0.
            verbose (int): 0 is silent, 1 reports progress.  Defaults to 1.

        Raises:
            ValueError: if ``qggmrf_nbr_wts`` is set, in which case use
                ``nbr_weight_time``, or if ``denoise_slab_gb`` is not positive.

        Example:
            >>> mace.set_params(mace_prior_weight=0.5, rho_mann=0.5, dejitter=True)
        """
        if 'mace_prior_weight' in kwargs:
            _normalize_prior_weights(kwargs['mace_prior_weight'])
        if 'denoise_slab_gb' in kwargs and not float(kwargs['denoise_slab_gb']) > 0:
            raise ValueError(f"denoise_slab_gb must be positive; got {kwargs['denoise_slab_gb']!r}.")
        if 'qggmrf_nbr_wts' in kwargs:
            raise ValueError(
                'qggmrf_nbr_wts cannot be set on MACE4DModel.  Its three entries name '
                'the row, column, and slice directions of one volume, but the three '
                'hyperplane volumes are (t, x, y), (t, y, z), and (t, x, z), so the '
                'column direction is x in one volume and y in another.  Only the frame '
                'direction means the same thing in all three volumes.  Set '
                'nbr_weight_time instead, which weights a frame neighbor against a '
                'spatial neighbor.')
        # This model runs no reconstruction of its own, so setting sigma_prox does
        # not disable an auto-regularization.
        sigma_prox_given = 'sigma_prox' in kwargs
        sigma_prox = kwargs.pop('sigma_prox', None)
        if sigma_prox_given:
            super().set_params(no_warning=True, no_compile=no_compile, sigma_prox=sigma_prox)
        if kwargs:
            super().set_params(no_warning=no_warning, no_compile=no_compile, **kwargs)

    def set_device_pool(self, devices=None):
        """
        Set the devices :meth:`recon` spreads its work over.

        Each frame's data-fit task runs on one device of the pool, assigned
        round robin, and any device takes the denoising tasks.  The pool
        takes effect at the next :meth:`recon` call.

        Args:
            devices (optional): None for every GPU or the CPU; ``'cpu'`` or
                ``'gpu'``; a count of devices; a list of device indices; or a
                list of devices, one worker per entry.  The CPU may be
                repeated; a repeated GPU is refused.  None and a count are
                capped by the count ``MBIRTORCH_NUM_DEVICES`` pins.  See
                :func:`mbirtorch.mace.resolve_device_pool`.

        Raises:
            ValueError: if the pool names a GPU more than once, or if a
                requested count exceeds the devices available.

        Example:
            >>> mace.set_device_pool(2)
        """
        self._devices = resolve_device_pool(devices)

    @property
    def devices(self):
        """The device pool.  When no pool has been set, this is the default pool."""
        return list(self._devices) if self._devices is not None else resolve_device_pool(None)

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=10,
              stop_threshold_change_pct=0.2, init_dir=None, log_dir=None):
        """
        Reconstruct one volume per time frame from the sinogram of the scan.

        Args:
            sinogram (numpy or tensor): the full sinogram, of shape
                ``(num_views, num_det_rows, num_det_channels)``.
            weights (numpy or tensor, optional): positive weights of the
                sinogram's shape.  Defaults to None, unit weights.
            init_recon (numpy, optional): the initial 4D image, of shape
                ``(num_frames,) + recon_shape``.  Defaults to None.  The
                image is then read from ``init_dir`` when one is there, and
                is otherwise computed by reconstructing each frame alone.
            max_iterations (int, optional): consensus iterations.  Defaults
                to 10.
            stop_threshold_change_pct (float, optional): stop when the percent
                change of the consensus image in one iteration falls below
                this.  0 runs every iteration.  Defaults to 0.2.
            init_dir (str, optional): directory of the cached initial image
                ``init_recon.npy``, read when present and written otherwise.
                Defaults to None, no cache.
            log_dir (str, optional): directory for ``run_info.txt``,
                ``timing_log.csv``, and ``task_log.csv``.  Defaults to None,
                no log files.

        Returns:
            (recon, recon_dict): the 4D reconstruction as a numpy array of
            shape ``(num_frames,) + recon_shape``, and a dict with the run
            settings under ``'recon_params'``, the per-iteration timing under
            ``'timing'``, a completion time under ``'notes'``, and a copy of
            the model's parameters under ``'model_params'``.

        Raises:
            ValueError: if ``sinogram``, ``weights``, or ``init_recon`` has
                the wrong shape, or if the initial image is constant so that
                no denoiser noise level can be estimated.

        Warns:
            UserWarning: if the temporal filter is on but the frames are too
                few for it, fewer than ``frames_per_rotation``, in which case
                the run proceeds with the filter off and the run settings
                say so.

        Example:
            >>> recon_4d, recon_dict = mace.recon(sinogram, weights=weights, log_dir='./logs')
        """
        num_frames = self.num_frames
        beta = _normalize_prior_weights(self.get_params('mace_prior_weight'))
        verbose = self.get_params('verbose')
        rho_mann = float(self.get_params('rho_mann'))
        dejitter = bool(self.get_params('dejitter'))
        prox_warm_start = bool(self.get_params('prox_warm_start'))
        denoiser_warm_start = bool(self.get_params('denoiser_warm_start'))

        sinogram = self._validate_sinogram(sinogram, 'sinogram')
        if weights is not None:
            weights = self._validate_sinogram(weights, 'weights')
        # An initial image the caller supplies stays the caller's.  One that
        # recon reads or computes belongs to recon and may be written.
        init_is_own = init_recon is None
        if init_recon is not None:
            init_recon = self._validate_init_recon(init_recon)
            init_source = 'provided by caller'

        # The filter is turned off when the frame count is below the period, or when
        # the matrix would remove every mode and zero the whole reconstruction.
        filter_matrix = None
        dejitter_note = None
        if dejitter:
            if num_frames < self.frames_per_rotation:
                dejitter_note = (f'turned off: {num_frames} frames are fewer than the filter period '
                                 f'of {self.frames_per_rotation}')
            else:
                filter_matrix = temporal_filter_matrix(num_frames, period=self.frames_per_rotation)
                if not bool(torch.any(filter_matrix != 0)):
                    filter_matrix = None
                    dejitter_note = (f'turned off: the filter removes every mode of {num_frames} '
                                     f'frames at a period of {self.frames_per_rotation}')
            if filter_matrix is None:
                dejitter = False
                warnings.warn(f'The temporal filter was {dejitter_note}.  Use more frames, or set '
                              'dejitter=False to silence this warning.')
            else:
                dejitter_note = _describe_filter(filter_matrix, num_frames, self.frames_per_rotation)
                if self.get_params('dejitter_verbose'):
                    self.logger.info(f'[MACE] Temporal filter: {dejitter_note}')

        pool = self.devices
        frame_devices = [pool[t % len(pool)] for t in range(num_frames)]
        if verbose:
            counts = [frame_devices.count(d) for d in pool]
            self.logger.info(f'[MACE] {len(pool)} worker(s) on {[str(d) for d in pool]}; '
                             f'prox frames per worker: {counts}.')
            self.logger.info(f'[MACE] Start 4D reconstruction with {num_frames} time frames.')

        frame_agents = [
            ForwardProxAgent(self.model_list[t], sinogram[self.view_slices[t]],
                             weights=None if weights is None else weights[self.view_slices[t]],
                             sigma_prox=self.get_params('sigma_prox'),
                             inner_iterations=self.get_params('prox_num_iterations'),
                             device=frame_devices[t],
                             partition_advance=self.get_params('prox_partition_advance'),
                             use_warm_start=prox_warm_start)
            for t in range(num_frames)]

        if init_recon is None:
            if init_dir is not None:
                init_recon = self._load_cached_init(init_dir)
            if init_recon is not None:
                init_source = f"cached ({os.path.join(init_dir, 'init_recon.npy')})"
            else:
                init_recon = self._compute_init_recon(frame_agents, pool, init_dir)
                init_source = (f'computed ({num_frames} frames, {_INIT_ITERATIONS} '
                               'iterations each)')
        x0 = torch.as_tensor(init_recon, dtype=torch.float32).contiguous()

        given_sigma = self.get_params('sigma_noise')
        if given_sigma is None:
            global_sigma = self._estimate_global_sigma(init_recon, pool[0])
            sigma_source = 'estimated from the initial image'
        else:
            global_sigma = float(given_sigma)
            sigma_source = 'set by sigma_noise'
        if not np.isfinite(global_sigma) or global_sigma <= 0:
            raise ValueError(
                f'The denoiser noise level {sigma_source} is {global_sigma}, which cannot be '
                'used.  A constant initial image gives this value.  Supply a non-constant '
                'init_recon, omit init_recon and let the model compute the per-frame '
                'initialization, or set sigma_noise to a positive value.')
        if verbose:
            self.logger.info(f'[MACE] Global denoiser sigma = {global_sigma:.6g} '
                             f'({sigma_source})')

        data_fit = _DataFitAgent(frame_agents, frame_devices, x0, filter_matrix,
                                 keep_stack=dejitter or prox_warm_start, adopt_stack=init_is_own)
        iteration_counts = []
        counts_lock = threading.Lock()
        priors = []
        batch_sizes = []
        # The denoisers receive filtered inputs when the filter is on, so
        # their strength is set from the filtered initial image.
        image_for_statistics = (x0 if filter_matrix is None
                                else apply_temporal_filter(x0, filter_matrix, axis=0))
        sigma_x, sigma_x_source, spread = self._denoiser_sigma_x(
            image_for_statistics, global_sigma, pool[0])
        for _, axis in _ORIENTATIONS:
            image_shape, params, batch_size = self._configure_orientation(
                axis, image_for_statistics, global_sigma, sigma_x)
            batch_sizes.append(batch_size)
            make = self._stack_denoiser_factory(image_shape, params, global_sigma, batch_size,
                                                iteration_counts, counts_lock)
            priors.append(HyperplaneAgent(axis, make, batch_size=batch_size,
                                          filter_matrix=filter_matrix,
                                          use_warm_start=denoiser_warm_start))
        del image_for_statistics
        if verbose:
            self.logger.info(f'[MACE] Denoiser sigma_x = {sigma_x:.6g} ({sigma_x_source}); '
                             f'batch sizes [xyt, yzt, xzt] = {batch_sizes} at '
                             f"{float(self.get_params('denoise_slab_gb')):g} GB per slab")

        run_settings = self._run_settings(pool, init_source, global_sigma, sigma_source,
                                          weights, max_iterations, stop_threshold_change_pct,
                                          sigma_x, sigma_x_source, spread, batch_sizes)
        run_settings['dejitter'] = dejitter
        if dejitter_note is not None:
            run_settings['temporal filter'] = dejitter_note

        timing_log_path = task_log_path = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            _write_run_info(os.path.join(log_dir, 'run_info.txt'), run_settings)
            timing_log_path = os.path.join(log_dir, 'timing_log.csv')
            with open(timing_log_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writeheader()
            task_log_path = os.path.join(log_dir, 'task_log.csv')
            with open(task_log_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=_TASK_FIELDS).writeheader()

        timing_rows = []
        loop = MACE([data_fit] + priors, x0, mu=beta, rho=rho_mann, devices=pool)
        # The loop holds its own copies, so the initial image is released
        # here.
        del x0, init_recon

        def after_step(iteration, x_bar):
            rows = loop.info['tasks'][-1]
            prox_total = sum(end - start for k, _, _, start, end in rows if k == 0)
            denoise_total = sum(end - start for k, _, _, start, end in rows if k > 0)
            makespan = max(end for _, _, _, _, end in rows)
            with counts_lock:
                counts = list(iteration_counts)
                iteration_counts.clear()
            mean_iterations = float(np.mean(counts)) if counts else float('nan')
            change_pct = loop.info['change_pct'][-1]
            timing_row = dict(zip(_TIMING_FIELDS,
                                  [iteration + 1, prox_total, denoise_total, makespan,
                                   loop.info['time'][-1], change_pct, mean_iterations]))
            timing_rows.append(timing_row)
            if timing_log_path is not None:
                with open(timing_log_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=_TIMING_FIELDS).writerow(timing_row)
                with open(task_log_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=_TASK_FIELDS)
                    for k, task_index, device_index, start, end in rows:
                        if k == 0:
                            kind, index, part = 'prox', task_index, ''
                        else:
                            kind, index, part = 'denoise', k - 1, task_index
                        writer.writerow(dict(zip(_TASK_FIELDS,
                                                 [iteration + 1, kind, index, part, device_index,
                                                  round(start, 3), round(end, 3)])))
            if verbose:
                self.logger.info(
                    f'[MACE] Iteration {iteration + 1}/{max_iterations}: prox={prox_total:.2f}s, '
                    f'denoise={denoise_total:.2f}s, makespan={makespan:.2f}s, '
                    f"total={loop.info['time'][-1]:.2f}s, change={change_pct:.4f}%, "
                    f'denoiser iterations={mean_iterations:.1f}')

        with loop:
            x_bar, _ = loop.run(max_iterations=max_iterations,
                                stop_threshold_change_pct=stop_threshold_change_pct,
                                callback=after_step)
        # The result shares storage with the loop's average buffer.  The loop
        # is not used after this call returns.
        result = x_bar.cpu().numpy()
        if verbose:
            self.logger.info('[MACE] Reconstruction complete.')

        run_settings['iterations completed'] = len(timing_rows)
        if log_dir is not None:
            _write_run_info(os.path.join(log_dir, 'run_info.txt'), run_settings)
        recon_dict = {
            'recon_params': run_settings,
            'timing': timing_rows,
            'notes': 'Reconstruction completed: {}\n\n'.format(datetime.datetime.now()),
            'model_params': self.params.copy(),
        }
        return result, recon_dict

    def _prior_params(self):
        """Return the qGGMRF parameters given to every denoiser.

        ``sharpness`` is not among them.  Setting it on a denoiser would turn
        that denoiser's own automatic regularization back on.  The frame axis
        is the row axis of every hyperplane volume, so ``nbr_weight_time`` is
        the first neighbor weight.
        """
        p, q, T = self.get_params(['p', 'q', 'T'])
        nbr_weight_time = float(self.get_params('nbr_weight_time'))
        return dict(p=float(p), q=float(q), T=float(T),
                    qggmrf_nbr_wts=[nbr_weight_time, 1.0, 1.0])

    def _denoiser_sigma_x(self, image, global_sigma, device):
        """Return the prior strength of all three denoisers, where it came
        from, and the spread of the image along time against space.

        One strength is used for the whole 4D volume, because the three
        hyperplane priors add up to one 4D prior only when they share it.  It
        is estimated from the image as a stack of frames, so that every
        neighbor difference the estimate reads lies inside one frame.
        Setting ``sigma_x`` disables the estimate.
        """
        num_frames = int(image.shape[0])
        frame_shape = tuple(int(n) for n in image.shape[1:])
        denoiser = QGGMRFDenoiser(frame_shape)
        denoiser.configure_devices(devices=[device])
        denoiser.set_params(no_warning=True, verbose=0, sigma_noise=global_sigma,
                            sigma_y=global_sigma, sharpness=self.get_params('sharpness'),
                            **self._prior_params())
        chosen = denoiser.subsample_views(np.arange(num_frames))
        spread = _spread_along_time(image, chosen, global_sigma)
        if not self.get_params('auto_regularize_flag'):
            return float(self.get_params('sigma_x')), 'set by sigma_x', spread
        regularization = denoiser.auto_set_regularization_params_from_stack(image)
        return (float(regularization['sigma_x']), 'estimated from the initial image', spread)

    def _configure_orientation(self, axis, x0, sigma, sigma_x):
        """Return the volume shape, the denoiser parameters, and the batch
        size of one orientation.

        Every orientation is given the same noise level and the same prior
        strength, so that the three priors add up to one 4D prior.  The batch
        size comes from :func:`_slab_batch_size`.
        """
        image_shape = tuple(int(x0.shape[d]) for d in _permutation(axis)[1:])
        # A subset with fewer than about 64 pixels makes the line search
        # compute zero over zero on flat regions.
        num_pixels = image_shape[0] * image_shape[1]
        default_subsets = int(QGGMRFDenoiser(image_shape).get_params('granularity')[0])
        num_subsets = max(1, min(default_subsets, num_pixels // 64))
        batch_size = _slab_batch_size(int(x0.shape[axis]), image_shape,
                                      float(self.get_params('denoise_slab_gb')))
        params = dict(sigma_noise=sigma, sigma_y=sigma, sigma_x=sigma_x,
                      granularity=[num_subsets], partition_sequence=[0],
                      auto_regularize_flag=False, **self._prior_params())
        return image_shape, params, batch_size

    @staticmethod
    def _stack_denoiser_factory(image_shape, params, sigma, batch_size, iteration_counts, lock):
        """Return a ``make_stack_denoiser(device)`` for
        :class:`HyperplaneAgent`.  Each denoiser it returns is pinned to its
        device, uses the given parameters, sweeps every slab at
        ``batch_size`` volumes, and records the iteration count of every
        volume it sweeps."""
        def make_stack_denoiser(device):
            denoiser = QGGMRFDenoiser(image_shape)
            denoiser.configure_devices(devices=[device])
            denoiser.set_params(no_warning=True, verbose=0, **params)

            def padded(stack, count):
                """Return the stack with its last volume repeated up to
                ``count`` volumes.  A short last slab is then swept at the
                shape of the others, so no second variant is compiled."""
                short = count - int(stack.shape[0])
                if short <= 0:
                    return stack
                return torch.cat([stack, stack[-1:].expand(short, *stack.shape[1:])])

            def denoise(stack, init_stack=None):
                real = int(stack.shape[0])
                count = real if batch_size is None else max(int(batch_size), real)
                # The agent hands over its own copies of the slab, and nothing reads
                # them again, so the sweep writes them in place.
                out, info = denoiser.denoise_stack(
                    padded(stack, count), sigma_noise=sigma,
                    init_stack=None if init_stack is None else padded(init_stack, count),
                    max_iterations=_DENOISE_MAX_ITERATIONS,
                    stop_threshold_change_pct=_DENOISE_STOP_THRESHOLD_PCT,
                    batch_size=count, overwrite_input=True)
                with lock:
                    iteration_counts.extend(int(n) for n in info['num_iterations'][:real])
                return out[:real]
            return denoise
        return make_stack_denoiser

    @staticmethod
    def _estimate_global_sigma(init_recon, device):
        """Return the noise level used by all three denoisers, estimated from
        the initial image."""
        init_recon = np.asarray(init_recon, dtype=np.float32)
        image_3d = init_recon.reshape(-1, init_recon.shape[2], init_recon.shape[3])
        denoiser = QGGMRFDenoiser(image_3d.shape)
        denoiser.configure_devices(devices=[device])
        return float(denoiser.estimate_image_noise_std(image_3d))

    def _compute_init_recon(self, frame_agents, pool, init_dir):
        """Reconstruct each frame alone, on the frame's device, with the
        sinograms the frame agents already placed, and cache the result."""
        verbose = self.get_params('verbose')
        stop_threshold = self.get_params('prox_stop_threshold')
        if verbose:
            self.logger.info(f'[MACE] Computing the initial reconstruction on {len(pool)} worker(s)...')
        t0 = time.perf_counter()
        volumes = [None] * self.num_frames

        def reconstruct(frames):
            for t in frames:
                agent = frame_agents[t]
                volume, _ = agent.model.recon(
                    agent.sinogram, weights=agent.weights, max_iterations=_INIT_ITERATIONS,
                    stop_threshold_change_pct=stop_threshold, logfile_path=None, print_logs=False)
                volumes[t] = np.asarray(volume, dtype=np.float32)

        # Frames are grouped by pool entry, as the loop's workers are, so
        # that the frames of one worker run one after another on one thread.
        by_worker = {}
        for t in range(self.num_frames):
            by_worker.setdefault(t % len(pool), []).append(t)
        if len(by_worker) == 1:
            reconstruct(next(iter(by_worker.values())))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(by_worker)) as executor:
                futures = [executor.submit(reconstruct, frames) for frames in by_worker.values()]
                for future in futures:
                    future.result()
        init_recon = np.stack(volumes)
        if init_dir is not None:
            os.makedirs(init_dir, exist_ok=True)
            np.save(os.path.join(init_dir, 'init_recon.npy'), init_recon)
        if verbose:
            self.logger.info(f'[MACE] Initialization done in {time.perf_counter() - t0:.2f} sec.')
        return init_recon

    def _load_cached_init(self, init_dir):
        """Return the image in ``init_dir/init_recon.npy``, or None when the
        file is missing.  When the file cannot be loaded or has the wrong
        shape, a warning is issued and the result is None."""
        path = os.path.join(init_dir, 'init_recon.npy')
        if not os.path.isfile(path):
            return None
        try:
            init_recon = self._validate_init_recon(np.load(path))
        except (ValueError, OSError) as e:
            warnings.warn(f'init_dir has an invalid initialization image ({e}); recomputing.')
            return None
        if self.get_params('verbose'):
            self.logger.info(f'[MACE] Using cached init from {path}.')
        return init_recon

    def _validate_sinogram(self, sinogram, name):
        """Return the array unchanged.  Raises ValueError when its shape is
        not the sinogram shape."""
        if not torch.is_tensor(sinogram):
            sinogram = np.asarray(sinogram)
        shape = tuple(int(n) for n in sinogram.shape)
        if shape != self.sinogram_shape:
            raise ValueError(f"{name} shape {shape} does not match the model's sinogram shape "
                             f'{self.sinogram_shape}.')
        return sinogram

    def _expected_init_shape(self):
        return (self.num_frames,) + self.recon_shape

    def _validate_init_recon(self, init_recon):
        """Return the initial image as float32 numpy.  Raises ValueError on a
        wrong shape."""
        init_recon = np.asarray(init_recon, dtype=np.float32)
        expected = self._expected_init_shape()
        if init_recon.shape != expected:
            raise ValueError(f'init_recon shape {init_recon.shape} does not match expected {expected}.')
        return init_recon

    def _run_settings(self, pool, init_source, global_sigma, sigma_source, weights,
                      max_iterations, stop_threshold_change_pct, sigma_x, sigma_x_source,
                      spread, batch_sizes):
        """Return the settings of a run, for ``run_info.txt`` and the result
        dict."""
        from . import __version__
        beta = _normalize_prior_weights(self.get_params('mace_prior_weight'))
        sigma_prox = self.get_params('sigma_prox')
        return {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mbirtorch version': __version__,
            'time frames': self.num_frames,
            'frame shape': self.recon_shape,
            'views per frame': self.view_slices[0].stop - self.view_slices[0].start,
            'workers': f'{len(pool)}: ' + ', '.join(str(d) for d in pool),
            'init source': init_source,
            'weights': 'unit (weights=None)' if weights is None else 'supplied by caller',
            'beta [fwd, xyt, yzt, xzt]': [round(float(b), 4) for b in beta],
            'rho_mann': self.get_params('rho_mann'),
            'max_iterations': max_iterations,
            'stop_threshold_change_pct': stop_threshold_change_pct,
            'prox_num_iterations': self.get_params('prox_num_iterations'),
            'prox_stop_threshold': self.get_params('prox_stop_threshold'),
            'prox_partition_advance': self.get_params('prox_partition_advance'),
            'prox_warm_start': self.get_params('prox_warm_start'),
            'denoiser_warm_start': self.get_params('denoiser_warm_start'),
            'sigma_prox': 'auto' if sigma_prox is None else sigma_prox,
            'denoiser sigma (global)': float(global_sigma),
            'denoiser sigma source': sigma_source,
            'denoiser sigma_x': float(sigma_x),
            'denoiser sigma_x source': sigma_x_source,
            'denoiser sharpness': self.get_params('sharpness'),
            'denoiser stop_threshold_change_pct': _DENOISE_STOP_THRESHOLD_PCT,
            'nbr_weight_time': float(self.get_params('nbr_weight_time')),
            'nbr_weight_time note': ('1.0 weights a frame neighbor 1.5 times a spatial '
                                     'neighbor; 2/3 weights them equally'),
            'temporal over spatial spread': spread['ratio'],
            'spread read from': (f"{spread['frames read']} frames, stride {spread['stride']}, "
                                 f"block {spread['block']}"),
            'denoise batch sizes [xyt, yzt, xzt]': list(batch_sizes),
            'denoise slab budget (GB)': float(self.get_params('denoise_slab_gb')),
            'dejitter': self.get_params('dejitter'),
            'frames_per_rotation': self.frames_per_rotation,
            'frame_overlap_factor': self.frame_overlap_factor,
        }
