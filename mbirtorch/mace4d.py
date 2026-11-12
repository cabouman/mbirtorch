"""This module performs 4D reconstruction of one continuous scan of a moving object.

The module divides the scan into overlapping angular windows, one for each time frame.  It
reconstructs one volume for each frame.  :class:`MACE4DModel` holds the volumes in one array
of shape ``(frames, x, y, z)``.  It computes them as the MACE consensus of a data-fit agent
and three qGGMRF denoisers.  The overlapping windows produce a periodic modulation along the
frame axis.  :func:`sor_filter_matrix` builds the filter that removes this modulation.
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
from .vcd_utils import gen_set_of_pixel_partitions, named_rng, named_seed

# The maximum iterations and the stop threshold of each denoiser call.
_DENOISE_MAX_ITERATIONS = 15
_DENOISE_STOP_THRESHOLD_PCT = 0.05
# The size in bytes of each slab when the filter is applied to the data-fit outputs.
_FILTER_SLAB_BYTES = 64 * 2 ** 20
# The relative singular value below which sor_filter_matrix leaves out a harmonic.  A harmonic is
# left out when it is zero at the frame angles or repeats other harmonics there, as
# sin(3 theta) does at 6 frames per rotation.  A model holds its angles in float32.  The
# rounding of these angles gives such a harmonic a relative singular value of up to about 1e-5.
_HARMONIC_RANK_TOL = 1e-4

# The name of each denoiser orientation and the spatial axis that it holds fixed.  The axis
# index uses the order (frames, x, y, z).
_ORIENTATIONS = [('XY-t', 3), ('YZ-t', 1), ('XZ-t', 2)]

_TIMING_FIELDS = ['iteration', 'prox_total_sec', 'denoise_total_sec', 'makespan_sec',
                  'iteration_total_sec', 'consensus_change_pct', 'denoise_mean_iterations']
_TASK_FIELDS = ['iteration', 'kind', 'index', 'part', 'worker', 'start_sec', 'end_sec']


def apply_temporal_filter(x, matrix, axis=0):
    """
    This function applies a filter matrix along one axis of a tensor.

    Args:
        x (torch.Tensor): The array to filter.
        matrix (torch.Tensor): A square matrix from :func:`sor_filter_matrix`.  Its size
            equals the length of ``axis``.  The function moves it to the device and dtype of
            ``x``.
        axis (int, optional): The axis along which the matrix acts.  Defaults to 0.

    Returns:
        torch.Tensor: A new tensor with the shape of ``x``.
    """
    matrix = matrix.to(device=x.device, dtype=x.dtype)
    moved = x.movedim(axis, 0)
    filtered = torch.tensordot(matrix, moved, dims=1)
    return filtered.movedim(0, axis)


def sor_filter_matrix(angles, order):
    """
    This function returns the filter matrix P of Sensor Orthogonal Reconstruction (SOR).  P
    acts along the frame axis.  The filter removes the part of each voxel that
    varies with the sensor angle theta of the frame.  At each voxel, it removes the
    least-squares fit of cos(m theta) and sin(m theta) for m = 1 to p, and it keeps the mean
    over the frames.  :class:`MACE4DModel` uses this filter when ``dejitter`` is on.  Use
    :func:`apply_temporal_filter` to apply P to another array.

    Args:
        angles (array): The sensor angle theta of each frame, in radians.  It is the center
            angle of the views of the frame.  It needs at least 2 frames.
        order (int): The highest harmonic p of theta that the filter removes.  A larger p
            removes changes that vary faster with the angle.

    Returns:
        torch.Tensor: The float32 matrix P of shape (num_frames, num_frames), on the CPU.
    """
    theta = np.asarray(angles, dtype=np.float64)
    if theta.ndim != 1 or len(theta) < 2:
        raise ValueError('angles must be a 1D array of at least 2 angles; '
                         f'got shape {theta.shape}.')

    # The columns of harmonics are cos(m theta) and sin(m theta) for m = 1 to p.  Their means
    # are removed, so the filter keeps the mean over the frames.  P = I - Q Q^T, where the
    # columns of Q are an orthonormal basis of the harmonics.
    harmonics = np.stack([f(m * theta) for m in range(1, int(order) + 1)
                          for f in (np.cos, np.sin)], axis=1)
    harmonics -= harmonics.mean(axis=0)
    basis, singular_values, _ = np.linalg.svd(harmonics, full_matrices=False)
    basis = basis[:, singular_values > _HARMONIC_RANK_TOL * singular_values.max()]
    matrix = np.eye(len(theta)) - basis @ basis.T
    return torch.as_tensor(matrix.astype(np.float32))


def _frame_angles(angles, view_slices):
    """This function returns the sensor angle of each frame, in radians.  It is the center
    angle of the views of the frame.

    Args:
        angles (array): The angle of each view of the scan, in radians.
        view_slices (list of slice): The views of each frame, as in ``MACE4DModel.view_slices``.
    """
    angles = np.asarray(angles, dtype=np.float64)
    # The angle of the summed unit vectors is the center of the views, also when the angles
    # wrap past 2 pi.
    return np.array([np.arctan2(np.sin(angles[s]).sum(), np.cos(angles[s]).sum())
                     for s in view_slices])


def _normalize_prior_weights(prior_weight):
    """This function returns the four agent weights ``[forward, xyt, yzt, xzt]``.

    A scalar prior weight ``w`` gives ``[1 - w, w/3, w/3, w/3]``.  A list of three prior
    weights gives the last three entries.
    """
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
    """This function writes each run setting to ``path`` as one ``key = value`` line."""
    width = max(len(key) for key in run_settings)
    lines = ['# MACE4DModel run settings']
    lines += ['{:<{width}} = {}'.format(key, value, width=width)
              for key, value in run_settings.items()]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _describe_filter(matrix, order):
    """This function returns one line of text that describes the filter.

    The line gives the harmonics of the frame angle that the filter removes.  It also gives
    the number of modes along the frame axis that the filter removes and keeps.
    """
    num_frames = int(matrix.shape[0])
    kept = int(round(float(matrix.trace())))
    removed = num_frames - kept
    return (f'removes the harmonics 1 to {order} of the frame angle; {removed} of '
            f'{num_frames} modes removed, {kept} kept')


def _permutation(axis):
    """This function returns the axis order that puts ``axis`` first and the frame axis
    second."""
    return (axis,) + tuple(d for d in range(4) if d != axis)


def _slab_batch_size(num_planes, volume_shape, slab_gb):
    """This function returns the number of hyperplanes in each denoiser task.

    The function first finds the largest number of hyperplanes that fits in ``slab_gb`` GB.
    It then divides the hyperplanes into slabs of nearly equal size.

    Args:
        num_planes (int): The number of hyperplanes in the orientation.
        volume_shape (tuple of int): The shape of one hyperplane volume.
        slab_gb (float): The memory limit of one slab, in units of ``2 ** 30`` bytes.

    Returns:
        int: The number of hyperplanes in each slab, between 1 and ``num_planes``.
    """
    num_planes = int(num_planes)
    volume_bytes = 4 * int(np.prod(volume_shape))
    fit = max(1, int(float(slab_gb) * 2 ** 30) // volume_bytes)
    num_slabs = -(-num_planes // min(num_planes, fit))
    return -(-num_planes // num_slabs)


def _centered_block(shape, num_frames, point_budget):
    """This function returns the slices of a block centered in an array of shape ``shape``.

    The block holds at most ``point_budget`` entries in total over ``num_frames`` frames.
    """
    total = int(num_frames) * int(np.prod(shape))
    scale = 1.0 if total <= point_budget else (point_budget / total) ** (1.0 / len(shape))
    block = []
    for extent in shape:
        width = int(min(int(extent), max(1, round(int(extent) * scale))))
        start = (int(extent) - width) // 2
        block.append(slice(start, start + width))
    return tuple(block)


def _rms_neighbor_difference(values, support, axes):
    """This function returns the RMS difference between neighboring entries along ``axes``
    inside ``support``.

    The function returns None when ``support`` contains no pair of neighbors.
    """
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
    """This function compares the RMS difference between neighbors along the frame axis with
    the RMS difference between neighbors along the three spatial axes.

    The function reads only the frames in ``chosen`` and only the voxels of the object.

    Args:
        image (tensor): The 4D image, of shape ``(frames, x, y, z)``.
        chosen (numpy): The indices of the frames to read.
        sigma_noise (float): The noise level.  It sets the threshold that separates the object
            from the background.
        point_budget (int, optional): The largest number of voxels to read over all chosen
            frames.

    Returns:
        dict: The ratio of the two RMS differences, the two RMS differences, the number of
        frames read, the stride between them, and the shape of the block.  The ratio is None
        when fewer than two frames are chosen or when the function does not find the object.
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
    """This class is used to apply the proximal maps of the data-fit terms of all frames.  The
    class acts as one agent of the MACE reconstruction of the 4D array.

    Each frame has one :class:`ForwardProxAgent`, which runs on the device of that frame.  When
    the temporal filter is on, the class applies the filter to the stack of outputs.
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
            # The class writes into init_stack in place from here on.
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
    This class is used to compute a space-time reconstruction of one continuous CT scan.

    The class assumes that the views are in time order.  It divides the scan into overlapping
    time frames of consecutive views.  :meth:`recon` returns one volume for each frame.  The
    overlapping frames produce a periodic modulation along the frame axis.  By default, a
    filter along the frame axis removes this modulation.

    Args:
        ct_model (TomographyModel): The model of the full scan.  It is a ConeBeamModel or a
            ParallelBeamModel with one angle for each view.
        frames_per_rotation (int, optional): The number of frames in each full rotation.
            Defaults to 6.
        frame_overlap_factor (float, optional): The number of frames that cover any one view.
            Each frame spans ``frame_overlap_factor * 360 / frames_per_rotation`` degrees.
            Defaults to 2.0.
        num_frames (int, optional): The number of frames to reconstruct, counted from the start
            of the scan.  None means every frame.  Defaults to None.

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
                        prox_partition_advance=1.0,
                        prox_warm_start=True, denoiser_warm_start=False, sigma_prox=None,
                        dejitter=True, sor_order=3, dejitter_verbose=0,
                        sigma_noise=None, nbr_weight_time=1.0, sharpness=0.0,
                        denoise_slab_gb=2.0)
        self._devices = None

    def refresh_device_bindings(self):
        """This method does nothing, because this model holds no projectors."""

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        This method sets reconstruction parameters by keyword.  The geometry and regularization
        parameters belong to ``ct_model``, so set them there.

        Args:
            no_warning (bool, optional): If True, the method does not check the parameter names
                and does not warn.  Defaults to False.
            no_compile (bool, optional): This argument has no effect.  It exists to match the
                base class.
            mace_prior_weight (float or list of 3 floats): The total weight of the three priors,
                or one weight each for XY-t, YZ-t, and XZ-t.  The data-fit agent gets the
                remainder of 1.  Defaults to 0.5.
            rho_mann (float): The step size of the consensus iteration.  Defaults to 0.5.
            prox_num_iterations (int): The number of iterations of each data-fit call.
                Defaults to 3.
            prox_partition_advance (float): The number of entries of the partition sequence
                that each data-fit call moves forward in each iteration.  Defaults to 1.0.
            prox_warm_start (bool): If True, each data-fit call starts from the previous output
                of its frame.  Defaults to True.
            denoiser_warm_start (bool): If True, each denoiser call starts from its previous
                output.  This option uses one more full-size array for each orientation.
                Defaults to False.
            denoise_slab_gb (float): The largest size of the stack of hyperplane volumes that
                one denoiser task processes.  The size is in units of ``2**30`` bytes.  A task
                holds about three arrays of that size.  Defaults to 2.0.
            sigma_prox (float or None): The strength of every data-fit call.  None sets it from
                the data.  Defaults to None.
            dejitter (bool): If True, the method applies the temporal filter along the frame
                axis.  Defaults to True.
            sor_order (int): The highest harmonic p of the frame angle that the temporal filter
                removes.  Defaults to 3.
            dejitter_verbose (int): If nonzero, the method logs the harmonics that the filter
                removes and the number of modes that it keeps.  Defaults to 0.
            sigma_noise (float or None): The noise level of all three denoisers.  None
                estimates it from the initial image.  Defaults to None.
            sigma_x (float): The prior strength of all three denoisers.  Setting it turns off
                the automatic estimate.  Defaults to the automatic estimate.
            sharpness (float): This value scales the automatic ``sigma_x`` of the denoisers.
                The prior of each frame takes its sharpness from ``ct_model``.  Defaults to 0.
            nbr_weight_time (float): The weight of a neighbor in time relative to a spatial
                neighbor in the priors of the denoisers.  The default of 1.0 gives a neighbor
                in time 1.5 times the weight of a spatial neighbor.  The value 2/3 makes the
                two weights equal.  Defaults to 1.0.
            verbose (int): The value 0 prints nothing, and 1 reports progress.  Defaults to 1.

        Raises:
            ValueError: If ``qggmrf_nbr_wts`` is set, or if ``denoise_slab_gb`` is not positive.
                Use ``nbr_weight_time`` in place of ``qggmrf_nbr_wts``.

        Example:
            >>> mace.set_params(mace_prior_weight=0.5, rho_mann=0.5, dejitter=True)
        """
        if kwargs.pop('prox_stop_threshold', None) is not None:
            warnings.warn('prox_stop_threshold has no effect and will be removed in a future release.',
                          FutureWarning, stacklevel=2)
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
        sigma_prox_given = 'sigma_prox' in kwargs
        sigma_prox = kwargs.pop('sigma_prox', None)
        if sigma_prox_given:
            super().set_params(no_warning=True, no_compile=no_compile, sigma_prox=sigma_prox)
        if kwargs:
            super().set_params(no_warning=no_warning, no_compile=no_compile, **kwargs)

    def set_device_pool(self, devices=None):
        """
        This method sets the devices that :meth:`recon` uses, starting with the next call to
        :meth:`recon`.  The method assigns the data-fit tasks of the frames to the devices in
        turn.  Any device can run the denoising tasks.

        Args:
            devices (optional): Any value that ``mbirtorch.mace.resolve_device_pool`` accepts.
                The value can be None for every GPU or else the CPU, ``'cpu'``, ``'gpu'``, a
                device count, a list of device indices, or a list of devices.

        Raises:
            ValueError: If the list names a GPU more than once, or if the requested count is
                larger than the number of available devices.

        Example:
            >>> mace.set_device_pool(2)
        """
        self._devices = resolve_device_pool(devices)

    @property
    def devices(self):
        """This property returns the list of devices that :meth:`recon` uses.

        If no devices have been set, the property returns the default list.
        """
        return list(self._devices) if self._devices is not None else resolve_device_pool(None)

    def recon(self, sinogram, weights=None, init_recon=None, max_iterations=10,
              stop_threshold_change_pct=0.2, init_dir=None, log_dir=None, seed=None):
        """
        This method reconstructs one volume for each time frame from the sinogram of the scan.

        Args:
            sinogram (numpy or tensor): The full sinogram, of shape
                ``(num_views, num_det_rows, num_det_channels)``.
            weights (numpy or tensor, optional): Positive weights with the shape of the
                sinogram.  None means that all weights are one.  Defaults to None.
            init_recon (numpy, optional): The initial 4D image, of shape
                ``(num_frames,) + recon_shape``.  If it is None, the method reads the image from
                ``init_dir`` when the file exists.  Otherwise, the method computes a direct
                reconstruction of each frame.  Defaults to None.
            max_iterations (int, optional): The number of consensus iterations.  Defaults to 10.
            stop_threshold_change_pct (float, optional): The method stops when the percent
                change of the consensus image in one iteration falls below this value.  The
                value 0 runs every iteration.  Defaults to 0.2.
            init_dir (str, optional): The directory of the cached initial image
                ``init_recon.npy``.  The method reads the file when it exists and writes it
                otherwise.  None means no cache.  Defaults to None.
            log_dir (str, optional): The directory for ``run_info.txt``, ``timing_log.csv``,
                and ``task_log.csv``.  None means no log files.  Defaults to None.
            seed (int, optional): The seed of every random draw in the run.  Two runs with the
                same seed on the same devices agree up to float rounding.  None means that the
                method draws a seed from the global np.random state.  Defaults to None.

        Returns:
            (recon, recon_dict): recon is the 4D reconstruction, a numpy array of shape
            ``(num_frames,) + recon_shape``.  recon_dict is a dict with these entries: the run
            settings under ``'recon_params'``, the timing of each iteration under
            ``'timing'``, the completion time under ``'notes'``, and a copy of the model
            parameters under ``'model_params'``.

        Raises:
            ValueError: If ``sinogram``, ``weights``, or ``init_recon`` has the wrong shape.
                The method also raises ValueError if the initial image is constant, because
                then it cannot estimate the noise level of the denoisers.

        Warns:
            UserWarning: If the temporal filter is on but it would remove all change between the
                frames, as it does with too few frames.  The run then continues with the filter
                off.

        Example:
            >>> recon_4d, recon_dict = mace.recon(sinogram, weights=weights, log_dir='./logs')
        """
        # This line is the only draw from the global random state.  Every other draw comes
        # from this seed.
        seed = int(np.random.randint(1 << 31)) if seed is None else int(seed)
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
        # The method does not write into an initial image that the caller supplied.
        init_is_own = init_recon is None
        if init_recon is not None:
            init_recon = self._validate_init_recon(init_recon)
            init_source = 'provided by caller'

        # The filter is turned off when it would remove all change between the frames.
        filter_matrix = None
        dejitter_note = None
        if dejitter:
            filter_matrix, dejitter_note = self._temporal_filter()
            if filter_matrix is None:
                dejitter = False
                warnings.warn(f'The temporal filter was {dejitter_note}.  Use more frames or a '
                              'smaller sor_order, or set dejitter=False to silence this warning.')
            elif self.get_params('dejitter_verbose'):
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
                             use_warm_start=prox_warm_start,
                             seed=named_seed(seed, t))
            for t in range(num_frames)]

        if init_recon is None:
            if init_dir is not None:
                init_recon = self._load_cached_init(init_dir)
            if init_recon is not None:
                init_source = f"cached ({os.path.join(init_dir, 'init_recon.npy')})"
            else:
                init_recon = self._compute_init_recon(frame_agents, pool, init_dir)
                init_source = f'computed ({num_frames} frames, direct reconstruction)'
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
        # When the filter is on, the denoiser strength is computed from the filtered initial
        # image.
        image_for_statistics = (x0 if filter_matrix is None
                                else apply_temporal_filter(x0, filter_matrix, axis=0))
        sigma_x, sigma_x_source, spread = self._denoiser_sigma_x(
            image_for_statistics, global_sigma, pool[0])
        for name, axis in _ORIENTATIONS:
            image_shape, params, batch_size, partition = self._configure_orientation(
                axis, image_for_statistics, global_sigma, sigma_x,
                named_rng(seed, name))
            batch_sizes.append(batch_size)
            make = self._stack_denoiser_factory(image_shape, params, global_sigma, batch_size,
                                                partition, iteration_counts, counts_lock)
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
                                          sigma_x, sigma_x_source, spread, batch_sizes, seed)
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
        # x_bar shares storage with the average buffer of the MACE loop.
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

    def _temporal_filter(self):
        """This method returns the matrix of the temporal filter and one line of text that
        describes the filter.

        The matrix is the SOR filter matrix of the frame angles.  The matrix is None when the
        filter would remove all change between the frames.
        """
        order = int(self.get_params('sor_order'))
        if self.num_frames > 1:
            required_params, _, _ = self.ct_model.get_all_params()
            angles = _frame_angles(required_params['angles'], self.view_slices)
            matrix = sor_filter_matrix(angles, order)
            # The trace is the number of modes that the filter keeps.  One of them is the mean
            # over the frames.
            if round(float(matrix.trace())) > 1:
                return matrix, _describe_filter(matrix, order)
        return None, (f'turned off: at order {order} it would remove all change between the '
                      f'{self.num_frames} frames')

    def _prior_params(self):
        """This method returns the qGGMRF parameters of every denoiser.

        The frame axis is the row axis of every hyperplane volume.
        """
        p, q, T = self.get_params(['p', 'q', 'T'])
        nbr_weight_time = float(self.get_params('nbr_weight_time'))
        return dict(p=float(p), q=float(q), T=float(T),
                    qggmrf_nbr_wts=[nbr_weight_time, 1.0, 1.0])

    def _denoiser_sigma_x(self, image, global_sigma, device):
        """This method returns the prior strength ``sigma_x`` that all three denoisers share.

        The method also returns a description of where ``sigma_x`` came from.  Third, the method
        returns the result of ``_spread_along_time``.  That result compares the variation of
        the image in time with the variation of the image in space.
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

    def _configure_orientation(self, axis, x0, sigma, sigma_x, rng=None):
        """This method returns the volume shape, the denoiser parameters, the batch size, and
        the pixel partition of one orientation.
        """
        image_shape = tuple(int(x0.shape[d]) for d in _permutation(axis)[1:])
        # With fewer than about 64 pixels in a subset, the line search divides zero by zero.
        num_pixels = image_shape[0] * image_shape[1]
        default_subsets = int(QGGMRFDenoiser(image_shape).get_params('granularity')[0])
        num_subsets = max(1, min(default_subsets, num_pixels // 64))
        batch_size = _slab_batch_size(int(x0.shape[axis]), image_shape,
                                      float(self.get_params('denoise_slab_gb')))
        params = dict(sigma_noise=sigma, sigma_y=sigma, sigma_x=sigma_x,
                      granularity=[num_subsets], partition_sequence=[0],
                      auto_regularize_flag=False, **self._prior_params())
        partition = gen_set_of_pixel_partitions(image_shape, [num_subsets],
                                                use_ror_mask=False, rng=rng)[0]
        return image_shape, params, batch_size, partition

    @staticmethod
    def _stack_denoiser_factory(image_shape, params, sigma, batch_size, partition,
                                iteration_counts, lock):
        """This method returns a function ``make_stack_denoiser(device)`` for
        :class:`HyperplaneAgent`.

        That function builds a denoiser on ``device`` and returns a function that denoises a
        stack of hyperplane volumes.
        """
        def make_stack_denoiser(device):
            denoiser = QGGMRFDenoiser(image_shape)
            denoiser.configure_devices(devices=[device])
            denoiser.set_params(no_warning=True, verbose=0, **params)
            denoiser.initialize_denoiser(partition=partition)

            def padded(stack, count):
                """This function pads the stack to ``count`` volumes by repeating its last
                volume.

                With this padding, a short last slab does not need a second compiled version.
                """
                short = count - int(stack.shape[0])
                if short <= 0:
                    return stack
                return torch.cat([stack, stack[-1:].expand(short, *stack.shape[1:])])

            def denoise(stack, init_stack=None):
                real = int(stack.shape[0])
                count = real if batch_size is None else max(int(batch_size), real)
                # The slab is a copy that nothing reads again, so the denoiser overwrites it.
                out, info = denoiser.denoise_stack(
                    padded(stack, count), sigma_noise=sigma,
                    init_stack=None if init_stack is None else padded(init_stack, count),
                    max_iterations=_DENOISE_MAX_ITERATIONS,
                    stop_threshold_change_pct=_DENOISE_STOP_THRESHOLD_PCT,
                    batch_size=count, overwrite_input=True,
                    do_initialization=False)
                with lock:
                    iteration_counts.extend(int(n) for n in info['num_iterations'][:real])
                return out[:real]
            return denoise
        return make_stack_denoiser

    @staticmethod
    def _estimate_global_sigma(init_recon, device):
        """This method estimates the noise level of the denoisers from the initial image."""
        init_recon = np.asarray(init_recon, dtype=np.float32)
        image_3d = init_recon.reshape(-1, init_recon.shape[2], init_recon.shape[3])
        denoiser = QGGMRFDenoiser(image_3d.shape)
        denoiser.configure_devices(devices=[device])
        return float(denoiser.estimate_image_noise_std(image_3d))

    def _compute_init_recon(self, frame_agents, pool, init_dir):
        """This method computes a direct reconstruction of each frame on the device of that
        frame.

        The method does not use the weights.  When ``init_dir`` is given, the method saves the
        result there.
        """
        verbose = self.get_params('verbose')
        if verbose:
            self.logger.info(f'[MACE] Computing the initial reconstruction on {len(pool)} worker(s)...')
        t0 = time.perf_counter()
        volumes = [None] * self.num_frames

        def reconstruct(frames):
            for t in frames:
                agent = frame_agents[t]
                volume = agent.model.recon_direct(agent.sinogram)
                volumes[t] = np.asarray(volume, dtype=np.float32)

        # The frames of one device run one after another on one thread.
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
        """This method returns the image in ``init_dir/init_recon.npy``.

        If the file is missing, the method returns None.  If the file is not valid, the method
        warns and returns None.
        """
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
        """This method checks the shape of the array and returns the array unchanged.

        The method raises ValueError when the shape of the array is not the sinogram shape.
        """
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
        """This method returns the initial image as a float32 numpy array.

        The method raises ValueError when the image has the wrong shape.
        """
        init_recon = np.asarray(init_recon, dtype=np.float32)
        expected = self._expected_init_shape()
        if init_recon.shape != expected:
            raise ValueError(f'init_recon shape {init_recon.shape} does not match expected {expected}.')
        return init_recon

    def _run_settings(self, pool, init_source, global_sigma, sigma_source, weights,
                      max_iterations, stop_threshold_change_pct, sigma_x, sigma_x_source,
                      spread, batch_sizes, seed):
        """This method returns the settings of a run."""
        from . import __version__
        beta = _normalize_prior_weights(self.get_params('mace_prior_weight'))
        sigma_prox = self.get_params('sigma_prox')
        return {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mbirtorch version': __version__,
            'seed': seed,
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
            'denoiser pixel partitions': 'one per orientation, drawn once from the seed',
            'dejitter': self.get_params('dejitter'),
            'sor_order': self.get_params('sor_order'),
            'frames_per_rotation': self.frames_per_rotation,
            'frame_overlap_factor': self.frame_overlap_factor,
        }
