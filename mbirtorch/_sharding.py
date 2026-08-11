"""Device placement, safe transfer, and per-device threaded execution --
ported from the mbirjax._sharding package (placement.py, transfer.py,
thread_execution.py).  A single process drives every device, bands move
device-to-device, and one python thread per device issues the work -- the
substrate the jax fbp-filter parallelism study selected and the torch
substrate spikes revalidated (measurements in the plans repo).

The jax global sharded array (make_array_from_single_device_arrays) has no
counterpart here -- DTensor is deliberately avoided as immature for these
index-heavy kernels -- so the
device form of a sharded array here is a :class:`Shards` container: the
per-device tensors plus their :class:`Placement`.  A single device is the
trivial 1-shard case, and the placement functions keep returning PLAIN
tensors there,
so the n=1 reconstruction path is unchanged.

Under view/slice sharding the only data that crosses the recon<->sino
boundary is voxel cylinders (the sinogram is written locally on its
view-shard and never moves).  Two shapes of that crossing exist, and they
differ in which axis of the cylinder is cut.  The banded adjoint pair cuts
the SLICE axis:

  - ``broadcast_band_to_views`` (forward / all-gather): copy a slice-band
    from its slice-owner to every view-owner.
  - ``sum_band_to_owner`` (back / reduce-scatter): sum each view-owner's
    band partials onto the band's slice-owner.

``gather_column_band`` cuts the PIXEL axis instead: it assembles one batch of
pixel columns at every slice on one view-owner.  A geometry whose slices
project onto a range of detector rows needs the whole slice axis before it
can produce any of its own rows, so a slice band buys it nothing, and the
forward driver gathers columns for it when that path is switched on.  A
row-aligned geometry can produce its rows from a band and takes the same
gather anyway, because its kernel is markedly faster on the wider block of
values.  Only the forward has the second shape; the back projection reduces
through ``sum_band_to_owner`` either way.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class Placement:
    """Defines how one array type is distributed across devices: the mapping
    from each device to the contiguous block (shard) of the array it owns,
    determined by a sharded axis and a device list.

    The placement also answers "what is on the devices?" for its sharded
    axis: when ``real_size`` (the problem-owned axis length, from the model
    params) is given and does not divide the device count, the DEVICE form of
    the axis is the next multiple of the device count (``padded_size``), with
    the tail zero-filled and kept exactly inert by the model (entry zero-fill
    + masking).  Problem-owned shapes stay in the model params; the padded
    device shape lives only here.  (Verbatim mbirjax semantics; the jax mesh
    and NamedSharding serve its SPMD compiler and have no counterpart in this
    explicit-placement design.)

    Args:
        devices (sequence of torch.device or str): the devices this array
            type lives on.  A single device is the trivial (1-shard)
            placement.
        axis (int): the axis of this array type that is partitioned across
            the devices (may be negative; resolved against an array's rank
            where used).  Recon-like -> the slice axis (-1); sino-like -> the
            view axis (0).
        real_size (int or None): the problem-owned length of the sharded axis
            (e.g. num_views for a sino placement).  When given,
            ``padded_size`` is the device-form length (the smallest multiple
            of the device count >= real_size); when None, padding is
            unknown/unsupported and only the divisible case is valid.
    """

    def __init__(self, devices, axis, real_size=None):
        self.devices = [torch.device(d) for d in devices]
        if len(self.devices) < 1:
            raise ValueError("Placement requires at least one device.")
        self.axis = axis
        self.real_size = int(real_size) if real_size is not None else None
        if self.real_size is None:
            self.padded_size = None
        else:
            n = len(self.devices)
            self.padded_size = ((self.real_size + n - 1) // n) * n

    @property
    def n_devices(self):
        return len(self.devices)

    @property
    def is_trivial(self):
        """True when this placement is a single device (1 shard)."""
        return len(self.devices) == 1

    @property
    def is_padded(self):
        """True when the device form of the sharded axis is longer than the
        problem's real axis (real_size does not divide the device count)."""
        return self.padded_size is not None and self.padded_size > self.real_size

    def real_mask(self, ndim):
        """Broadcastable indicator of the REAL entries of a device-form array
        under this placement.

        Returns None when nothing is padded (the common case -- callers use
        None as "no masking needed").  Otherwise returns a host NumPy boolean
        array of rank ``ndim`` that is ``padded_size`` long on this
        placement's shard axis and 1 elsewhere, True on the real entries and
        False on the zero-filled padding, for excluding padding from
        statistical reductions.
        """
        if not self.is_padded:
            return None
        mask_shape = [1] * ndim
        mask_shape[self.axis % ndim] = self.padded_size
        return (np.arange(self.padded_size) < self.real_size).reshape(mask_shape)

    def shard_ranges(self, size):
        """The half-open axis range each device owns when an axis of length
        ``size`` is split into equal contiguous blocks (one per device).

        Args:
            size (int): the length of the sharded axis to split.  Must be
                divisible by the device count (the sharding contract).

        Returns:
            list of (device, (start, end)): the half-open block owned by each
            device, in device order.
        """
        n = len(self.devices)
        if size % n != 0:
            raise ValueError(
                f"Cannot evenly shard axis of size {size} across {n} devices.")
        block = size // n
        return [(self.devices[i], (i * block, (i + 1) * block)) for i in range(n)]

    def padded_shard_ranges(self):
        """``shard_ranges`` over the device-form (padded) axis length, plus
        each shard's count of REAL (problem-owned) entries.

        Returns:
            list of (device, (start, end), n_valid): the half-open global
            block each device owns on the padded axis, and how many of its
            entries are real (the rest, ``end - start - n_valid``, are
            zero-filled padding at the end of the axis).  Requires
            ``real_size`` to have been given.
        """
        if self.padded_size is None:
            raise ValueError("padded_shard_ranges requires real_size to be set.")
        ranges = self.shard_ranges(self.padded_size)
        return [(dev, (start, end), max(0, min(end, self.real_size) - start))
                for dev, (start, end) in ranges]


class Shards:
    """The device form of a sharded array: one tensor per device plus the
    placement that says which global block each covers.

    This is the explicit-placement stand-in for jax's shard-backed global
    array.  It is
    a plain container -- no arithmetic; the drivers and the VCD loop operate on
    the per-device tensors directly.  ``gather()`` is the host exit
    (:math:`\\to` numpy, concatenated on the sharded axis, padding NOT
    cropped -- the model's _gather_sinogram / _gather_recon own the crop).
    """

    def __init__(self, tensors, placement):
        if len(tensors) != placement.n_devices:
            raise ValueError(
                f"{len(tensors)} shard tensors for {placement.n_devices} devices.")
        self.tensors = list(tensors)
        self.placement = placement
        # Identity check: each shard must live on its placement's device (a
        # cheap catch for a shard built or moved onto the wrong device, which
        # would otherwise surface as a distant cross-device RuntimeError or a
        # silent host-side slowdown).  Unindexed device forms ('cuda') match
        # any index of their type; both-indexed forms must agree.
        for t, d in zip(self.tensors, placement.devices):
            if t.device.type != d.type or (
                    t.device.index is not None and d.index is not None
                    and t.device.index != d.index):
                raise ValueError(
                    f'Shard on {t.device} does not match its placement '
                    f'device {d}.')

    @property
    def dtype(self):
        """The shards' common dtype (from the first shard)."""
        return self.tensors[0].dtype

    def gather(self):
        """Concatenate the shards on the host along the sharded axis."""
        parts = [t.detach().cpu().numpy() for t in self.tensors]
        return np.concatenate(parts, axis=self.placement.axis % parts[0].ndim)


# ── safe transfer (the mbirjax transfer.py port) ──────────────────────────────
# jax's device_put silently corrupted device-resident transfers on some GPUs
# (L40S); torch's tensor.to() has no known analog, but the empirical probe is
# kept as near-free paranoia: we test the actual hardware once per device set
# instead of assuming, and route through host memory if a copy ever fails to
# round-trip.
_warned_host_bounce = False


def is_dev2dev_safe(devices) -> bool:
    """Empirically test that a direct device-to-device copy is correct.

    Moves a small known array from devices[0] to devices[1] and checks the
    value survives.  Returns True for a single device (nothing to move).
    """
    devices = [torch.device(d) for d in devices]
    if len(devices) < 2:
        return True
    probe = torch.tensor([1.0, 2.0, 3.0, 4.0])
    src = probe.to(devices[0])
    dst = src.to(devices[1])
    return bool(torch.equal(dst.cpu(), probe))


def move_shard(x, target, dev2dev_safe=True):
    """Place tensor ``x`` on ``target``, choosing a hardware-safe path.

    The single cross-device transfer primitive.  A move to the tensor's own
    device is a no-op (``tensor.to`` returns self), so the single-device case
    carries no overhead.

    Args:
        x (tensor): possibly resident on another device.
        target (torch.device): destination.
        dev2dev_safe (bool): the cached result of :func:`is_dev2dev_safe` for
            this device set.  When False, route through host memory (always
            correct, one warning per process).
    """
    if dev2dev_safe:
        return x.to(target)
    global _warned_host_bounce
    if not _warned_host_bounce:
        _warned_host_bounce = True
        warnings.warn(
            "Direct device-to-device transfer failed the round-trip probe on "
            "this hardware; routing cross-device transfers through host "
            "memory.  This is correct but slower.", stacklevel=2)
    return torch.as_tensor(x.detach().cpu().numpy()).to(target)


def sum_band_to_owner(partials, owner, dev2dev_safe=True):
    """Move per-device partials onto ``owner`` and sum them there.

    The cross-device reduce used by sharded back projection: under
    view-sharding each device computed only a partial back projection (its
    own views' contribution) for some band of slices; the true value is the
    sum over devices, formed and left resident on the band's slice-owner.
    """
    contribs = [move_shard(p, owner, dev2dev_safe=dev2dev_safe) for p in partials]
    total = contribs[0]
    for c in contribs[1:]:
        total = total + c
    return total


def broadcast_band_to_views(band, view_owners, dev2dev_safe=True):
    """Copy a slice-band cylinder from its slice-owner to every view-owner.

    The adjoint of :func:`sum_band_to_owner`: broadcast (copy to N devices)
    is the transpose of the reduce (sum from N devices), which is what keeps
    forward and back projection adjoint under sharding.

    Returns:
        dict {view_owner: tensor}: the band resident on each view-owner.
    """
    return {dev: move_shard(band, dev, dev2dev_safe=dev2dev_safe)
            for dev in view_owners}


def gather_column_band(shard_tensors, p0, p1, target, dev2dev_safe=True):
    """Gather one batch of pixel columns, at EVERY slice, onto ``target``.

    The forward's second transfer primitive, built from :func:`move_shard`
    exactly as :func:`broadcast_band_to_views` is.  Each slice-owner holds
    the same pixel columns for its own slices, so moving every owner's
    ``[p0:p1]`` rows to one device and concatenating them along the slice
    axis assembles those columns' whole cylinder there.

    This is the cross-device shape a geometry needs when one recon slice
    projects onto a RANGE of detector rows: such a view-owner cannot produce
    any of its own rows from a slice band, because every slice contributes to
    the rows it owns.  It takes a narrow column of pixels at every slice
    instead.  What one gather costs is then set by the width of the column
    batch and not by the device count, which is what makes the shape usable
    at volumes where a whole assembled cylinder would not fit.  A row-aligned
    geometry, which could work from a band, takes the same gather for a
    performance reason instead: what it gets back is a full-width block of
    values, which is the width regime its kernel is efficient in.

    The concatenation is in shard order, which is global slice order, and it
    keeps the device form's padded slice tail rather than trimming it.  The
    tail is held at zero by the model, a zero voxel contributes nothing
    through a projection, and the geometry bodies anchor their z geometry on
    the real slice count from the params rather than on the width of the
    array they are handed -- so the tail is inert, and trimming it would only
    force a non-contiguous copy inside the projector.

    This changes which device assembles which voxels, never which device
    produces which sinogram rows, so it has no adjoint of its own: the back
    projection is untouched and still reduces through
    :func:`sum_band_to_owner`.

    Args:
        shard_tensors (sequence of tensor): the slice-sharded cylinders, each
            (num_pixels, local_slices), in global slice order.
        p0 (int): first pixel column of the batch.
        p1 (int): one past the last pixel column of the batch.
        target (torch.device): the view-owner the cylinder is assembled on.
        dev2dev_safe (bool): forwarded to :func:`move_shard`.

    Returns:
        tensor: (p1 - p0, total_slices) on ``target``.
    """
    pieces = [move_shard(t[p0:p1], target, dev2dev_safe=dev2dev_safe)
              for t in shard_tensors]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


# ── per-device threaded execution (the mbirjax thread_execution.py port) ──────
def device_pool(n):
    """A reusable thread pool for repeated :func:`run_per_device` calls.

    Use as a context manager; pass the pool as ``executor=`` so a loop of
    many per-device fan-outs (e.g. per-band streaming) reuses one pool.
    """
    return ThreadPoolExecutor(max_workers=n)


def run_per_device(devices, worker_fn, executor=None):
    """Run worker_fn once per device, each in its own thread.

    Torch ops need no thread-local default device: tensors carry their
    device, so a torch op dispatches correctly from any thread.  A RAW
    kernel launch does not.  A Triton launch targets the launching thread's
    current CUDA device and that device's current stream, and a fresh
    thread's current device is 0, so a kernel body called from these workers
    must bracket its own launch in ``with torch.cuda.device(...)`` (the
    wrappers in triton_cone.py and triton_parallel.py do; the measured
    defect is in the kernel-sharding findings in the plans repo).
    Results are returned in DEVICE order (result[i] corresponds to
    devices[i]), not completion order.  No synchronization is performed --
    callers that need values materialized (before assembling or reading)
    synchronize explicitly, which lets a caller overlap the next band's
    transfer with the current band's compute.

    A single device short-circuits to a direct call on the calling thread --
    no pool, no thread hop -- so code written uniformly over shards costs a
    plain function call at n=1.

    Args:
        devices (sequence): devices to run on; one thread per device.
        worker_fn (callable): worker_fn(i, device) -> result.
        executor (ThreadPoolExecutor, optional): a pool to reuse (e.g. from
            :func:`device_pool`); when None a private pool is created and
            closed for this call.
    """
    devices = list(devices)
    n = len(devices)
    if n == 1:
        return [worker_fn(0, devices[0])]

    def _run(i):
        return worker_fn(i, devices[i])

    if executor is not None:
        return list(executor.map(_run, range(n)))
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(_run, range(n)))


def exchange_qggmrf_halos(recon_shards, dev2dev_safe=True):
    """The boundary-slice exchange for the sharded qGGMRF prior.

    Each slice-shard's inter-slice term needs the slice immediately beyond
    each of its boundaries: the LAST slice of the shard to its left and the
    FIRST slice of the shard to its right (each a (num_pixels,) column,
    moved to the shard's device).  True volume edges get None, which the
    prior maps to the reflected boundary condition -- so a single shard
    reproduces the single-device result exactly.

    Args:
        recon_shards (Shards): the slice-sharded flat recon, each tensor
            (num_pixels, local_slices).
        dev2dev_safe (bool): forwarded to :func:`move_shard`.

    Returns:
        (left_halos, right_halos): lists in device order, entries (num_pixels,)
        tensors on the receiving shard's device, or None at the true edges.
    """
    tensors = recon_shards.tensors
    devs = recon_shards.placement.devices
    n = len(tensors)
    left = [None] + [move_shard(tensors[i][:, -1].contiguous(), devs[i + 1],
                                dev2dev_safe) for i in range(n - 1)]
    right = [move_shard(tensors[i + 1][:, 0].contiguous(), devs[i],
                        dev2dev_safe) for i in range(n - 1)] + [None]
    return left, right
