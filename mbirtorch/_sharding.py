"""Device placement, safe transfer, and per-device threaded execution.
A single process drives every device, bands move device-to-device, and one
python thread per device issues the work -- the
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
view-shard and never moves).  The two directions cut a different axis of the
cylinder.

The back projection cuts the SLICE axis.  ``sum_band_to_owner`` (a
reduce-scatter) sums each view-owner's band partials onto the band's
slice-owner.

The forward projection cuts the PIXEL axis.  ``transfer_cylinder_batch``
assembles one batch of full-height voxel cylinders on one view-owner.  A
geometry whose slices project onto a range of detector rows needs the whole
slice axis before it can produce any of its own rows, so a slice band buys it
nothing.  A row-aligned geometry could produce its rows from a band, and takes
the same cylinder transfer because its kernel is markedly faster on the wider
block of values.
"""

import contextlib
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class Placement:
    """Defines how one array type is distributed across devices: the mapping
    from each device to the contiguous block (shard) of the array it owns,
    determined by a sharded axis and a device list.

    The device form of the sharded axis is exactly as long as the problem's
    own axis.  Each device owns one contiguous block, the block lengths
    differ by at most one, and the longer blocks come first.  A device count
    larger than the axis length leaves the trailing devices with empty
    blocks.  The axis is never padded: the device form is a list of
    per-device tensors, so blocks of unequal length are allowed.

    Args:
        devices (sequence of torch.device or str): the devices this array
            type lives on.  A single device is the trivial (1-shard)
            placement.
        axis (int): the axis of this array type that is partitioned across
            the devices (may be negative; resolved against an array's rank
            where used).  Recon-like -> the slice axis (-1); sino-like -> the
            view axis (0).
        axis_len (int or None): the length of the sharded axis (e.g.
            num_views for a sino placement).  When given, it is the axis
            length :meth:`shard_ranges` splits by default.
    """

    def __init__(self, devices, axis, axis_len=None):
        self.devices = [torch.device(d) for d in devices]
        if len(self.devices) < 1:
            raise ValueError("Placement requires at least one device.")
        self.axis = axis
        self.axis_len = int(axis_len) if axis_len is not None else None

    @property
    def n_devices(self):
        return len(self.devices)

    @property
    def is_trivial(self):
        """True when this placement is a single device (1 shard)."""
        return len(self.devices) == 1

    def shard_ranges(self, axis_len=None):
        """The half-open axis range each device owns when an axis of length
        ``axis_len`` is split into contiguous blocks, one per device.

        The block lengths differ by at most one, and the longer blocks come
        first.  That is ``numpy.array_split``'s convention, which the index
        arithmetic below uses directly, and it matches the convention the
        slice bands already follow.  An axis length smaller than the device
        count gives the trailing devices empty ranges.

        Args:
            axis_len (int, optional): the length of the sharded axis to
                split.  Defaults to this placement's ``axis_len``.

        Returns:
            list of (device, (start, end)): the half-open block owned by each
            device, in device order.
        """
        if axis_len is None:
            axis_len = self.axis_len
        if axis_len is None:
            raise ValueError(
                'shard_ranges needs an axis length.  This placement was built '
                'without axis_len, so pass the length explicitly, as in '
                'shard_ranges(num_slices).')
        blocks = np.array_split(np.arange(int(axis_len)), len(self.devices))
        bounds = np.cumsum([0] + [len(b) for b in blocks])
        return [(dev, (int(bounds[i]), int(bounds[i + 1])))
                for i, dev in enumerate(self.devices)]


class Shards:
    """The device form of a sharded array: one tensor per device plus the
    placement that says which global block each covers.

    This is the explicit-placement stand-in for jax's shard-backed global
    array.  It is
    a plain container -- no arithmetic; the drivers and the VCD loop operate on
    the per-device tensors directly.  ``gather()`` is the host exit
    (:math:`\\to` numpy, concatenated on the sharded axis).
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


# ── safe transfer ─────────────────────────────────────────────────────────────
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


#: How many bytes of one arriving partial the reduce moves at a time.  A
#: REASONED default, not a measured knee -- the cluster measurement of this
#: change comes after it.  The slab has to be large enough that the fixed
#: cost of one step (a python call, one device-to-device copy, one add) stays
#: small beside the step's own work: at 64 MiB the copy and the add take
#: hundreds of microseconds on any device-to-device link, against tens of
#: microseconds of launch and dispatch, so the host stays well ahead of the
#: devices and the streaming overhead is a few percent of the reduce.  And it
#: has to be small enough to be negligible beside the band it streams: a
#: production band is gigabytes, so the slab is well under one percent of it.
#: A band smaller than one slab moves in a single piece, which is exactly
#: what the reduce did before, so nothing changes at small sizes.
REDUCE_SLAB_BYTES = 64 * 2 ** 20


def reduce_slab_rows(num_rows, row_bytes):
    """How many rows of a band partial :func:`sum_band_to_owner` moves per
    step, given the bytes in one row of it.

    Shared with the memory ledger, which prices the transient this bounds:
    the size the code moves and the size the model charges must not be able
    to drift apart.
    """
    # Never zero: the answer is a loop step, and a step of zero is an error
    # even where the range it walks is empty.
    if row_bytes <= 0:
        return max(1, int(num_rows))
    return max(1, min(int(num_rows), int(REDUCE_SLAB_BYTES) // int(row_bytes)))


def sum_band_to_owner(partials, owner, dev2dev_safe=True):
    """Move per-device partials onto ``owner`` and sum them there.

    The cross-device reduce used by sharded back projection: under
    view-sharding each device computed only a partial back projection (its
    own views' contribution) for some band of slices; the true value is the
    sum over devices, formed and left resident on the band's slice-owner.

    The sum is STREAMED in row slabs, and that is what bounds the owner's
    peak.  Moving every partial across first and then summing them held n
    whole bands on the owner at once, and because one band is the whole shard
    by default, that transient did not shrink as devices were added: n
    devices each holding a band of 1/n of the volume is the same number of
    bytes at every device count.  Streaming leaves the owner holding its
    running total and one bounded slab per source instead, so what it holds
    ABOVE the total is a fixed number of bytes rather than a share of the
    volume.

    The summation order is unchanged.  Every element is still accumulated in
    the order the partials are given, so the result is bit for bit what the
    unstreamed reduce produced: streaming partitions the elements, and no
    element's own sequence of additions is touched.

    The partials are read and never written.  The first one is copied (or
    moved) to make the running total, so a caller may still use its arrays
    after the call.

    Args:
        partials (list of tensor): one band partial per contributing device,
            all of the same shape, summed in the order given.
        owner (torch.device): the band's slice-owner, where the sum is formed
            and left resident.
        dev2dev_safe (bool): forwarded to :func:`move_shard`.
    """
    if len(partials) == 1:
        return move_shard(partials[0], owner, dev2dev_safe=dev2dev_safe)
    total = move_shard(partials[0], owner, dev2dev_safe=dev2dev_safe)
    if total is partials[0]:
        # The first partial already lives on the owner, so move_shard handed
        # back the caller's own tensor.  Accumulate into a copy of it rather
        # than writing through to an array the caller still holds.
        total = total.clone()
    num_rows = int(total.shape[0])
    row_bytes = (total.numel() // max(1, num_rows)) * total.element_size()
    step = reduce_slab_rows(num_rows, row_bytes)
    for start in range(0, num_rows, step):
        stop = min(start + step, num_rows)
        # Rows, not slices: a partial is (pixels, slices) with the slices
        # contiguous, so a block of ROWS is a contiguous piece and each
        # transfer stays a single flat copy.  Every source's transfer for
        # this slab is issued BEFORE any of them is consumed, so copies from
        # different devices still overlap each other the way they did when
        # whole bands were moved up front.
        slabs = [move_shard(p[start:stop], owner, dev2dev_safe=dev2dev_safe)
                 for p in partials[1:]]
        rows = total[start:stop]
        for slab in slabs:
            rows.add_(slab)
        # Released here: the next iteration's list comprehension is evaluated
        # BEFORE `slabs` is rebound, so without this the previous slabs stay
        # live on the owner through the next slab's transfers, doubling the
        # very transient this loop exists to bound.
        slabs = None
    return total


def transfer_cylinder_batch(shard_tensors, p0, p1, target, dev2dev_safe=True):
    """Assemble one batch of FULL-HEIGHT voxel cylinders on ``target``.

    The forward's transfer primitive, built from :func:`move_shard`.  Each
    slice-owner holds the same pixels for its own slices, so moving every
    owner's ``[p0:p1]`` rows to one device and concatenating them along the
    slice axis assembles those pixels' whole cylinders there.

    This is the cross-device shape a geometry needs when one recon slice
    projects onto a RANGE of detector rows: such a view-owner cannot produce
    any of its own rows from a slice band, because every slice contributes to
    the rows it owns.  It takes a batch of whole cylinders instead.  What one
    transfer costs is then set by the width of the pixel batch and not by the
    device count, which is what makes the shape usable at volumes where the
    whole cylinder array would not fit.  A row-aligned geometry, which could
    work from a band, takes the same transfer for a performance reason
    instead: what it gets back is a full-width block of values, which is the
    width regime its kernel is efficient in.

    The concatenation is in shard order, which is global slice order, so each
    assembled cylinder covers the whole slice axis exactly once.  The shards
    may differ in length, and a shard that owns no slices contributes a
    zero-width piece, which the concatenation accepts.

    This changes which device assembles which voxels, never which device
    produces which sinogram rows, so it has no adjoint of its own: the back
    projection is untouched and still reduces through
    :func:`sum_band_to_owner`.

    Args:
        shard_tensors (sequence of tensor): the slice-sharded cylinders, each
            (num_pixels, local_slices), in global slice order.
        p0 (int): first pixel of the batch.
        p1 (int): one past the last pixel of the batch.
        target (torch.device): the view-owner the cylinders are assembled on.
        dev2dev_safe (bool): forwarded to :func:`move_shard`.

    Returns:
        tensor: (p1 - p0, total_slices) on ``target``.
    """
    pieces = [move_shard(t[p0:p1], target, dev2dev_safe=dev2dev_safe)
              for t in shard_tensors]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


# ── copy streams for the cylinder transfer (CUDA only) ───────────────────────
# One extra CUDA stream per device, used for nothing but the cylinder
# transfer's cross-device copies.  A stream runs its work in the order it was
# given, one item at a time, so copies left on the stream a device projects on
# can only take turns with those projections however early they are issued --
# torch issues a cross-device copy on the SOURCE device's current stream and
# orders the DESTINATION device's current stream behind it, and for the
# transfer's worker threads both of those are the default stream the device
# projects on.  A stream of their own is what lets a copy and a projection run
# at once.
#
# Cached per device index and created once, the way projectors.py caches its
# compiled bodies: the lock is taken only to CREATE a stream, so the worker
# threads that ask for one every batch find it already there and stay
# lock-free.
_COPY_STREAMS = {}
_COPY_STREAM_LOCK = threading.Lock()


def copy_stream(device):
    """The dedicated copy stream for ``device``, or None when it has none.

    None is returned for every non-CUDA device, and it is the signal the
    callers below read as "this device has no streams to arrange": each of
    them then does the plain synchronous thing, which is what the CPU and MPS
    paths have always done.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    index = (device.index if device.index is not None
             else torch.cuda.current_device())
    stream = _COPY_STREAMS.get(index)
    if stream is None:
        with _COPY_STREAM_LOCK:
            stream = _COPY_STREAMS.get(index)
            if stream is None:
                stream = torch.cuda.Stream(device=index)
                _COPY_STREAMS[index] = stream
    return stream


def _transfer_stream_devices(shard_tensors, target):
    """The distinct CUDA devices one transfer touches: every shard's device
    and the target it assembles on.  Ordered by device index so that the
    nested stream contexts are always entered in the same order."""
    seen = {}
    for dev in [t.device for t in shard_tensors] + [torch.device(target)]:
        if dev.type == 'cuda':
            index = (dev.index if dev.index is not None
                     else torch.cuda.current_device())
            seen[index] = torch.device('cuda', index)
    return [seen[index] for index in sorted(seen)]


def open_copy_streams(devices):
    """Let the copy streams start: each waits for its device's compute stream.

    The shards a transfer reads were written by earlier kernels on the compute
    stream, and a copy stream knows nothing of that stream's ordering, so
    without this a copy could read a shard before the kernel that filled it
    had finished.  Called once per forward rather than per batch: it orders
    the copy stream behind everything queued so far, which covers every batch
    that follows.
    """
    for dev in devices:
        stream = copy_stream(dev)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(torch.device(dev)))


def close_copy_streams(devices):
    """The other half of :func:`open_copy_streams`: each compute stream waits
    for its copy stream.

    A copy READS a slice-owner's shard, and whatever writes that shard next
    runs on the compute stream.  Nothing else orders those two, so without
    this a later update could overwrite a shard while a copy was still
    reading it.
    """
    for dev in devices:
        stream = copy_stream(dev)
        if stream is not None:
            torch.cuda.current_stream(torch.device(dev)).wait_stream(stream)


def transfer_cylinder_batch_async(shard_tensors, p0, p1, target,
                                  dev2dev_safe=True):
    """:func:`transfer_cylinder_batch`, issued on the copy streams.

    The values are the same either way; what this adds is that the copies do
    not go into the queue the projections run in, so one transfer can be
    moving while an earlier batch is projected.

    Returns:
        (tensor, ready): the assembled cylinder batch, and an event that fires
        once its copies have landed -- or None for the event off CUDA, where
        the copies are already finished by the time this returns.
    """
    stream = copy_stream(target)
    if stream is None:
        return transfer_cylinder_batch(shard_tensors, p0, p1, target,
                                       dev2dev_safe), None
    # BOTH ends of every copy have to be on a copy stream: torch issues the
    # copy on the source's current stream and orders the destination's current
    # stream behind it, so leaving either end on its default stream would put
    # the copy straight back in the queue the projections run in.
    with contextlib.ExitStack() as stack:
        for dev in _transfer_stream_devices(shard_tensors, target):
            stack.enter_context(torch.cuda.stream(copy_stream(dev)))
        cylinder = transfer_cylinder_batch(shard_tensors, p0, p1, target,
                                           dev2dev_safe)
        ready = torch.cuda.Event()
        ready.record(stream)
    # The cylinder batch was allocated on the copy stream and is read on the
    # compute stream.  Without this the caching allocator would be free to hand
    # its block to the next transfer the moment python drops the name, while
    # the projection was still reading it.  This covers the arriving pieces
    # too: they are allocated and concatenated on the one copy stream, and the
    # only one that ever escapes is the single-shard case, where the piece IS
    # the cylinder batch returned here.
    cylinder.record_stream(torch.cuda.current_stream(torch.device(target)))
    return cylinder, ready


def wait_for_cylinder_batch(target, ready):
    """Hold ``target``'s compute stream until one batch's copies have landed.

    The event is per batch and is waited on immediately before the projection
    that reads that batch.  Waiting on the copy stream as a whole instead
    would also wait for the batch transferred ahead, which is exactly the work
    meant to be moving during this projection, and the overlap would collapse
    back into taking turns.
    """
    if ready is not None:
        torch.cuda.current_stream(torch.device(target)).wait_event(ready)


# ── per-device threaded execution ─────────────────────────────────────────────
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

    A shard that holds no slices counts as absent.  Such a shard gets None
    on both of its sides.  The neighbor that does hold slices also gets None
    in that direction, so the prior applies the reflected boundary condition
    at the last real slice.

    Args:
        recon_shards (Shards): the slice-sharded flat recon, each tensor
            (num_pixels, local_slices).
        dev2dev_safe (bool): forwarded to :func:`move_shard`.

    Returns:
        (left_halos, right_halos): lists in device order, entries (num_pixels,)
        tensors on the receiving shard's device, or None at a true edge and at
        a boundary with a shard that holds no slices.
    """
    tensors = recon_shards.tensors
    devs = recon_shards.placement.devices
    n = len(tensors)
    # A boundary carries a halo only when the shards on both of its sides
    # hold slices.  A shard with no slices comes last, because the slice axis
    # is split with the longer blocks first, so no halo ever has to come
    # from beyond one.
    joined = [t.shape[1] > 0 and u.shape[1] > 0
              for t, u in zip(tensors[:-1], tensors[1:])]
    left = [None] + [move_shard(tensors[i][:, -1].contiguous(), devs[i + 1],
                                dev2dev_safe) if joined[i] else None
                     for i in range(n - 1)]
    right = [move_shard(tensors[i + 1][:, 0].contiguous(), devs[i],
                        dev2dev_safe) if joined[i] else None
             for i in range(n - 1)] + [None]
    return left, right
