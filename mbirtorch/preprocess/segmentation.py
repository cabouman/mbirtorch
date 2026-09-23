import numpy as np
import torch
import mbirtorch.preprocess as mtp
from mbirtorch import _sharding

# Largest number of elements in one chunk of the per-shard histogram passes.
_HISTOGRAM_CHUNK_ELEMENTS = 1 << 24


def _shard_valid_masks(valid_mask, placement, ndim):
    """Split a broadcastable host mask into one piece per shard.

    Returns a list of numpy arrays, or a list of Nones when there is no mask.
    """
    if valid_mask is None:
        return [None] * placement.n_devices
    mask = np.asarray(valid_mask)
    axis = placement.axis % ndim
    pieces = []
    for _dev, (s0, s1) in placement.shard_ranges():
        if mask.shape[axis] == 1:
            pieces.append(mask)
        else:
            sel = [slice(None)] * ndim
            sel[axis] = slice(s0, s1)
            pieces.append(mask[tuple(sel)])
    return pieces


def _shard_chunks(tensor, mask_piece):
    """Yield (chunk, mask_chunk) pairs over the leading axis.

    Each chunk holds at most _HISTOGRAM_CHUNK_ELEMENTS elements.  The mask
    chunk is a torch bool tensor broadcast to the chunk's shape, or None.
    """
    per_row = max(1, int(np.prod(tensor.shape[1:], dtype=np.int64)))
    step = max(1, _HISTOGRAM_CHUNK_ELEMENTS // per_row)
    m = None
    if mask_piece is not None:
        m = torch.as_tensor(np.asarray(mask_piece), dtype=torch.bool,
                            device=tensor.device)
    for i in range(0, tensor.shape[0], step):
        chunk = tensor[i:i + step]
        if m is None:
            yield chunk, None
        else:
            mc = m if m.shape[0] == 1 else m[i:i + step]
            yield chunk, mc.expand(chunk.shape)


def _sharded_masked_histogram(shards, valid_mask, num_bins):
    """Histogram the valid entries of a sharded volume, one shard at a time.

    The volume is never gathered.  Only the per-shard count tables leave a
    device.  The binning here is a truncated float32 multiply, so a few values
    near a bin edge can land in a different bin than np.histogram would choose.

    Returns:
        (hist, bin_edges): host numpy arrays, int64 counts and numpy's edges.

    Raises:
        ValueError: if the valid entries are empty or span a degenerate range.
    """
    masks = _shard_valid_masks(valid_mask, shards.placement,
                               shards.tensors[0].ndim)

    # Pass 1 takes the masked min and max per chunk and combines them on the host.
    lo, hi = np.inf, -np.inf
    for t, mp in zip(shards.tensors, masks):
        for chunk, mc in _shard_chunks(t, mp):
            vals = chunk.reshape(-1) if mc is None else chunk[mc]
            if vals.numel() == 0:
                continue          # The shard is empty, or the mask excludes it.
            lo = min(lo, float(vals.min()))
            hi = max(hi, float(vals.max()))

    # A degenerate range raises rather than binning.  numpy expands a zero
    # width range when it derives edges, so the counts and edges would disagree.
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(
            'The sharded volume has no valid entries to histogram: every '
            'shard was empty or entirely excluded by valid_mask.')
    if hi <= lo:
        raise ValueError(
            f'The valid entries span the degenerate range [{lo}, {hi}] '
            f'(min == max), so there are no intensity classes to separate.  '
            f'Segmentation needs a volume that takes more than one value.')

    # Pass 2 counts each chunk into num_bins buckets on the device and sums the
    # counts on the host.  The value hi lands in the last bin, which is closed.
    hist = np.zeros(num_bins, dtype=np.int64)
    scale = num_bins / (hi - lo)
    for t, mp in zip(shards.tensors, masks):
        for chunk, mc in _shard_chunks(t, mp):
            vals = chunk.reshape(-1) if mc is None else chunk[mc]
            vals = vals[(vals >= lo) & (vals <= hi)]
            if vals.numel() == 0:
                continue
            idx = torch.clamp(((vals - lo) * scale).to(torch.int64),
                              max=num_bins - 1)
            hist += torch.bincount(idx, minlength=num_bins).cpu().numpy()

    # The edges come from np.histogram on an empty array, so the sharded and
    # unsharded paths use the same edge arithmetic.
    edges_dtype = shards.dtype
    empty = np.empty(0, dtype=str(edges_dtype).replace('torch.', ''))
    _, bin_edges = np.histogram(empty, bins=num_bins, range=(float(lo), float(hi)))
    return hist, bin_edges


def _masked_histogram(image, valid_mask, num_bins, xp):
    """Histogram the valid entries of ``image`` over their own min to max range.

    Invalid entries are replaced by a finite sentinel above the range, which
    ``histogram`` then drops.  The constants are typed to ``image.dtype`` so
    that ``where`` does not upcast and double the size of the temporaries.
    """
    inf = xp.asarray(xp.inf, dtype=image.dtype)
    lo = xp.min(xp.where(valid_mask, image, inf))
    hi = xp.max(xp.where(valid_mask, image, -inf))
    sentinel = hi + xp.maximum(xp.abs(hi), xp.asarray(1.0, dtype=image.dtype))  # Finite and above hi.
    # The range endpoints are Python floats.  numpy computes the bin edges in
    # the dtype of the range, and float32 endpoints would shift some edges.
    return xp.histogram(xp.where(valid_mask, image, sentinel), bins=num_bins, range=(float(lo), float(hi)))


def multi_threshold_otsu(image, classes=2, num_bins=1024, valid_mask=None):
    """
    Segment an image into multiple intensity classes using Otsu's method.

    This function computes optimal threshold values that divide an image into the specified
    number of classes by minimizing the intra-class variance. It returns `classes - 1` thresholds
    that can be used to partition the image intensity range into `classes` distinct segments.

    Args:
        image (np.ndarray, torch.Tensor, or Shards):
            Input image of floating-point values.
        classes (int, optional):
            Number of classes to divide the image into. Must be ≥ 2. Defaults to 2.
        num_bins (int, optional):
            Number of bins to use when constructing the image histogram. Defaults to 1024.
        valid_mask (array or None, optional):
            Broadcastable boolean mask, True on the entries to include.  Used e.g. to restrict the
            histogram range and counts to a region of interest.  None includes everything.

    Returns:
        list of float:
            A list of `classes - 1` threshold values, given in increasing order. These thresholds
            can be used to separate the image into `classes` distinct intensity regions.

    Example:
        >>> thresholds = multi_threshold_otsu(image, classes=4)
        >>> # Resulting thresholds will split image into 4 intensity regions
    """
    if classes < 2:
        raise ValueError("Number of classes must be at least 2")

    if num_bins < classes:
        raise ValueError("Number of bins must be at least equal to number of classes")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()

    if isinstance(image, _sharding.Shards):
        hist, bin_edges = _sharded_masked_histogram(image, valid_mask, num_bins)
    elif valid_mask is not None:
        hist, bin_edges = _masked_histogram(image, np.asarray(valid_mask), num_bins, np)
    else:
        hist, bin_edges = np.histogram(image, bins=num_bins, range=(float(np.min(image)), float(np.max(image))))

    thresholds = _otsu_thresholds_dp(hist, classes - 1)

    # bin_edges[t] is the cut for boundary t.  Values below it fall in bins
    # less than t, which are the lower classes.
    scaled_thresholds = [bin_edges[t] for t in thresholds]

    return scaled_thresholds


def _otsu_thresholds_dp(hist, num_thresholds):
    """Find multi-threshold Otsu boundaries by dynamic programming.

    The returned boundaries are half open.  Boundary ``t`` means that bin ``t``
    starts the next class, so the matching threshold value is ``bin_edges[t]``.

    Args:
        hist (ndarray): Histogram counts of the image.
        num_thresholds (int): Number of thresholds to find.

    Returns:
        list of int: strictly increasing boundary indices in ``[1, len(hist) - 1]``.
    """
    if num_thresholds == 0:
        return []

    hist = np.asarray(hist, dtype=np.float64)
    num_bins = len(hist)
    # The bin coordinates are centered and scaled to [-1, 1].  This leaves the
    # thresholds unchanged and keeps the moment prefix sums well conditioned.
    half_span = max((num_bins - 1) / 2.0, 1.0)     # The max guards a one bin histogram.
    bin_coord = (np.arange(num_bins, dtype=np.float64) - (num_bins - 1) / 2.0) / half_span

    # Each prefix sum has a leading zero, so the moment of bins [a, b) is P[b] - P[a].
    m0 = np.concatenate(([0.0], np.cumsum(hist)))
    m1 = np.concatenate(([0.0], np.cumsum(bin_coord * hist)))
    m2 = np.concatenate(([0.0], np.cumsum(bin_coord * bin_coord * hist)))

    # Entry [a, b] of each outer difference is the moment of bins [a, b).
    int_m0 = m0[None, :] - m0[:, None]
    int_m1 = m1[None, :] - m1[:, None]
    int_m2 = m2[None, :] - m2[:, None]

    # The within class cost of bins [a, b) is M2 - M1^2/M0, and an empty interval costs
    # zero.  Entries with a >= b are infinite, so the boundaries must increase.
    mean_sq_term = np.divide(int_m1 ** 2, int_m0, out=np.zeros_like(int_m0), where=int_m0 > 0)
    cost = np.maximum(int_m2 - mean_sq_term, 0.0)      # Clip negative rounding residue.
    invalid = ~np.triu(np.ones((num_bins + 1, num_bins + 1), dtype=bool), k=1)   # a >= b
    cost[invalid] = np.inf

    # best[b] is the least cost of covering bins [0, b) with the classes so far.
    # Each stage adds one class and records the argmin for the backtrack.
    best = cost[0, :].copy()                           # One class covering [0, b).
    split_of = np.zeros((num_thresholds, num_bins + 1), dtype=np.int64)
    for stage in range(num_thresholds):
        total = best[:, None] + cost                   # total[s, b]
        split_of[stage] = np.argmin(total, axis=0)
        best = np.min(total, axis=0)

    # The backtrack starts at b = num_bins and walks the stages in reverse.
    boundaries = []
    b = num_bins
    for stage in range(num_thresholds - 1, -1, -1):
        b = int(split_of[stage][b])
        boundaries.append(b)
    return boundaries[::-1]


def segment_plastic_metal(recon, num_metal, radial_margin=None, top_margin=None, bottom_margin=None):
    """
    Segment a reconstruction into plastic and multiple metal masks using multi-threshold Otsu.

    ``recon`` may be a host numpy array, a torch tensor, or a sharded volume (a ``Shards``
    container); the class masks are returned in the same form as the input, on the same
    devices.  A sharded volume is processed shard by shard where it sits: only the small
    histogram tables travel, never the volume.

    Args:
        recon (np.ndarray, torch.Tensor, or Shards): Reconstructed volume.
        num_metal (int): Number of metal materials to segment.
        radial_margin (int or None, optional): Margin in pixels to subtract from the cylindrical mask
            radius.  None (default) uses a size-relative margin, max(2, min(10, min(rows, cols) // 25)):
            identical to the former fixed 10 for volumes 250 pixels and wider, and proportionally
            smaller for small volumes (where a fixed 10 would cut real object).
        top_margin (int or None, optional): Number of slices to mask out from the top of the volume.
            None (default) uses max(2, min(10, num_slices // 25)) with the same rationale.
        bottom_margin (int or None, optional): Number of slices to mask out from the bottom.
            None (default) as for top_margin.

    Returns:
        tuple: ``(plastic_mask, metal_masks, plastic_scale, metal_scales)``.  Each mask has the same
        type as ``recon``: numpy in gives numpy out, tensor in gives tensor out, ``Shards`` in gives
        ``Shards`` out.

            - plastic_mask (np.ndarray, torch.Tensor, or Shards): Binary mask for plastic regions.
            - metal_masks (list): One binary mask per metal region, each in the same form as ``recon``.
            - plastic_scale (float): Scaling factor for the plastic region.
            - metal_scales (list of float): One scaling factor per metal region.
    """
    if num_metal <= 0:
        raise ValueError("num_metal must be positive")

    is_shards = isinstance(recon, _sharding.Shards)
    if is_shards and recon.placement.is_trivial:
        # There is one shard, so the tensor path handles it.
        pl = recon.placement
        plastic_mask, metal_masks, plastic_scale, metal_scales = segment_plastic_metal(
            recon.tensors[0], num_metal, radial_margin=radial_margin,
            top_margin=top_margin, bottom_margin=bottom_margin)
        return (_sharding.Shards([plastic_mask], pl),
                [_sharding.Shards([m], pl) for m in metal_masks],
                plastic_scale, metal_scales)

    # The default margins scale with the volume, so the mask never removes a
    # large fraction of a small field of view.
    shape = recon.tensors[0].shape if is_shards else recon.shape
    num_slices = recon.placement.axis_len if is_shards else recon.shape[2]
    if radial_margin is None:
        radial_margin = max(2, min(10, min(shape[0], shape[1]) // 25))
    if top_margin is None:
        top_margin = max(2, min(10, num_slices // 25))
    if bottom_margin is None:
        bottom_margin = max(2, min(10, num_slices // 25))

    # The mask removes flash at the boundary of the recon.
    recon = mtp.apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=top_margin,
                                       bottom_margin=bottom_margin)

    thresholds = multi_threshold_otsu(recon, classes=num_metal + 2)

    is_torch = isinstance(recon, torch.Tensor)

    # Plastic is the lowest class.
    plastic_low_threshold = thresholds[0]
    plastic_metal_threshold = thresholds[1]

    def _tensor_class_mask(vol, lower, upper):
        return ((vol > lower) & (vol <= upper)).to(vol.dtype)

    def class_mask(lower, upper):
        if is_shards:
            # The mask is built on each shard's own device and comes back
            # sharded the same way as the volume.
            return _sharding.Shards(
                [_tensor_class_mask(t, lower, upper) for t in recon.tensors],
                recon.placement)
        if is_torch:
            return _tensor_class_mask(recon, lower, upper)
        in_class = (recon > lower) & (recon <= upper)
        return np.where(in_class, np.float32(1.0), np.float32(0.0)).astype(recon.dtype)

    plastic_mask = class_mask(plastic_low_threshold, plastic_metal_threshold)
    plastic_scale = mtp.compute_scaling_factor(recon, plastic_mask)

    metal_masks = []
    metal_scales = []
    for i in range(1, num_metal + 1):
        lower = thresholds[i]
        upper = thresholds[i + 1] if i + 1 < len(thresholds) else np.inf
        metal_mask = class_mask(lower, upper)
        metal_masks.append(metal_mask)
        metal_scales.append(mtp.compute_scaling_factor(recon, metal_mask))

    return plastic_mask, metal_masks, plastic_scale, metal_scales
