import numpy as np
import torch
import mbirtorch.preprocess as mtp
from mbirtorch import _sharding

# Chunk bound for the per-shard histogram passes, in elements: keeps each
# temporary well under a GB and each chunk's counts exact in int64.
_HISTOGRAM_CHUNK_ELEMENTS = 1 << 24


def _shard_valid_masks(valid_mask, placement, ndim):
    """Split a broadcastable host mask into per-shard pieces: the placement's
    axis is sliced to each shard's range; size-1 (broadcast) axes stay whole.
    Returns a list of numpy arrays, or Nones when there is no mask."""
    if valid_mask is None:
        return [None] * placement.n_devices
    mask = np.asarray(valid_mask)
    axis = placement.axis % ndim
    pieces = []
    for _dev, (s0, s1) in placement.shard_ranges(placement.padded_size):
        if mask.shape[axis] == 1:
            pieces.append(mask)
        else:
            sel = [slice(None)] * ndim
            sel[axis] = slice(s0, s1)
            pieces.append(mask[tuple(sel)])
    return pieces


def _shard_chunks(tensor, mask_piece):
    """Yield (chunk, mask_chunk) pairs over the leading axis, each chunk at
    most _HISTOGRAM_CHUNK_ELEMENTS elements.  The mask chunk is a torch bool
    tensor broadcast to the chunk's shape, or None."""
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
    """Histogram of the valid entries of a sharded volume, computed shard by
    shard on each shard's own device.  Only the small per-shard count tables
    ever leave a device; the volume is never gathered.

    A histogram is a sum of counts, so summing the per-shard tables gives
    exactly the histogram of the whole volume (counts are exact integers).
    The range is the masked min/max, combined across shards on the host.

    Returns:
        (hist, bin_edges): host numpy arrays, matching the single-array path
        (int64 counts; edges computed by np.histogram's own edge arithmetic).
    """
    masks = _shard_valid_masks(valid_mask, shards.placement,
                               shards.tensors[0].ndim)

    # Pass 1: masked min and max per chunk, combined on the host.
    lo, hi = np.inf, -np.inf
    for t, mp in zip(shards.tensors, masks):
        for chunk, mc in _shard_chunks(t, mp):
            vals = chunk.reshape(-1) if mc is None else chunk[mc]
            if vals.numel() == 0:
                continue          # a shard that is entirely padding
            lo = min(lo, float(vals.min()))
            hi = max(hi, float(vals.max()))

    # Pass 2: count per chunk into num_bins buckets (exact int64 on device),
    # summed on the host.  Values are mapped to buckets by the same linear
    # map np.histogram uses; hi lands in the last (closed) bin.
    hist = np.zeros(num_bins, dtype=np.int64)
    scale = num_bins / (hi - lo) if hi > lo else 0.0
    for t, mp in zip(shards.tensors, masks):
        for chunk, mc in _shard_chunks(t, mp):
            vals = chunk.reshape(-1) if mc is None else chunk[mc]
            vals = vals[(vals >= lo) & (vals <= hi)]
            if vals.numel() == 0:
                continue
            idx = torch.clamp(((vals - lo) * scale).to(torch.int64),
                              max=num_bins - 1)
            hist += torch.bincount(idx, minlength=num_bins).cpu().numpy()

    # Edges from np.histogram itself (on an empty array), so both paths use
    # numpy's exact edge arithmetic.
    edges_dtype = shards.dtype
    empty = np.empty(0, dtype=str(edges_dtype).replace('torch.', ''))
    _, bin_edges = np.histogram(empty, bins=num_bins, range=(float(lo), float(hi)))
    return hist, bin_edges


def _masked_histogram(image, valid_mask, num_bins, xp):
    """Histogram of the valid entries of ``image``, matching ``histogram(image, num_bins,
    range=(min, max))`` computed over the valid entries only.

    The range comes from masked min/max, and invalid entries are pushed to a finite sentinel ABOVE the
    range, which ``histogram``'s range semantics then drop -- so the bin EDGES and counts are exactly
    those of the valid entries (no post-hoc count correction needed).  ``xp`` is the array module
    (currently always ``np``, on the host).  The infinity/one constants are typed to ``image.dtype``
    so the ``where`` cannot
    silently upcast (a float64 scalar would double the full-size temporaries).
    """
    inf = xp.asarray(xp.inf, dtype=image.dtype)
    lo = xp.min(xp.where(valid_mask, image, inf))
    hi = xp.max(xp.where(valid_mask, image, -inf))
    sentinel = hi + xp.maximum(xp.abs(hi), xp.asarray(1.0, dtype=image.dtype))  # finite, strictly > hi
    # Python-float range endpoints: numpy computes the bin edges in the RANGE's dtype, so float
    # endpoints give the f64-computed, image-dtype-cast edges, whereas f32 endpoints would shift some
    # edges by a few ULP.
    return xp.histogram(xp.where(valid_mask, image, sentinel), bins=num_bins, range=(float(lo), float(hi)))


def multi_threshold_otsu(image, classes=2, num_bins=1024, valid_mask=None):
    """
    Segment an image into multiple intensity classes using Otsu's method.

    This function computes optimal threshold values that divide an image into the specified
    number of classes by minimizing the intra-class variance. It returns `classes - 1` thresholds
    that can be used to partition the image intensity range into `classes` distinct segments.

    The histogram is computed on the host; a torch tensor input is gathered to the host first.

    Args:
        image (np.ndarray or torch.Tensor):
            Input image of floating-point values.
        classes (int, optional):
            Number of classes to divide the image into. Must be ≥ 2. Defaults to 2.
        num_bins (int, optional):
            Number of bins to use when constructing the image histogram. Defaults to 1024.
        valid_mask (array or None, optional):
            Broadcastable boolean mask, True on the entries to include.  Used e.g. to exclude the
            zero-padded entries of a padded volume so the histogram range and counts match the
            unpadded volume exactly.  None includes everything.

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

    # Compute the histogram of the valid entries.
    if isinstance(image, _sharding.Shards):
        # Sharded volume: histogram shard by shard on each shard's device.
        hist, bin_edges = _sharded_masked_histogram(image, valid_mask, num_bins)
    elif valid_mask is not None:
        hist, bin_edges = _masked_histogram(image, np.asarray(valid_mask), num_bins, np)
    else:
        # Python-float range endpoints for the same reason as in _masked_histogram (edge consistency).
        hist, bin_edges = np.histogram(image, bins=num_bins, range=(float(np.min(image)), float(np.max(image))))

    # Find the optimal thresholds (half-open class-boundary bin indices) by dynamic programming
    thresholds = _otsu_thresholds_dp(hist, classes - 1)

    # Convert boundary indices to image values.  bin_edges[t] is the exact cut for boundary t: values
    # below it fall precisely in bins < t (the lower classes), matching the histogram split.
    scaled_thresholds = [bin_edges[t] for t in thresholds]

    return scaled_thresholds


def _otsu_thresholds_dp(hist, num_thresholds):
    """
    Multi-threshold Otsu via dynamic programming.

    Otsu's criterion minimizes the total within-class variance: for thresholds t_1 < ... < t_k, class
    ``c`` spans the bin interval ``[t_c, t_{c+1})`` (with t_0 = 0 and t_{k+1} = num_bins) and the
    objective is

        sum over classes of   sum_{i in class} (i - class mean)^2 * hist[i].

    The objective is separable over the classes, so the optimal thresholds solve the classic 1-D
    segmentation DP

        D[c][b] = min over s < b of  D[c-1][s] + cost(s, b),

    where ``cost(a, b)`` is the within-class term of a single class spanning bins ``[a, b)``.  The cost
    is O(1) from prefix sums of the histogram's zeroth/first/second moments, each DP stage is one
    vectorized (B+1)^2 min-reduction, and the thresholds come from an argmin backtrack: O(k B^2)
    float64 NumPy with no recursion and no per-bin Python loops, exact for every ``num_thresholds``.

    Threshold convention: returned values are half-open class boundaries -- threshold ``t`` means bin
    ``t`` starts the next class.  The consistent threshold VALUE is therefore the left bin edge
    ``bin_edges[t]``: values below it fall exactly in bins < t (the lower classes).

    Args:
        hist (ndarray): Histogram of the image (counts; any nonnegative dtype).
        num_thresholds (int): Number of thresholds to find (k = classes - 1).

    Returns:
        list of int: strictly increasing boundary indices in ``[1, len(hist) - 1]``.
    """
    if num_thresholds == 0:
        return []

    hist = np.asarray(hist, dtype=np.float64)
    num_bins = len(hist)
    # Bin coordinates centered and scaled to [-1, 1].  The within-class variance is shift-invariant,
    # and a uniform scale multiplies EVERY interval cost by the same factor, so the DP's argmin -- and
    # hence the returned thresholds -- is unchanged in exact arithmetic.  Centering removes the
    # mean-offset cancellation in the prefix-difference arithmetic below; scaling keeps the moment
    # prefix sums O(total count) instead of O(count * bins^2) (~1e15 for a 1e9-voxel volume), so all
    # magnitudes stay comfortably conditioned in float64.
    half_span = max((num_bins - 1) / 2.0, 1.0)     # max() guards the degenerate 1-bin histogram
    bin_coord = (np.arange(num_bins, dtype=np.float64) - (num_bins - 1) / 2.0) / half_span

    # Moment prefix sums, each with a leading 0 so that P[j] = (sum over bins i < j) and the moment of
    # any bin interval [a, b) is P[b] - P[a].  m0 = counts, m1 = first moment, m2 = second moment.
    m0 = np.concatenate(([0.0], np.cumsum(hist)))
    m1 = np.concatenate(([0.0], np.cumsum(bin_coord * hist)))
    m2 = np.concatenate(([0.0], np.cumsum(bin_coord * bin_coord * hist)))

    # Moments of every candidate class interval at once: outer differences, entry [a, b] = P[b] - P[a]
    # = the moment of bins [a, b).
    int_m0 = m0[None, :] - m0[:, None]
    int_m1 = m1[None, :] - m1[:, None]
    int_m2 = m2[None, :] - m2[:, None]

    # Within-class cost of the interval [a, b): expanding sum (i - mean)^2 h_i with mean = M1/M0 gives
    # M2 - M1^2/M0.  Empty (zero-count) intervals cost 0 (an empty class contributes no variance);
    # structurally invalid entries (a >= b) are +inf so the argmin can never produce non-increasing
    # boundaries.
    mean_sq_term = np.divide(int_m1 ** 2, int_m0, out=np.zeros_like(int_m0), where=int_m0 > 0)
    cost = np.maximum(int_m2 - mean_sq_term, 0.0)      # clip tiny negative rounding residue
    invalid = ~np.triu(np.ones((num_bins + 1, num_bins + 1), dtype=bool), k=1)   # a >= b
    cost[invalid] = np.inf

    # DP stages: best[b] = minimal cost of covering bins [0, b) with the current number of classes.
    # Each stage adds one class: total[s, b] = (best cover of [0, s) so far) + (one new class [s, b));
    # the argmin over s is recorded per b for the backtrack.
    best = cost[0, :].copy()                           # one class: [0, b)
    split_of = np.zeros((num_thresholds, num_bins + 1), dtype=np.int64)
    for stage in range(num_thresholds):
        total = best[:, None] + cost                   # total[s, b]
        split_of[stage] = np.argmin(total, axis=0)
        best = np.min(total, axis=0)

    # Backtrack from the full range [0, num_bins): the last stage's argmin at b = num_bins is the last
    # threshold; each recovered threshold then indexes the previous stage's argmin row.
    boundaries = []
    b = num_bins
    for stage in range(num_thresholds - 1, -1, -1):
        b = int(split_of[stage][b])
        boundaries.append(b)
    return boundaries[::-1]


def segment_plastic_metal(recon, num_metal, radial_margin=None, top_margin=None, bottom_margin=None,
                          valid_mask=None, num_real_slices=None):
    """
    Segment a reconstruction into plastic and multiple metal masks using multi-threshold Otsu.

    ``recon`` may be a host numpy array, a torch tensor, or a sharded volume (a ``Shards``
    container); the class masks are returned in the same form as the input, on the same
    devices.  A sharded volume is processed shard by shard where it sits: only the small
    histogram tables travel, never the volume.  For a volume whose slice axis is
    zero-padded, pass ``valid_mask`` / ``num_real_slices`` so the padded slices are
    excluded from the histogram, the bottom margin lands on the real bottom slices, and
    the class masks are zero on padding (otherwise a threshold interval spanning 0 would
    include the padded voxels and bias the scaling factors).

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
        valid_mask (array or None, optional): Broadcastable boolean mask, True on real voxels.
        num_real_slices (int or None, optional): Real slice count (see ``apply_cylindrical_mask``).

    Returns:
        Tuple[ndarray, List[ndarray], float, List[float]]:
            - plastic_mask (ndarray): Binary mask for plastic regions.
            - metal_masks (List[ndarray]): List of binary masks for each metal region.
            - plastic_scale (float): Scaling factor for plastic region.
            - metal_scales (List[float]): List of scaling factors for each metal region.
    """
    if num_metal <= 0:
        raise ValueError("num_metal must be positive")

    is_shards = isinstance(recon, _sharding.Shards)
    if is_shards and recon.placement.is_trivial:
        # One shard: compute through the tensor path, return in the form given.
        pl = recon.placement
        plastic_mask, metal_masks, plastic_scale, metal_scales = segment_plastic_metal(
            recon.tensors[0], num_metal, radial_margin=radial_margin,
            top_margin=top_margin, bottom_margin=bottom_margin,
            valid_mask=valid_mask, num_real_slices=num_real_slices)
        return (_sharding.Shards([plastic_mask], pl),
                [_sharding.Shards([m], pl) for m in metal_masks],
                plastic_scale, metal_scales)

    # Size-relative default margins: the former fixed 10 at production sizes, smaller for small
    # volumes so the mask never carves off a large fraction of the field of view.
    shape = recon.tensors[0].shape if is_shards else recon.shape
    if num_real_slices is not None:
        num_slices = num_real_slices
    else:
        num_slices = recon.placement.real_size if is_shards else recon.shape[2]
    if radial_margin is None:
        radial_margin = max(2, min(10, min(shape[0], shape[1]) // 25))
    if top_margin is None:
        top_margin = max(2, min(10, num_slices // 25))
    if bottom_margin is None:
        bottom_margin = max(2, min(10, num_slices // 25))

    # Remove any flash from the boundary of the recon
    recon = mtp.apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=top_margin,
                                       bottom_margin=bottom_margin, num_real_slices=num_real_slices)

    # Compute thresholds using multi-threshold Otsu (padded voxels excluded via valid_mask)
    thresholds = multi_threshold_otsu(recon, classes=num_metal + 2, valid_mask=valid_mask)

    is_torch = isinstance(recon, torch.Tensor)
    if is_shards:
        shard_masks = _shard_valid_masks(valid_mask, recon.placement,
                                         recon.tensors[0].ndim)

    # Plastic: lowest class
    plastic_low_threshold = thresholds[0]
    plastic_metal_threshold = thresholds[1]

    # Masks are 1 inside the class interval AND on a real voxel: padded voxels are exactly 0, so a class
    # interval spanning 0 would otherwise mark them, biasing compute_scaling_factor's denominator.
    def _tensor_class_mask(vol, lower, upper, mask):
        in_class = (vol > lower) & (vol <= upper)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(np.asarray(mask), device=vol.device)
            in_class = in_class & mask.bool()
        return in_class.to(vol.dtype)

    def class_mask(lower, upper):
        if is_shards:
            # Shard by shard, each on its own device; the mask comes back
            # sharded the same way as the volume.
            return _sharding.Shards(
                [_tensor_class_mask(t, lower, upper, mp)
                 for t, mp in zip(recon.tensors, shard_masks)],
                recon.placement)
        if is_torch:
            return _tensor_class_mask(recon, lower, upper, valid_mask)
        in_class = (recon > lower) & (recon <= upper)
        if valid_mask is not None:
            in_class = in_class & np.asarray(valid_mask).astype(bool)
        return np.where(in_class, np.float32(1.0), np.float32(0.0)).astype(recon.dtype)

    plastic_mask = class_mask(plastic_low_threshold, plastic_metal_threshold)
    plastic_scale = mtp.compute_scaling_factor(recon, plastic_mask)

    # Metal masks and scaling
    metal_masks = []
    metal_scales = []
    for i in range(1, num_metal + 1):  # start from index 1
        lower = thresholds[i]
        upper = thresholds[i + 1] if i + 1 < len(thresholds) else np.inf
        metal_mask = class_mask(lower, upper)
        metal_masks.append(metal_mask)
        metal_scales.append(mtp.compute_scaling_factor(recon, metal_mask))

    return plastic_mask, metal_masks, plastic_scale, metal_scales
