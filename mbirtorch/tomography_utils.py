"""Direct-recon filter machinery.

``generate_direct_recon_filter`` builds the space-domain filter in numpy.
``apply_row_filter`` applies it as a per-row 'valid' convolution with the
(2C-1)-tap filter (output length == channels), computed with torch.fft in
row batches.
"""

import numpy as np
import torch

# Detector rows convolved per batch; bounds the FFT work area.  The value comes
# from an H100 sweep in an earlier implementation and is kept as a starting
# point; re-sweep as part of the torch tuning.
ROW_FILTER_BATCH = 1024


def generate_direct_recon_filter(num_channels, filter_name="ramp"):
    """The space-domain direct-recon filter of length 2*num_channels - 1."""
    supported_filters = ["ramp"]
    if filter_name not in supported_filters:
        raise ValueError(f"Unsupported filter. Supported filters are: "
                         f"{', '.join(supported_filters)}.")
    n = np.arange(-num_channels + 1, num_channels)
    recon_filter = (1 / 2) * np.sinc(n) - (1 / 4) * (np.sinc(n / 2)) ** 2
    return recon_filter.astype(np.float32)


def apply_row_filter(block, filter_arr, row_weight=None):
    """Convolve every detector row of ``block`` with ``filter_arr`` ('valid').

    Optional ``row_weight`` supplies the FDK cosine pre-weighting: a
    view-INDEPENDENT (rows, channels) map that multiplies each detector row
    BEFORE convolution.  It is applied inside the batched loop (a bounded
    gather per window), so the peak stays at the input+output floor -- no
    full-sinogram out-of-place copy.  ``None`` (the default) is pure FBP,
    leaving the parallel-beam path unchanged.

    Args:
        block: (views, rows, channels) float32 tensor.
        filter_arr: (2*channels - 1,) float32 tensor (pre-scaled by the caller).
        row_weight: optional (rows, channels) float32 tensor (FDK cosine
            pre-weighting), broadcast over views, applied to each row before
            filtering.  None (default) = no pre-weighting (FBP).

    Returns:
        (views, rows, channels) tensor.
    """
    n_views, n_rows, n_channels = block.shape
    filt_len = filter_arr.shape[0]
    full_len = n_channels + filt_len - 1
    rows = block.reshape(-1, n_channels)
    total_rows = rows.shape[0]

    # FFT of the filter once; each row batch shares it.
    filt_f = torch.fft.rfft(filter_arr, n=full_len)
    # Compute the indices to match the 'valid' output of a scipy convolution.
    # A length-C (C = num_channels) signal convolved with a length-(2C-1) filter
    # is the C samples of the full linear convolution starting at index C - 1, so
    # the valid slice is filtered[C-1 : C-1+C].
    start = n_channels - 1
    out = torch.empty_like(rows)
    # Never zero: a view-shard that owns no views has no rows to filter, and
    # a loop step of zero is an error even where the range it walks is empty.
    batch = max(1, min(ROW_FILTER_BATCH, total_rows))
    for r0 in range(0, total_rows, batch):
        window = rows[r0:r0 + batch]
        if row_weight is not None:
            # Flattened row k is detector row k % n_rows: gather the window's
            # weights (a (batch, channels) transient, bounded by the batch).
            det_rows = torch.arange(r0, r0 + window.shape[0],
                                    device=block.device) % n_rows
            window = window * row_weight[det_rows]
        # Do the convolution in frequency space, then convert back and place the result.
        win_f = torch.fft.rfft(window, n=full_len, dim=1)
        filtered = torch.fft.irfft(win_f * filt_f, n=full_len, dim=1)
        out[r0:r0 + window.shape[0]] = filtered[:, start:start + n_channels]
    return out.reshape(n_views, n_rows, n_channels)
