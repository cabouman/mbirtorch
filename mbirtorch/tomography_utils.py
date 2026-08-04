"""Direct-recon filter machinery, ported from mbirjax.tomography_utils.

``generate_direct_recon_filter`` is a verbatim numpy port.  ``apply_row_filter``
keeps the same contract (per-row 'valid' convolution with the (2C-1)-tap
filter, output length == channels) but implements it with torch.fft in
row batches, replacing jax's fftconvolve + scan machinery.
"""

import numpy as np
import torch

# Detector rows convolved per batch; bounds the FFT work area (the mbirjax
# constant, kept -- its H100 sweep rationale is backend-independent enough for
# Phase 1; re-sweep in Phase 2).
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


def apply_row_filter(block, filter_arr):
    """Convolve every detector row of ``block`` with ``filter_arr`` ('valid').

    Args:
        block: (views, rows, channels) float32 tensor.
        filter_arr: (2*channels - 1,) float32 tensor (pre-scaled by the caller).

    Returns:
        (views, rows, channels) tensor.
    """
    n_views, n_rows, n_channels = block.shape
    filt_len = filter_arr.shape[0]
    full_len = n_channels + filt_len - 1
    rows = block.reshape(-1, n_channels)
    total_rows = rows.shape[0]

    # FFT of the filter once; row batches share it.  'valid' output of a
    # length-C signal against a length-(2C-1) filter is the C samples of the
    # full linear convolution starting at index filt_len - 1... for len(filter)
    # >= len(signal), scipy's 'valid' starts at len(signal) - 1.  Here
    # filt_len = 2C - 1 >= C, so the valid slice is full[C-1 : C-1+C].
    filt_f = torch.fft.rfft(filter_arr, n=full_len)
    start = n_channels - 1
    out = torch.empty_like(rows)
    batch = min(ROW_FILTER_BATCH, total_rows)
    for r0 in range(0, total_rows, batch):
        window = rows[r0:r0 + batch]
        win_f = torch.fft.rfft(window, n=full_len, dim=1)
        full = torch.fft.irfft(win_f * filt_f, n=full_len, dim=1)
        out[r0:r0 + window.shape[0]] = full[:, start:start + n_channels]
    return out.reshape(n_views, n_rows, n_channels)
