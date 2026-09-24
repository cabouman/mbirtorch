"""Plots of a factorization for the command line."""
import os

_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#8B4513", "#555555",
           "#7F00FF"]


def _short(source, n=60):
    """The basename of a source label ('path' or 'path:group'), keeping the group; a drive letter's colon is left
    alone because what follows it is a path."""
    head, sep, tail = source.rpartition(":")
    if sep and tail and not any(c in tail for c in "/\\") and os.path.exists(head):
        s = os.path.basename(head) + ":" + tail
    else:
        s = os.path.basename(source)
    return s if len(s) <= n else "..." + s[-n:]


def plot_factorization(base, source, W4, H, bin_indices):
    """Write <base>_spectra.png (the rows of H and the mean-pixel spectrum) and <base>_maps.png (the maps of every
    view). Uses matplotlib's object interface, so the caller's pyplot state and backend are untouched.

    Returns:
        list: the two paths.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from .outputs import mean_pixel_spectrum
    R = H.shape[0]
    total, contrib, n_mat = mean_pixel_spectrum(W4.reshape(-1, R), H)
    ncol = 2 if R > 5 else 1
    fig = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(fig)
    ax, ax2 = fig.subplots(2, 1, sharex=True)
    for r in range(R):
        ax.plot(bin_indices, H[r], color=_COLORS[r % len(_COLORS)], lw=1.2, label=f"component {r}")
    ax.set_ylabel("attenuation per unit map value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=ncol)
    ax.set_title(f"rows of H, rank {R}: {_short(source)}\n(when maps are proportional, the split among rows is "
                 "arbitrary)", fontsize=11)
    ax2.plot(bin_indices, total, color="black", lw=1.6, label=f"total, average of {n_mat:,} material pixels")
    for r in range(R):
        ax2.plot(bin_indices, contrib[r], color=_COLORS[r % len(_COLORS)], lw=1.0, alpha=0.9,
                 label=f"component {r} share")
    ax2.set_xlabel("source wavelength index")
    ax2.set_ylabel("attenuation of the average material pixel")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, ncol=ncol)
    ax2.set_title("mean-pixel spectrum: independent of the split among components", fontsize=11)
    p1 = base + "_spectra.png"
    fig.savefig(p1, dpi=130)
    V = W4.shape[0]
    fig = Figure(figsize=(3.2 * R, 3.2 * V), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(V, R, squeeze=False)
    for v in range(V):
        for r in range(R):
            im = axes[v, r].imshow(W4[v, :, :, r], cmap="magma")
            axes[v, r].set_title(f"view {v}, component {r}" if V > 1 else f"component {r}", fontsize=10)
            axes[v, r].axis("off")
            fig.colorbar(im, ax=axes[v, r], fraction=0.046)
    fig.suptitle(f"material maps (W): {_short(source)}", fontsize=11)
    p2 = base + "_maps.png"
    fig.savefig(p2, dpi=110)
    return [p1, p2]
