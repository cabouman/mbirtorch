"""Agents for the MACE loop: callables mapping a volume tensor to a volume
tensor on a fixed device.

The two model-based agents, ``ForwardProxAgent`` and ``QGGMRFDenoiserAgent``,
are defined in the package module :mod:`mbirtorch.mace` and are imported here
under the names the scripts use.  The DRUNet agent stays in this file because
it needs ``deepinv``, which is not a dependency of the package.

Every knob is bound at construction so that each agent is a fixed operator
across the loop.  The one deliberate exception is the forward agent's
partition schedule: it walks the model's partition sequence coarse to fine as
the loop's iteration count grows, and settles on the finest partitions for
the rest of the run (operators may follow a schedule early, but must be
fixed in the tail so the equilibrium is well defined).  Warm starts are the
other piece of cross-call state: each agent initializes its inner solve from
its own previous OUTPUT, which converges to the consensus; the input
converges to consensus plus a nonzero dual offset, so it is the wrong warm
start.  Neither affects which operator is being approximated, only how
accurately a fixed number of inner iterations approximates it.
"""

import torch

from mbirtorch.mace import ForwardProxAgent, QGGMRFDenoiserAgent  # noqa: F401


def load_drunet(device):
    """Load the pretrained grayscale DRUNet once (weights auto-download on
    the first call).  deepinv supplies the network; it is confined to this
    module so the weight source stays swappable."""
    from deepinv.models import DRUNet
    net = DRUNet(in_channels=1, out_channels=1, pretrained='download',
                 device=device)
    net.eval()
    return net


class DRUNetAgent:
    """Pretrained DRUNet as the prior agent.

    DRUNet is a Gaussian denoiser conditioned on a continuous noise level, so
    ``sigma_noise`` (in recon units) plays the same strength role as it does
    for the qGGMRF agent.  The network is trained on images in [0, 1], so a
    fixed intensity scale c is applied around every call:
    D_recon(v) = D(c v, c sigma) / c.  The scale is chosen once, from the
    initial reconstruction, and must stay fixed across the loop -- rescaling
    per call would make the agent a different operator each iteration.

    The 2D network denoises the volume's slices along ``slice_axis``
    independently (with one slice this is exactly the 2D problem), and each
    slice's two dimensions are reflect-padded to multiples of 8 as the U-Net
    requires.  Multi-slice fusion uses three of these agents, one per axis;
    the intensity scale and the mask are applied to the VOLUME, so they are
    shared by all orientations.  The network is feedforward, so there is no
    inner solve to warm-start.

    Args:
        net: the loaded network (see :func:`load_drunet`), shared between
            agents so the weights load once.
        sigma_noise (float): denoising strength in recon units.
        intensity_scale (float): the fixed scale c mapping recon values
            into the network's [0, 1] range.
        ror_mask (tensor, optional): (rows, cols, 1) mask; outside it the
            output keeps the input values, matching a reconstruction model
            that only updates inside the region of reconstruction.
        slice_batch (int, optional): slices per network call.
        slice_axis (int, optional): axis whose slices are denoised (0, 1,
            or 2).  Defaults to 2, the slice axis of a (rows, cols, slices)
            volume.
    """

    def __init__(self, net, sigma_noise, intensity_scale, ror_mask=None,
                 slice_batch=8, slice_axis=2):
        self.net = net
        self.sigma_noise = float(sigma_noise)
        self.intensity_scale = float(intensity_scale)
        self.ror_mask = ror_mask
        self.slice_batch = slice_batch
        self.slice_axis = slice_axis

    def __call__(self, v, iteration=0):
        import torch.nn.functional as functional
        x = torch.moveaxis(self.intensity_scale * v,
                           self.slice_axis, 0).unsqueeze(1)
        height, width = x.shape[-2:]
        pad_rows = (-height) % 8
        pad_cols = (-width) % 8
        if pad_rows or pad_cols:
            x = functional.pad(x, (0, pad_cols, 0, pad_rows), mode='reflect')
        sigma_scaled = self.intensity_scale * self.sigma_noise
        with torch.no_grad():
            y = torch.cat([self.net(batch, sigma_scaled)
                           for batch in x.split(self.slice_batch)])
        y = torch.moveaxis(y[..., :height, :width].squeeze(1),
                           0, self.slice_axis)
        y = y / self.intensity_scale
        if self.ror_mask is not None:
            y = v + self.ror_mask * (y - v)
        return y
