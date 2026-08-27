"""Agents for the MACE loop: callables mapping a volume tensor to a volume
tensor on a fixed device.

Every knob is bound at construction so that each agent is a fixed operator
across the loop.  The one deliberate exception is the forward agent's
partition schedule: it walks the model's partition sequence coarse to fine as
the cumulative iteration count grows, and settles on the finest partitions
for the rest of the run (operators may follow a schedule early, but must be
fixed in the tail so the equilibrium is well defined).  Warm starts are the
other piece of cross-call state: each agent initializes its inner solve from
its own previous OUTPUT, which converges to the consensus; the input
converges to consensus plus a nonzero dual offset, so it is the wrong warm
start.  Neither affects which operator is being approximated, only how
accurately a fixed number of inner iterations approximates it.
"""

import torch

import mbirtorch


class ForwardProxAgent:
    """Proximal map of the tomography data-fit term, via
    :meth:`TomographyModel.prox_map`.

    The sinogram, weights, and sigma_prox are bound at construction.  Each
    call runs a fixed number of warm-started VCD iterations, and the
    cumulative count is passed as ``first_iteration`` so the model's
    partition sequence advances coarse to fine across calls.  Use one agent
    per model instance: the agent relies on the model's cached prox
    initialization.

    Args:
        model (TomographyModel): the projection model; its device layout is
            settled on the first call (or beforehand by
            ``configure_devices``).
        sinogram: measured sinogram (numpy or tensor).
        weights (optional): sinogram weights, as for ``prox_map``.
        sigma_prox (float, optional): proximal strength.  None uses the
            model's auto value.
        inner_iterations (int, optional): VCD iterations per call.
        init_recon (optional): warm start for the first call.  None lets
            prox_map fall back to its own direct-recon initialization.
    """

    def __init__(self, model, sinogram, weights=None, sigma_prox=None,
                 inner_iterations=3, init_recon=None):
        self.model = model
        self.sinogram = sinogram
        self.weights = weights
        self.sigma_prox = sigma_prox
        self.inner_iterations = inner_iterations
        self._previous_output = init_recon
        self._iterations_done = 0

    def __call__(self, v):
        first = self._iterations_done
        output, _ = self.model.prox_map(
            v, self.sinogram, sigma_prox=self.sigma_prox,
            weights=self.weights, init_recon=self._previous_output,
            do_initialization=(first == 0),
            max_iterations=first + self.inner_iterations,
            first_iteration=first,
            stop_threshold_change_pct=0.0,
            print_logs=False, output_sharded=True)
        self._previous_output = output
        self._iterations_done = first + self.inner_iterations
        return output


class QGGMRFDenoiserAgent:
    """qGGMRF prior agent: the proximal map of the qGGMRF regularizer, via
    :meth:`QGGMRFDenoiser.denoise`.

    The prior parameters are pinned at construction (auto-regularization
    off), so the agent is the same operator on every call and ``sigma_noise``
    is its one strength knob -- the same role sigma plays for a
    noise-conditioned network denoiser.

    Args:
        image_shape (tuple): volume shape (rows, cols, slices).
        sigma_noise (float): denoising strength, in recon units.
        pinned_params (dict, optional): prior parameters to fix, e.g.
            ``{'sigma_x': ...}`` read from a direct recon's
            regularization_params.
        inner_iterations (int, optional): VCD sweeps per call.
        like_model (TomographyModel, optional): model whose device layout to
            share, so volumes pass between the agents without leaving the
            devices.
        use_ror_mask (bool, optional): restrict updates to the inscribed
            ellipse, matching a reconstruction model that does the same.
    """

    def __init__(self, image_shape, sigma_noise, pinned_params=None,
                 inner_iterations=8, like_model=None, use_ror_mask=False):
        self.model = mbirtorch.QGGMRFDenoiser(tuple(int(n) for n in image_shape))
        if like_model is not None:
            self.model.configure_devices(like=like_model)
        self.model.set_params(no_warning=True, verbose=0)
        if pinned_params:
            self.model.set_params(no_warning=True, **pinned_params)
        self.model.set_params(no_warning=True, auto_regularize_flag=False)
        self.sigma_noise = float(sigma_noise)
        self.inner_iterations = inner_iterations
        self.use_ror_mask = use_ror_mask
        self._previous_output = None

    def __call__(self, v):
        output, _ = self.model.denoise(
            v, sigma_noise=self.sigma_noise,
            use_ror_mask=self.use_ror_mask,
            init_image=self._previous_output,
            max_iterations=self.inner_iterations,
            stop_threshold_change_pct=0.0,
            print_logs=False, output_sharded=True)
        self._previous_output = output
        return output


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

    Slices of the (rows, cols, slices) volume are denoised independently by
    the 2D network (with one slice this is exactly the 2D problem), and rows
    and cols are reflect-padded to multiples of 8 as the U-Net requires.
    The network is feedforward, so there is no inner solve to warm-start.

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
    """

    def __init__(self, net, sigma_noise, intensity_scale, ror_mask=None,
                 slice_batch=8):
        self.net = net
        self.sigma_noise = float(sigma_noise)
        self.intensity_scale = float(intensity_scale)
        self.ror_mask = ror_mask
        self.slice_batch = slice_batch

    def __call__(self, v):
        import torch.nn.functional as functional
        rows, cols, num_slices = v.shape
        x = (self.intensity_scale * v).permute(2, 0, 1).unsqueeze(1)
        pad_rows = (-rows) % 8
        pad_cols = (-cols) % 8
        if pad_rows or pad_cols:
            x = functional.pad(x, (0, pad_cols, 0, pad_rows), mode='reflect')
        sigma_scaled = self.intensity_scale * self.sigma_noise
        with torch.no_grad():
            y = torch.cat([self.net(batch, sigma_scaled)
                           for batch in x.split(self.slice_batch)])
        y = y[..., :rows, :cols].squeeze(1).permute(1, 2, 0)
        y = y / self.intensity_scale
        if self.ror_mask is not None:
            y = v + self.ror_mask * (y - v)
        return y
