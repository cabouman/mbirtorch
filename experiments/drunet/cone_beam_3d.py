"""The 3D noisy cone-beam problem: the same generator and noise model as
cone_beam_2d.py (see the comments there for the transmission-noise
derivation), without the mid-slice restriction."""

import numpy as np

import mbirtorch


def get_data(num_views=128, num_det_rows=128, num_det_channels=128,
             dosage=500.0):
    phantom, sinogram, params = mbirtorch.generate_demo_data(
        model_type='cone', object_type='shepp-logan',
        num_views=num_views, num_det_rows=num_det_rows,
        num_det_channels=num_det_channels, target_max_attenuation=6.0)

    noise_std = np.sqrt(np.exp(sinogram) / dosage)
    rng = np.random.default_rng(0)
    sinogram = sinogram + noise_std * rng.standard_normal(
        sinogram.shape).astype(np.float32)
    return phantom, sinogram, params
