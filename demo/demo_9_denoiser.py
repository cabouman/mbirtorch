"""Demo 9: the qGGMRF denoiser.

The denoiser is the reconstruction's image model used on its own: it takes
a noisy 3D image and returns a smoothed one that preserves edges.  No
geometry or sinogram is involved.  Its one knob is sigma_noise, your
estimate of the noise standard deviation: larger values smooth more.
"""

import numpy as np
import mbirtorch

# Make a clean phantom and add noise with a known standard deviation.
shape = (128, 128, 128)
noise_std = 0.1
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(shape)
noisy = phantom + noise_std * np.random.default_rng(0).standard_normal(shape).astype(np.float32)

# Denoise.  Try sigma_noise above and below the true noise level to see
# over- and under-smoothing.
denoiser = mbirtorch.QGGMRFDenoiser(shape)
denoised, denoise_dict = denoiser.denoise(noisy, sigma_noise=noise_std)

def nrmse(image):
    return np.linalg.norm(image - phantom) / np.linalg.norm(phantom)

print(f'Error of the noisy image:    {nrmse(noisy):.3f}')
print(f'Error of the denoised image: {nrmse(denoised):.3f}')

mbirtorch.slice_viewer(noisy, denoised, data_dicts=[None, denoise_dict], vmin=0.0, vmax=1.2,
                       title='Noisy image (left) and qGGMRF denoised image (right)')
