"""Demo 13: dehydrating hyperspectral neutron data.

A hyperspectral neutron scan records, at every detector pixel, the transmission in each of many wavelength bins.
A sample made of a few materials has a low-rank attenuation X = W H: a map W of how much of each material every
pixel crosses, and a spectrum H per material.  dehydrate estimates W and H by maximum likelihood on the Poisson
counts, estimating the number of materials when it is not given; rehydrate multiplies them back into denoised data.
At a low dose most bins hold a few counts or none, where the least-squares (L2) baseline breaks down.
"""

import matplotlib.pyplot as plt
import numpy as np
import mbirtorch.hsnt as hsnt

# Simulate one projection of three materials (Ni, Cu and Al) at about 3 open-beam counts per pixel and bin.
material_basis, wavelengths = hsnt.load_material_basis()
np.random.seed(0)
noisy, _, truth = hsnt.generate_hyper_data(material_basis, detector_rows=64, detector_columns=64, dosage_rate=3,
                                           material_density={"Ni": 0.25, "Cu": 0.25, "Al": 0.75}, verbose=0)

# Dehydrate by maximum likelihood (rank estimated) and rehydrate; denoise with the L2 baseline for comparison.
dehydrated = hsnt.dehydrate(noisy, 'attenuation')
denoised = hsnt.rehydrate(dehydrated)
denoised_l2 = hsnt.l2_hyper_denoise(noisy, 'attenuation', num_materials=3, verbose=0)


def snr_db(attenuation):
    """SNR of the transmission against the noiseless truth."""
    error = np.linalg.norm(np.exp(-attenuation) - np.exp(-truth))
    return 20 * np.log10(np.linalg.norm(np.exp(-truth)) / error)


for label, attenuation in (('noisy data', noisy), ('L2 baseline', denoised_l2), ('maximum likelihood', denoised)):
    print(f'Transmission SNR, {label + ":":19s} {snr_db(attenuation):5.1f} dB')

# One pixel's transmission spectrum, and the material maps of the dehydrated data.
pixel = (0, 10, 32)
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(wavelengths, np.exp(-noisy[pixel]), '.', color='0.7', ms=2, label='noisy')
ax.plot(wavelengths, np.exp(-denoised_l2[pixel]), color='#E69F00', label='L2 baseline')
ax.plot(wavelengths, np.exp(-denoised[pixel]), color='#0072B2', label='maximum likelihood')
ax.plot(wavelengths, np.exp(-truth[pixel]), 'k--', label='truth')
ax.set(xlabel='wavelength (Angstrom)', ylabel='transmission', ylim=(0, 1.5), title=f'pixel {pixel[1:]}')
ax.legend()

subspace_data = dehydrated[0][0]
fig, axes = plt.subplots(1, subspace_data.shape[-1], figsize=(4 * subspace_data.shape[-1], 4), constrained_layout=True)
for k, ax in enumerate(np.atleast_1d(axes)):
    ax.imshow(subspace_data[:, :, k], cmap='magma')
    ax.set_title(f'material map {k}')
    ax.axis('off')
plt.show()
