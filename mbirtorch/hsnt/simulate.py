import numpy as np

from .denoise import rehydrate


def generate_hyper_data(material_basis, num_angles=1, detector_rows=64, detector_columns=64, dosage_rate=300,
                        material_density=None, verbose=1, *, noisy=True):
    """
    Simulate hyperspectral neutron counts for :math:`N_m=3` materials (Ni, Cu, Al) and :math:`N_k` wavelength bins.

    The open beam is noiseless, dosage_rate counts in every pixel and bin; the sample counts are Poisson. Divide the
    counts by the open beam to get the transmission that dehydrate takes with dataset_type='transmission'; a zero
    count is a zero transmission, which the likelihood handles.

    Args:
        material_basis: ndarray of shape :math:`(N_m, N_k)`, where rows are material linear attenuation coefficient spectra.
        num_angles: Number of view angles :math:`(N_v)`. Defaults to 1.
        detector_rows: Number of rows in the detector :math:`(N_r)`. Defaults to 64.
        detector_columns: Number of columns in the detector :math:`(N_c)`. Defaults to 64.
        dosage_rate: Neutron dosage rate during hyperspectral data collection. Defaults to 300.
        material_density: Material density (vol. fraction) for Ni, Cu, and Al. Defaults to {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}.
        verbose: Verbosity level. If 0, prints nothing; if 1, prints details; if >1, also generates plots. Defaults to 1.
        noisy: Whether to generate noisy data (keyword only). Defaults to True.

    Returns:
        A list in the form [counts, open_beam, angles, gt_hyper_projection].
            - counts: Poisson counts through the sample, shape :math:`(N_v, N_r, N_c, N_k)`, in the dtype of
              material_basis.
            - open_beam: Open-beam counts of the same shape, dosage_rate everywhere.
            - angles: ndarray of view angles in radians.
            - gt_hyper_projection: Ground truth noiseless attenuation of the same shape.

    """
    if material_basis.shape[0] != 3:
        raise ValueError("material_basis must have exactly 3 rows (Ni, Cu, Al).")

    if detector_rows < 3 or detector_columns < 2:
        raise ValueError("detector_rows must be ≥3 and detector_columns ≥2.")
    if dosage_rate <= 0:
        raise ValueError("dosage_rate must be positive.")

    if material_density is None:
        material_density = {"Ni": 0.2, "Cu": 0.2, "Al": 1.0}
    required = {"Ni", "Cu", "Al"}
    missing = required - set(material_density)
    if missing:
        raise KeyError(f"material_density missing keys: {sorted(missing)}")

    if np.any(material_basis < 0):
        raise ValueError("material_basis should be non-negative attenuation coefficients.")

    number_of_materials = material_basis.shape[0]
    number_of_wavelengths = material_basis.shape[1]

    angles = np.linspace(0, np.pi, num_angles)

    # The phantom is three stacked bars, one of each material.
    height = detector_rows // 3
    width = detector_columns // 2
    # -(width // 2), not -width // 2: floor division makes the latter asymmetric for an odd
    # width, and the square root then goes negative at one end.
    thickness = 20 * np.sqrt((width//2)**2 - np.linspace(-(width // 2), width // 2, width)**2)/ width
    material_projection = np.zeros((num_angles, detector_rows, detector_columns, number_of_materials),
                                   dtype=material_basis.dtype)
    material_projection[:, :height, width // 2:width + width // 2, 0] = material_density["Ni"] * thickness
    material_projection[:, 2 * height:, width // 2:width + width // 2, 1] = material_density["Cu"] * thickness
    material_projection[:, height:2 * height, width // 2:width + width // 2, 2] = material_density["Al"] * thickness

    gt_hyper_projection = rehydrate([material_projection, material_basis, 'attenuation'])

    open_beam = np.full(gt_hyper_projection.shape, dosage_rate, dtype=material_basis.dtype)

    expected_counts = np.exp(-gt_hyper_projection) * open_beam
    expected_counts = np.nan_to_num(expected_counts, nan=0, posinf=0, neginf=0)

    # The measured counts are Poisson distributed; the open beam is taken as noiseless.
    counts = np.random.poisson(expected_counts) if noisy else expected_counts
    counts = counts.astype(material_basis.dtype)

    if verbose >= 1:
        print("generate_hyper_data(): ")
        print("   -Shape of material_basis (linear attenuation coefficients for Ni, Cu, and Al):", material_basis.shape)
        print("   -Shape of material_projection (density of Ni, Cu, and Al):", material_projection.shape)
        print("   -Shape of counts and open beam: ", counts.shape)

    if verbose > 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(material_basis.T)
        plt.xlabel("wavelength index")
        plt.ylabel("linear attenuation ($cm^{-1}$)")
        plt.title("Material basis functions (Ni, Cu, Al)")
        plt.legend(["Ni", "Cu", "Al"])

    return [counts, open_beam, angles, gt_hyper_projection]



def synthetic_material_basis(num_wavelengths=200):
    """
    Three synthetic attenuation spectra, one per material, for simulated scans.

    Each spectrum falls slowly with wavelength and drops sharply at one wavelength of its own,
    the shape of a Bragg edge.  The values are of order one and are not those of any real
    material; a material map gives the attenuation per unit length by scaling them.

    Args:
        num_wavelengths (int, optional): The number of wavelength bins.  Defaults to 200.

    Returns:
        numpy.ndarray: The spectra, float32, shape (3, num_wavelengths), all values positive.
    """
    w = np.linspace(0.0, 1.0, num_wavelengths, dtype=np.float32)
    return np.stack([1.5 - 0.6 * w + 0.8 * (w < 0.3),
                     1.0 - 0.4 * w + 1.0 * (w < 0.55),
                     0.6 - 0.2 * w + 0.5 * (w < 0.8)]).astype(np.float32)


def gen_material_phantom(recon_shape, density=0.08):
    """
    A phantom of three materials: three cylinders along the slice axis, spaced evenly around
    the center of the volume, one of each material.

    Args:
        recon_shape (tuple of int): (num_rows, num_cols, num_slices).
        density (float, optional): The value inside each cylinder, which scales that material's
            spectrum to the attenuation per voxel.  Defaults to 0.08.

    Returns:
        numpy.ndarray: One map per material, float32, shape (3,) + recon_shape, ``density``
        inside that material's cylinder and 0 elsewhere.
    """
    num_rows, num_cols, num_slices = recon_shape
    rows = np.arange(num_rows, dtype=np.float32)[:, None, None]
    cols = np.arange(num_cols, dtype=np.float32)[None, :, None]
    slices = np.arange(num_slices, dtype=np.float32)[None, None, :]
    half = min(num_rows, num_cols) / 2.0
    ring, radius = 0.45 * half, 0.22 * half
    in_height = np.abs(slices - (num_slices - 1) / 2.0) <= 0.4 * num_slices
    maps = np.zeros((3,) + tuple(recon_shape), dtype=np.float32)
    for k, angle in enumerate(np.deg2rad([90.0, 210.0, 330.0])):
        center_row = (num_rows - 1) / 2.0 - ring * np.sin(angle)
        center_col = (num_cols - 1) / 2.0 + ring * np.cos(angle)
        inside = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2
        maps[k][np.broadcast_to(inside & in_height, recon_shape)] = density
    return maps


def generate_hyper_sinogram(ct_model, material_maps, material_basis, dosage_rate=300, noisy=True, seed=None):
    """
    Simulate a hyperspectral scan of a volume of several materials.

    Each material map is projected with the model, and the projections are combined with the
    spectra into the attenuation at every wavelength.  The open beam is ``dosage_rate`` counts in
    every pixel and bin, and the counts through the sample are Poisson with that mean times the
    transmission.  Divide the counts by the open beam to get the transmission that
    :func:`dehydrate` takes with ``dataset_type='transmission'``.

    Args:
        ct_model (TomographyModel): The model of the scan.
        material_maps (numpy.ndarray): One volume per material, shape (num_materials,) +
            recon_shape, the attenuation per unit length at a spectrum value of 1.
        material_basis (numpy.ndarray): The spectra, shape (num_materials, num_wavelengths).
        dosage_rate (float, optional): Open beam counts per pixel and bin.  Defaults to 300.
        noisy (bool, optional): If True, the counts are Poisson; if False, their mean.  Defaults
            to True.
        seed (int, optional): Seed of the Poisson draw.  Defaults to None.

    Returns:
        tuple: (counts, open_beam, attenuation), each of shape (num_views, num_det_rows,
        num_det_channels, num_wavelengths), float32: the counts through the sample, the open
        beam, and the noiseless attenuation.
    """
    material_maps = np.asarray(material_maps, dtype=np.float32)
    material_basis = np.asarray(material_basis, dtype=np.float32)
    if material_maps.shape[0] != material_basis.shape[0]:
        raise ValueError(f'material_maps has {material_maps.shape[0]} materials and material_basis '
                         f'{material_basis.shape[0]}; they must match.')
    material_sinograms = np.stack([np.asarray(ct_model.forward_project(volume)) for volume in material_maps],
                                  axis=-1)
    attenuation = (material_sinograms @ material_basis).astype(np.float32)
    open_beam = np.full(attenuation.shape, dosage_rate, dtype=np.float32)
    expected_counts = np.exp(-attenuation) * open_beam
    if noisy:
        counts = np.random.default_rng(seed).poisson(expected_counts).astype(np.float32)
    else:
        counts = expected_counts.astype(np.float32)
    return counts, open_beam, attenuation
