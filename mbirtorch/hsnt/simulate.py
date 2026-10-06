import numpy as np

from .denoise import rehydrate


def generate_hyper_data(material_basis, num_angles=1, detector_rows=64, detector_columns=64, dosage_rate=300,
                        material_density=None, verbose=1, *, noisy=True):
    """
    Simulate noisy hyperspectral neutron attenuation data for :math:`N_m=3` materials (Ni, Cu, Al) and :math:`N_k` wavelength bins.

    The open beam is noiseless, dosage_rate counts in every pixel and bin; the sample counts are Poisson; the
    transmission is floored at 1e-30 before the logarithm, so a zero count becomes an attenuation of about 69.

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
        A list in the form [noisy_hyper_projection, angles, gt_hyper_projection].
            - noisy_hyper_projection: Simulated noisy hyperspectral data of shape :math:`(N_v, N_r, N_c, N_k)`, in
              the dtype of material_basis.
            - angles: ndarray of view angles in radians.
            - gt_hyper_projection: Ground truth noiseless hyperspectral data of same shape.

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

    epsilon = 1e-30
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

    noiseless_open_beam = dosage_rate * np.ones((detector_rows, detector_columns, number_of_wavelengths),
                                                dtype=material_basis.dtype)

    noiseless_object_scan = np.exp(-gt_hyper_projection) * noiseless_open_beam
    noiseless_object_scan = np.nan_to_num(noiseless_object_scan, nan=0, posinf=0, neginf=0)

    # The measured counts are Poisson distributed; the open beam is taken as noiseless.
    noisy_object_scan = np.random.poisson(noiseless_object_scan) if noisy else noiseless_object_scan

    ratio = (noisy_object_scan / noiseless_open_beam).astype(material_basis.dtype)
    ratio[ratio < epsilon] = epsilon
    noisy_hyper_projection = -np.log(ratio)

    if verbose >= 1:
        print("generate_hyper_data(): ")
        print("   -Shape of material_basis (linear attenuation coefficients for Ni, Cu, and Al):", material_basis.shape)
        print("   -Shape of material_projection (density of Ni, Cu, and Al):", material_projection.shape)
        print("   -Shape of hyperspectral data: ", noisy_hyper_projection.shape)

    if verbose > 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(material_basis.T)
        plt.xlabel("wavelength index")
        plt.ylabel("linear attenuation ($cm^{-1}$)")
        plt.title("Material basis functions (Ni, Cu, Al)")
        plt.legend(["Ni", "Cu", "Al"])

    return [noisy_hyper_projection, angles, gt_hyper_projection]

