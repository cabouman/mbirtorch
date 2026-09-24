"""Compare the NNAL solvers with the L2 baseline on the three-material phantom.

Simulates one low-dose projection of Ni, Cu and Al, factorizes it with the L2 baseline (scikit-learn NMF) and with
the joint-Newton, multiplicative and block-Newton solvers, prints each fit's NNAL and L2 losses, and writes the
recovered spectra and maps, each matched to the true materials by least squares, as PNG files.

    python experiments/hsnt/compare_solvers.py [-o OUTPUT_DIR]
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from mbirtorch.hsnt import (compare_spectra, generate_hyper_data, l2_dehydrate, load_material_basis,
                            nnal_factorization, stable_nnal)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default=".", help="directory for the PNG files (default: current)")
    out_dir = parser.parse_args().output
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector_rows, detector_columns = 64, 64
    material_density = {"Ni": 0.25, "Cu": 0.25, "Al": 0.75}     # volume fractions
    num_materials_fit = 3
    np.random.seed(129)

    material_basis, _ = load_material_basis()
    num_materials_true = material_basis.shape[0]
    noisy, _, _ = generate_hyper_data(material_basis, num_angles=1, detector_rows=detector_rows,
                                      detector_columns=detector_columns, dosage_rate=3,
                                      material_density=material_density, noisy=True, verbose=0)
    noisy = torch.nan_to_num(torch.tensor(noisy, dtype=torch.float32, device=device), nan=0.0, posinf=0.0,
                             neginf=0.0)
    T = torch.exp(-noisy).reshape(-1, noisy.shape[-1])
    print(f"Range of T: {torch.min(T).item():.2g} to {torch.max(T).item():.2g}")

    # The true maps: the simulator's rectangles of half-cylinder thickness (generate_hyper_data returns no maps).
    height, width = detector_rows // 3, detector_columns // 2
    thickness = 20 * np.sqrt((width // 2) ** 2 - np.linspace(-(width // 2), width // 2, width) ** 2) / width
    maps_true = np.zeros((1, detector_rows, detector_columns, num_materials_true))
    cols = slice(width // 2, width + width // 2)
    maps_true[:, :height, cols, 0] = material_density["Ni"] * thickness
    maps_true[:, 2 * height:, cols, 1] = material_density["Cu"] * thickness
    maps_true[:, height:2 * height, cols, 2] = material_density["Al"] * thickness
    maps_true = maps_true.reshape(-1, num_materials_true)

    fits = {}
    start = time.time()
    W, H, _ = l2_dehydrate(noisy.cpu().numpy(), dataset_type="attenuation", num_materials=num_materials_fit,
                           safety_factor=1, verbose=0)
    fits["L2 (scikit-learn NMF)"] = (torch.tensor(W.reshape(T.shape[0], -1), dtype=T.dtype, device=device),
                                     torch.tensor(H, dtype=T.dtype, device=device))
    print(f"L2 baseline: {time.time() - start:.1f} s")
    for method, label in (("joint_newton", "joint Newton"), ("multiplicative", "multiplicative"),
                          ("block_newton", "block Newton")):
        start = time.time()
        W, H, steps = nnal_factorization(T, method=method, num_materials=num_materials_fit, max_steps=1000,
                                         rel_tol=1e-6)
        fits[label] = (W, H)
        print(f"{label}: {time.time() - start:.1f} s, {steps} steps")

    print()
    for label, (W, H) in fits.items():
        nnal = stable_nnal(W @ H, T).item()
        l2 = torch.linalg.norm(torch.log(T) + W @ H).item()
        print(f"{label:24s} NNAL loss {nnal:.6g}   L2 loss {l2:.6g}")

    # Match each fit to the true materials: theta maps the true spectra onto the rows of H by least squares.
    basis_t = torch.tensor(material_basis, dtype=T.dtype, device=device)
    spectra, maps = [], [maps_true.reshape(detector_rows, detector_columns, -1)]
    for W, H in fits.values():
        theta = torch.linalg.lstsq(H.T, basis_t.T)[0].T.cpu().numpy()
        W, H = W.cpu().numpy(), H.cpu().numpy()
        spectra.append(theta @ H)
        maps.append((W @ np.linalg.pinv(theta)).reshape(detector_rows, detector_columns, -1))

    compare_spectra(spectra_groups=spectra, ground_truth=material_basis, labels=["Ni", "Cu", "Al"],
                    subtitles=list(fits), title="Material attenuation spectra", x_label="Wavelength index",
                    y_label="Attenuation", y_lim=(0, 1.1),
                    filename=os.path.join(out_dir, "compare_solvers_spectra.png"))

    row_max = maps_true.max(axis=0).reshape(1, 1, -1)
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Material maps (RGB = Ni, Cu, Al), scaled by the true maximum")
    for i, (image, title) in enumerate(zip(maps, ["Ground truth"] + list(fits))):
        ax = fig.add_subplot(1, len(maps), i + 1)
        ax.set_title(title)
        ax.imshow(np.clip(image / row_max, 0, 1))
    fig.savefig(os.path.join(out_dir, "compare_solvers_maps.png"))


if __name__ == "__main__":
    main()
