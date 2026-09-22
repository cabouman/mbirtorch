# mbirtorch

[![CI](https://github.com/cabouman/mbirtorch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cabouman/mbirtorch/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mbirtorch/badge/?version=latest)](https://mbirtorch.readthedocs.io/en/latest/)

MBIRTorch: Model-Based Iterative Reconstruction (MBIR) for tomographic reconstruction using [PyTorch](https://pytorch.org/). 

Features include:
* Multiple geometries:  parallel beam, cone beam (including curved detector and helical), translation mode, and multi-axis parallel. 
* Preprocessing routines for NSI and Zeiss scanners.
* Utilities for metal artifact reduction and stripe removal.  
* Informative demos and extensive documentation. 
* Seamless operation on 1 or more GPUs, Mac MPS, or CPU. 
* Compiled torch and Triton kernels for efficiency.

Available on PyPI via 
```bash
pip install mbirtorch
```

Reconstruct in one line:
```python
import mbirtorch
recon, recon_dict = mbirtorch.recon_simple_parallel(sinogram, angles)
```

Full documentation at [https://mbirtorch.readthedocs.io/](https://mbirtorch.readthedocs.io/)

## Hyperspectral neutron data (hsnt)

`mbirtorch.hsnt` factorizes hyperspectral neutron transmission data into material maps and spectra by maximum
likelihood (Poisson counts), estimates the number of materials, and denoises by dehydration and rehydration.
From the shell:

    mbirtorch-hsnt inspect data.h5
    mbirtorch-hsnt convert sample_tifs/ --open-beam open_beam/ -o sample.h5
    mbirtorch-hsnt dehydrate sample.h5 -o results/
    mbirtorch-hsnt rehydrate results/sample_dehydrated.h5 -o results/

See the Hyperspectral CT page of the documentation.
