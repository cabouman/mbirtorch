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
Full documentation at [https://mbirtorch.readthedocs.io/](https://mbirtorch.readthedocs.io/)
