.. _InstallationDocs:

============
Installation
============

Install ``MBIRTorch`` from PyPI into a Python 3.11 or later environment::

    pip install mbirtorch

The standard ``torch`` dependency installs automatically.  On a Linux
machine with an NVIDIA GPU, the default torch wheel includes CUDA support;
no separate CUDA variant of ``MBIRTorch`` is needed.

**Installing from source**

Install from source to modify the package or to run its test suite.

1. Download the source code

Move to a directory of your choice and run the following two commands::

    git clone https://github.com/cabouman/mbirtorch.git
    cd mbirtorch

2. Install the conda environment and package

We provide bash scripts that do a clean install of ``MBIRTorch`` in a new
conda environment::

    cd dev_scripts
    source clean_install_all.sh

This creates a conda environment named ``mbirtorch``, installs the package
in editable mode with its test and documentation dependencies, and builds
the documentation.

To install into an existing environment instead, run this from the
repository root::

    pip install .

**Optional Pixi development environment**

For contributors who use `Pixi <https://pixi.sh>`__, ``MBIRTorch`` also provides an
optional reproducible development environment, defined by ``pixi.toml`` and pinned by
``pixi.lock`` at the repository root.  This does not replace the conda installation
workflow above.

For the default CPU environment on Linux or Apple Silicon macOS::

    pixi run smoke
    pixi run test-fast

For a CUDA-enabled Linux system::

    pixi run -e cuda smoke-torch
    pixi run -e cuda test-fast

The ``cuda`` environment uses the CUDA 13 build of torch, which needs an NVIDIA driver
that supports CUDA 13.  On a machine whose driver supports only CUDA 12, use the
``cuda12`` environment instead::

    pixi run -e cuda12 smoke-torch
    pixi run -e cuda12 test-fast

Additional useful tasks include::

    pixi run test
    pixi run docs

**Verifying the installation**

The tests are not part of the installed package, so run them from a source
checkout.  From the repository root::

    pytest tests
