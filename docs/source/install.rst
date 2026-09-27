.. _InstallationDocs:

============
Installation
============

Install from PyPI
-----------------

MBIRTorch needs Python 3.11 or later.  Install it with::

    pip install mbirtorch

On a Linux machine with an NVIDIA GPU, the default torch wheel includes CUDA support, and
reconstructions run on the GPU.  On Apple Silicon macOS, the default torch wheel includes Metal
support, and reconstructions run on the GPU.  No separate GPU variant of MBIRTorch is needed.


Verify the installation
-----------------------

Run the first demo script, as described in :doc:`quick_start`.  It needs no data and takes about
a minute.


Install from source
-------------------

Install from source to modify the package or to run its test suite.  First clone the repository::

    git clone https://github.com/cabouman/mbirtorch.git
    cd mbirtorch

Then install in one of two ways.

**A new conda environment.**  The script below creates a conda environment named ``mbirtorch``,
installs the package in editable mode with its test and documentation dependencies, and builds
the documentation::

    cd dev_scripts
    source clean_install_all.sh

The script deletes any existing conda environment named ``mbirtorch`` before creating the new one.

**An existing environment.**  From the repository root, install the package in editable mode with
its test dependencies::

    pip install -e ".[test]"

Either way, run the tests from the repository root::

    pytest tests


Pixi environment
----------------

For contributors who use `Pixi <https://pixi.sh>`__, the files ``pixi.toml`` and ``pixi.lock`` at
the repository root define a pinned development environment.  This is an alternative to the conda
environment above, not a replacement for it.

On Linux or Apple Silicon macOS, the default environment uses the CPU build of torch::

    pixi run smoke
    pixi run test-fast

On a Linux machine with an NVIDIA driver for CUDA 13, use the ``cuda`` environment.  If the driver
supports only CUDA 12, use ``cuda12`` instead::

    pixi run -e cuda smoke-torch
    pixi run -e cuda test-fast

The tasks ``test`` and ``docs`` run the full test suite and build the documentation.
