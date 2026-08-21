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

**Verifying the installation**

The tests are not part of the installed package, so run them from a source
checkout.  From the repository root::

    pytest tests
