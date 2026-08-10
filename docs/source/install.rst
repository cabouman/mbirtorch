.. _InstallationDocs:

============
Installation
============

``MBIRTorch`` is currently installed from source on GitHub.  Installation
from PyPI with ``pip install mbirtorch`` will be available once the first
release is published.

**Installing from source**

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

**Installing with pip**

Alternatively, install into an existing Python (>= 3.11) environment from
the repository root::

    pip install .

The standard ``torch`` dependency installs automatically.  On a Linux
machine with an NVIDIA GPU, the default torch wheel includes CUDA support;
no separate CUDA variant of ``MBIRTorch`` is needed.

**Verifying the installation**

From the repository root, run the self-contained test suite::

    pytest tests
