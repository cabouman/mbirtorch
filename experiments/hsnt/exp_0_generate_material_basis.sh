#!/bin/bash
# Generates binaries/material_basis.npy in a conda env of its own, "bem", which holds ORNL's
# braggedgemodeling package. That package must not go into the mbirtorch env: its conda-forge
# scipy conflicts with torch's OpenMP runtime and torch then fails to import.
set -e
cd "$(dirname "$0")"
conda env list | grep -q '^bem ' || conda create -y -n bem -c conda-forge python=3.11 numpy braggedgemodeling
conda run -n bem python exp_0_generate_material_basis.py
