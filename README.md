# mbirtorch

[![CI](https://github.com/cabouman/mbirtorch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cabouman/mbirtorch/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mbirtorch/badge/?version=latest)](https://mbirtorch.readthedocs.io/en/latest/)

A PyTorch port of [mbirjax](https://github.com/cabouman/mbirjax): high-performance
tomographic reconstruction with multi-GPU support.

This repo is the home for the port evaluated in
`mbirjax_plans/plans/torch_port/port_plan.md` (checked out parallel to this
repo).  The plan defines the motivation, the parity gates against mbirjax,
the incremental phase plan, and a progress record; its findings pages and the
supporting scripts live alongside it under `mbirjax_plans/plans/`.

## Setup

Create the conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate mbirtorch
pip install -e ".[test]"
```

## Tests

```bash
dev_scripts/run_tests.sh
```

## Caches

mbirtorch keeps one on-disk cache: compiled `torch.compile` artifacts, under
`~/.mbirtorch/torch_cache`.  It exists to make cold starts fast -- with it, a
fresh process reuses prior compilations instead of recompiling (roughly 14 s
down to 2 s for a first small reconstruction).  It grows with the number of
distinct compiled shapes and typically stays in the tens of megabytes; it is
never cleaned automatically.  To remove it:

```python
import mbirtorch
mbirtorch.clear_cache()   # deletes ~/.mbirtorch entirely (recreated empty)
```

The location can be redirected by setting the `TORCHINDUCTOR_CACHE_DIR`
environment variable before the first compile (e.g. to node-local or scratch
storage on a cluster, where home quotas are tight); `clear_cache()` does not
touch a redirected location.

Everything else the package caches is in-memory only and is freed with the
objects that hold it (e.g. the per-model pixel-index cache); nothing besides
`~/.mbirtorch` is written to disk.
