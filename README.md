# mbirtorch

A PyTorch port of [mbirjax](https://github.com/cabouman/mbirjax) (Phase 0 scaffold).

This repo is the home for the port evaluated in
`mbirjax_plans/plans/torch_port/port_plan.md` (checked out parallel to this
repo).  The plan defines the motivation, the parity gates against mbirjax, and
the incremental phase plan.  The current state is Phase 0: the repo scaffold
plus de-risking benchmarks; the benchmarks live with the plan, in
`mbirjax_plans/plans/experiments/torch_port/`.

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
