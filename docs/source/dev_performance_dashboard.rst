Performance Dashboard
=====================

The performance dashboard records the run time, peak memory, and **correctness** of MBIRTorch's
operators over time.  It covers both CPU and GPU.

**Live dashboard:** https://cabouman.github.io/mbirtorch_metrics/

The dashboard is updated automatically.  Once a day, a scheduled job checks the tracked branches of
MBIRTorch.  The job measures each branch that has new commits since it was last measured.  The job
then pushes the new measurements to the
`mbirtorch_metrics <https://github.com/cabouman/mbirtorch_metrics>`__ repository.  Each push starts
a GitHub Action that rebuilds the dashboard and publishes it at the link above.  You do not need to
run anything to read the dashboard.

What it measures
----------------

For each branch it measures, the job runs these operators: the direct-reconstruction filter,
forward projection, back projection, the iterative VCD reconstruction, and the qGGMRF denoiser.
Each operator runs over a range of problem sizes and device counts, on both CPU and GPU.  For every
configuration, the job records the following:

- **run time**, the minimum over the timed calls that follow one warm-up call;
- **peak memory**; and
- a numeric **fingerprint** of the output, used to detect correctness changes.

The job also runs the branch's unit tests and records any failures.

How to read it
--------------

The live dashboard includes its own reading guide.  To open the guide, expand the
**"How to read this dashboard"** panel at the top of the dashboard.  The guide describes the tiles,
the red correctness banner, the History and Scaling views, the colors and marks, and what counts as
a correctness divergence.  This page does not repeat the guide, so there is only one copy to
maintain.

Running it yourself
-------------------

The dashboard is a single self-contained HTML page built from a YAML time series, so it needs no
server.  The `mbirtorch_metrics <https://github.com/cabouman/mbirtorch_metrics>`__ repository
contains the measurement engine, the scheduled job, and the build script.  Its ``README`` explains
how to build the dashboard locally.  The guides in its ``action_scripts/`` and ``tooling/`` folders
explain how runs are measured, checked for regressions, and scheduled.
