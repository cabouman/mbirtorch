.. _ProjectorKernels:

=================
Projector kernels
=================

The two projectors are the heart of the reconstruction: **forward projection**
turns a volume of voxels into a sinogram (what the detector would have seen), and
**back projection** turns a sinogram into a volume (spreading each measurement back
along its ray).  Every iteration of the reconstruction calls both, so their speed
sets the speed of the whole solver.

This page describes how those kernels are built and where they live in the code.  It
picks up *below* the multi-device level: :doc:`dev_sharding_overview` explains how a
reconstruction is split across devices into bands, and this page explains the kernel
that then runs on each band or shard.  There are two layers.  The **torch bodies**
are written in ordinary PyTorch, compiled with ``torch.compile``, and run everywhere
-- CPU, MPS, any GPU, every geometry; they are the baseline, the permanent fallback,
and the value reference.  On top of them, on CUDA, sits a layer of hand-written
**Triton kernels** that replace the bodies where they have been measured to win.
The design notes and every measured number are in the plans repo under
``plans/torch_port/``.


The torch bodies
----------------

**Division of labor.**  The geometry-agnostic driver (``projectors.py``) owns
iteration and memory: the view-batch loops, the transient memory budget, per-device
compiled instances, and the compile lock.  The shared fan arithmetic that the
geometries have in common lives in ``horizontal_fan.py``.  Each geometry model
contributes only its own coordinate math and its per-view-batch bodies
(``parallel_beam.py``, ``cone_beam.py``).

**Batched over views.**  The driver walks the views in batches with a plain Python
loop, calling the geometry's body once per batch.  ``view_batch_size`` is the single
memory/speed knob: the dominant transient of a torch body is proportional to the
batch, and the driver's budget may cap the realized batch below the requested one.
A hand-written kernel declares its own, much smaller, per-view cost, so the same
driver runs both kinds of body without change.

**Compiled once per shape, with a fallback.**  The bodies are compiled with
``torch.compile`` and cached per function at module level; the measured chain-level
wins range from 1.7x on CPU to 22x on CUDA, with the fan chain's peak-memory
transients collapsing by an order of magnitude.  A compile failure falls back to
eager execution -- silently for the caller, but recorded, so exotic backends keep
working.  Compilation events are serialized process-wide behind a lock, because
concurrent cold compiles from the per-device worker threads crash the compiler
stack; steady-state execution takes no lock.

**Geometry decides whether a slice can stream.**  For parallel beam a detector row
looks straight across at exactly one recon slice, so the bodies can work on a band
of rows and produce just those slices -- this is the ``rows_track_slices`` case, and
it is what lets the multi-device back projection stream one slice band at a time.
For cone beam a single recon slice spreads across a *range* of detector rows, so a
view-owner must read the full detector rows and hold whole voxel cylinders; the
back driver still bands the *output* slice axis through the ``slice_start`` and
``band_slices`` arguments each body carries.


The Triton kernels
------------------

The hand-written kernels are written in `Triton <https://triton-lang.org>`__, which
compiles GPU kernels from plain Python at import time -- no separate CUDA build
step, and the code ships in the package like any other module (``triton_cone.py``,
``triton_parallel.py``).

**A kernel is an alternative body, never a new driver.**  Each kernel wrapper has
the same signature as the torch body it replaces, so the driver's view loop, the
multi-device seams, and everything downstream pass through unchanged.  The
torch bodies stay compiled in at every call site as the fallback and the value
reference, so the kernel path can be turned off at any time with no loss of
function.

**What the kernels fuse.**  The torch back body materializes a large per-view gather
transient; the Triton back kernel instead accumulates its output tile in registers
across the view reduction and the tap loops, so that transient is never written.
The torch forward body materializes scaled-value and detector-column transients and
then scatter-adds them; the Triton forward kernel keeps its detector column in
registers and scatters straight into the sinogram with atomic adds.  In both
directions the tap weights are derived in-kernel from the same eager geometry
precomputes the torch bodies use, so the tap axis is never materialized -- and
curved detectors and helical z shifts need no kernel code at all, because they
enter through those precomputes alone.

**The parallel forward goes further: sort, then multiply.**  The atomic adds above
are the parallel forward's natural bottleneck, so its default kernel reorders each
view's pixels by detector channel before projecting them (a per-view ``argsort``,
recomputed at every call).  Sorted, a small tile of pixels lands in a narrow window
of adjacent channels, and the kernel accumulates the whole tile into that window
with a small full-precision matrix multiply -- full precision because the
tensor-core default would not hold the value contract below -- then scatters only
the finished window: a few atomic adds where the per-tap kernel issued one per
pixel per tap.  A tile whose sorted span exceeds the window (a sparse or scattered
pixel set is the ordinary cause) falls back tap by tap for that tile alone, so
correctness never rests on the sort; only speed does.  The per-tap parallel forward
remains in the module behind an environment switch (see below), and cone beam keeps
its per-tap forward: a sorted cone variant is a design sketch in the plans repo,
not yet built.

**Parallel beam is the degenerate case.**  Apart from the sorted forward just
described, the parallel kernels are the cone kernels with the vertical fan deleted:
detector row *r* is recon slice *r*, so the row axis rides through both kernels
unchanged.  With the vertical fan goes the cone kernels'
rounding carve-out (see below), so the parallel kernels reproduce their torch bodies
to float summation order alone.  The parallel geometry is also cheaper in a way the
kernels exploit: its footprint terms depend on the view angle alone, so they are
passed as per-view scalars rather than per-pixel planes.

**The cone rounding carve-out.**  The cone kernels form the cone-angle divisor as
``sqrt(1 + (v/SDD)^2)`` where the torch bodies compute ``cos(atan2(v, SDD))`` -- the
forms are algebraically identical and differ by a couple of float rounding steps --
and their detector-row center can round a ``.5`` tie differently.  Both deviations
are bounded by the value contract the self-checks enforce (relative 1e-4, checked at
the squared-coefficient path where the gap is largest).


Availability: probe the hardware, never trust a list
----------------------------------------------------

Whether a kernel may replace its torch body is decided by two gates, both automatic
(``kernel_availability.py``):

1. **The process-wide probe.**  ``triton_available()`` compiles and runs a trivial
   Triton kernel end to end and caches the answer for the life of the process.  It
   fails closed: no CUDA platform, no ``triton`` import, a compile error, or a wrong
   result all mean the torch bodies are used, with the reason recorded in the
   returned ``(usable, reason)`` pair.
2. **The per-kernel, per-device value self-check.**  On first use, each of the four
   kernel slots (cone back, cone forward, parallel back, parallel forward) is run
   against its torch body on a tiny problem *on the actual device that will run it*,
   and falls back if the results differ by more than the 1e-4 contract.  The answer
   is cached per device for the process.  The parallel forward slot holds two
   kernels behind one wrapper -- sorted and per-tap -- and the check exercises the
   one its switch selects at that moment (see below).

This differs from MBIRJAX, which enables its custom kernels only on an allowlist of
GPU models.  MBIRTorch instead checks the hardware it is on: the kernels default on
wherever both gates pass, on any architecture, and a toolchain that miscompiles a
kernel is caught by the value check even on a GPU no one has tested.  The gates
answer *correctness* only; the performance record behind the default-on decision was
measured on H100 (the composed gates in the plans repo), so on a very different GPU
the kernels are still safe to use but their speed is unmeasured.

Selection happens in each geometry's ``_view_batch_bodies``: if a kernel's gates
pass it is chosen, otherwise the torch body is.  Every kernel is on by default,
each having passed a composed performance gate -- a full reconstruction comparison
against the kill-switch variant -- before earning that default; the sorted parallel
forward passed its own such gate against the per-tap route.

**One multi-GPU subtlety, fixed and guarded.**  A Triton launch targets the
launching thread's current device, and the per-device worker threads launch from
threads whose current device is 0.  The kernel wrappers therefore bracket every
launch on the tensors' own device; ``tests/test_kernels_sharded.py`` holds that
contract.


Turning the kernels off
-----------------------

Set ``MBIRTORCH_DISABLE_TRITON=1`` in the environment to disable every hand-written
kernel at once and run on the compiled torch bodies.  This is a diagnostic escape
hatch and a bisection handle, not a tuning knob: the self-checks above are the
automatic guard users rely on, and they need no environment variable.  The variable
is read inside the availability probe, whose result is cached per process, so it
must be set *before the first model is built*; changing it afterwards has no effect.
The torch bodies are the permanent value reference, so results stay correct with the
kernels off -- only speed changes.

``MBIRTORCH_SORTED_FORWARD=0`` is a narrower handle: every kernel stays on, but the
parallel forward runs its per-tap kernel instead of the sorted one.  Unlike the
process-wide variable above, this one is read at every call, so it can flip mid-run
around a single block.  That freedom has one edge: the first-use self-check
exercised whichever kernel the switch selected then, so a kernel swapped in
mid-process has not been self-checked in this process -- set the variable before the
first projection when that guard matters.

To retire a single kernel permanently, delete the selection line that chooses it in
the geometry's ``_view_batch_bodies``; every call site keeps the torch body it was
always compiled with, so nothing else changes.
