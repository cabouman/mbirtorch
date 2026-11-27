# Reconstruction from disk to disk

Each reconstruction script in this folder is for a scan that is too large for the GPUs at
hand.  The script reads one part of the sinogram from a file, reconstructs the slices that
this part determines, writes them to the output file, and continues with the next part.  The
volume exists only on disk.

| file | purpose |
|---|---|
| `recon_on_disk_parallel.py` | MBIR of a parallel-beam scan, one band of slices at a time.  The result is that of `recon_split_sino`. |
| `recon_direct_on_disk_parallel.py` | Filtered back projection of a parallel-beam scan, one band of slices at a time.  The result is that of `recon_direct`. |
| `recon_direct_on_disk_cone.py` | FDK reconstruction of a circular cone-beam scan, one slab of slices at a time.  The result is that of `recon_direct`. |
| `make_synthetic_parallel_sinogram.py` | Phantom sinogram for the parallel-beam scripts, and comparison of a reconstruction with the phantom. |
| `make_synthetic_cone_sinogram.py` | Phantom sinogram for the cone-beam script, and comparison of a reconstruction with the phantom. |
| `check_recon_direct_on_disk.py` | Comparison of the two direct scripts with `recon_direct` of the whole volume, on small scans. |
| `on_disk_common.py` | Code that the reconstruction scripts share. |

`python <script> --help` lists the options of each script.

## The functions

Each reconstruction script defines a function with the name of the script.  The function takes
the model of the whole scan, the name of the sinogram file, and the name of the output file.
The devices of the model and its `memory_check_margin` apply to every part.  This form lets the
functions become the methods `recon_on_disk` and `recon_direct_on_disk` of the model.  The
command line of a script builds the model from its options and calls the function.

```python
import numpy as np
import mbirtorch
from recon_direct_on_disk_cone import recon_direct_on_disk_cone

angles = np.linspace(0, 2 * np.pi, 1601, endpoint=False)
model = mbirtorch.ConeBeamModel((1601, 3048, 3048), angles, source_detector_dist=5080, source_iso_dist=2540)
recon_direct_on_disk_cone(model, 'sino.npy', 'recon.h5')
```

The functions use two private functions of the package:
`TomographyModel._fits_available_devices` is the memory check, and
`mbirtorch.utilities._recon_attributes` gives the attributes of the HDF5 output.

## What the reconstruction scripts share

Each script reconstructs the scan in parts.  A part is a range of slices.  The parallel-beam
scripts call a part a band, and the cone-beam script calls it a slab.  `on_disk_common.py`
holds the code that does not depend on the geometry or on the reconstruction method: the
options for the files, the devices, and the memory check, the reader of the sinogram file, the
writer of the output file, the progress file, the choice of the number of parts, and the loop
over the parts.  It also holds the direct reconstruction by parts, which both direct scripts
call.  Each script holds its own options, its model, and its division of the scan into parts.
The MBIR script also holds its reconstruction of one band and the blending of two bands.  The
scripts therefore read and write the same files, and they record and resume a run in the same
way.

The sinogram is a `.npy` file, which is memory mapped, or an HDF5 file.  Its axes are (views,
detector rows, detector channels) in float32.  The script reads only the detector rows that
it needs for one part.

`--slices-per-section` sets the largest number of slices of a part.  It has the name of the
parameter of `recon_split_sino`.  Without it, the script takes the smallest number of parts
for which the memory check of the package accepts every part on the GPUs.  The check accepts
a part if (1 + margin) times its estimated peak memory is at most the free memory of each GPU.
`--margin` sets the margin, which is 0.3 by default.  On a device that is not a GPU, the check
has no limit, and `--slices-per-section` is required.

A `.h5` output is in the layout of `mbirtorch.export_recon_hdf5`: dataset `recon` with axes
(slice, col, row).  `mbirtorch.import_recon_hdf5` reads it.  A `.npy` output is a memory
mapped array with axes (slice, row, col).

A progress file beside the output, `<output name>_progress.json`, records the settings of the
run and each finished part.  The same command with `--resume` continues an interrupted run
from the first unfinished part, with the number of parts of the interrupted run.  The script
refuses to resume a run with other settings.  The script flushes the output file before it
records a part, so the output of a process that is killed during a reconstruction holds every
recorded part.  The HDF5 library does not guarantee that an `.h5` file can be opened again if
the process is killed while it writes to the file.  A `.npy` output does not have this limit.

## Parallel beam: MBIR, `recon_on_disk_parallel.py`

In parallel-beam geometry, a detector row is a reconstruction slice.  For each band of
detector rows, the script reads the rows from the sinogram file, reconstructs them with
`ParallelBeamModel.recon`, blends the first slices of the band with the last slices of the
previous band, and writes the band.  The bands and the result are those of
`recon_split_sino`.

```bash
python make_synthetic_parallel_sinogram.py 180,64,128 sino.npy
python recon_on_disk_parallel.py --sino sino.npy --slices-per-section 32 --output recon.h5
python make_synthetic_parallel_sinogram.py --check recon.h5
```

This example has two bands.  It runs in about one minute on a CPU with four cores.

A resumed run does not give exactly the result of a run without a stop.  The bands after the
stop use other random pixel partitions, as a second run of the same command does.  If the
process stopped while it wrote a band, the resumed run also blends the seam of that band
twice, and the slices of the seam then have other blending weights.

## Parallel beam: direct reconstruction, `recon_direct_on_disk_parallel.py`

For each band of detector rows, the script reads the rows from the sinogram file,
reconstructs their slices with `ParallelBeamModel.recon_direct`, and writes the band.  A slice
depends only on its own detector row, so the bands need no overlap and no blending.  The result
equals `recon_direct` of the whole volume up to the rounding of float32 arithmetic.

```bash
python make_synthetic_parallel_sinogram.py 180,64,128 sino.npy
python recon_direct_on_disk_parallel.py --sino sino.npy --slices-per-section 32 --output recon.h5
python make_synthetic_parallel_sinogram.py --check recon.h5
```

## Cone beam: direct reconstruction, `recon_direct_on_disk_cone.py`

A slab is a range of slices, and a band is a range of detector rows.  The FDK filter acts
along each detector row alone, and the back projection of a voxel reads only the detector
rows that the voxel touches.  For each slab, the script reads the band of rows that the slab
touches, reconstructs the slab with `ConeBeamModel.recon_direct`, and writes the slab.  The
result equals `recon_direct` of the whole volume up to the rounding of float32 arithmetic.  The
slabs need no overlap and no blending.  The bands of neighboring slabs overlap, so the script
reads some detector rows more than once.

```bash
python make_synthetic_cone_sinogram.py 360,192,192 sino.npy --source-detector-dist 320 --source-iso-dist 160
python recon_direct_on_disk_cone.py --sino sino.npy --source-detector-dist 320 --source-iso-dist 160 --output recon.h5
python make_synthetic_cone_sinogram.py --check recon.h5
```

The two distances are required.  The other geometry options have the names of the
`ConeBeamModel` parameters, and an option that is not given keeps the default of the
package.  The generator and the reconstruction must be given the same geometry options.  For
a run on a CPU, add `--devices cpu --compile-off --slices-per-section 48` to the second
command above, and add `--devices cpu --compile-off` to the first.

The script supports circular scans only.

## Tests

`python check_recon_direct_on_disk.py` runs on a CPU in about one minute.  It checks that each
direct script gives the result of `recon_direct` of the whole volume: for two parallel-beam
scans and ten cone-beam geometries, with five sizes of a part each.  With the cone-beam script,
it also checks that the `.npy` and `.h5` outputs are equal, and that a run that was stopped or
killed and then resumed gives the result of a run without a stop.  These checks run the code of
`on_disk_common.py`, which all reconstruction scripts use.

The folder has no test of the code that only `recon_on_disk_parallel.py` holds.

No GPU has run the scripts of this folder.
