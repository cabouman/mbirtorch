# Reconstruction that streams from disk to disk

Each reconstruction script in this folder is for a scan that is too large for the GPUs at
hand.  The script reads one part of the sinogram from a file, reconstructs the slices that
this part determines, writes them to the output file, and continues with the next part.  The
volume exists only on disk.

| file | purpose |
|---|---|
| `stream_recon_parallel.py` | Iterative reconstruction of a parallel-beam scan, one band of detector rows at a time. |
| `make_synthetic_sinogram.py` | Phantom sinogram for the parallel-beam script, and comparison of a reconstruction with the phantom. |
| `stream_fdk_cone.py` | FDK reconstruction of a circular cone-beam scan, one slab of slices at a time. |
| `make_synthetic_cone_sinogram.py` | Phantom sinogram for the cone-beam script, and comparison of a reconstruction with the phantom. |
| `check_stream_fdk_cone.py` | Comparison of `stream_fdk_cone.py` with the FDK reconstruction of the whole volume, on small scans. |
| `stream_common.py` | Code that the two reconstruction scripts share. |

`python <script> --help` lists the options of each script.

The two reconstruction scripts use two private functions of the package:
`TomographyModel._fits_available_devices` is the memory check, and
`mbirtorch.utilities._recon_attributes` gives the attributes of the HDF5 output.

## What the two reconstruction scripts share

Each script reconstructs the scan in parts.  A part is a band of the parallel-beam script or
a slab of the cone-beam script.  `stream_common.py` holds the code that does not depend on
the geometry or on the reconstruction method: the options for the files, the devices, and the
memory check, the reader of the sinogram file, the writer of the output file, the progress
file, and the loop over the parts.  Each script holds its own options, its model, its
division of the scan into parts, and its reconstruction of one part.  The parallel-beam
script also holds the blending of two bands.  The two scripts therefore read and write the
same files, and they record and resume a run in the same way.

The sinogram is a `.npy` file, which is memory mapped, or an HDF5 file.  Its axes are (views,
detector rows, detector channels) in float32.  The script reads only the detector rows that
it needs for one part.

A `.h5` output is in the layout of `mbirtorch.export_recon_hdf5`: dataset `recon` with axes
(slice, col, row).  `mbirtorch.import_recon_hdf5` reads it.  A `.npy` output is a memory
mapped array with axes (slice, row, col).

A progress file beside the output, `<output name>_progress.json`, records each finished
part.  The same command with `--resume` continues an interrupted run from the first
unfinished part, with the number of parts of the interrupted run.  The script flushes the
output file before it records a part, so the output of a process that is killed during a
reconstruction holds every recorded part.  The HDF5 library does not guarantee that an `.h5`
file can be opened again if the process is killed while it writes to the file.  A `.npy`
output does not have this limit.

## Parallel beam: `stream_recon_parallel.py`

In parallel-beam geometry, a detector row is a reconstruction slice.  For each band of
detector rows, the script reads the rows from the sinogram file, reconstructs them with
`ParallelBeamModel.recon`, blends the first slices of the band with the last slices of the
previous band, and writes the band.  The bands and the result are those of
`recon_split_sino`.  The script chooses the number of bands with the memory check of the
package, or `--slices-per-section` sets the size of a band.  Without a GPU,
`--slices-per-section` is required.

```bash
python make_synthetic_sinogram.py 180,64,128 sino.npy
python stream_recon_parallel.py --sino sino.npy --angle-span 180 --slices-per-section 32 --output recon.h5
python make_synthetic_sinogram.py --check recon.h5
```

This example has two bands.  It runs in about one minute on a CPU with four cores.

A resumed run does not give exactly the result of a run without a stop.  The bands after the
stop use other random pixel partitions, as a second run of the same command does.  If the
process stopped while it wrote a band, the resumed run also blends the seam of that band
twice, and the slices of the seam then have other blending weights.

The script and the generator come from `experiments/parallel_4k`.  The generator is an
unchanged copy.  The part of the script that the cone-beam script also needs is now in
`stream_common.py`.  The script differs from the original in five points:

- It flushes an `.h5` output when it creates the file, and it flushes the output after each
  band.  The `.h5` output of the original cannot be opened after the process is killed, and
  `--resume` then fails.
- A new run replaces the progress file of an earlier run before it creates the output file.
  With the original, `--resume` after a new run that was killed in its first band skips the
  bands that the earlier run recorded.  In a `.npy` output, the slices of these bands are
  then zero.
- A resumed run keeps the number of bands of the interrupted run.  The original chooses the
  number again, and it refuses to resume if the choice differs.
- It writes the attributes of `export_recon_hdf5`.  `import_recon_hdf5` then reads the file
  without a warning, and it returns the parameters of the model and of the reconstruction.
- It records the peak GPU memory of each band in the progress file.  For this, it releases the
  unused GPU memory of torch before each band.

The README of `experiments/parallel_4k` gives the measured memory and time of the original
script for 900 views of 500 x 4096 pixels on one GPU, and the modeled memory for a
4096 x 4096 detector.

## Cone beam: `stream_fdk_cone.py`

A slab is a range of slices, and a band is a range of detector rows.  The FDK filter acts
along each detector row alone, and the back projection of a voxel reads only the detector
rows that the voxel touches.  For each slab, the script reads the band of rows that the slab
touches, reconstructs the slab with `ConeBeamModel.recon_fdk`, and writes the slab.  The
result equals `recon_fdk` of the whole volume up to the rounding of float32 arithmetic.  The
slabs need no overlap and no blending.

```bash
python make_synthetic_cone_sinogram.py 360,192,192 sino.npy --source-detector-dist 320 --source-iso-dist 160
python stream_fdk_cone.py --sino sino.npy --source-detector-dist 320 --source-iso-dist 160 --output recon.h5
python make_synthetic_cone_sinogram.py --check recon.h5
```

The two distances are required.  The other geometry options have the names of the
`ConeBeamModel` parameters, and an option that is not given keeps the default of the
package.  The generator and the reconstruction must be given the same geometry options.

`--num-slabs` sets the number of slabs.  Without it, the script takes the smallest number of
slabs that the memory check of the package accepts on the GPUs, with the margin of
`--margin`.  On a device that is not a GPU, `--num-slabs` is required.  For a run on a CPU,
add `--devices cpu --compile-off --num-slabs 4` to the second command above, and add
`--devices cpu --compile-off` to the first.

The script supports circular scans only.

## Tests

`python check_stream_fdk_cone.py` runs on a CPU in less than one minute.  It checks that
`stream_fdk_cone.py` gives the result of `recon_fdk` of the whole volume for ten geometries,
that the `.npy` and `.h5` outputs are equal, and that a run that was stopped or killed and
then resumed gives the result of a run without a stop.  The last checks run the code of
`stream_common.py`, which both reconstruction scripts use.

The folder has no test of the code that only `stream_recon_parallel.py` holds.

No GPU has run the scripts of this folder.  The original parallel-beam script in
`experiments/parallel_4k` has run on one GPU.
