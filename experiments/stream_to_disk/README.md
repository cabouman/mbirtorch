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

This README has one section for each reconstruction script:
[parallel beam](#large-parallel-beam-reconstruction-in-slice-bands) and
[cone-beam FDK](#cone-beam-fdk-reconstruction-in-slabs).

The two parallel-beam files are unchanged copies of the files in `experiments/parallel_4k`,
made on 2026-11-26.  The parallel-beam section is the README of that folder.
`stream_fdk_cone.py` imports the file reader and the file writer of
`stream_recon_parallel.py`.

A test of the parallel-beam script on a CPU on 2026-11-26 showed two limits.  The copy does
not change them.

- The `.h5` output has no `format` attribute.  `import_recon_hdf5` reads the file in the
  right axis order, and it warns that the file is in the old layout.
- The script does not flush the output file after each band.  After a process that ended
  without cleanup during the second band, the `.h5` output could not be opened, and
  `--resume` failed.

## Large parallel-beam reconstruction in slice bands

`stream_recon_parallel.py` reconstructs a parallel-beam scan that is too large for the GPUs
at hand, one band of detector rows at a time.  The case it was written for is a 4096 x 4096
detector with 900 views, which gives a 4096 x 4096 x 4096 volume of 275 GB in float32 and a
60 GB sinogram.  `make_synthetic_sinogram.py` writes a phantom sinogram for tests and checks a
reconstruction against the phantom.

### How it works

In parallel beam geometry a detector row is a reconstruction slice, so the problem separates
exactly into bands of rows.  The harness reads one band of sinogram rows from disk,
reconstructs it with `ParallelBeamModel.recon` on the GPUs, blends the seam with the previous
band's last slices read back from the output file, writes the band, and moves on.  The host
holds one band of the sinogram, its weights, and one reconstructed band.  The volume exists
only on disk.

The bands are the ones `recon_split_sino` would choose: each extends 5 rows past its seams,
and the band size comes from the library's memory model for the visible GPUs, with a safety
margin as a parameter (`--margin`, default 0.3; the library itself uses 0.15).  The regularization parameters come from a view subsample of the whole sinogram with the
weights of those views, as in `recon` and `recon_split_sino`.  The output equals
`recon_split_sino`'s to floating point round-off (3e-7 NRMSE measured on the same seed).  A progress file beside the output records each finished band, and `--resume`
continues an interrupted run.  The harness uses one private library call,
`_fits_available_devices`, to size the bands.

### Running it

```bash
python stream_recon_parallel.py --sino sino.npy --angles angles.npy --weights transmission --output recon.h5
```

The sinogram array has axes (views, detector rows, detector channels) in float32.  A `.npy`
file is memory mapped and an HDF5 file is read as hyperslabs, so in both cases only one
band is in memory.  The angles are radians, one per view, or `--angle-span 180` makes
equally spaced views over 180 degrees.  Weights are `none`, `transmission`, or
`transmission_root`, computed per band, or a `.npy` file read per band.  `--num-gpus` pins
the GPU count; `--slices-per-section` overrides the band size.

For a test without data, make a phantom sinogram first, then check the result:

```bash
python make_synthetic_sinogram.py 900,4096,4096 sino_4k.npy
python stream_recon_parallel.py --sino sino_4k.npy --angle-span 180 --output recon_4k.h5
python make_synthetic_sinogram.py --check recon_4k.h5
```

The phantom is a low dynamic range Shepp-Logan head.  Its sinogram is made by forward
projecting the phantom band by band through the library's own parallel beam model, and the
file is written band by band, so a 4K sinogram needs no more host memory than one band.

### How many GPUs

The peak memory on one GPU is set by the number of slices that GPU holds at a time, which is
the band's slice count divided by the GPU count.  The library's memory model prices the 4K
case at about 0.27 GiB per slice per GPU, with the compiled prior and the Triton kernels.
The table gives the modeled peak of the worst GPU for a given number of bands and GPUs, in
GiB.  A configuration fits when 1.15 times the peak is below the card's free memory, which
is about 79 GiB on an 80 GB card.

| bands | slices per band | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 1078 | 540 | 270 | 136 |
| 2 | 2053 | 541 | 271 | 136 | 69 |
| 3 | 1376 | 363 | 182 | 92 | 46.5 |
| 4 | 1034 | 273 | 137 | 69 | 35 |
| 5 | 830 | 219 | 110 | 56 | 29 |
| 8 | 522 | 139 | 70 | 36 | 20 |
| 16 | 266 | 71 | 36 | 20 | 13 |
| 24 | 181 | 49 | 25 | 14 | 9.5 |

So on 80 GB cards the 4K case needs about 17 bands in total, spread over the GPUs: 3 bands
on 8 GPUs, 5 on 4, 9 on 2, and 17 on 1.  On 40 GB cards the counts are about double.  The
harness makes this choice itself from the free memory it measures at run time, and its
larger margin gives a few more bands than these counts.

### Host memory

The harness holds one band of the sinogram, its weights, and one reconstructed band.  With
5 bands at 4K that is about 90 GB, so a 4-GPU node, or one GPU with 17 bands, can run the 4K
case.  For comparison, `recon_split_sino` stitches all the bands on the host and would need
about 2.8 volumes, 750 to 810 GB at 4K.

### Measured baseline, 2026-09-23

900 views x 500 rows x 4096 channels, one A100-SXM4-40GB (gilbreth-n010), no weights, seed
0, stop threshold 0.2 percent, torch 2.14.0+cu130, mbirtorch 0.1.1 at commit 7267c47.  The
sinogram, the 33.6 GB result, the logs, and the driver script are in
`/depot/bouman/users/buzzard/results/mbirtorch_parallel_4k/`.

| quantity | measured |
|---|---|
| bands chosen | 6 of about 84 slices (93-row models), margin 0.3 |
| iterations per band | 10, 8, 7, 7, 8, 10 |
| one iteration on a 93-row band | about 26 s |
| setup per band (FBP start, Hessian, initial projection) | about 70 s |
| compile, paid once | about 60 s |
| write to depot per band | 25 to 34 s, about 190 MB/s |
| whole reconstruction | 2004 s; the sinogram took 241 s to make |
| GPU peak | 22.6 GiB allocated, 29.3 GiB reserved, of 39.5 GiB |
| host RSS peak | 31 GB |
| phantom NRMSE on five slices | 0.047 to 0.079 |

Scaled linearly in slices, the 4K case on one A100-40GB would be about 44 bands and 4 to 6
hours, depending on whether bands stop at 8 iterations or run to 15.  Per slice that is
about four times slower than the H100 estimate below.  The card accounts for part of it,
since an A100 has under half the H100's memory bandwidth.  The narrow bands may account for
the rest, because the kernels tile 256 rows at a time and a 93-row band fills them poorly.
An H100 run at full width would separate the two.  The memory model sized the bands for an
allocated peak near 30 GiB, and the run measured 22.6, so at this width the model is
conservative by about a quarter.

### Time estimates for 4K on H100 cards, not yet measured

These extrapolate measured reconstructions of a 1024 x 1008 x 992 volume on H100 cards (the
mbirtorch performance dashboard, September 2026).  The 4K case has 61 times the projection
work and 69 times the voxels of that cell.  One iteration there costs about 4.9 s on one
H100, so one 4K iteration should cost about 5 minutes on one H100, and 15 iterations:

| GPUs (H100) | bands | estimated time |
|---:|---:|---|
| 1 | 17 | about 1.5 hours |
| 4 | 5 | about 25 to 30 minutes, using the measured 3.2x speedup at 4 GPUs |
| 8 | 3 | about 15 to 20 minutes, assuming the scaling continues |

The A100 baseline above suggests these may be optimistic by a factor near two.

### Reading the output

A `.h5` output is in the layout of `mbirtorch.export_recon_hdf5`, dataset `recon` with axes
(slice, col, row), read back with `mbirtorch.import_recon_hdf5`.  A `.npy` output is a memory
mapped array with axes (slice, row, col).  The progress file beside the output records each
band's read, reconstruction, and write times, its iteration count, and the peak host memory.

## Cone-beam FDK reconstruction in slabs

`stream_fdk_cone.py` reconstructs a circular cone-beam scan by FDK when the reconstruction of
the whole volume fits neither on the GPUs nor in host memory.  The case it was written for is
a 3048 x 3048 detector with 1601 views, which gives a 3048 x 3048 x 3048 volume of 113 GB in
float32 and a 59 GB sinogram.  `make_synthetic_cone_sinogram.py` writes a phantom sinogram for
tests and checks a reconstruction against the phantom.  `check_stream_fdk_cone.py` compares
the script with `recon_fdk` of the whole volume on small scans.

No GPU has run `stream_fdk_cone.py` yet.  The section "What is tested" gives the evidence.

### How it works

A slab is a range of slices, and a band is a range of detector rows.  The FDK filter acts
along each detector row alone, and the back projection of a voxel reads only the detector
rows that the voxel touches.  For this reason, a slab can be reconstructed from the band of
rows that its voxels touch.  For each slab, the script reads the band from the sinogram file,
reconstructs the slab with `ConeBeamModel.recon_fdk` on the GPUs, and writes the slab to the
output file.  The host holds one band of the sinogram and one slab of the volume.

The result equals `recon_fdk` of the whole volume up to the rounding of float32 arithmetic.
The slabs need no overlap and no blending, because FDK computes each voxel from the sinogram
alone.

The bands of neighboring slabs overlap.  A voxel at height z above the central plane, with
magnification m, lies at height m z on the detector.  The magnification is larger for the
voxels that are nearer to the source, so one slice touches a range of detector rows, and this
range grows with z.  For this reason, the script reads and filters some detector rows more
than once, and the largest band does not shrink in proportion to the size of the slabs.

The table gives the bands for the example of this section.  The example is a made-up scan of
the size above with a source-to-detector distance of 5080 ALU, a source-to-axis distance of
2540 ALU, and the defaults of the package for all other parameters.  Its magnification at the
axis is 2, and its fan ratio is 0.30.  The fan ratio is half of the detector width divided by
the source-to-detector distance.

| slabs | largest slab (slices) | largest band (rows) | rows of all bands, relative to the detector |
|---:|---:|---:|---:|
| 1 | 3048 | 3048 | 1.00 |
| 2 | 1524 | 1527 | 1.00 |
| 3 | 1016 | 1456 | 1.22 |
| 5 | 610 | 1076 | 1.53 |
| 6 | 508 | 1066 | 1.67 |
| 8 | 381 | 941 | 1.93 |
| 12 | 254 | 871 | 2.48 |
| 16 | 191 | 795 | 3.01 |
| 32 | 96 | 771 | 5.17 |

### Running it

```bash
python stream_fdk_cone.py --sino sino.npy --source-detector-dist 5080 --source-iso-dist 2540 --output recon.h5
```

The sinogram array has axes (views, detector rows, detector channels) in float32.  A `.npy`
file is memory mapped and an HDF5 file is read as hyperslabs, so in both cases only one band
is in memory.  The two distances are required, in ALU.  The other geometry options have the
names of the `ConeBeamModel` parameters: `--delta-det-channel`, `--delta-det-row`,
`--det-channel-offset`, `--det-row-offset`, `--curved-detector`, `--delta-voxel`,
`--recon-shape`, and `--recon-slice-offset`.  An option that is not given keeps the default
of the package, and `--param NAME=VALUE` sets any other model parameter.  The angles are
radians in a `.npy` file given with `--angles`, one per view, or `--angle-span` makes equally
spaced views over that many degrees (default 360).  `--num-gpus` pins the GPU count, and
`--devices` gives the list of devices.

For a test without data, make a phantom sinogram first, then check the result.  The two
scripts must be given the same geometry options.

```bash
python make_synthetic_cone_sinogram.py 1601,3048,3048 sino.npy --source-detector-dist 5080 --source-iso-dist 2540
python stream_fdk_cone.py --sino sino.npy --source-detector-dist 5080 --source-iso-dist 2540 --output recon.h5
python make_synthetic_cone_sinogram.py --check recon.h5
```

The phantom is that of the parallel-beam generator.  The generator forward projects the
phantom one slab of slices at a time through a cone-beam model of the slab and of its band,
and it adds the result to the rows of the band in the sinogram file.  The host holds one slab
of the phantom and one band of the sinogram.  `--slab-slices` sets the number of slices of a
slab (default 64).  The generator has run only on small scans on a CPU.

A small example with the fan ratio of the example above runs on a CPU.  It takes about 5
minutes and about 6 GB of host memory.

```bash
python make_synthetic_cone_sinogram.py 360,192,192 sino.npy --source-detector-dist 320 --source-iso-dist 160 --devices cpu --compile-off
python stream_fdk_cone.py --sino sino.npy --source-detector-dist 320 --source-iso-dist 160 --output recon.h5 --devices cpu --compile-off --num-slabs 4
python make_synthetic_cone_sinogram.py --check recon.h5
```

The check reports an NRMSE of 0.157 for the central slice and of 0.222 for the checked
slices nearest to the two ends of the volume.  FDK is an approximate reconstruction of a
cone-beam scan, and its error grows toward the two ends of the rotation axis.  The NRMSE is
therefore larger than for the parallel-beam script.

`check_stream_fdk_cone.py` needs no data and no GPU.  It takes less than one minute on a CPU.

```bash
python check_stream_fdk_cone.py
```

### How many slabs

`--num-slabs` sets the number of slabs.  Without it, the script chooses the smallest number
of slabs for which the memory check of the package accepts every slab on the visible GPUs.
The check accepts a slab if (1 + margin) times the modeled peak memory of its reconstruction
is at most the free memory of each GPU.  `--margin` sets the margin (default 0.3; the package
itself uses 0.15).  If two or more GPUs are visible and `--devices` is not given, the package
runs the same check with this margin before it reconstructs each slab.  On a device that is
not a GPU, the check has no limit, and `--num-slabs` is required.

The table gives the choice of the script for the example at the default margin.  The values
come from the memory model of the package for an assumed free memory.  They are not runs.
The table assumes 38 GiB of free memory for a 40 GB card and 79 GiB for an 80 GB card.

| GPUs | free memory per GPU (GiB) | slabs | largest slab (slices) | largest band (rows) | modeled peak per GPU (GiB) | band and slab on the host (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38 | 16 | 191 | 795 | 29.1 | 23 |
| 2 | 38 | 5 | 610 | 1076 | 28.7 | 44 |
| 4 | 38 | 3 | 1016 | 1456 | 22.4 | 66 |
| 1 | 79 | 5 | 610 | 1076 | 57.2 | 44 |
| 2 | 79 | 3 | 1016 | 1456 | 44.7 | 66 |
| 4 | 79 | 2 | 1524 | 1527 | 30.6 | 86 |

The default margin comes from one GPU test of 2026-11-26.  A study script reconstructed the
example with the same slabs and bands on H100 cards, with mbirtorch 0.2.0 and torch 2.14.1.
It limited torch to 38 GiB on each GPU, and it used the default settings of the memory
allocator of torch.  It kept the sinogram in host memory and wrote no file.

| GPUs | slabs | modeled peak per GPU (GiB) | limit / modeled peak | result |
|---:|---:|---:|---:|---|
| 2 | 8 | 20.6 | 1.85 | finished in 210 s; 20.9 GiB allocated, 29.8 GiB reserved |
| 2 | 6 | 25.4 | 1.49 | finished in 187 s; 25.9 GiB allocated, 37.1 GiB reserved |
| 1 | 16 | 29.1 | 1.31 | finished in 403 s; 29.8 GiB allocated, 31.3 GiB reserved |
| 1 | 12 | 31.9 | 1.19 | out of memory |
| 2 | 4 | 33.5 | 1.13 | out of memory |

The allocated memory was within 2.4 percent of the modeled peak.  The memory that torch
reserved was up to 44 percent larger than the allocated memory, and the limit applies to the
reserved memory.  The margin of the package, 0.15, accepts the case of 12 slabs on one GPU,
which ran out of memory.  A margin between 0.20 and 0.30 separates the three cases that
finished from the two that failed, and the default of 0.3 is at the upper end of this
range.  No run has tested the choice for two 40 GB cards in the table above (5 slabs, limit /
modeled peak 1.32).

The script sets the allocator option `expandable_segments:True`, as the parallel-beam script
does.  In an iterative reconstruction, this option halved the memory that torch reserved and
did not use.  The test above did not set the option, so its effect on FDK by slabs has not
been measured.

### Host memory and time

The host holds one band of the sinogram and one slab of the volume.  For the example, the
last column of the first table under "How many slabs" gives the size of the two together.
The package uses more host memory during a reconstruction, and this amount has not been
measured for this script.  The progress file records the peak host memory after each slab.

The time of this script has not been measured on a GPU.  The second table under "How many
slabs" gives the reconstruction times of the GPU test.  This script also reads the bands and
writes the volume.  At the write speed of the parallel-beam baseline above, about 190 MB/s,
the write of the 113 GB volume of the example takes about 10 minutes.

### Output and resume

A `.h5` output is in the layout of `mbirtorch.export_recon_hdf5`: dataset `recon` with axes
(slice, col, row) and the attributes of that function.  `mbirtorch.import_recon_hdf5` reads
it without a warning.  A `.npy` output is a memory mapped array with axes (slice, row, col).

After each slab, the script flushes the output file and then replaces the progress file
beside the output.  The same command with `--resume` continues an interrupted run from the
first unfinished slab.  This also holds for a run whose process ended without cleanup during
a write.  A resumed run keeps the number of slabs of the interrupted run.  It stops if the
shapes, the geometry, the angles, or the filter differ from those of the interrupted run.

For each slab, the progress file records the slices, the band, the times to read, to
reconstruct, and to write, the peak host memory, and the peak memory that torch allocated and
reserved on each GPU.

### What is tested

All tests of this script ran on a CPU, with mbirtorch 0.2.0 at commit dfd0c2b and torch
2.15.0.  `check_stream_fdk_cone.py` passes with these results:

- The script was run on ten geometries, each with five numbers of slabs from 1 to 48.  The
  result differs from `recon_fdk` of the whole volume by at most 5.4e-6 of the largest
  value.  The geometries cover fan ratios from 0.10 to 0.45, a tall volume, detector offsets
  and unequal pitches, a square support, thick and thin slices, a curved detector, a volume
  of given shape, pitch, and offset, and views over 200 degrees.
- A forward projection of each slab shows that its band holds every detector row that the
  slab touches.  As a control, a slab that is reconstructed without the first and the last
  row that it touches differs from the whole reconstruction by 4.1e-2 or more.
- The `.npy` output equals the `.h5` output.  A run that was stopped and resumed, and a run
  whose process was ended without cleanup and that was resumed, both equal the run without a
  stop, for both file types.  A sinogram in an HDF5 file gives the same result as the `.npy`
  sinogram.
- The phantom sinogram that the generator makes one slab at a time differs from the forward
  projection of the whole phantom by 1.3e-6 of the largest value.

The choice of the number of slabs was run on a CPU with made-up numbers of GPUs and made-up
free memory.  Its modeled peaks equal those of the GPU test for the same slabs.

The following has not been tested, because no GPU has run the scripts: the choice of the
number of slabs on real GPUs, the reconstruction on more than one GPU, the allocator option,
the time to read and to write the files, and the sinogram generator at the size of the
example.  The GPU test of 2026-11-26 covers the method and not this script.  In that test, FDK by slabs differed from `recon_fdk` of the whole
volume of a measured scan by at most 1.3e-5 of the largest value, and it took 1.3 to 2.2
times as long.

### Limits

- The script supports circular scans only.  For a helical scan, `recon_fdk` scales each slice
  by a factor that depends on the number of detector rows of the model, and the model of a
  band has fewer rows.
- The script gives what `recon_fdk` gives.  At commit dfd0c2b, `recon_fdk` assumes equally
  spaced views over the full angular range, and it applies no short-scan weighting.
- The script uses two private calls of the package.  `TomographyModel._fits_available_devices`
  is the memory check, and `mbirtorch.utilities._recon_attributes` gives the attributes of
  the HDF5 output.
