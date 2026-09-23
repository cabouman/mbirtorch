# Large parallel-beam reconstruction in slice bands

These scripts reconstruct a parallel-beam scan that is too large for the GPUs at hand.  The
case they were written for is a 4096 x 4096 detector with 900 views.  That scan gives a
4096 x 4096 x 4096 volume of 275 GB in float32 and a 60 GB sinogram.

In parallel beam geometry a detector row is a reconstruction slice, so the problem separates
exactly into bands of rows.  Both reconstruction scripts work band by band and blend a short
overlap at each seam, and the result is approximately the same as a single reconstruction of
the whole volume.

| script | what it does |
|---|---|
| `recon_parallel_split.py` | calls the library's `recon_split_sino`, which reconstructs the bands and stitches them on the host; also prices a scan with `--plan` |
| `stream_recon_parallel.py` | the streaming harness: reconstructs and writes one band at a time, so the host never holds the volume |
| `make_synthetic_sinogram.py` | writes a phantom sinogram for tests, and checks a reconstruction against the phantom |
| `gautschi_4k.sbatch` | the benchmark job for the Purdue gautschi cluster |

## Running them

Give the sinogram, the angles, and an output path:

```bash
python stream_recon_parallel.py --sino sino.npy --angles angles.npy --weights transmission --output recon.h5
```

The sinogram array has axes (views, detector rows, detector channels) in float32.  A `.npy`
file is memory mapped and read band by band.  An HDF5 file is read band by band by the
streaming harness and whole by `recon_parallel_split.py`.  The angles are radians, one per
view, or `--angle-span 180` makes equally spaced views over 180 degrees.

`recon_parallel_split.py --plan` prints the memory plan and exits: the part count the
library will choose, the peak memory per GPU, and the host memory the run needs.  With
`--shape 900,4096,4096` it needs no sinogram file.  It runs on any machine; without a GPU,
give the card size with `--gpu-memory-gb` and the count with `--num-gpus`.

For a test without data, make a phantom sinogram first, then check the result:

```bash
python make_synthetic_sinogram.py 900,4096,4096 sino_4k.npy
python stream_recon_parallel.py --sino sino_4k.npy --angle-span 180 --output recon_4k.h5
python make_synthetic_sinogram.py --check recon_4k.h5
```

The phantom is a low dynamic range Shepp-Logan head.  Its sinogram is made by forward
projecting the phantom band by band through the library's own parallel beam model, and the
file is written band by band, so a 4K sinogram needs no more host memory than one band.

## How many GPUs

The peak memory on one GPU is set by the number of slices that GPU holds at a time, which is
the band's slice count divided by the GPU count.  The library's memory model prices the 4K
case at about 0.27 GiB per slice per GPU.  The model was checked against measured peaks at
the 2048 class and agreed within 7 percent.  The table gives the modeled peak of the worst GPU
for a given number of parts and GPUs, in GiB.  A configuration fits when 1.15 times the peak
is below the card's free memory, which is about 79 GiB on an 80 GB card.

| parts | slices per part | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 1078 | 540 | 270 | 136 |
| 2 | 2053 | 541 | 271 | 136 | 69 |
| 3 | 1376 | 363 | 182 | 92 | 46.5 |
| 4 | 1034 | 273 | 137 | 69 | 35 |
| 5 | 830 | 219 | 110 | 56 | 29 |
| 8 | 522 | 139 | 70 | 36 | 20 |
| 16 | 266 | 71 | 36 | 20 | 13 |
| 24 | 181 | 49 | 25 | 14 | 9.5 |

So on 80 GB cards the run needs about 17 slice bands in total, spread over the GPUs:

| GPUs | parts on 80 GB cards | parts on 40 GB cards | parts on 24 GB cards |
|---:|---:|---:|---:|
| 1 | 17 (241 slices each) | 36 | not priced |
| 2 | 9 (456) | 18 | 35 |
| 4 | 5 (820) | 9 | 17 |
| 8 | 3 (1366) | 5 | 9 |

These are the counts `--plan` reports for nominal cards.  The library makes the choice itself
when `--slices-per-part` is not given, from the free memory it measures at run time, so a
count can differ by one from the table.  The streaming harness uses the same check with a
larger safety margin (`--margin`, default 0.3), so it uses a few more bands.

## Host memory

`recon_split_sino` stitches the parts on the host, and the stitch is its host memory peak.
During the last stitch three volumes are in memory: the list of parts, the volume stitched so
far, and the new stitched volume.  At 4K that is 730 GB with 3 parts and 810 GB with 17.  The
sinogram adds 60 GB unless it is memory mapped from a `.npy` file, and the weights add another
60 GB.  `--plan` prints this number.  A node with 1 TB of host memory runs the 4K case this
way; a node with 512 GB does not, whatever its GPU count.

On gautschi the `ai` partition allocates 9200 MB of host memory per CPU and 14 CPUs per GPU,
and it refuses `--mem`.  Eight GPUs are therefore the only allocation with enough host
memory, about 1030 GB, for the stitch.

The streaming harness avoids the limit.  It reads one band of sinogram rows from disk,
reconstructs it with `ParallelBeamModel.recon`, blends the seam with the previous band's last
slices read back from the output file, writes the band, and moves on.  The host holds one
band of the sinogram, its weights, and one reconstructed band.  With 5 bands at 4K that is
about 90 GB, so a 4-GPU half node, or one GPU with 17 bands, can run the 4K case.

The bands are the ones `recon_split_sino` chooses, and the output equals its output to
floating point round-off (measured 3e-7 NRMSE on the same seed).  A progress file beside the
output records each finished band, and `--resume` continues an interrupted run.  The harness
uses one private library call, `_fits_available_devices`, to size the bands.  Exposing that
check, and a slab-wise gather of a reconstruction from the GPUs to disk, are the two
interfaces that would let the harness use public API only and hold slabs rather than a whole
band on the host.

## Time

The estimates below extrapolate measured reconstructions of a 1024 x 1008 x 992 volume on
H100 cards (the mbirtorch performance dashboard, September 2026).  The 4K case has 61 times
the projection work and 69 times the voxels of that cell.  One iteration there costs about
4.9 s on one H100, so one 4K iteration should cost about 5 minutes on one H100.  A
reconstruction of 15 iterations then takes:

| GPUs (H100) | bands | estimated time |
|---:|---:|---|
| 1 | 17 | about 1.5 hours |
| 4 | 5 | about 25 to 30 minutes, using the measured 3.2x speedup at 4 GPUs |
| 8 | 3 | about 15 to 20 minutes, assuming the scaling continues |

Add a few minutes for reading the sinogram and writing the 275 GB result.  A 40 GB A100 runs
these kernels roughly two times slower than an H100.  The batch file measures the actual
times on gautschi, and its results supersede this table.

## Reading the output

Both reconstruction scripts write HDF5 in the layout of `mbirtorch.export_recon_hdf5`:
dataset `recon` with axes (slice, col, row), read back with `mbirtorch.import_recon_hdf5`.
The streaming harness also accepts a `.npy` output path, which gives a memory mapped array
with axes (slice, row, col).  A JSON file beside the output records the timings, the band
ranges, the iteration count of each band, the peak host memory, and the peak memory of each
GPU.
