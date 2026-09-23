# Large parallel-beam reconstruction in slice bands

`stream_recon_parallel.py` reconstructs a parallel-beam scan that is too large for the GPUs
at hand, one band of detector rows at a time.  The case it was written for is a 4096 x 4096
detector with 900 views, which gives a 4096 x 4096 x 4096 volume of 275 GB in float32 and a
60 GB sinogram.  `make_synthetic_sinogram.py` writes a phantom sinogram for tests and checks a
reconstruction against the phantom.

## How it works

In parallel beam geometry a detector row is a reconstruction slice, so the problem separates
exactly into bands of rows.  The harness reads one band of sinogram rows from disk,
reconstructs it with `ParallelBeamModel.recon` on the GPUs, blends the seam with the previous
band's last slices read back from the output file, writes the band, and moves on.  The host
holds one band of the sinogram, its weights, and one reconstructed band.  The volume exists
only on disk.

The bands are the ones `recon_split_sino` would choose: each extends 5 rows past its seams,
and the band size comes from the library's memory model for the visible GPUs, with a safety
margin as a parameter (`--margin`, default 0.3; the library itself uses 0.15).  The output
equals `recon_split_sino`'s to floating point round-off (3e-7 NRMSE measured on the same
seed).  A progress file beside the output records each finished band, and `--resume`
continues an interrupted run.  The harness uses one private library call,
`_fits_available_devices`, to size the bands.

## Running it

```bash
python stream_recon_parallel.py --sino sino.npy --angles angles.npy --weights transmission --output recon.h5
```

The sinogram array has axes (views, detector rows, detector channels) in float32.  A `.npy`
file is memory mapped and an HDF5 file is read as hyperslabs, so in both cases only one
band is in memory.  The angles are radians, one per view, or `--angle-span 180` makes
equally spaced views over 180 degrees.  Weights are `none`, `transmission`, or
`transmission_root`, computed per band, or a `.npy` file read per band.  `--num-gpus` pins
the GPU count; `--slices-per-part` overrides the band size.

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

## Host memory

The harness holds one band of the sinogram, its weights, and one reconstructed band.  With
5 bands at 4K that is about 90 GB, so a 4-GPU node, or one GPU with 17 bands, can run the 4K
case.  For comparison, `recon_split_sino` stitches all the bands on the host and would need
about 2.8 volumes, 750 to 810 GB at 4K.

## Measured baseline, 2026-09-23

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

## Time estimates for 4K on H100 cards, not yet measured

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

## Reading the output

A `.h5` output is in the layout of `mbirtorch.export_recon_hdf5`, dataset `recon` with axes
(slice, col, row), read back with `mbirtorch.import_recon_hdf5`.  A `.npy` output is a memory
mapped array with axes (slice, row, col).  The progress file beside the output records each
band's read, reconstruction, and write times, its iteration count, and the peak host memory.
