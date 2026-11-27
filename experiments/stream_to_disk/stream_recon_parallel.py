"""Stream a large parallel-beam reconstruction through the GPUs, one band of slices at a time.

This is a proof of concept for a volume that fits neither on the GPUs nor in host memory
as a whole.  One band of detector rows is in play at a time.  The harness reads the band's
rows of the sinogram from disk, reconstructs them on the GPUs with
``ParallelBeamModel.recon``, blends the band's first slices with the previous band's last
slices at the seam, writes the band to the output file, and moves on.  The host holds one
band of the sinogram, its weights, and one reconstructed band.  The volume exists only on
disk.

The bands and the result are those of ``recon_split_sino``.  The band size comes from the
library's memory model for the visible GPUs.  Every band extends ``half_overlap`` rows
past its seams, so consecutive bands share ``2 * half_overlap`` slices, and the seam is
blended over those slices with ``stitch_arrays`` and its weights.  The regularization
parameters are set once, from a view subsample of the whole sinogram with the weights of
those views, exactly as ``recon`` and ``recon_split_sino`` set them, and copied into every
band's model.  With the same seed and bands, the output equals ``recon_split_sino``'s to
floating point round-off.

    python stream_recon_parallel.py --sino sino.npy --angles angles.npy --weights transmission --output recon.h5
    python stream_recon_parallel.py --sino sino.npy --angles angles.npy --output recon.h5 --resume

For a test without data, ``make_synthetic_sinogram.py`` writes a phantom sinogram and checks a
reconstruction against the phantom.

Inputs.  The sinogram is a ``.npy`` file, which is memory mapped so that only a band's rows
are read, or an HDF5 file, from which a band's rows are read as a hyperslab.  Its axes are
(views, detector rows, detector channels) in float32.  The angles are radians in a ``.npy``
file, or ``--angle-span`` gives equally spaced views over that many degrees.  Weights are
``none``, ``transmission`` or ``transmission_root`` computed per band, or a ``.npy`` file
read per band.

Output.  A ``.h5`` path gives an HDF5 file in the layout of ``export_recon_hdf5``: dataset
``recon`` with axes (slice, col, row), readable with ``import_recon_hdf5``.  A ``.npy``
path gives a memory mapped array with axes (slice, row, col).  Both are written band by
band.  A progress file beside the output records each finished band, and ``--resume``
continues an interrupted run from the first unfinished band.

Private library calls.  The band size uses ``TomographyModel._fits_available_devices``,
the memory check ``recon_split_sino`` itself uses.  The attributes of the HDF5 output come from
``mbirtorch.utilities._recon_attributes``, which ``stream_common.py`` calls.  Everything else
is public API.

Shared code.  ``stream_common.py`` holds the code that this script shares with
``stream_fdk_cone.py``: the reader of the sinogram file, the writer of the output file, the
progress file, and the loop over the bands.
"""

import argparse
import os
import time

# stream_common sets the allocator option of torch, so this import comes before that of torch.
from stream_common import (
    ProgressFile, SinogramSource, add_angle_arguments, add_device_arguments, add_input_arguments,
    add_output_arguments, device_list, finish_output, open_output, print_sizes, print_summary, run_parts,
    smallest_fitting_count, view_angles)

import numpy as np
import torch

import mbirtorch

DEFAULT_ANGLE_SPAN_DEG = 180.0
# The name of a part of the run in the messages and in the help texts.
PART_NAME = 'band'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    src = add_input_arguments(p)
    add_angle_arguments(src, DEFAULT_ANGLE_SPAN_DEG)
    src.add_argument('--weights', default='none',
                     help="'none' (default), 'transmission', 'transmission_root', or a .npy file")
    out = add_output_arguments(p, PART_NAME)
    out.add_argument('--log', default='~/.mbirtorch/logs/stream_recon_parallel.log',
                     help='merged reconstruction log (default ~/.mbirtorch/logs/stream_recon_parallel.log)')
    out.add_argument('--quiet', action='store_true', help='do not print the per-iteration log lines')
    dev = add_device_arguments(p, PART_NAME)
    dev.add_argument('--slices-per-section', type=int,
                     help='slices per band (default: the most the memory model says fit the GPUs)')
    dev.add_argument('--half-overlap', type=int, default=5,
                     help='rows kept past each seam on each side (default 5)')
    rec = p.add_argument_group('reconstruction')
    rec.add_argument('--max-iterations', type=int, default=15)
    rec.add_argument('--stop-pct', type=float, default=0.2,
                     help='stop when the relative change per iteration is below this percent '
                          '(default 0.2; 0 runs exactly max-iterations)')
    rec.add_argument('--sharpness', type=float, default=1.0)
    rec.add_argument('--snr-db', type=float, default=30.0)
    rec.add_argument('--positivity', action='store_true')
    rec.add_argument('--seed', type=int, help='numpy seed for the pixel partitions')
    return p.parse_args(argv)


# ── the bands ────────────────────────────────────────────────────────────────
def band_ranges(num_rows, num_parts, half_overlap):
    """The kept slice range and the model row range of each band, as recon_split_sino
    tiles them: nearly equal kept ranges, each model extended half_overlap rows past its
    interior seams."""
    base, extra = divmod(num_rows, num_parts)
    ranges, start = [], 0
    for index in range(num_parts):
        stop = start + base + (1 if index < extra else 0)
        model_lo, model_hi = max(start - half_overlap, 0), min(stop + half_overlap, num_rows)
        ranges.append(((start, stop), (model_lo, model_hi)))
        start = stop
    return ranges


def worst_part_rows(num_rows, num_parts, half_overlap):
    biggest_kept = -(-num_rows // num_parts)
    if num_parts == 1:
        return biggest_kept
    if num_parts == 2:
        return biggest_kept + half_overlap
    return biggest_kept + 2 * half_overlap


def band_model(full_model, num_band_rows):
    """A copy of the full model covering ``num_band_rows`` detector rows, and so that many
    slices, with the full model's regularization parameters fixed."""
    recon_rows, recon_cols = full_model.get_params('recon_shape')[:2]
    model = mbirtorch.copy_ct_model(full_model, new_num_det_rows=num_band_rows, no_warning=True)
    model.set_params(no_warning=True, auto_regularize_flag=False,
                     recon_shape=(recon_rows, recon_cols, num_band_rows))
    return model


def choose_num_parts(full_model, num_rows, half_overlap, margin, weights_supplied):
    """The fewest bands whose largest band's modeled peak fits the GPUs with ``margin``.
    This is the choice recon_split_sino makes, with the margin as a parameter."""
    if not torch.cuda.is_available():
        raise SystemExit('without CUDA the memory model cannot read a GPU budget; '
                         'give --slices-per-section')
    max_parts = max(1, num_rows // (2 * half_overlap))

    def fits(num_parts):
        model = band_model(full_model, worst_part_rows(num_rows, num_parts, half_overlap))
        model.memory_check_margin = margin
        call_arrays = {'weights': 'supplied'} if weights_supplied else {}
        return model._fits_available_devices(**call_arrays)

    num_parts = smallest_fitting_count(fits, max_parts)
    if num_parts is None:
        raise SystemExit(f'no band of at least {2 * half_overlap} slices fits the GPUs at margin {margin}')
    return num_parts


def blend_ramp(half_overlap):
    """The ramp recon_split_sino hands to stitch_arrays: at most 4 blended slices, even."""
    ramp = min(4, half_overlap)
    return ramp - ramp % 2


# ── the run ──────────────────────────────────────────────────────────────────
def main(argv=None):
    args = parse_args(argv)
    devices = device_list(args)

    t_start = time.time()
    source = SinogramSource(args.sino, args.dataset)
    num_views, num_rows = source.shape[:2]
    angles = view_angles(args, num_views)
    weights_source = None
    if args.weights.endswith('.npy'):
        weights_source = SinogramSource(args.weights)
        if weights_source.shape != source.shape:
            raise ValueError(f'weights shape {weights_source.shape} differs from {source.shape}')
    weights_supplied = args.weights != 'none'

    # The full model carries the geometry and the regularization.  No data of its size is
    # ever placed on a device: the regularization statistics run on the view subsample
    # recon would take, with the weights of those views, exactly as recon does.
    full = mbirtorch.ParallelBeamModel(source.shape, angles)
    full.set_params(sharpness=args.sharpness, snr_db=args.snr_db,
                    positivity_flag=args.positivity, verbose=1, no_warning=True)
    recon_shape = tuple(int(s) for s in full.get_params('recon_shape'))
    sample = source.view_subsample()
    if weights_source is not None:
        sample_weights = weights_source.view_subsample()
    elif weights_supplied:
        sample_weights = mbirtorch.gen_weights(sample, weight_type=args.weights)
    else:
        sample_weights = None
    regularization = full.auto_set_regularization_params(sample, weights=sample_weights)
    del sample, sample_weights
    print_sizes(source.shape, recon_shape)
    print('regularization from the whole sinogram: '
          + ', '.join(f'{k}={v:.4g}' for k, v in regularization.items()), flush=True)

    # The progress file makes the run resumable and records each band's cost.
    progress = ProgressFile(args.output, PART_NAME, args.resume)
    ho = args.half_overlap
    given = None if args.slices_per_section is None else max(1, -(-num_rows // args.slices_per_section))
    num_parts = progress.num_parts(
        given, lambda: choose_num_parts(full, num_rows, ho, args.margin, weights_supplied))
    if num_parts > 1 and num_rows // num_parts < 2 * ho:
        raise SystemExit(f'{num_parts} bands leave fewer than {2 * ho} slices each; use fewer bands '
                         'or a smaller --half-overlap')
    # Each entry of the plan holds the kept slices of a band and the rows of its model.
    plan = [(lo, hi, mlo, mhi) for (lo, hi), (mlo, mhi) in band_ranges(num_rows, num_parts, ho)]
    print(f'{num_parts} band(s) of about {-(-num_rows // num_parts)} slices, overlap {ho} rows per seam',
          flush=True)
    config = {'sinogram_shape': list(source.shape), 'recon_shape': list(recon_shape),
              'num_parts': num_parts, 'half_overlap': ho, 'weights': args.weights,
              'max_iterations': args.max_iterations, 'stop_pct': args.stop_pct,
              'sharpness': args.sharpness, 'snr_db': args.snr_db,
              'regularization': regularization}
    sink = open_output(args.output, recon_shape, progress, config)

    if args.seed is not None:
        np.random.seed(args.seed)
    log_path = os.path.expanduser(args.log)
    part_logs = [f'{log_path}.part{k}' for k in range(num_parts)]
    ramp = blend_ramp(ho)

    def read(k):
        """Return the sinogram rows of the model of band k, and their weights."""
        _lo, _hi, mlo, mhi = plan[k]
        sino_band = source.read_rows(mlo, mhi)
        if weights_source is not None:
            weights = weights_source.read_rows(mlo, mhi)
        elif weights_supplied:
            weights = mbirtorch.gen_weights(sino_band, weight_type=args.weights)
        else:
            weights = None
        return sino_band, weights

    def reconstruct(k, data):
        """Return the reconstruction of band k, and its number of iterations for the record."""
        sino_band, weights = data
        _lo, _hi, mlo, mhi = plan[k]
        model = band_model(full, mhi - mlo)
        if devices:
            model.configure_devices(devices=devices)
        band, band_dict = model.recon(sino_band, weights=weights, max_iterations=args.max_iterations,
                                      stop_threshold_change_pct=args.stop_pct,
                                      logfile_path=part_logs[k], print_logs=not args.quiet)
        return band, {'iterations': int(band_dict['recon_params']['num_iterations'])}

    def write(k, band):
        # The band's first 2 * ho slices are the seam it shares with the previous band, whose
        # last 2 * ho slices are already on disk.  The blend is stitch_arrays over those two
        # slabs, so it is the blend recon_split_sino computes.  The rest of the band is
        # written as it is; its own last 2 * ho slices wait for the next band's blend.
        mlo = plan[k][2]
        if k > 0:
            seam = 2 * ho
            previous_tail = sink.read(mlo, mlo + seam)
            blended = mbirtorch.stitch_arrays([previous_tail, band[:, :, :seam]],
                                              overlap=seam, axis=2, ramp_overlap=ramp)
            sink.write(blended, mlo)
            sink.write(band[:, :, seam:], mlo + seam)
        else:
            sink.write(band, mlo)

    run_parts(plan, read, reconstruct, sink, progress, args.max_parts, write=write)

    recon_params = dict(config, method='MBIR by bands (stream_recon_parallel.py)',
                        iterations_per_part=[entry['iterations'] for entry in progress.done])
    finish_output(sink, progress, full, recon_params)
    source.close()
    if weights_source is not None:
        weights_source.close()
    if progress.finished:
        labels = [f'stream_recon_parallel: band {k + 1} of {num_parts} (slices {lo}-{hi - 1})'
                  for k, (lo, hi, _mlo, _mhi) in enumerate(plan)]
        mbirtorch.merge_log_files(log_path, zip(labels, part_logs))
    print_summary(progress, time.time() - t_start, args.output)


if __name__ == '__main__':
    main()
