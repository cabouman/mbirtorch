"""Reconstruct a large parallel-beam scan by MBIR, one band of slices at a time.

This script is for a volume that fits neither on the GPUs nor in host memory as a whole.  In
parallel beam, detector row r gives slice r.  For each band of detector rows, the script reads
the rows from the sinogram file, reconstructs them on the GPUs with ``ParallelBeamModel.recon``,
blends the first slices of the band with the last slices of the previous band, and writes the
band to the output file.  The host holds one band of the sinogram, its weights, and one
reconstructed band.  The volume exists only on disk.

The bands and the result are those of ``recon_split_sino``.  Every band extends
``half_overlap`` rows past its seams, so consecutive bands share ``2 * half_overlap`` slices,
and the seam is blended over those slices with ``stitch_arrays`` and its weights.  The
regularization parameters are set once, from a view subsample of the whole sinogram with the
weights of those views, as ``recon`` and ``recon_split_sino`` set them, and copied into the model
of every band.  With the same seed and bands, the output equals the output of
``recon_split_sino`` up to floating point rounding.

    python recon_on_disk_parallel.py --sino sino.npy --angles angles.npy --weights transmission --output recon.h5
    python recon_on_disk_parallel.py --sino sino.npy --angles angles.npy --output recon.h5 --resume

The function ``recon_on_disk_parallel`` does the reconstruction.  It takes the
``ParallelBeamModel`` of the whole scan, so that it can become a method of the model.

Inputs.  The sinogram is a ``.npy`` file, which is memory mapped so that only a band's rows
are read, or an HDF5 file, from which a band's rows are read as a hyperslab.  Its axes are
(views, detector rows, detector channels) in float32.  The angles are radians in a ``.npy``
file, or ``--angle-span`` gives equally spaced views over that many degrees.  Weights are
``none``, ``transmission`` or ``transmission_root`` computed per band, or a ``.npy`` file
read per band.

Bands.  ``--slices-per-section`` sets the largest number of slices that a band gives to the
volume.  Without it, the script uses the smallest number of bands for which the memory check of
the package accepts the model of every band on the GPUs, with the margin of ``--margin``.  On a
device that is not a GPU, ``--slices-per-section`` is required.

Output.  A ``.h5`` path gives an HDF5 file in the layout of ``export_recon_hdf5``: dataset
``recon`` with axes (slice, col, row), readable with ``import_recon_hdf5``.  A ``.npy``
path gives a memory mapped array with axes (slice, row, col).  Both are written band by
band.  A progress file beside the output records each finished band, and ``--resume``
continues an interrupted run from the first unfinished band.

Shared code.  ``on_disk_common.py`` holds the reader and writer of the files, the progress file,
the choice of the number of bands, and the loop over the bands.  This script holds the bands,
the model of one band, and the blending of two bands.  The choice of the number of bands uses
``TomographyModel._fits_available_devices``, which is the memory check of the package and a
private function.
"""

import argparse
import os
import time

# on_disk_common sets the allocator option of torch, so this import comes before that of torch.
from on_disk_common import (
    ProgressFile, add_angle_arguments, add_device_arguments, add_input_arguments, add_output_arguments,
    apply_device_options, choose_num_parts, compile_mode, finish_output, model_settings, num_parts_for,
    open_output, open_sinogram, print_plan, print_sizes, print_summary, read_sinogram_shape, run_parts,
    view_angles)

import numpy as np

import mbirtorch

DEFAULT_ANGLE_SPAN_DEG = 180.0
# The name of a part of the run in the messages and in the help texts.
PART_NAME = 'band'
# The name of the method in the attributes of an HDF5 output.
METHOD = 'MBIR by bands (recon_on_disk_parallel.py)'
# These file types hold weights.  Any other value of the weights is a weight type of gen_weights.
WEIGHT_FILE_TYPES = ('.npy', '.h5', '.hdf5')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    src = add_input_arguments(p)
    add_angle_arguments(src, DEFAULT_ANGLE_SPAN_DEG)
    src.add_argument('--weights', default='none',
                     help="'none' (default), 'transmission', 'transmission_root', or a .npy file")
    out = add_output_arguments(p, PART_NAME)
    out.add_argument('--log', default='~/.mbirtorch/logs/recon_on_disk_parallel.log',
                     help='merged reconstruction log (default %(default)s)')
    out.add_argument('--quiet', action='store_true', help='do not print the per-iteration log lines')
    dev = add_device_arguments(p, PART_NAME)
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


# ── the model of the scan and the bands ──────────────────────────────────────
def scan_model(args, sinogram_shape):
    """Return the ParallelBeamModel of the whole scan, with the angles of the options.  No array
    of the size of this model is placed on a device."""
    return mbirtorch.ParallelBeamModel(sinogram_shape, view_angles(args, sinogram_shape[0]),
                                       compile_mode=compile_mode(args))


def band_plan(num_rows, num_parts, half_overlap):
    """Return one (slice_lo, slice_hi, row_lo, row_hi) tuple for each band, as recon_split_sino
    divides the rows.  [slice_lo, slice_hi) are the slices that the band gives to the volume.  The
    numbers of these slices of two bands differ by at most one.  [row_lo, row_hi) are the rows of
    the model of the band, which extend half_overlap rows past each seam with another band."""
    base, extra = divmod(num_rows, num_parts)
    plan, start = [], 0
    for index in range(num_parts):
        stop = start + base + (1 if index < extra else 0)
        plan.append((start, stop, max(start - half_overlap, 0), min(stop + half_overlap, num_rows)))
        start = stop
    return plan


def band_model(base, slice_lo, slice_hi, row_lo, row_hi):
    """Return the model of the band (slice_lo, slice_hi, row_lo, row_hi) of base.  The model
    covers the detector rows [row_lo, row_hi), and so the same number of slices.  It keeps the
    regularization parameters of base, the devices that were set for base with
    configure_devices, and the margin of the memory check of base."""
    num_band_rows = row_hi - row_lo
    recon_rows, recon_cols = base.get_params('recon_shape')[:2]
    model = mbirtorch.copy_ct_model(base, new_num_det_rows=num_band_rows, no_warning=True)
    model.set_params(no_warning=True, auto_regularize_flag=False,
                     recon_shape=(recon_rows, recon_cols, num_band_rows))
    model.memory_check_margin = base.memory_check_margin
    return model


def blend_ramp(half_overlap):
    """The ramp recon_split_sino hands to stitch_arrays: at most 4 blended slices, even."""
    ramp = min(4, half_overlap)
    return ramp - ramp % 2


def open_weights(weights, model):
    """Return the SinogramSource of the weights if weights names a file, and None otherwise."""
    if weights is None or os.path.splitext(weights)[1].lower() not in WEIGHT_FILE_TYPES:
        return None
    return open_sinogram(weights, None, model)


# ── the reconstruction ───────────────────────────────────────────────────────
def recon_on_disk_parallel(model, sino_file, output_file, weights=None, half_overlap=5, max_iterations=15,
                           stop_threshold_change_pct=0.2, logfile_path='~/.mbirtorch/logs/recon.log',
                           print_logs=True, slices_per_section=None, dataset=None, resume=False, max_parts=None):
    """Reconstruct a parallel-beam scan by MBIR, one band of slices at a time, from the sinogram
    file sino_file to the output file output_file.  The result is that of recon_split_sino.

    model is the ParallelBeamModel of the whole scan.  Its devices and the margin of its memory
    check apply to every band.  If its auto_regularize_flag is True, the function sets its
    regularization parameters from a view subsample of the sinogram file.  weights is None, a
    weight type of gen_weights, or the name of a file of weights with the shape of the sinogram.
    half_overlap, max_iterations, stop_threshold_change_pct, logfile_path, print_logs, and
    slices_per_section are the arguments of recon_split_sino.  If slices_per_section is None,
    the function takes the smallest number of bands that the memory check accepts.  dataset
    names the dataset of an HDF5 sinogram.  With resume, the function continues the interrupted
    run of output_file.  The function stops before the band of index max_parts if max_parts is
    not None."""
    t_start = time.time()
    source = open_sinogram(sino_file, dataset, model)
    num_rows = source.shape[1]
    weights_source = open_weights(weights, model)
    recon_shape = tuple(int(s) for s in model.get_params('recon_shape'))

    # No data of the size of the model is placed on a device.  The regularization statistics
    # use the view subsample that recon would take, with the weights of those views.
    sample = source.view_subsample()
    if weights_source is not None:
        sample_weights = weights_source.view_subsample()
    elif weights is not None:
        sample_weights = mbirtorch.gen_weights(sample, weight_type=weights)
    else:
        sample_weights = None
    regularization = model.auto_set_regularization_params(sample, weights=sample_weights)
    del sample, sample_weights
    print_sizes(source.shape, recon_shape)
    print('regularization from the whole sinogram: '
          + ', '.join(f'{k}={v:.4g}' for k, v in regularization.items()), flush=True)

    # The progress file makes the run resumable and records each band's cost.
    progress = ProgressFile(output_file, PART_NAME, resume)
    ho = half_overlap

    def plan_of(count):
        return band_plan(num_rows, count, ho)

    def part_model_of(*band):
        return band_model(model, *band)

    call_arrays = {'weights': 'supplied'} if weights is not None else {}
    num_parts = progress.num_parts(
        num_parts_for(slices_per_section, num_rows),
        lambda: choose_num_parts(model, plan_of, part_model_of, max(1, num_rows // (2 * ho)), PART_NAME,
                                 workload='recon', **call_arrays))
    if num_parts > 1 and num_rows // num_parts < 2 * ho:
        raise SystemExit(f'{num_parts} bands leave fewer than {2 * ho} slices each; use fewer bands '
                         'or a smaller half_overlap')
    plan = plan_of(num_parts)
    print_plan(plan, PART_NAME, num_rows)
    print(f'overlap {ho} rows per seam', flush=True)
    config = {'num_parts': num_parts, 'half_overlap': ho, 'weights': weights, 'max_iterations': max_iterations,
              'stop_threshold_change_pct': stop_threshold_change_pct, 'model_params': model_settings(model)}
    sink = open_output(output_file, recon_shape, progress, config)

    log_path = os.path.expanduser(logfile_path)
    part_logs = [f'{log_path}.part{k}' for k in range(num_parts)]
    ramp = blend_ramp(ho)

    def read(k):
        """Return the sinogram rows of the model of band k, and their weights."""
        _lo, _hi, row_lo, row_hi = plan[k]
        sino_band = source.read_rows(row_lo, row_hi)
        if weights_source is not None:
            band_weights = weights_source.read_rows(row_lo, row_hi)
        elif weights is not None:
            band_weights = mbirtorch.gen_weights(sino_band, weight_type=weights)
        else:
            band_weights = None
        return sino_band, band_weights

    def reconstruct(k, data):
        """Return the reconstruction of band k, and its number of iterations for the record."""
        sino_band, band_weights = data
        band, band_dict = band_model(model, *plan[k]).recon(
            sino_band, weights=band_weights, max_iterations=max_iterations,
            stop_threshold_change_pct=stop_threshold_change_pct, logfile_path=part_logs[k],
            print_logs=print_logs)
        return band, {'iterations': int(band_dict['recon_params']['num_iterations'])}

    def write(k, band):
        # The band's first 2 * ho slices are the seam it shares with the previous band, whose
        # last 2 * ho slices are already on disk.  The blend is stitch_arrays over those two
        # slabs, so it is the blend recon_split_sino computes.  The rest of the band is
        # written as it is; its own last 2 * ho slices wait for the next band's blend.
        row_lo = plan[k][2]
        if k > 0:
            seam = 2 * ho
            previous_tail = sink.read(row_lo, row_lo + seam)
            blended = mbirtorch.stitch_arrays([previous_tail, band[:, :, :seam]],
                                              overlap=seam, axis=2, ramp_overlap=ramp)
            sink.write(blended, row_lo)
            sink.write(band[:, :, seam:], row_lo + seam)
        else:
            sink.write(band, row_lo)

    run_parts(plan, read, reconstruct, sink, progress, max_parts, write=write)

    recon_params = {'method': METHOD, 'num_parts': num_parts, 'half_overlap': ho, 'weights': weights,
                    'max_iterations': max_iterations, 'stop_threshold_change_pct': stop_threshold_change_pct,
                    'iterations_per_part': [entry['iterations'] for entry in progress.done]}
    finish_output(sink, progress, model, recon_params)
    source.close()
    if weights_source is not None:
        weights_source.close()
    if progress.finished:
        labels = [f'recon_on_disk_parallel: band {k + 1} of {num_parts} (slices {lo}-{hi - 1})'
                  for k, (lo, hi, _row_lo, _row_hi) in enumerate(plan)]
        mbirtorch.merge_log_files(log_path, zip(labels, part_logs))
    print_summary(progress, time.time() - t_start, output_file)


def main(argv=None):
    args = parse_args(argv)
    model = scan_model(args, read_sinogram_shape(args.sino, args.dataset))
    model.set_params(sharpness=args.sharpness, snr_db=args.snr_db, positivity_flag=args.positivity, verbose=1,
                     no_warning=True)
    apply_device_options(model, args)
    if args.seed is not None:
        np.random.seed(args.seed)
    recon_on_disk_parallel(model, args.sino, args.output, weights=None if args.weights == 'none' else args.weights,
                           half_overlap=args.half_overlap, max_iterations=args.max_iterations,
                           stop_threshold_change_pct=args.stop_pct, logfile_path=args.log,
                           print_logs=not args.quiet, slices_per_section=args.slices_per_section,
                           dataset=args.dataset, resume=args.resume, max_parts=args.max_parts)


if __name__ == '__main__':
    main()
