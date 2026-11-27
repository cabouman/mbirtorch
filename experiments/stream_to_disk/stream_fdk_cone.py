"""Reconstruct a large circular cone-beam scan by FDK, one slab of slices at a time.

This script is for a scan whose FDK reconstruction fits neither on the GPUs nor in host memory
as a whole.  A slab is a range of slices, and a band is a range of detector rows.  The FDK
filter acts along each detector row alone, and the back projection of a voxel reads only the
rows that the voxel touches.  For this reason, a slab can be reconstructed from the band of
rows that its voxels touch.  For each slab, the script reads the band from the sinogram file,
reconstructs the slab on the GPUs with ``ConeBeamModel.recon_fdk``, and writes the slab to the
output file.  The host holds one band of the sinogram and one slab of the volume.  The volume
exists only on disk.

The result equals the FDK reconstruction of the whole volume, up to the rounding of float32
arithmetic.  The slabs need no overlap and no blending.  The bands of neighboring slabs
overlap, so the script reads and filters some detector rows more than once.  The script
supports circular scans only.

    python stream_fdk_cone.py --sino sino.npy --source-detector-dist 5080 --source-iso-dist 2540 --output recon.h5

For a test without data, ``make_synthetic_cone_sinogram.py`` writes a phantom sinogram and
checks a reconstruction against the phantom.  ``check_stream_fdk_cone.py`` compares this script
with ``recon_fdk`` of the whole volume on small scans.

Inputs.  The sinogram is a ``.npy`` file, which is memory mapped, or an HDF5 file.  In both
cases the script reads only the rows of one band at a time.  The axes are (views, detector
rows, detector channels) in float32.  The angles are radians in a ``.npy`` file, or
``--angle-span`` gives equally spaced views over that many degrees.  The geometry options have
the names of the ``ConeBeamModel`` parameters, and an option that is not given keeps the
default of the package.  ``--param`` sets any other parameter of the model.

Slabs.  ``--num-slabs`` sets the number of slabs.  Without it, the script uses the smallest
number of slabs for which the memory check of the package accepts every slab on the GPUs.  The
check accepts a slab if (1 + margin) times the estimated peak memory of its reconstruction is
at most the free memory of each GPU.  ``--margin`` sets the margin.  If two or more GPUs are
visible and ``--devices`` is not given, the package runs the same check with this margin before
it reconstructs each slab.  On a device that is not a GPU, the check has no limit, and
``--num-slabs`` is required.

Output.  A ``.h5`` path gives an HDF5 file in the layout of ``export_recon_hdf5``: dataset
``recon`` with axes (slice, col, row), readable with ``import_recon_hdf5``.  A ``.npy`` path
gives a memory mapped array with axes (slice, row, col).  The script writes one slab at a
time.  A progress file beside the output records each finished slab.  The same command with
``--resume`` continues an interrupted run from the first unfinished slab.

Private library calls.  The choice of the number of slabs uses
``TomographyModel._fits_available_devices``, which is the memory check of the package.  The
attributes of the HDF5 output come from ``mbirtorch.utilities._recon_attributes``, which
``stream_common.py`` calls.  Everything else is public API.

Shared code.  ``stream_common.py`` holds the code that this script shares with
``stream_recon_parallel.py``: the reader of the sinogram file, the writer of the output file,
the progress file, and the loop over the slabs.
"""

import argparse
import hashlib
import json
import time

# stream_common sets the allocator option of torch, so this import comes before that of torch.
from stream_common import (
    ProgressFile, SinogramSource, add_angle_arguments, add_device_arguments, add_input_arguments,
    add_output_arguments, device_list, finish_output, open_output, print_sizes, print_summary, run_parts,
    smallest_fitting_count, view_angles)

import numpy as np

import mbirtorch

DEFAULT_ANGLE_SPAN_DEG = 360.0
# The name of a part of the run in the messages and in the help texts.
PART_NAME = 'slab'
# This many rows are added at each end of a band, against the rounding of the row index.
BAND_SAFETY_ROWS = 1
# These model parameters determine the result.  The progress file records them, so that
# --resume can refuse to continue a run that had other settings.
RESULT_PARAMS = ('source_detector_dist', 'source_iso_dist', 'use_curved_detector',
                 'delta_det_channel', 'delta_det_row', 'det_channel_offset', 'det_row_offset',
                 'delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect', 'recon_slice_offset',
                 'use_ror_mask')
# These geometry options are passed to the model under the same name when they are given.
GEOMETRY_OPTIONS = ('delta_det_channel', 'delta_det_row', 'det_channel_offset', 'det_row_offset',
                    'delta_voxel', 'recon_slice_offset')


def shape_from_text(text):
    """Return the tuple of three positive integers that text holds, such as '1601,3048,3048'."""
    try:
        shape = tuple(int(value) for value in text.split(','))
    except ValueError:
        shape = ()
    if len(shape) != 3 or min(shape) < 1:
        raise argparse.ArgumentTypeError(f'{text!r} is not three positive integers separated by commas')
    return shape


def add_geometry_arguments(parser):
    """Add the options that describe the scan and the volume to parser.  The sinogram generator
    uses the same options, so that both scripts build the same model."""
    geo = parser.add_argument_group('geometry (distances are in ALU; an option that is not given '
                                    'keeps the default of the package)')
    geo.add_argument('--source-detector-dist', type=float, help='distance from the source to the detector (required)')
    geo.add_argument('--source-iso-dist', type=float, help='distance from the source to the rotation axis (required)')
    geo.add_argument('--delta-det-channel', type=float, help='detector pitch along the channels')
    geo.add_argument('--delta-det-row', type=float, help='detector pitch along the rows')
    geo.add_argument('--det-channel-offset', type=float, help='offset of the detector center along the channels')
    geo.add_argument('--det-row-offset', type=float, help='offset of the detector center along the rows')
    geo.add_argument('--curved-detector', action='store_true', help='the detector is a cylinder around the source')
    add_angle_arguments(geo, DEFAULT_ANGLE_SPAN_DEG)
    geo.add_argument('--delta-voxel', type=float, help='voxel pitch')
    geo.add_argument('--recon-shape', type=shape_from_text, metavar='ROWS,COLS,SLICES', help='shape of the volume')
    geo.add_argument('--recon-slice-offset', type=float,
                     help='position of the center of the volume along the rotation axis')
    geo.add_argument('--param', action='append', default=[], metavar='NAME=VALUE',
                     help='another model parameter, for example voxel_slice_aspect=1.5 or use_ror_mask=false; '
                          'VALUE is read as JSON; the option can be repeated')
    geo.add_argument('--compile-off', action='store_true',
                     help='build the models without torch.compile, for a test on a CPU')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    add_input_arguments(p)
    add_geometry_arguments(p)
    add_output_arguments(p, PART_NAME)
    dev = add_device_arguments(p, PART_NAME)
    dev.add_argument('--num-slabs', type=int,
                     help='number of slabs (default: the smallest number that the memory check accepts)')
    rec = p.add_argument_group('reconstruction')
    rec.add_argument('--filter-name', default='ramp', help='filter of recon_fdk (default %(default)s)')
    args = p.parse_args(argv)
    require_distances(p, args)
    return args


def require_distances(parser, args):
    """Stop with a usage error if one of the two distances of the scan is missing."""
    if args.source_detector_dist is None or args.source_iso_dist is None:
        parser.error('--source-detector-dist and --source-iso-dist are required')


# ── the model of the scan ────────────────────────────────────────────────────
def scan_model(args, sinogram_shape):
    """Return the ConeBeamModel of the whole scan and the whole volume that the geometry options
    describe.  No array of the size of this model is placed on a device."""
    required = {'geometry_type': 'ConeBeamModel', 'sinogram_shape': tuple(sinogram_shape),
                'angles': view_angles(args, sinogram_shape[0]),
                'source_detector_dist': args.source_detector_dist, 'source_iso_dist': args.source_iso_dist,
                'use_curved_detector': args.curved_detector,
                'compile_mode': 'off' if args.compile_off else 'auto'}
    optional = {name: getattr(args, name) for name in GEOMETRY_OPTIONS if getattr(args, name) is not None}
    if args.recon_shape is not None:
        optional['recon_shape'] = args.recon_shape
    for item in args.param:
        name, separator, value = item.partition('=')
        if not separator:
            raise SystemExit(f'--param {item!r} is not of the form NAME=VALUE')
        try:
            optional[name] = json.loads(value)
        except ValueError:
            optional[name] = value
    # build_model applies the volume parameters after the automatic volume of the package.
    return mbirtorch.build_model(required, optional)


# ── the slabs and their bands ────────────────────────────────────────────────
def slab_edges(num_slices, num_slabs):
    """Return the num_slabs + 1 slice indices that divide num_slices slices into slabs.  The
    numbers of slices of two slabs differ by at most one."""
    return [int(round(k * num_slices / num_slabs)) for k in range(num_slabs + 1)]


def band_rows(model, slice_lo, slice_hi):
    """Return the range [row_lo, row_hi) of detector rows that the voxels of slices
    [slice_lo, slice_hi) of model touch in the back projection.  The range is (0, 0) if these
    voxels touch no detector row.  The model must describe a circular scan."""
    num_rows = int(model.get_params('sinogram_shape')[1])
    num_slices = int(model.get_params('recon_shape')[2])
    delta_det_row, det_row_offset = model.get_params(['delta_det_row', 'det_row_offset'])
    delta_slice = model.get_params('delta_voxel') * model.get_params('voxel_slice_aspect')
    z_shift = float(np.asarray(model.get_params('view_params_array'))[0, 1])
    z_center = model.get_params('recon_slice_offset') - z_shift
    # The magnification of every voxel lies between these two bounds.
    mag_min, mag_max = model.pixel_magnification_bounds()
    if not np.isfinite(mag_max):
        return 0, num_rows
    # A voxel at height z with magnification m has its center at detector height m * z.  The
    # extreme heights belong to the first and to the last slice of the slab.
    z_first = z_center + delta_slice * (slice_lo - (num_slices - 1) / 2.0)
    z_last = z_center + delta_slice * (slice_hi - 1 - (num_slices - 1) / 2.0)
    heights = [mag * z for mag in (mag_min, mag_max) for z in (z_first, z_last)]
    # The footprint of a voxel extends (w + 1) / 2 rows to each side of its center, where w is
    # the height of the voxel on the detector in rows.
    half_footprint = (mag_max * delta_slice / delta_det_row + 1.0) / 2.0
    center_row = (num_rows - 1) / 2.0 + det_row_offset / delta_det_row
    row_lo = int(np.floor(min(heights) / delta_det_row + center_row - half_footprint)) - BAND_SAFETY_ROWS
    row_hi = int(np.ceil(max(heights) / delta_det_row + center_row + half_footprint)) + 1 + BAND_SAFETY_ROWS
    row_lo, row_hi = max(row_lo, 0), min(row_hi, num_rows)
    return (row_lo, row_hi) if row_hi > row_lo else (0, 0)


def part_model(base, row_lo, row_hi, slice_lo, slice_hi):
    """Return the model for detector rows [row_lo, row_hi) and slices [slice_lo, slice_hi) of
    base.  Each voxel and each detector element of the part has the same position as in base.
    The part uses the devices that were set for base with configure_devices, and it uses the
    margin of the memory check of base."""
    num_rows_base = base.get_params('sinogram_shape')[1]
    delta_det_row = base.get_params('delta_det_row')
    num_recon_rows, num_recon_cols, num_slices_base = base.get_params('recon_shape')
    delta_slice = base.get_params('delta_voxel') * base.get_params('voxel_slice_aspect')
    num_rows, num_slices = row_hi - row_lo, slice_hi - slice_lo
    det_row_offset = base.get_params('det_row_offset') + (
        (num_rows_base - 1) / 2.0 - ((num_rows - 1) / 2.0 + row_lo)) * delta_det_row
    slice_offset = base.get_params('recon_slice_offset') + delta_slice * (
        slice_lo + (num_slices - 1) / 2.0 - (num_slices_base - 1) / 2.0)
    model = mbirtorch.copy_ct_model(base, new_num_det_rows=num_rows, no_warning=True)
    model.set_params(det_row_offset=det_row_offset, recon_slice_offset=slice_offset,
                     recon_shape=(num_recon_rows, num_recon_cols, num_slices))
    model.memory_check_margin = base.memory_check_margin
    return model


def slab_plan(model, num_slabs):
    """Return one (slice_lo, slice_hi, row_lo, row_hi) tuple for each of the num_slabs slabs of
    the volume of model.  The rows are the band of the slab."""
    edges = slab_edges(int(model.get_params('recon_shape')[2]), num_slabs)
    return [(slice_lo, slice_hi) + band_rows(model, slice_lo, slice_hi)
            for slice_lo, slice_hi in zip(edges[:-1], edges[1:])]


def largest_slabs(plan):
    """Return the slabs of plan that need the most memory.  A slab is left out if another slab
    has at least as many slices and at least as many band rows."""
    sizes = {(slice_hi - slice_lo, row_hi - row_lo): (slice_lo, slice_hi, row_lo, row_hi)
             for slice_lo, slice_hi, row_lo, row_hi in plan if row_hi > row_lo}
    return [slab for size, slab in sizes.items()
            if not any(other != size and other[0] >= size[0] and other[1] >= size[1] for other in sizes)]


def plan_fits(model, plan):
    """Return True if the memory check of the package accepts the FDK reconstruction of every
    slab of plan.  The check uses the margin of model."""
    return all(part_model(model, row_lo, row_hi, slice_lo, slice_hi)._fits_available_devices(workload='direct')
               for slice_lo, slice_hi, row_lo, row_hi in largest_slabs(plan))


def choose_num_slabs(model):
    """Return the smallest number of slabs whose plan passes the memory check of the package.
    The check uses the margin of model.  Use this function when the user gave no number of
    slabs.  The devices of model must be GPUs."""
    if any(device.type != 'cuda' for device in model.sino_placement.devices):
        raise SystemExit('the memory check has no limit for a device that is not a GPU; give --num-slabs')
    num_slices = int(model.get_params('recon_shape')[2])
    num_slabs = smallest_fitting_count(lambda count: plan_fits(model, slab_plan(model, count)), num_slices)
    if num_slabs is None:
        raise SystemExit('the band of detector rows of a single slice does not fit the GPUs at margin '
                         f'{model.memory_check_margin}')
    return num_slabs


# ── the run ──────────────────────────────────────────────────────────────────
def main(argv=None):
    args = parse_args(argv)
    devices = device_list(args)

    t_start = time.time()
    source = SinogramSource(args.sino, args.dataset)
    num_rows = source.shape[1]
    full = scan_model(args, source.shape)
    if devices:
        full.configure_devices(devices=devices)
    # Every slab takes this margin, both for the choice of the number of slabs and for the memory
    # check of the package before each reconstruction.
    full.memory_check_margin = args.margin
    recon_shape = tuple(int(s) for s in full.get_params('recon_shape'))
    num_slices = recon_shape[2]
    print_sizes(source.shape, recon_shape)

    # The progress file makes the run resumable and records the cost of each slab.
    progress = ProgressFile(args.output, PART_NAME, args.resume)
    num_slabs = progress.num_parts(args.num_slabs, lambda: choose_num_slabs(full))
    if not 1 <= num_slabs <= num_slices:
        raise SystemExit(f'the number of slabs must be between 1 and the number of slices, {num_slices}')
    plan = slab_plan(full, num_slabs)
    largest_band = max(row_hi - row_lo for _lo, _hi, row_lo, row_hi in plan)
    print(f'{num_slabs} slab(s) of about {-(-num_slices // num_slabs)} slices; the largest band has '
          f'{largest_band} of {num_rows} detector rows', flush=True)
    angles = np.asarray(full.get_params('view_params_array'))[:, 0]
    config = {'sinogram_shape': list(source.shape), 'recon_shape': list(recon_shape), 'num_parts': num_slabs,
              'filter_name': args.filter_name,
              'angles_sha256': hashlib.sha256(np.ascontiguousarray(angles, dtype=np.float32).tobytes()).hexdigest(),
              'model_params': {name: full.get_params(name) for name in RESULT_PARAMS}}
    sink = open_output(args.output, recon_shape, progress, config)

    def read(k):
        """Return the band of detector rows of slab k, or None if the slab touches no row."""
        _slice_lo, _slice_hi, row_lo, row_hi = plan[k]
        return source.read_rows(row_lo, row_hi) if row_hi > row_lo else None

    def reconstruct(k, band):
        """Return the FDK reconstruction of slab k from its band.  This script records no value
        of its own for a slab."""
        slice_lo, slice_hi, row_lo, row_hi = plan[k]
        if band is None:
            # No detector row measures this slab, so its back projection is zero.
            return np.zeros(recon_shape[:2] + (slice_hi - slice_lo,), dtype=np.float32), {}
        part = part_model(full, row_lo, row_hi, slice_lo, slice_hi)
        return part.recon_fdk(band, filter_name=args.filter_name), {}

    run_parts(plan, read, reconstruct, sink, progress, args.max_parts)

    recon_params = dict(config, method='FDK by slabs (stream_fdk_cone.py)')
    del recon_params['model_params']
    finish_output(sink, progress, full, recon_params)
    source.close()
    print_summary(progress, time.time() - t_start, args.output)


if __name__ == '__main__':
    main()
