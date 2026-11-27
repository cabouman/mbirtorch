"""Reconstruct a large circular cone-beam scan by FDK, one slab of slices at a time.

This script is for a scan whose FDK reconstruction fits neither on the GPUs nor in host memory
as a whole.  A slab is a range of slices, and a band is a range of detector rows.  The FDK
filter acts along each detector row alone, and the back projection of a voxel reads only the
rows that the voxel touches.  For this reason, a slab can be reconstructed from the band of
rows that its voxels touch.  For each slab, the script reads the band from the sinogram file,
reconstructs the slab on the GPUs with ``ConeBeamModel.recon_direct``, and writes the slab to
the output file.  The host holds one band of the sinogram and one slab of the volume.  The
volume exists only on disk.

The result equals the FDK reconstruction of the whole volume, up to the rounding of float32
arithmetic.  The slabs need no overlap and no blending.  The bands of neighboring slabs
overlap, so the script reads and filters some detector rows more than once.  The script
supports circular scans only.

    python recon_direct_on_disk_cone.py --sino sino.npy --source-detector-dist 5080 --source-iso-dist 2540 \
        --output recon.h5

The function ``recon_direct_on_disk_cone`` does the reconstruction.  It takes the
``ConeBeamModel`` of the whole scan, so that it can become a method of the model.

Inputs.  The sinogram is a ``.npy`` file, which is memory mapped, or an HDF5 file.  In both
cases the script reads only the rows of one band at a time.  The axes are (views, detector
rows, detector channels) in float32.  The angles are radians in a ``.npy`` file, or
``--angle-span`` gives equally spaced views over that many degrees.  The geometry options have
the names of the ``ConeBeamModel`` parameters, and an option that is not given keeps the
default of the package.  ``--param`` sets any other parameter of the model.

Slabs.  ``--slices-per-section`` sets the largest number of slices of a slab.  Without it, the
script uses the smallest number of slabs for which the memory check of the package accepts
every slab on the GPUs.  The check accepts a slab if (1 + margin) times the estimated peak
memory of its reconstruction is at most the free memory of each GPU.  ``--margin`` sets the
margin.  If two or more GPUs are visible and ``--devices`` is not given, the package runs the
same check with this margin before it reconstructs each slab.  On a device that is not a GPU,
the check has no limit, and ``--slices-per-section`` is required.

Output.  A ``.h5`` path gives an HDF5 file in the layout of ``export_recon_hdf5``: dataset
``recon`` with axes (slice, col, row), readable with ``import_recon_hdf5``.  A ``.npy`` path
gives a memory mapped array with axes (slice, row, col).  A progress file beside the output
records each finished slab.  The same command with ``--resume`` continues an interrupted run
from the first unfinished slab.

Shared code.  ``on_disk_common.py`` holds the direct reconstruction by parts, which
``recon_direct_on_disk_parallel.py`` also uses, and the reader and writer of the files.  This
script holds the model of the scan, the bands of the slabs, and the model of one slab.  The
choice of the number of slabs uses ``TomographyModel._fits_available_devices``, which is the
memory check of the package and a private function.
"""

import argparse
import json

# on_disk_common sets the allocator option of torch, so this import comes before that of torch.
from on_disk_common import (
    add_angle_arguments, add_device_arguments, add_input_arguments, add_output_arguments,
    apply_device_options, compile_mode, read_sinogram_shape, recon_direct_on_disk, view_angles)

import numpy as np

import mbirtorch

DEFAULT_ANGLE_SPAN_DEG = 360.0
# The name of a part of the run in the messages and in the help texts.
PART_NAME = 'slab'
# The name of the method in the attributes of an HDF5 output.
METHOD = 'FDK by slabs (recon_direct_on_disk_cone.py)'
# This many rows are added at each end of a band, against the rounding of the row index.
BAND_SAFETY_ROWS = 1
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    add_input_arguments(p)
    add_geometry_arguments(p)
    add_output_arguments(p, PART_NAME)
    add_device_arguments(p, PART_NAME)
    rec = p.add_argument_group('reconstruction')
    rec.add_argument('--filter-name', default='ramp', help='filter of recon_direct (default %(default)s)')
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
                'compile_mode': compile_mode(args)}
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


def part_model(base, slice_lo, slice_hi, row_lo, row_hi):
    """Return the model for slices [slice_lo, slice_hi) and detector rows [row_lo, row_hi) of
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


# ── the reconstruction ───────────────────────────────────────────────────────
def recon_direct_on_disk_cone(model, sino_file, output_file, filter_name='ramp', slices_per_section=None,
                              dataset=None, resume=False, max_parts=None):
    """Reconstruct a circular cone-beam scan by FDK, one slab of slices at a time, from the
    sinogram file sino_file to the output file output_file.

    model is the ConeBeamModel of the whole scan.  Its devices and the margin of its memory check
    apply to every slab.  slices_per_section sets the largest number of slices of a slab.  If it
    is None, the function takes the smallest number of slabs that the memory check accepts.  The
    other arguments are those of on_disk_common.recon_direct_on_disk."""
    z_shifts = np.asarray(model.get_params('view_params_array'))[:, 1]
    if np.ptp(z_shifts) > 0:
        raise SystemExit('recon_direct_on_disk_cone supports circular scans only')
    recon_direct_on_disk(model, sino_file, output_file, lambda count: slab_plan(model, count),
                         lambda *part: part_model(model, *part), PART_NAME, METHOD, filter_name=filter_name,
                         slices_per_section=slices_per_section, dataset=dataset, resume=resume,
                         max_parts=max_parts)


def main(argv=None):
    args = parse_args(argv)
    model = scan_model(args, read_sinogram_shape(args.sino, args.dataset))
    apply_device_options(model, args)
    recon_direct_on_disk_cone(model, args.sino, args.output, filter_name=args.filter_name,
                              slices_per_section=args.slices_per_section, dataset=args.dataset,
                              resume=args.resume, max_parts=args.max_parts)


if __name__ == '__main__':
    main()
