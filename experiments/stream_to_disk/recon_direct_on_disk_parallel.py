"""Reconstruct a large parallel-beam scan by filtered back projection, one band of slices at a time.

This script is for a volume that fits neither on the GPUs nor in host memory as a whole.  In
parallel beam, detector row r gives slice r.  For each band of detector rows, the script reads
the rows from the sinogram file, reconstructs the slices of these rows on the GPUs with
``ParallelBeamModel.recon_direct``, and writes them to the output file.  The host holds one band
of the sinogram and one band of the volume.  The volume exists only on disk.

The result equals the direct reconstruction of the whole volume, up to the rounding of float32
arithmetic.  The bands need no overlap and no blending.

    python recon_direct_on_disk_parallel.py --sino sino.npy --angles angles.npy --output recon.h5

The function ``recon_direct_on_disk_parallel`` does the reconstruction.  It takes the
``ParallelBeamModel`` of the whole scan, so that it can become a method of the model.

Inputs.  The sinogram is a ``.npy`` file, which is memory mapped, or an HDF5 file.  In both
cases the script reads only the rows of one band at a time.  The axes are (views, detector
rows, detector channels) in float32.  The angles are radians in a ``.npy`` file, or
``--angle-span`` gives equally spaced views over that many degrees.

Bands.  ``--slices-per-section`` sets the largest number of slices of a band.  Without it, the
script uses the smallest number of bands for which the memory check of the package accepts
every band on the GPUs, with the margin of ``--margin``.  On a device that is not a GPU,
``--slices-per-section`` is required.

Output.  A ``.h5`` path gives an HDF5 file in the layout of ``export_recon_hdf5``: dataset
``recon`` with axes (slice, col, row), readable with ``import_recon_hdf5``.  A ``.npy`` path
gives a memory mapped array with axes (slice, row, col).  A progress file beside the output
records each finished band.  The same command with ``--resume`` continues an interrupted run
from the first unfinished band.

Shared code.  ``on_disk_common.py`` holds the direct reconstruction by parts, which
``recon_direct_on_disk_cone.py`` also uses.  ``recon_on_disk_parallel.py`` holds the model of
the scan, the bands, and the model of one band.  The choice of the number of bands uses
``TomographyModel._fits_available_devices``, which is the memory check of the package and a
private function.
"""

import argparse

# on_disk_common sets the allocator option of torch, so this import comes before that of torch.
from on_disk_common import (
    add_angle_arguments, add_device_arguments, add_input_arguments, add_output_arguments,
    apply_device_options, read_sinogram_shape, recon_direct_on_disk)

from recon_on_disk_parallel import DEFAULT_ANGLE_SPAN_DEG, PART_NAME, band_model, band_plan, scan_model

# The name of the method in the attributes of an HDF5 output.
METHOD = 'FBP by bands (recon_direct_on_disk_parallel.py)'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    src = add_input_arguments(p)
    add_angle_arguments(src, DEFAULT_ANGLE_SPAN_DEG)
    add_output_arguments(p, PART_NAME)
    add_device_arguments(p, PART_NAME)
    rec = p.add_argument_group('reconstruction')
    rec.add_argument('--filter-name', default='ramp', help='filter of recon_direct (default %(default)s)')
    return p.parse_args(argv)


def recon_direct_on_disk_parallel(model, sino_file, output_file, filter_name='ramp', slices_per_section=None,
                                  dataset=None, resume=False, max_parts=None):
    """Reconstruct a parallel-beam scan by filtered back projection, one band of slices at a time,
    from the sinogram file sino_file to the output file output_file.

    model is the ParallelBeamModel of the whole scan.  Its devices and the margin of its memory
    check apply to every band.  slices_per_section sets the largest number of slices of a band.
    If it is None, the function takes the smallest number of bands that the memory check
    accepts.  The other arguments are those of on_disk_common.recon_direct_on_disk."""
    num_rows = int(model.get_params('sinogram_shape')[1])
    # A band of the direct reconstruction reads only its own rows, so it has no overlap.
    recon_direct_on_disk(model, sino_file, output_file, lambda count: band_plan(num_rows, count, 0),
                         lambda *band: band_model(model, *band), PART_NAME, METHOD, filter_name=filter_name,
                         slices_per_section=slices_per_section, dataset=dataset, resume=resume,
                         max_parts=max_parts)


def main(argv=None):
    args = parse_args(argv)
    model = scan_model(args, read_sinogram_shape(args.sino, args.dataset))
    apply_device_options(model, args)
    recon_direct_on_disk_parallel(model, args.sino, args.output, filter_name=args.filter_name,
                                  slices_per_section=args.slices_per_section, dataset=args.dataset,
                                  resume=args.resume, max_parts=args.max_parts)


if __name__ == '__main__':
    main()
