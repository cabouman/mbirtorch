"""Headless tests for mbirtorch/viewers/geometry_scene.py.

For each of the six scan geometries of ``geometry_probe.py``,
``test_projection_matches_projector`` forward-projects single voxels with the
real projector, measures where each footprint lands on the detector, and
compares that with ``GeometryScene.project_points``.  A viewer that failed
this test would draw a geometry the projector does not use.
"""

import pytest

import geometry_probe as probe
from mbirtorch.viewers.geometry_scene import GeometryScene

# The gate: the largest allowed difference between a measured footprint
# centroid and the scene's prediction, in detector pixels.  Half a pixel is the
# tolerance the projection tests use.
GATE_PIXELS = 0.5

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

# Filled in by the gate test and printed by this file's __main__ block, so the
# run record can quote a number per geometry.
MAX_ERRORS = {}


def build_scene(cfg):
    """Build the model of one probe configuration and a scene from it."""
    model = probe.build_model(cfg)
    return model, GeometryScene.from_model(model)


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_projection_matches_projector(name):
    """Every projected voxel center lands where the projector puts it.

    The comparison uses footprint centroids, which are only meaningful when the
    footprint is at least about as wide as a detector pixel.  Every probe
    configuration keeps the voxel pitch near the detector pitch for that
    reason, and the probe's own record explains the failure mode.
    """
    cfg = CONFIGS_BY_NAME[name]
    model, scene = build_scene(cfg)

    max_row_error = 0.0
    max_channel_error = 0.0
    num_pairs = 0
    for voxel in probe.probe_voxels(cfg['recon_shape']):
        measured = probe.measure_footprint_centroids(model, cfg, voxel)
        center = scene.voxel_centers([voxel])
        for view, (reason, row, channel) in enumerate(measured):
            if reason is not None:
                continue
            row_pred, channel_pred = scene.project_points(center, view)
            row_error = abs(row - float(row_pred[0]))
            channel_error = abs(channel - float(channel_pred[0]))
            assert row_error <= GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: row {row} against '
                f'predicted {float(row_pred[0])}')
            assert channel_error <= GATE_PIXELS, (
                f'{name} voxel {voxel} view {view}: channel {channel} against '
                f'predicted {float(channel_pred[0])}')
            max_row_error = max(max_row_error, row_error)
            max_channel_error = max(max_channel_error, channel_error)
            num_pairs += 1

    assert num_pairs > 0
    MAX_ERRORS[name] = (num_pairs, max_row_error, max_channel_error)


if __name__ == '__main__':
    # Print the per-geometry gate errors for the run record.
    for cfg in probe.CONFIGS:
        test_projection_matches_projector(cfg['name'])
    print(f'{"geometry":16s} {"pairs":>6s} {"max row err":>12s} '
          f'{"max chan err":>13s}')
    for name, (pairs, row_error, channel_error) in MAX_ERRORS.items():
        print(f'{name:16s} {pairs:6d} {row_error:12.4f} {channel_error:13.4f}')
