"""Demo 6: sparse view selection.

Before a scan, pick the few view angles that reconstruct an object best.
Given a reference object like the one to be scanned, get_opt_views chooses
angles from a set of candidates by minimizing the view covariance loss.  The
reconstruction from the chosen views is then compared with one from the same
number of evenly spaced views.
"""

import numpy as np
import mbirtorch

# The candidate scan: a cone-beam scan with 128 candidate angles over a full
# rotation, of which 25 are to be chosen.  The reference object is an asymmetric
# polygon, so some view angles matter more than others.
num_candidate_views = 128
num_selected_views = 25

reference, _, params = mbirtorch.gen_demo_data(object_type='polygon', model_type='cone',
                                               num_views=num_candidate_views, num_det_rows=16,
                                               num_det_channels=128)
candidate_angles = params['angles']
ct_model = mbirtorch.ConeBeamModel((num_candidate_views, 16, 128), candidate_angles,
                                   source_detector_dist=params['source_detector_dist'],
                                   source_iso_dist=params['source_iso_dist'])
ct_model.set_params(verbose=0)

# Select the views.  r_1 is the fraction of voxels sampled and r_2 the fraction
# of candidate views tried per step; larger values are slower and more exact.
# With priority_order the chosen views come most important first.
view_inds, vcl = mbirtorch.get_opt_views(ct_model, reference, num_selected_views,
                                         r_1=0.01, r_2=0.5, priority_order=True, seed=42)
selected_angles = candidate_angles[view_inds]
print('Selected angles in degrees, most important first:', np.round(np.degrees(selected_angles)))

# Reconstruct the reference from the selected views and from evenly spaced views.
recons, dicts = {}, {}
for name, angles in (('evenly spaced', np.linspace(0, 2 * np.pi, num_selected_views, endpoint=False)),
                     ('selected', selected_angles)):
    model = mbirtorch.copy_ct_model(ct_model, angles)
    model.set_params(verbose=0)
    recons[name], dicts[name] = model.recon(model.forward_project(reference))
    nrmse = np.linalg.norm(recons[name] - reference) / np.linalg.norm(reference)
    print(f'{name} views: normalized RMS error {nrmse:.3f}')

# Show the selected angles on the reference, and the reconstructions.
mbirtorch.show_image_with_projection_rays(reference[:, :, 0], rotation_angles_rad=selected_angles,
                                          title='Reference object with the selected view angles')
mbirtorch.slice_viewer(reference, recons['evenly spaced'], recons['selected'],
                       data_dicts=[None, dicts['evenly spaced'], dicts['selected']], vmin=0.0, vmax=1.0,
                       slice_label=['GROUND TRUTH', 'EVENLY SPACED VIEWS', 'SELECTED VIEWS'],
                       title='Ground truth (left), evenly spaced views (middle), selected views (right)')
