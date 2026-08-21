"""Demo 5: direct reconstruction (FBP) versus model-based reconstruction (MBIR).

Filtered back projection (FBP) is fast and works well when there are many
views and little noise.  Model-based iterative reconstruction (MBIR) costs
more computation but stays good when the data gets hard.

This demo runs both methods twice: once with plenty of views, where FBP is
serviceable, and once with very few views, where FBP breaks down and the
MBIR advantage is unmistakable.  The sparse case is meant to be
illustrative, not practical.
"""

import numpy as np
import mbirtorch

# Problem size.
num_det_rows = 128
num_det_channels = 128

def make_data(num_views):
    phantom, sinogram, params = mbirtorch.generate_demo_data(
        model_type='parallel', object_type='shepp-logan',
        num_views=num_views, num_det_rows=num_det_rows,
        num_det_channels=num_det_channels)
    return phantom, sinogram, params['angles']

def nrmse(recon, phantom):
    return np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)

# Case 1: plenty of views (128).  FBP is serviceable; MBIR is cleaner.
phantom, sinogram, angles = make_data(num_views=128)
model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
fbp_many = model.direct_recon(sinogram)
mbir_many, _ = model.recon(sinogram)
print(f'128 views:  FBP error {nrmse(fbp_many, phantom):.3f},  '
      f'MBIR error {nrmse(mbir_many, phantom):.3f}')

# Case 2: very few views (16).  FBP produces streaks; MBIR holds up.
phantom, sinogram, angles = make_data(num_views=16)
model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
fbp_sparse = model.direct_recon(sinogram)
mbir_sparse, mbir_sparse_dict = model.recon(sinogram)
print(f'16 views:   FBP error {nrmse(fbp_sparse, phantom):.3f},  '
      f'MBIR error {nrmse(mbir_sparse, phantom):.3f}')

# View the sparse-view case: phantom, FBP, MBIR.
mbirtorch.slice_viewer(
    phantom, fbp_sparse, mbir_sparse,
    data_dicts=[None, None, mbir_sparse_dict], vmin=0.0, vmax=1.2,
    title='16 views: phantom (left), FBP with streaks (center), MBIR (right)', block=False)
mbirtorch.slice_viewer(sinogram, title='Sinogram', slice_axis=0)

# The practical rule: with many views and low noise, FBP is fast and good
# enough, and MBIR uses it internally as a starting point.  With few views,
# high noise, or metal, MBIR is worth its computation.
