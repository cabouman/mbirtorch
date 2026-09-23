"""Demo 12: the differentiable projectors.

Reconstruct by gradient descent instead of by calling recon().  The
reconstruction is an ordinary torch tensor with requires_grad=True, the
forward projector is differentiable, and Adam minimizes the data fit loss
||A x - y||^2.  This is the pattern you would use to put the physics
operator inside a deep-learning pipeline.
"""

import numpy as np
import torch
import mbirtorch

num_views = 32
num_det_rows = 16
num_det_channels = 64
num_steps = 150

# Make a phantom and project it to get a synthetic sinogram.
phantom, sinogram, params = mbirtorch.generate_demo_data(
    model_type='parallel', object_type='shepp-logan',
    num_views=num_views, num_det_rows=num_det_rows,
    num_det_channels=num_det_channels)

# The differentiable projectors run on one device only.
ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, params['angles'])
ct_model.configure_devices(1)
recon_shape = ct_model.get_params('recon_shape')

y = torch.as_tensor(sinogram, dtype=torch.float32,
                    device=ct_model.torch_device)
x = torch.zeros(tuple(recon_shape), dtype=torch.float32,
                device=ct_model.torch_device, requires_grad=True)
optimizer = torch.optim.Adam([x], lr=0.1)

for step in range(num_steps):
    optimizer.zero_grad()
    residual = mbirtorch.forward_project_differentiable(ct_model, x) - y
    loss = torch.sum(residual ** 2)
    loss.backward()
    optimizer.step()
    if step % 25 == 0 or step == num_steps - 1:
        print(f'step {step:3d}   data fit loss {float(loss.detach()):.4f}')

recon = x.detach().cpu().numpy()
nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
print(f'Normalized RMS error between reconstruction and phantom: {nrmse:.3f}')

mbirtorch.slice_viewer(
    phantom, recon, vmin=0.0,
    title='Phantom (left) and gradient descent reconstruction (right)',
    block=False)
