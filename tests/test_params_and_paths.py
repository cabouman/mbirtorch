"""Regression tests: set_params semantics, VCD-loop paths, and the
compile/eager and weights equivalences the suite previously never asserted."""

import numpy as np
import torch

import mbirtorch


def _small_model(device="cpu", **kwargs):
    sino_shape = (24, 16, 16)
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
    m = mbirtorch.ParallelBeamModel(sino_shape, angles, **kwargs)
    m.configure_devices(devices=[device])
    m.set_params(no_warning=True, verbose=0)
    return m


def _box_problem(model):
    recon_shape = model.get_params('recon_shape')
    phantom = np.zeros(tuple(recon_shape), dtype=np.float32)
    r0, c0, s0 = [max(1, n // 4) for n in recon_shape]
    phantom[r0:-r0, c0:-c0, s0:-s0] = 1.0
    return phantom, model.forward_project(phantom)


def test_multi_step_geometry_change_allowed():
    # mbirjax defers validation to recon entry, so a transiently-inconsistent
    # state between set_params and auto_set_recon_geometry must not raise.
    model = _small_model()
    new_shape = (30, 20, 20)
    new_angles = np.linspace(0, np.pi, new_shape[0], endpoint=False)
    model.set_params(sinogram_shape=new_shape, angles=new_angles)
    model.auto_set_recon_geometry()
    assert model.get_params('recon_shape')[2] == new_shape[1]
    sino = model.forward_project(np.ones(tuple(model.get_params('recon_shape')),
                                         dtype=np.float32))
    assert sino.shape == new_shape


# ── VCD-loop paths ─────────────────────────────────────────────────────────────
def test_positivity_path(device):
    model = _small_model(device)
    model.set_params(no_warning=True, positivity_flag=True)
    phantom, sinogram = _box_problem(model)
    # Negative-going noise would drive an unconstrained recon negative.
    rng = np.random.RandomState(3)
    noisy = sinogram + 0.1 * np.max(sinogram) * rng.randn(*sinogram.shape).astype(np.float32)
    np.random.seed(0)
    recon, recon_dict = model.recon(noisy, max_iterations=3,
                                    stop_threshold_change_pct=0.0)
    assert float(recon.min()) >= -1e-5
    fm = recon_dict['recon_params']['fm_rmse']
    assert fm[-1] < fm[0]


def test_restart_contract():
    model = _small_model()
    _, sinogram = _box_problem(model)
    np.random.seed(11)
    r3, d3 = model.recon(sinogram, max_iterations=3, stop_threshold_change_pct=0.0)
    # Restart: two more iterations continuing the partition sequence.
    np.random.seed(12)
    rr, dr = model.recon(sinogram, init_recon=r3, max_iterations=5,
                         first_iteration=3, stop_threshold_change_pct=0.0)
    assert dr['recon_params']['num_iterations'] == 2
    # The restart continues improving on the run it resumed.
    assert dr['recon_params']['fm_rmse'][-1] <= d3['recon_params']['fm_rmse'][-1] + 1e-6


def test_weights_none_equals_explicit_ones():
    model = _small_model()
    _, sinogram = _box_problem(model)
    np.random.seed(5)
    r_none, _ = model.recon(sinogram, max_iterations=2, stop_threshold_change_pct=0.0)
    np.random.seed(5)
    r_ones, _ = model.recon(sinogram, weights=np.ones_like(sinogram),
                            max_iterations=2, stop_threshold_change_pct=0.0)
    rel = np.max(np.abs(r_none - r_ones)) / max(np.max(np.abs(r_ones)), 1e-30)
    assert rel < 1e-5, rel


def test_compile_on_off_value_equality():
    results = {}
    for mode in ("auto", "off"):
        model = _small_model(compile_mode=mode)
        _, sinogram = _box_problem(model)
        np.random.seed(7)
        recon, _ = model.recon(sinogram, max_iterations=2,
                               stop_threshold_change_pct=0.0)
        results[mode] = recon
    rel = (np.max(np.abs(results["auto"] - results["off"]))
           / max(np.max(np.abs(results["off"])), 1e-30))
    assert rel < 1e-4, rel


class _LargestAllocation(torch.utils._python_dispatch.TorchDispatchMode):
    """The largest NEW array allocated inside the block, in bytes.

    An in-place op hands back the buffer it was given, so an output that
    shares storage with an input is not an allocation and is not counted.
    Used to hold a claim about what a piece of arithmetic does NOT allocate,
    which is the claim the memory ledger's charges rest on.
    """

    def __init__(self):
        self.nbytes = 0

    @staticmethod
    def _tensors(obj):
        if isinstance(obj, torch.Tensor):
            return [obj]
        if isinstance(obj, (tuple, list)):
            return [t for item in obj
                    for t in _LargestAllocation._tensors(item)]
        if isinstance(obj, dict):
            return [t for item in obj.values()
                    for t in _LargestAllocation._tensors(item)]
        return []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        held = {t.data_ptr() for t in self._tensors(args)}
        held |= {t.data_ptr() for t in self._tensors(kwargs or {})}
        out = func(*args, **(kwargs or {}))
        for t in self._tensors(out):
            if t.data_ptr() not in held:
                self.nbytes = max(self.nbytes, t.numel() * t.element_size())
        return out
