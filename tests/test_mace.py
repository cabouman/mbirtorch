"""Tests for the MACE consensus loop and its agents.

The loop adds each agent's output into the consensus as it arrives, so its
update is checked against the plain formulas written out in full.  The
threaded path is checked with two workers on the CPU and the checkpoint by
a round trip.  The agents are checked with stub denoisers.  The whole loop
is checked by the equality gate.  There the proximal map of the data-fit
term and the qGGMRF denoiser at matched strengths must reproduce the
standard reconstruction.
"""

import math

import numpy as np
import pytest
import torch

import mbirtorch
import mbirtorch.mace as mace_module
from mbirtorch.mace import (MACE, ForwardProxAgent, HyperplaneAgent, QGGMRFDenoiserAgent, Task,
                            resolve_device_pool)
from mbirtorch.mace4d import apply_temporal_filter, temporal_filter_matrix
from mbirtorch.tomography_model import default_devices, gpu_devices


def _rel_max(out, ref):
    out = np.asarray(out.detach().cpu() if torch.is_tensor(out) else out, dtype=np.float64)
    ref = np.asarray(ref.detach().cpu() if torch.is_tensor(ref) else ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


# ── the device pool ──────────────────────────────────────────────────────────
def _indexed(device):
    """A device with an explicit index, as the pool resolver reports them."""
    device = torch.device(device)
    return device if device.type == 'cpu' or device.index is not None else torch.device(device.type, 0)


def test_device_pool_resolves_every_form(monkeypatch):
    """Each accepted form resolves to the devices it names, with an explicit
    index on every device that takes one, and the refused forms raise by
    name.  A GPU named twice is refused and a repeated CPU is kept.
    MBIRTORCH_NUM_DEVICES pins the count for the whole process, and the
    automatic forms honor it.

    The conftest fixture pins every test to one device, which is what keeps
    the suite deterministic on a multi-GPU host.  A test of the pool's
    hardware forms has to opt out of it, and doing so explicitly keeps the
    pin's reach visible.
    """
    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES', raising=False)
    defaults = [_indexed(d) for d in default_devices()]
    assert resolve_device_pool(None) == defaults
    assert resolve_device_pool('cpu') == [torch.device('cpu')]
    assert resolve_device_pool(1) == defaults[:1]
    assert resolve_device_pool([0]) == defaults[:1]
    assert resolve_device_pool(['cpu', 'cpu']) == [torch.device('cpu'), torch.device('cpu')]
    assert resolve_device_pool([torch.device('cpu')]) == [torch.device('cpu')]
    if gpu_devices():
        assert resolve_device_pool('gpu') == [_indexed(d) for d in gpu_devices()]
    else:
        with pytest.raises(ValueError, match='no GPU'):
            resolve_device_pool('gpu')
    with pytest.raises(ValueError, match='requested'):
        resolve_device_pool(len(defaults) + 1)
    with pytest.raises(ValueError, match='out of range'):
        resolve_device_pool([len(defaults)])
    with pytest.raises(ValueError, match="'cpu' or 'gpu'"):
        resolve_device_pool('tpu')
    with pytest.raises(ValueError, match='boolean'):
        resolve_device_pool(True)
    with pytest.raises(ValueError, match='at least one'):
        resolve_device_pool([])
    # A device without an index names the same device as index 0.  The two
    # spellings therefore name one GPU, which the pool refuses, and a task
    # pinned either way matches.
    with pytest.raises(ValueError, match='more than once'):
        resolve_device_pool([torch.device('cuda'), 'cuda:0'])
    assert Task(lambda d: None, device='cuda').device == torch.device('cuda', 0)

    # Two workers on one GPU share it and gain nothing, and on an Apple GPU
    # they fail inside Metal, so a pool that names a GPU twice raises and
    # names the device.  Repeating the CPU stays the way a pool runs more
    # than one worker on a machine with no GPU.
    assert resolve_device_pool(['cpu', 'cpu', 'cpu']) == [torch.device('cpu')] * 3
    for spelling in ('cuda:0', 'mps'):
        with pytest.raises(ValueError, match='more than once'):
            resolve_device_pool([spelling, spelling])
    # The refusal reads the whole pool, not just neighboring entries.
    with pytest.raises(ValueError, match='cuda:1'):
        resolve_device_pool(['cuda:1', 'cpu', 'cuda:1'])
    if gpu_devices():
        gpu = _indexed(gpu_devices()[0])
        assert resolve_device_pool([gpu]) == [gpu]

    # With the count pinned, None gives the pinned count and a larger count
    # raises with a message that names the variable.  A list of devices is
    # the caller's and keeps every device it names.
    four = [torch.device('cuda', index) for index in range(4)]
    monkeypatch.setattr(mace_module, 'default_devices', lambda: tuple(four))
    monkeypatch.setenv('MBIRTORCH_NUM_DEVICES', '2')
    assert resolve_device_pool(None) == four[:2]
    assert resolve_device_pool(2) == four[:2]
    with pytest.raises(ValueError, match='MBIRTORCH_NUM_DEVICES'):
        resolve_device_pool(3)
    assert resolve_device_pool([f'cuda:{index}' for index in range(4)]) == four
    monkeypatch.delenv('MBIRTORCH_NUM_DEVICES')
    assert resolve_device_pool(None) == four
    assert resolve_device_pool(3) == four[:3]


# ── stub agents ──────────────────────────────────────────────────────────────
class _AffineAgent:
    """An agent whose output is an affine map of its input, split into tasks
    along axis 0 at the given cut points.  Optionally carries a warm-start
    state."""

    def __init__(self, scale, offset, cuts=(), device=None, with_state=False):
        self.scale, self.offset, self.cuts = scale, offset, tuple(cuts)
        self.device = None if device is None else torch.device(device)
        self.with_state = with_state
        self._previous_output = None

    def _apply(self, w):
        """The whole output for the whole input, from the previous output."""
        out = self.scale * w + self.offset
        if self.with_state and self._previous_output is not None:
            out = out + 0.01 * self._previous_output.to(out.device)
        return out

    def __call__(self, w, iteration=0):
        out = self._apply(w)
        if self.with_state:
            self._previous_output = out.clone()
        return out

    def tasks(self, w, iteration=0):
        # The whole output is formed here from the previous state, and each
        # task hands over its region of it, so the state carried is the
        # output of this call and the tasks read the output of the last one.
        output = self._apply(w)
        if self.with_state:
            self._previous_output = output.clone()
        bounds = [0] + sorted(self.cuts) + [int(w.shape[0])]
        tasks = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            region = (slice(start, stop),) + (slice(None),) * (w.ndim - 1)

            def run(device, region=region, output=output):
                return output[region].to(device).to(output.device)
            tasks.append(Task(run, device=self.device, region=region))
        return tasks

    def state_dict(self):
        return {'previous_output': None if self._previous_output is None else self._previous_output.clone()}

    def load_state_dict(self, state):
        self._previous_output = None if state['previous_output'] is None else state['previous_output'].clone()


def _random_cuts(rng, length, count):
    """count - 1 distinct cut points inside [1, length), sorted, which divide
    axis 0 into count regions."""
    return sorted(rng.choice(np.arange(1, length), size=count - 1, replace=False).tolist())


def _plain_update(agents, W, x_bar_prev, mu, rho, iteration):
    """One step of the consensus update written in full, for comparison."""
    X = [agent(w, iteration) for agent, w in zip(agents, W)]
    z = sum(m * (2.0 * x - w) for m, x, w in zip(mu, X, W))
    W_new = [w + 2.0 * rho * (z - x) for w, x in zip(W, X)]
    x_bar = sum(m * x for m, x in zip(mu, X))
    change = float(torch.linalg.vector_norm(x_bar - x_bar_prev) / torch.linalg.vector_norm(x_bar_prev))
    spread = [float(torch.linalg.vector_norm(x - x_bar_prev) / torch.linalg.vector_norm(x_bar_prev)) for x in X]
    return W_new, x_bar, 100.0 * change, spread


# ── the loop ─────────────────────────────────────────────────────────────────
def test_folding_update_equals_the_plain_formulas(device):
    """The update applied region by region as the tasks complete equals the
    update written out in full on the same inputs.  Four quantities are
    compared: the agent inputs, the consensus average, the change statistic,
    and the per-agent spread.  The four agents have random regions that cover
    the array, so the pieces arrive in an order the plain formulas never
    see."""
    rng = np.random.default_rng(3)
    torch.manual_seed(3)
    shape = (8, 5, 6, 4)
    x0 = torch.randn(shape, device=device)
    agents = [_AffineAgent(0.7, 0.2, _random_cuts(rng, shape[0], 3)),
              _AffineAgent(-0.4, 0.5, _random_cuts(rng, shape[0], 2)),
              _AffineAgent(1.1, -0.3, []),
              _AffineAgent(0.5, 0.1, _random_cuts(rng, shape[0], 4))]
    mu = [0.5, 1 / 6, 1 / 6, 1 / 6]
    rho = 0.5
    loop = MACE(agents, x0, mu=mu, rho=rho)
    W_ref = [x0.clone() for _ in agents]
    x_bar_ref = x0.clone()
    worst = {'W': 0.0, 'x_bar': 0.0, 'change': 0.0, 'spread': 0.0}
    for iteration in range(3):
        change_pct = loop.step()
        W_ref, x_bar_ref, change_ref, spread_ref = _plain_update(agents, W_ref, x_bar_ref, mu, rho, iteration)
        worst['W'] = max(worst['W'], max(_rel_max(w, w_ref) for w, w_ref in zip(loop.W, W_ref)))
        worst['x_bar'] = max(worst['x_bar'], _rel_max(loop.x_bar, x_bar_ref))
        worst['change'] = max(worst['change'], abs(change_pct - change_ref) / change_ref)
        worst['spread'] = max(worst['spread'], max(abs(a - b) / b for a, b in zip(loop.info['spread'][-1], spread_ref)))
    print(f"folded vs plain update on {device}: W {worst['W']:.2e}, x_bar {worst['x_bar']:.2e}, "
          f"change {worst['change']:.2e}, spread {worst['spread']:.2e}")
    assert worst['W'] < 1e-6 and worst['x_bar'] < 1e-6
    assert worst['change'] < 1e-5 and worst['spread'] < 1e-5
    assert loop.iteration == 3 and len(loop.info['change_pct']) == 3


def test_two_workers_share_the_queue():
    """A pool of two CPU workers runs every task once.  Both workers take
    tasks that accept any device from the shared queue.  A task fixed to the
    CPU runs on a CPU worker.  The result equals the inline run."""
    torch.manual_seed(4)
    x0 = torch.randn(6, 4, 5)
    agents = [_AffineAgent(0.6, 0.1, [2, 4]),
              _AffineAgent(-0.3, 0.4, [1, 3, 5]),
              _AffineAgent(0.9, 0.0, [3], device='cpu')]
    inline = MACE(agents, x0, rho=0.4)
    for _ in range(2):
        inline.step()
    reference = inline.x_bar.clone()

    with MACE(agents, x0, rho=0.4, devices=['cpu', 'cpu']) as threaded:
        for _ in range(2):
            threaded.step()
    rel = _rel_max(threaded.x_bar, reference)
    print(f"two workers vs inline: rel_max = {rel:.2e}")
    assert rel < 1e-6


def test_a_failed_step_leaves_the_loop_refusing_further_steps():
    """A task that raises leaves some inputs with a partial update.  The
    error propagates, the loop refuses further steps and state saves, and
    loading a saved state makes it usable again."""
    torch.manual_seed(8)
    x0 = torch.randn(4, 3)

    class Failing(_AffineAgent):
        def tasks(self, w, iteration=0):
            tasks = super().tasks(w, iteration)
            tasks[-1] = Task(lambda device: 1 / 0, device=None, region=tasks[-1].region)
            return tasks

    good = MACE([_AffineAgent(0.5, 0.1, [2])], x0)
    good.step()
    saved = good.state_dict()
    for devices in (None, ['cpu', 'cpu']):
        with MACE([Failing(0.5, 0.1, [2])], x0, devices=devices) as loop:
            with pytest.raises(ZeroDivisionError):
                loop.step()
            with pytest.raises(RuntimeError, match='inconsistent'):
                loop.step()
            with pytest.raises(RuntimeError, match='inconsistent'):
                loop.state_dict()
            loop.load_state_dict(saved)
            loop.agents[0] = _AffineAgent(0.5, 0.1, [2])
            loop.step()
            assert loop.iteration == 2


def test_checkpoint_round_trip(device):
    """Three steps, a saved state, a new loop that loads it, and two more
    steps equal five steps in one loop, with agents that carry warm-start
    state of their own."""
    torch.manual_seed(5)
    x0 = torch.randn(6, 4, 3, device=device)

    def make_agents():
        return [_AffineAgent(0.6, 0.1, [2], with_state=True), _AffineAgent(-0.3, 0.3, [3], with_state=True)]
    mu, rho = [0.4, 0.6], 0.45
    straight = MACE(make_agents(), x0, mu=mu, rho=rho)
    for _ in range(5):
        straight.step()

    first = MACE(make_agents(), x0, mu=mu, rho=rho)
    for _ in range(3):
        first.step()
    state = first.state_dict()
    resumed = MACE(make_agents(), x0, mu=mu, rho=rho)
    resumed.load_state_dict(state)
    assert resumed.iteration == 3
    for _ in range(2):
        resumed.step()
    rel = max(_rel_max(resumed.x_bar, straight.x_bar),
              max(_rel_max(a, b) for a, b in zip(resumed.W, straight.W)))
    print(f"checkpoint round trip on {device}: rel_max = {rel:.2e}")
    assert rel < 1e-6
    assert resumed.iteration == 5 and len(resumed.info['change_pct']) == 5


# ── the agents ───────────────────────────────────────────────────────────────
class _ConstantStackDenoiser:
    """A stack denoiser that adds a constant and checks that each stack has
    the hyperplane index first and the given frame count second."""

    def __init__(self, constant, frames):
        self.constant = constant
        self.frames = frames

    def __call__(self, stack, init_stack=None):
        assert stack.ndim == 4 and stack.shape[1] == self.frames, tuple(stack.shape)
        if init_stack is not None:
            assert tuple(init_stack.shape) == tuple(stack.shape)
        return stack + self.constant


@pytest.mark.parametrize('axis', [1, 2, 3])
def test_hyperplane_agent_equals_the_unbatched_operation(axis, device):
    """The agent cuts the array into slabs along the axis, permutes each slab,
    denoises it, and permutes it back.  With a batch size that does not
    divide the slab count, the result equals the stack denoiser applied to
    the whole array at once.  A denoiser that adds a constant makes this data
    movement, so the check is exact; the denoiser itself checks that the
    frame index comes second in every stack it receives.  With a filter
    matrix, each slab is filtered along the frame axis before denoising:
    with the zero matrix of three frames at period 6 the identity denoiser
    returns zeros, and with twelve frames the result equals the filter
    applied to the whole array."""
    torch.manual_seed(7)
    w = torch.randn(3, 5, 6, 7, device=device)
    agent = HyperplaneAgent(axis, lambda dev: _ConstantStackDenoiser(2.5, frames=3), batch_size=4)
    tasks = agent.tasks(w, 0)
    assert len(tasks) == math.ceil(w.shape[axis] / 4)
    assert all(task.device is None for task in tasks)
    out = agent(w, 0)
    assert torch.equal(out, w + 2.5)
    # Through the loop, with two workers, the same.
    with MACE([agent], w, devices=['cpu', 'cpu']) as loop:
        loop.step()
    assert torch.equal(loop.x_bar, w + 2.5)

    torch.manual_seed(12)
    filtered_in = torch.randn(3, 4, 4, 5)
    agent = HyperplaneAgent(3, lambda dev: (lambda stack: stack), filter_matrix=temporal_filter_matrix(3, 6))
    assert torch.equal(agent(filtered_in, 0), torch.zeros_like(filtered_in))
    matrix = temporal_filter_matrix(12, 6)
    filtered_in = torch.randn(12, 4, 4, 5)
    agent = HyperplaneAgent(2, lambda dev: (lambda stack: stack), filter_matrix=matrix, batch_size=3)
    rel = _rel_max(agent(filtered_in, 0), apply_temporal_filter(filtered_in, matrix, axis=0))
    print(f"hyperplane filter vs direct filter: rel_max = {rel:.2e}")
    assert rel < 1e-6


@pytest.mark.parametrize('axis', [1, 2, 3])
def test_hyperplane_agent_never_hands_the_denoiser_a_view_of_its_input(axis):
    """The stack denoiser may write the stack it is given, so the agent hands
    it a copy.  Moving an axis of length one to the front of an array that is
    already on the worker's device leaves a view of that array, the one case
    where the copy has to be made on purpose: a denoiser that zeroes its
    stack in place must leave the agent's input, the loop's state, as it was,
    with the filter off and with the warm start on or off."""
    shape = [3, 4, 5, 6]
    shape[axis] = 1
    torch.manual_seed(3)
    w = torch.randn(*shape)
    before = w.clone()

    def zeroing(stack, init_stack=None):
        stack.zero_()
        if init_stack is not None:
            init_stack.zero_()
        return stack

    for warm in (False, True):
        agent = HyperplaneAgent(axis, lambda dev: zeroing, use_warm_start=warm)
        for iteration in range(2):
            out = agent(w, iteration)
            assert torch.equal(out, torch.zeros_like(w))
            assert torch.equal(w, before), f'axis {axis}, warm start {warm}, call {iteration}'


# ── the equality gate: the whole loop against the standard reconstruction ────
def test_mace_reproduces_the_standard_reconstruction():
    """The proximal map of the data-fit term and the qGGMRF denoiser, at
    matched strengths and with the prior parameters pinned to the standard
    reconstruction's, solve the same objective as recon.  The reference is a
    100-iteration recon of a small 2D cone-beam problem, and the start is a
    30-iteration recon of it, which is more than 1 percent away.  In 30
    iterations the consensus must move to within 1 percent NRMSE of the
    reference.  The reference has to be converged well below the tolerance.
    A 30-iteration recon is not, so the consensus overtakes it and the
    distance to it grows with iterations."""
    num_views, det = 64, 64
    phantom, sinogram, params = mbirtorch.generate_demo_data(
        model_type='cone', object_type='shepp-logan', num_views=num_views,
        num_det_rows=det, num_det_channels=det, target_max_attenuation=6.0)
    noise_std = np.sqrt(np.exp(sinogram) / 500.0)
    rng = np.random.default_rng(0)
    sinogram = (sinogram + noise_std * rng.standard_normal(sinogram.shape)).astype(np.float32)
    sinogram = sinogram[:, det // 2:det // 2 + 1]

    def make_model():
        model = mbirtorch.ConeBeamModel(sinogram.shape, params['angles'],
                                        source_detector_dist=params['source_detector_dist'],
                                        source_iso_dist=params['source_iso_dist'])
        model.configure_devices(devices=['cpu'])
        model.set_params(no_warning=True, sharpness=1.0, verbose=0)
        return model
    weights = mbirtorch.gen_weights(sinogram, weight_type='transmission_root')

    model = make_model()
    np.random.seed(0)
    reference, reference_dict = model.recon(sinogram, weights=weights, max_iterations=100,
                                            stop_threshold_change_pct=0.0, print_logs=False,
                                            logfile_path=None, output_sharded=True)
    regularization = reference_dict['recon_params']['regularization_params']
    sigma = float(regularization['sigma_prox'])
    sigma_x = float(regularization['sigma_x'])

    np.random.seed(0)
    start, _ = make_model().recon(sinogram, weights=weights, max_iterations=30,
                                  stop_threshold_change_pct=0.0, print_logs=False,
                                  logfile_path=None, output_sharded=True)
    start_distance = float(torch.linalg.vector_norm(start - reference) / torch.linalg.vector_norm(reference))
    np.random.seed(0)
    forward = ForwardProxAgent(model, sinogram, weights=weights, sigma_prox=sigma, inner_iterations=3,
                               init_recon=start.clone())
    denoiser = QGGMRFDenoiserAgent(tuple(reference.shape), sigma_noise=sigma,
                                   pinned_params={'sigma_x': sigma_x}, inner_iterations=8,
                                   like_model=model, use_ror_mask=model.get_params('use_ror_mask'))
    trace = []
    with MACE([forward, denoiser], start, rho=0.5) as loop:
        x_bar, info = loop.run(
            max_iterations=30,
            callback=lambda i, xb: trace.append(float(torch.linalg.vector_norm(xb - reference)
                                                      / torch.linalg.vector_norm(reference))))
    print(f"equality gate: start {start_distance:.4f} from recon(100); NRMSE after 10/20/30 "
          f"iterations = {trace[9]:.4f} / {trace[19]:.4f} / {trace[29]:.4f}; "
          f"final spread {info['spread'][-1]}")
    assert start_distance > 0.01          # the loop has to move, not merely stay
    assert trace[-1] < 0.01
