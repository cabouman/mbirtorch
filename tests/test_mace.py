"""Tests for the MACE consensus loop and its agents.

The loop adds each agent's output into the consensus as it arrives, so its
update is checked against the plain formulas written out in full.  The
threaded path is checked with two workers on the CPU and the checkpoint by
a round trip.  The agents are checked with stubs that record what they were
asked to do.  The whole loop is checked by the equality gate.  There the
proximal map of the data-fit term and the qGGMRF denoiser at matched
strengths must reproduce the standard reconstruction.
"""

import math
import threading
import time
import warnings

import numpy as np
import pytest
import torch

import mbirtorch
from mbirtorch import _sharding
from mbirtorch.mace import (MACE, ForwardProxAgent, HyperplaneAgent, QGGMRFDenoiserAgent, Task,
                            mace, resolve_device_pool)
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


def test_device_pool_resolves_every_form():
    """Each accepted form resolves to the devices it names, with an explicit
    index on every device that takes one; the refused forms raise by name."""
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
    # A device without an index names the same device as index 0, so the two
    # spellings resolve to one entry and a task pinned either way matches.
    assert resolve_device_pool([torch.device('cuda'), 'cuda:0']) == [torch.device('cuda', 0)] * 2
    assert Task(lambda d: None, device='cuda').device == torch.device('cuda', 0)


# ── stub agents ──────────────────────────────────────────────────────────────
class _AffineAgent:
    """An agent whose output is an affine map of its input, split into tasks
    along axis 0 at the given cut points.  Optionally records the thread that
    ran each task and carries a warm-start state."""

    def __init__(self, scale, offset, cuts=(), threads=None, device=None, with_state=False):
        self.scale, self.offset, self.cuts = scale, offset, tuple(cuts)
        self.threads = threads
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
                if self.threads is not None:
                    self.threads.append(threading.current_thread().name)
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
    threads = []
    agents = [_AffineAgent(0.6, 0.1, [2, 4], threads=threads),
              _AffineAgent(-0.3, 0.4, [1, 3, 5], threads=threads),
              _AffineAgent(0.9, 0.0, [3], threads=threads, device='cpu')]
    inline = MACE(agents, x0, rho=0.4)
    for _ in range(2):
        inline.step()
    reference = inline.x_bar.clone()

    del threads[:]
    with MACE(agents, x0, rho=0.4, devices=['cpu', 'cpu']) as threaded:
        for _ in range(2):
            threaded.step()
        rows = threaded.info['tasks'][-1]
    assert set(threads) == {'mace-worker-0', 'mace-worker-1'}
    assert len(rows) == 3 + 4 + 2 and sorted((row[0], row[1]) for row in rows) == \
        sorted((k, t) for k, n in enumerate((3, 4, 2)) for t in range(n))
    assert {row[2] for row in rows} == {0, 1}
    rel = _rel_max(threaded.x_bar, reference)
    print(f"two workers vs inline: rel_max = {rel:.2e}")
    assert rel < 1e-6


def test_task_fixed_to_a_device_outside_the_pool_is_refused():
    """A task pinned to a device the pool does not hold is refused by name.
    The pinned device is a CUDA device that this machine need not have; the
    check compares devices and moves no data, so it runs everywhere."""
    x0 = torch.zeros(4, 3)
    agent = _AffineAgent(1.0, 0.0, [], device=torch.device('cuda', 7))
    with MACE([agent], x0, devices=['cpu', 'cpu']) as loop:
        with pytest.raises(ValueError, match='not in the pool'):
            loop.step()


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


def test_x_bar_keeps_one_storage_across_steps():
    """The average handed to a callback is the loop's buffer.  It keeps one
    storage for the life of the loop, so a reference kept from an earlier
    step reads the latest average, never zeros."""
    torch.manual_seed(9)
    x0 = torch.randn(4, 3)
    kept = []
    with MACE([_AffineAgent(0.5, 0.1, []), _AffineAgent(-0.3, 0.2, [])], x0) as loop:
        loop.run(max_iterations=3, callback=lambda i, xb: kept.append(xb))
    assert all(k.data_ptr() == loop.x_bar.data_ptr() for k in kept)
    assert torch.equal(kept[0], loop.x_bar)
    assert not torch.equal(loop.x_bar, torch.zeros_like(x0))


def test_fold_after_all_folds_the_agents_pieces():
    """The output of an agent that keeps its task outputs and yields pieces
    once every task has run is added from those pieces, so the agent can
    transform the assembled output first.  Here the pieces are the outputs
    scaled by two."""
    torch.manual_seed(10)
    class Assembling(_AffineAgent):
        fold_after_all = True

        def tasks(self, w, iteration=0):
            self._held = {}
            tasks = []
            for start in range(0, int(w.shape[0]), 2):
                region = (slice(start, start + 2),) + (slice(None),) * (w.ndim - 1)

                def run(device, region=region, w=w):
                    self._held[region] = self._apply(w[region])
                    return None
                tasks.append(Task(run, device=None, region=region))
            return tasks

        def pieces(self):
            for region, out in self._held.items():
                yield region, 2.0 * out

    x0 = torch.randn(6, 3)
    plain = _AffineAgent(1.0, 0.4, [])   # the same map as 2 * (0.5 w + 0.2)
    reference = MACE([plain], x0)
    reference.step()
    for devices in (None, ['cpu', 'cpu']):
        with MACE([Assembling(0.5, 0.2, [])], x0, devices=devices) as loop:
            loop.step()
        rel = _rel_max(loop.x_bar, reference.x_bar)
        print(f"pieces vs plain agent, devices={devices}: rel_max = {rel:.2e}")
        assert rel < 1e-6


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

    # A loop with other weights keeps them and says so.
    other = MACE(make_agents(), x0, mu=[0.5, 0.5], rho=rho)
    with pytest.warns(UserWarning, match='keeps its own mu'):
        other.load_state_dict(state)
    assert other.mu == [0.5, 0.5]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        MACE(make_agents(), x0, mu=mu, rho=rho).load_state_dict(state)   # matching values: no warning


def test_mu_and_rho_change_between_steps():
    """The weights and the step are read at each step, so a change between
    steps takes effect on the next one."""
    torch.manual_seed(6)
    x0 = torch.randn(5, 4)
    agents = [_AffineAgent(0.5, 0.1, []), _AffineAgent(-0.2, 0.3, [])]
    loop = MACE(agents, x0, mu=[0.5, 0.5], rho=0.5)
    loop.step()
    loop.mu, loop.rho = [0.2, 0.8], 0.3
    loop.step()

    W_ref = [x0.clone(), x0.clone()]
    x_bar_ref = x0.clone()
    W_ref, x_bar_ref, _, _ = _plain_update(agents, W_ref, x_bar_ref, [0.5, 0.5], 0.5, 0)
    W_ref, x_bar_ref, _, _ = _plain_update(agents, W_ref, x_bar_ref, [0.2, 0.8], 0.3, 1)
    rel = max(_rel_max(loop.x_bar, x_bar_ref), max(_rel_max(w, w_ref) for w, w_ref in zip(loop.W, W_ref)))
    print(f"changed mu and rho vs plain formulas: rel_max = {rel:.2e}")
    assert rel < 1e-6


def test_run_stops_at_the_threshold_and_calls_back():
    """run stops when the percent change falls below the threshold, calls the
    callback after each step with the index just completed, and returns the
    average with the traces."""
    x0 = torch.ones(4, 3)
    agents = [_AffineAgent(1.0, 0.0, [])]   # the identity: the average never changes
    seen = []
    with MACE(agents, x0) as loop:
        x_bar, info = loop.run(max_iterations=10, stop_threshold_change_pct=1e-3,
                               callback=lambda i, xb: seen.append((i, float(xb.sum()))))
    assert loop.iteration == 1 and seen == [(0, float(x0.sum()))]
    assert info['change_pct'] == [0.0] and torch.equal(x_bar, x0)


def test_mace_wrapper_keeps_the_script_traces():
    """The one-call form returns the loop's traces plus the two the scripts
    read, derived from them."""
    torch.manual_seed(11)
    x0 = torch.randn(4, 3)
    x_bar, info = mace([_AffineAgent(0.5, 0.1, []), _AffineAgent(0.5, 0.2, [])], x0, num_iterations=3)
    assert x_bar.shape == x0.shape
    assert len(info['consensus_change']) == 3 and len(info['consensus_spread']) == 3
    assert info['consensus_change'] == [c / 100.0 for c in info['change_pct']]
    assert info['consensus_spread'] == [max(s) for s in info['spread']]


# ── the agents ───────────────────────────────────────────────────────────────
class _RecordingProxModel:
    """A stand-in for a tomography model whose prox_map records its
    arguments and returns a scaled input."""

    def __init__(self):
        self.calls = []

    def prox_map(self, prox_input, sinogram, **kwargs):
        self.calls.append(dict(kwargs, prox_input=prox_input))
        return 0.5 * prox_input, {}


@pytest.mark.parametrize('advance', [1.0, 0.25, 3.0])
def test_partition_advance_sets_the_prox_iteration_window(advance):
    """The call at loop iteration i runs prox_map from entry floor(i * advance)
    of the partition sequence for inner_iterations iterations, and
    initializes at iteration 0 only."""
    model = _RecordingProxModel()
    agent = ForwardProxAgent(model, sinogram=None, inner_iterations=3, partition_advance=advance)
    w = torch.ones(3, 3, 1)
    for iteration in range(6):
        agent(w, iteration)
    for iteration, call in enumerate(model.calls):
        first = int(math.floor(iteration * advance))
        assert call['first_iteration'] == first
        assert call['max_iterations'] == first + 3
        assert call['do_initialization'] == (iteration == 0)
        assert call['stop_threshold_change_pct'] == 0.0


def test_prox_agent_warm_start_uses_its_previous_output():
    """With the warm start on, init_recon is the agent's previous output,
    and the constructor's init_recon before any call.  With it off,
    init_recon is the input."""
    w1, w2 = torch.ones(2, 2, 1), 2.0 * torch.ones(2, 2, 1)
    start = 3.0 * torch.ones(2, 2, 1)

    model = _RecordingProxModel()
    warm = ForwardProxAgent(model, sinogram=None, init_recon=start, use_warm_start=True)
    out1 = warm(w1, 0)
    out2 = warm(w2, 1)
    assert torch.equal(model.calls[0]['init_recon'], start)
    assert torch.equal(model.calls[1]['init_recon'], out1)
    assert torch.equal(warm.state_dict()['previous_output'], out2)

    model = _RecordingProxModel()
    cold = ForwardProxAgent(model, sinogram=None, init_recon=start, use_warm_start=False)
    cold(w1, 0)
    cold(w2, 1)
    assert torch.equal(model.calls[0]['init_recon'], w1)
    assert torch.equal(model.calls[1]['init_recon'], w2)
    assert cold.state_dict()['previous_output'] is None


class _RecordingStackDenoiser:
    """A stack denoiser that adds a constant, records the init_stack it was
    given, and checks that each stack has the hyperplane index first and the
    given frame count second."""

    def __init__(self, constant, frames, delay=0.0):
        self.constant = constant
        self.frames = frames
        self.delay = delay
        self.init_stacks = []

    def __call__(self, stack, init_stack=None):
        assert stack.ndim == 4 and stack.shape[1] == self.frames, tuple(stack.shape)
        if init_stack is not None:
            assert tuple(init_stack.shape) == tuple(stack.shape)
        self.init_stacks.append(None if init_stack is None else init_stack.clone())
        if self.delay:
            time.sleep(self.delay)
        return stack + self.constant


@pytest.mark.parametrize('axis', [1, 2, 3])
def test_hyperplane_agent_equals_the_unbatched_operation(axis, device):
    """The agent cuts the array into slabs along the axis, permutes each slab,
    denoises it, and permutes it back.  With a batch size that does not
    divide the slab count, the result equals the stack denoiser applied to
    the whole array at once.  A denoiser that adds a constant makes this data
    movement, so the check is exact; the denoiser itself checks that the
    frame index comes second in every stack it receives."""
    torch.manual_seed(7)
    w = torch.randn(3, 5, 6, 7, device=device)
    agent = HyperplaneAgent(axis, lambda dev: _RecordingStackDenoiser(2.5, frames=3), batch_size=4)
    tasks = agent.tasks(w, 0)
    assert len(tasks) == math.ceil(w.shape[axis] / 4)
    assert all(task.device is None for task in tasks)
    out = agent(w, 0)
    assert torch.equal(out, w + 2.5)
    # Through the loop, with two workers, the same.
    with MACE([agent], w, devices=['cpu', 'cpu']) as loop:
        loop.step()
    assert torch.equal(loop.x_bar, w + 2.5)


def test_hyperplane_agent_applies_the_filter_along_the_frame_axis():
    """With a filter matrix, each slab is filtered along the frame axis before
    denoising.  With the zero matrix of three frames at period 6 the identity
    denoiser returns zeros, and with twelve frames the result equals the
    filter applied to the whole array."""
    torch.manual_seed(12)
    w = torch.randn(3, 4, 4, 5)
    agent = HyperplaneAgent(3, lambda dev: (lambda stack: stack), filter_matrix=temporal_filter_matrix(3, 6))
    assert torch.equal(agent(w, 0), torch.zeros_like(w))
    matrix = temporal_filter_matrix(12, 6)
    w = torch.randn(12, 4, 4, 5)
    agent = HyperplaneAgent(2, lambda dev: (lambda stack: stack), filter_matrix=matrix, batch_size=3)
    rel = _rel_max(agent(w, 0), apply_temporal_filter(w, matrix, axis=0))
    print(f"hyperplane filter vs direct filter: rel_max = {rel:.2e}")
    assert rel < 1e-6


@pytest.mark.parametrize('batch_size', [6, 2])
def test_hyperplane_agent_warm_start_passes_its_previous_output(batch_size):
    """With the warm start on, every task of the first call passes no
    init_stack, and every task of the second call passes the matching slab of
    the first call's output, permuted like the input.  This holds whether the
    call is one task or several.  With the warm start off, no init_stack is
    ever passed."""
    torch.manual_seed(13)
    w = torch.randn(3, 4, 5, 6)
    denoisers = []

    def make(device):
        denoisers.append(_RecordingStackDenoiser(1.0, frames=3))
        return denoisers[-1]
    warm = HyperplaneAgent(3, make, batch_size=batch_size, use_warm_start=True)
    tasks_per_call = len(warm.tasks(w, 0))
    first = warm(w, 0)
    warm(w, 1)
    recorded = denoisers[0].init_stacks
    assert len(recorded) == 2 * tasks_per_call
    assert all(init is None for init in recorded[:tasks_per_call])
    for task, init in zip(warm.tasks(w, 1), recorded[tasks_per_call:]):
        assert torch.equal(init, first[task.region].movedim(3, 0))
    assert torch.equal(warm.state_dict()['previous_output'], first)

    denoisers.clear()
    cold = HyperplaneAgent(3, make, batch_size=batch_size, use_warm_start=False)
    cold(w, 0)
    cold(w, 1)
    assert denoisers[0].init_stacks == [None] * (2 * tasks_per_call)
    assert cold.state_dict()['previous_output'] is None


def test_hyperplane_agent_makes_one_denoiser_per_worker():
    """Each worker thread that runs a task gets its own stack denoiser from
    make_stack_denoiser, so two workers on one device never share one.  The
    denoiser sleeps briefly so that neither worker can drain the queue before
    the other starts."""
    torch.manual_seed(14)
    w = torch.randn(3, 4, 4, 8)
    made = []

    def make(device):
        made.append(threading.current_thread().name)
        return _RecordingStackDenoiser(0.0, frames=3, delay=0.02)
    agent = HyperplaneAgent(3, make, batch_size=1, use_warm_start=True)
    with MACE([agent], w, devices=['cpu', 'cpu']) as loop:
        loop.step()
        loop.step()
    assert len(made) == len(set(made)) == 2, made


def test_agents_refuse_a_model_on_several_devices():
    """The agents return one tensor.  A denoiser configured on two devices
    returns its output divided across them, which the agent refuses by name
    rather than failing inside the loop."""
    shape = (8, 10, 12)
    agent = QGGMRFDenoiserAgent(shape, sigma_noise=0.1, pinned_params={'sigma_x': 0.05}, inner_iterations=1)
    agent.model.configure_devices(devices=['cpu', 'cpu'])
    with pytest.raises(ValueError, match='one device'):
        agent(torch.zeros(shape), 0)

    class DividedProxModel(_RecordingProxModel):
        def prox_map(self, prox_input, sinogram, **kwargs):
            placement = _sharding.Placement(['cpu', 'cpu'], axis=-1, axis_len=2)
            return _sharding.Shards([prox_input[..., :1], prox_input[..., 1:]], placement), {}
    forward = ForwardProxAgent(DividedProxModel(), sinogram=None)
    with pytest.raises(ValueError, match='one device'):
        forward(torch.zeros(2, 2, 2), 0)


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
