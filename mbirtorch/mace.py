"""The MACE consensus loop and the agents it runs.

Multi-agent consensus equilibrium (MACE) reconstructs one array from several
agents.  Each agent maps the array to the array it prefers, for example the
proximal map of a data-fit term or a denoiser.  An agent is a callable
``agent(w, iteration)`` from a tensor to a tensor of the same shape.  The
loop keeps one input ``W_k`` per agent.  At each step it evaluates every
agent, forms the consensus point ``z = sum_k mu_k (2 X_k - W_k)``, moves each
input by ``W_k += 2 rho (z - X_k)``, and reports the average
``x_bar = sum_k mu_k X_k``.  At a fixed point every agent output equals the
average.  With two agents, equal weights, and ``rho = 1/2`` the iteration is
the Douglas-Rachford (ADMM) form.

:class:`MACE` is the loop.  :class:`ForwardProxAgent`,
:class:`QGGMRFDenoiserAgent`, and :class:`HyperplaneAgent` are the agents
provided, and any callable of the same form serves as one.  :func:`mace`
runs the loop in one call.
"""

import copy
import functools
import math
import queue
import threading
import time
import warnings

import numpy as np
import torch

from . import _sharding
from ._memory_ledger import (DEVICE_COUNT_ENV_VAR, ELL1_CHUNK_BYTES, ELL1_MAX_CHUNKS,
                             pinned_device_count)
from .denoising import QGGMRFDenoiser
from .tomography_model import cpu_devices, default_devices, gpu_devices
from .vcd_utils import named_rng


def _canonical_device(device):
    """Return a ``torch.device`` with an explicit index for the device types
    that take one, so that ``'cuda'`` and ``'cuda:0'`` name the same
    device."""
    device = torch.device(device)
    if device.type in ('cuda', 'mps', 'xpu') and device.index is None:
        return torch.device(device.type, 0)
    return device


def _filter_along_axis(x, matrix, axis):
    """Apply a square matrix along one axis of a tensor, on the tensor's
    device and in its dtype."""
    matrix = matrix.to(device=x.device, dtype=x.dtype)
    moved = x.movedim(axis, 0)
    filtered = torch.tensordot(matrix, moved, dims=1)
    return filtered.movedim(0, axis)


def resolve_device_pool(devices=None):
    """
    Turn a description of a device pool into a list of ``torch.device``.

    Args:
        devices: one of the following.

            * None: every GPU, or the CPU when there is no GPU
              (:func:`~mbirtorch.tomography_model.default_devices`).
            * ``'cpu'``: the CPU device.  torch presents one CPU device
              however many cores the machine has, so this pool has one entry.
            * ``'gpu'``: every GPU; raises when there is none.
            * an integer ``n``: the first ``n`` default devices; raises when
              ``n`` exceeds them.
            * a sequence of integers: those indices into the default devices.
            * a sequence of devices (``torch.device`` or device strings):
              those devices, in that order.  The CPU may be repeated, which
              gives one worker per entry; a repeated GPU is refused.

    The automatic forms, None and a count, are capped by the device count
    ``MBIRTORCH_NUM_DEVICES`` pins, because that variable pins the count for
    the whole process and the reconstruction policy already reads it as an
    explicit choice.  A list of devices or of indices is the caller's and is
    not capped, and :func:`~mbirtorch.tomography_model.default_devices` and its siblings still report the
    hardware.

    Returns:
        list of torch.device

    Raises:
        ValueError: for a platform string other than ``'cpu'`` or ``'gpu'``,
            a request for GPUs when there are none, a count or index beyond
            the available devices, or a GPU named more than once.
    """
    return _reject_repeated_gpu(_device_pool(devices))


def _reject_repeated_gpu(pool):
    """Return ``pool`` unchanged, or raise when it names a GPU twice.

    Two workers on one CUDA device share one stream, and two workers on one
    Apple GPU fail inside Metal.  A repeated CPU entry is allowed, because
    repeating the CPU runs the threaded path on a machine with no GPU.
    """
    seen = set()
    for device in pool:
        if device.type == 'cpu':
            continue
        if device in seen:
            raise ValueError(
                f'the device pool names {device} more than once.  Two workers on one GPU '
                'share that GPU and gain nothing, and on an Apple GPU they fail inside '
                'Metal.  Repeat the CPU instead to run more than one worker.')
        seen.add(device)
    return pool


def _available_devices():
    """Return the default devices, capped by the count
    ``MBIRTORCH_NUM_DEVICES`` pins."""
    pool = [_canonical_device(d) for d in default_devices()]
    pinned = pinned_device_count()
    return pool if pinned is None else pool[:pinned]


def _device_pool(devices):
    """Return the pool of :func:`resolve_device_pool`, before the check for a
    repeated GPU."""
    if devices is None:
        return _available_devices()
    if isinstance(devices, torch.device):
        return [_canonical_device(devices)]
    if isinstance(devices, bool):
        raise ValueError(f'a device pool cannot be a boolean; got {devices!r}.')
    if isinstance(devices, str):
        platform = devices.lower()
        if platform == 'cpu':
            return list(cpu_devices())
        if platform == 'gpu':
            pool = [_canonical_device(d) for d in gpu_devices()]
            if not pool:
                raise ValueError("a 'gpu' device pool was requested but no GPU is available.")
            return pool
        raise ValueError(f"a device pool string must be 'cpu' or 'gpu'; got {devices!r}.")
    if isinstance(devices, (int, np.integer)):
        pool = _available_devices()
        count = int(devices)
        if not 1 <= count <= len(pool):
            pinned = pinned_device_count()
            pin_note = ('' if pinned is None else
                        f'  {DEVICE_COUNT_ENV_VAR} pins the count to {pinned}.')
            raise ValueError(f'{count} device(s) were requested, but {len(pool)} '
                             f'are available.{pin_note}')
        return pool[:count]
    devices = list(devices)
    if not devices:
        raise ValueError('a device pool must hold at least one device.')
    if any(isinstance(d, bool) for d in devices):
        raise ValueError(f'a device pool cannot hold a boolean; got {devices!r}.')
    if all(isinstance(d, (int, np.integer)) for d in devices):
        pool = [_canonical_device(d) for d in default_devices()]
        for index in devices:
            if not 0 <= int(index) < len(pool):
                raise ValueError(f'device index {int(index)} is out of range; {len(pool)} device(s) are available.')
        return [pool[int(index)] for index in devices]
    return [_canonical_device(d) for d in devices]


class Task:
    """
    One unit of an agent's work.

    Args:
        run (callable): ``run(device)`` computes the agent's output over the
            task's region on ``device`` and returns it on the device of the
            agent's input.
        device (torch.device, optional): the device the task must run on, or
            None for any device of the loop's pool.
        region (tuple of slice, optional): the part of the array the output
            covers, or None for the whole array.  Each entry is a slice or an
            integer, so that indexing the state with the region gives a view
            and the fold writes into the state.

    Raises:
        ValueError: if an entry of ``region`` is neither a slice nor an
            integer.
    """

    def __init__(self, run, device=None, region=None):
        self.run = run
        self.device = None if device is None else _canonical_device(device)
        if region is not None:
            region = tuple(region)
            for entry in region:
                if not isinstance(entry, (slice, int, np.integer)):
                    raise ValueError('a task region holds slices and integers only, so that it '
                                     f'indexes a view of the state; got {type(entry).__name__}.')
        self.region = region


def _tasks_of(agent, w, iteration):
    """Return the agent's own tasks, or one task over the whole array on the
    agent's device when it provides none."""
    if hasattr(agent, 'tasks'):
        return list(agent.tasks(w, iteration))
    device = getattr(agent, 'device', None)
    return [Task(lambda _device, agent=agent, w=w: agent(w, iteration), device=device, region=None)]


def _chunk_count(tensor):
    """Return the number of chunks to reduce a tensor in, so that the
    temporaries of the reduction stay near a fixed size."""
    n_bytes = tensor.numel() * tensor.element_size()
    return min(ELL1_MAX_CHUNKS, max(1, round(n_bytes / ELL1_CHUNK_BYTES)))


def _norms_of_change(new, previous):
    """Return ``(||new - previous||, ||previous||)``.  The sums are reduced a
    chunk at a time, so no full-size temporary is allocated."""
    new_flat = new.reshape(-1)
    previous_flat = previous.reshape(-1)
    n_chunks = _chunk_count(new_flat)
    diff_sq = 0.0
    prev_sq = 0.0
    for a, b in zip(torch.chunk(new_flat, n_chunks), torch.chunk(previous_flat, n_chunks)):
        diff_sq += float(torch.sum((a - b) ** 2))
        prev_sq += float(torch.sum(b * b))
    return math.sqrt(diff_sq), math.sqrt(prev_sq)


def _sum_of_squared_difference(x, reference):
    """Return ``sum((x - reference)^2)``, reduced a chunk at a time along the
    first axis.  ``reference`` may be a non-contiguous view, so the chunks
    are cut along an axis rather than over a flattened copy."""
    n_chunks = min(_chunk_count(x), max(1, int(x.shape[0]))) if x.ndim > 0 else 1
    total = 0.0
    for a, b in zip(torch.chunk(x, n_chunks, dim=0), torch.chunk(reference, n_chunks, dim=0)):
        total += float(torch.sum((a - b) ** 2))
    return total


class _Worker(threading.Thread):
    """One thread bound to one device, running the jobs it is handed."""

    def __init__(self, index, device):
        super().__init__(name=f'mace-worker-{index}', daemon=True)
        self.index = index
        self.device = device
        self._inbox = queue.Queue()

    def submit(self, job):
        self._inbox.put(job)

    def run(self):
        while True:
            job = self._inbox.get()
            if job is None:
                return
            job()


class MACE:
    """
    The MACE consensus iteration over a list of agents.

    The loop owns the state: one input ``W_k`` per agent and the consensus
    average, as tensors of the shape of ``x0`` on the device of ``x0``.
    :meth:`step` runs one iteration and :meth:`run` loops over it.  ``mu``
    and ``rho`` are plain attributes read at each step, so a caller may
    change them between steps.  :meth:`state_dict` and
    :meth:`load_state_dict` save and restore the state, so a long run can be
    resumed.  Each agent's output is added into the consensus as it arrives,
    so the loop holds three full-size arrays beyond the agent inputs,
    whatever the number of agents.

    Args:
        agents (list): the agents.  Each is a callable ``agent(w, iteration)``
            that returns a tensor of the shape of ``w`` on the device of
            ``w``.  An agent may provide ``tasks(w, iteration)``, a list of
            :class:`Task`, to split its work over regions and devices.  An
            agent may set ``fold_after_all = True`` and provide ``pieces()``,
            which yields ``(region, output)`` pairs once every task has run;
            its tasks then return nothing and the pieces are added instead.
            An agent may provide ``state_dict`` and ``load_state_dict`` for
            its own state.
        x0 (numpy or tensor): the initial array.  Every agent input starts
            here, and the state lives on its device.
        mu (list of float, optional): the agent weights, which should sum to
            one.  Defaults to equal weights.
        rho (float, optional): the step of the Mann iteration, in (0, 1).
            Defaults to 0.5.
        devices (optional): None runs every task inline, in order.  Otherwise
            a device pool in any form :func:`resolve_device_pool` accepts, with
            one worker thread per entry.  A GPU may not be named twice.  Each worker runs the tasks fixed to
            its device and then takes tasks that accept any device from a
            shared queue.  The workers start on the first step and stop in
            :meth:`close`.  A task with a fixed device must name a device of
            the pool.

    Attributes:
        W (list of torch.Tensor): the agent inputs.
        x_bar (torch.Tensor): the consensus average after the last step.  It
            is the loop's own buffer and the next step overwrites it, so a
            caller who keeps a copy clones it.
        iteration (int): steps completed.
        info (dict): per-iteration traces, one list entry per step.

            * ``change_pct``: the percent change of the average.
            * ``spread``: one value per agent, the norm of the agent's output
              minus the previous average, over the norm of that average.
            * ``time``: the wall time of the step in seconds.
            * ``tasks``: one row per task, ``(agent_index, task_index,
              device_index, start, end)``, with times in seconds from the
              start of the step.  In the inline mode the device index is 0.
    """

    def __init__(self, agents, x0, mu=None, rho=0.5, devices=None):
        self.agents = list(agents)
        num_agents = len(self.agents)
        if num_agents == 0:
            raise ValueError('MACE needs at least one agent.')
        if mu is None:
            mu = [1.0 / num_agents] * num_agents
        if len(mu) != num_agents:
            raise ValueError(f'mu must have one weight per agent; got {len(mu)} weights for {num_agents} agents.')
        self.mu = [float(m) for m in mu]
        self.rho = float(rho)

        if not torch.is_tensor(x0):
            x0 = torch.as_tensor(np.asarray(x0, dtype=np.float32))
        x0 = x0.detach().to(torch.float32)
        self._device = x0.device
        self.W = [x0.clone().contiguous() for _ in self.agents]
        self._x_bar = x0.clone().contiguous()
        self._x_bar_accum = torch.zeros_like(self._x_bar)
        self._z = torch.zeros_like(self._x_bar)
        self.iteration = 0
        self.info = {'change_pct': [], 'spread': [], 'time': [], 'tasks': []}

        self._pool = None if devices is None else resolve_device_pool(devices)
        self._workers = None
        self._lock = threading.Lock()
        # This is set when a step raised partway.  Some inputs then hold a partial
        # update, and the loop refuses further steps until a state is loaded.
        self._inconsistent = False

    @property
    def x_bar(self):
        return self._x_bar

    @property
    def devices(self):
        """The device pool, or None when tasks run inline."""
        return None if self._pool is None else list(self._pool)

    def _start_workers(self):
        self._workers = [_Worker(index, device) for index, device in enumerate(self._pool)]
        for worker in self._workers:
            worker.start()

    def close(self):
        """Stop the worker threads.  The state stays readable."""
        if self._workers is not None:
            for worker in self._workers:
                worker.submit(None)
            for worker in self._workers:
                worker.join()
            self._workers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def step(self):
        """
        Run one iteration and return the percent change of the average.

        Raises:
            RuntimeError: if an earlier step raised, since some agent inputs
                then hold a partial update.  :meth:`load_state_dict` restores
                a consistent state.
        """
        if self._inconsistent:
            raise RuntimeError('an earlier step raised partway through its update, so the state '
                               'is inconsistent; load a saved state before stepping again.')
        mu = list(self.mu)
        rho = self.rho
        iteration = self.iteration
        step_start = time.perf_counter()
        with torch.no_grad():
            jobs = []
            remaining = {}
            for k, agent in enumerate(self.agents):
                tasks = _tasks_of(agent, self.W[k], iteration)
                remaining[k] = len(tasks)
                jobs.extend((k, t, task) for t, task in enumerate(tasks))
            spread_sums = [0.0] * len(self.agents)
            rows = []
            context = dict(mu=mu, rho=rho, remaining=remaining, spread=spread_sums,
                           rows=rows, start=step_start)
            try:
                if self._pool is None:
                    for job in jobs:
                        device = job[2].device if job[2].device is not None else self._device
                        self._execute(job, device, 0, context)
                else:
                    self._run_threaded(jobs, context)
            except BaseException:
                self._inconsistent = True
                self._z.zero_()
                self._x_bar_accum.zero_()
                raise

            for k in range(len(self.agents)):
                self.W[k].add_(self._z, alpha=2.0 * rho)
            change_norm, previous_norm = _norms_of_change(self._x_bar_accum, self._x_bar)
            change = change_norm / previous_norm if previous_norm > 0 else math.inf
            spread = [math.sqrt(s) / previous_norm if previous_norm > 0 else math.inf
                      for s in spread_sums]
            # The average is copied into its buffer rather than swapped, so
            # that x_bar keeps one storage for the life of the loop.
            self._x_bar.copy_(self._x_bar_accum)
            self._x_bar_accum.zero_()
            self._z.zero_()

        self.iteration += 1
        change_pct = 100.0 * change
        self.info['change_pct'].append(change_pct)
        self.info['spread'].append(spread)
        self.info['time'].append(time.perf_counter() - step_start)
        self.info['tasks'].append(sorted(rows, key=lambda row: row[3]))
        return change_pct

    def _run_threaded(self, jobs, context):
        if self._workers is None:
            self._start_workers()
        shared = queue.Queue()
        fixed = {worker.index: [] for worker in self._workers}
        by_device = {}
        for worker in self._workers:
            by_device.setdefault(worker.device, []).append(worker.index)
        next_on_device = {device: 0 for device in by_device}
        for job in jobs:
            device = job[2].device
            if device is None:
                shared.put(job)
                continue
            candidates = by_device.get(device)
            if not candidates:
                raise ValueError(f'a task asks for device {device}, which is not in the pool {self._pool}.')
            # Workers that share a device take that device's fixed tasks in
            # turn.
            index = candidates[next_on_device[device] % len(candidates)]
            next_on_device[device] += 1
            fixed[index].append(job)

        errors = []
        done = threading.Event()
        pending = [len(self._workers)]

        def work_for(worker):
            def work():
                try:
                    for item in fixed[worker.index]:
                        self._execute(item, worker.device, worker.index, context)
                    while True:
                        try:
                            item = shared.get_nowait()
                        except queue.Empty:
                            break
                        self._execute(item, worker.device, worker.index, context)
                except BaseException as error:   # noqa: BLE001 - re-raised on the main thread
                    errors.append(error)
                finally:
                    with self._lock:
                        pending[0] -= 1
                        if pending[0] == 0:
                            done.set()
            return work

        for worker in self._workers:
            worker.submit(work_for(worker))
        done.wait()
        if errors:
            raise errors[0]

    def _execute(self, job, device, device_index, context):
        k, task_index, task = job
        agent = self.agents[k]
        start = time.perf_counter() - context['start']
        output = task.run(device)
        if getattr(agent, 'fold_after_all', False):
            with self._lock:
                context['remaining'][k] -= 1
                last = context['remaining'][k] == 0
            if last:
                for region, piece in agent.pieces():
                    self._fold(k, region, piece, context)
        else:
            self._fold(k, task.region, output, context)
        end = time.perf_counter() - context['start']
        with self._lock:
            context['rows'].append((k, task_index, device_index, start, end))

    def _fold(self, k, region, x, context):
        """Add one output region into the consensus, in place, under the
        lock."""
        index = (Ellipsis,) if region is None else tuple(region)
        x = x.detach().to(device=self._device, dtype=self._x_bar.dtype)
        mu_k = context['mu'][k]
        rho = context['rho']
        # The previous average is not written during a step, so the spread
        # term is reduced outside the lock.
        spread = _sum_of_squared_difference(x, self._x_bar[index])
        with self._lock:
            w = self.W[k][index]
            # z reads the agent's input before the input is overwritten below.
            self._z[index].add_(x, alpha=2.0 * mu_k).sub_(w, alpha=mu_k)
            self._x_bar_accum[index].add_(x, alpha=mu_k)
            w.sub_(x, alpha=2.0 * rho)
            context['spread'][k] += spread

    def run(self, max_iterations=30, stop_threshold_change_pct=0.0, callback=None):
        """
        Step until the percent change of the average falls below the
        threshold or ``max_iterations`` steps have run in total.

        Args:
            max_iterations (int, optional): the number of steps to reach,
                counted from the first step of this object or of the state
                it was loaded from.  Defaults to 30.
            stop_threshold_change_pct (float, optional): stop when the percent
                change of the average falls below this.  0 runs every step.
                Defaults to 0.0.
            callback (callable, optional): ``callback(iteration, x_bar)`` after
                each step, with the index of the step just completed.  The
                tensor is the loop's buffer, which the next step overwrites;
                a callback that keeps it clones it.

        Returns:
            (x_bar, info): the consensus average, which is the loop's buffer,
            and the traces.
        """
        while self.iteration < max_iterations:
            change_pct = self.step()
            if callback is not None:
                callback(self.iteration - 1, self._x_bar)
            if change_pct < stop_threshold_change_pct:
                break
        return self._x_bar, self.info

    def state_dict(self):
        """
        The state needed to resume: the agent inputs, the consensus average,
        the iteration count, the traces, and each agent's own state where the
        agent provides ``state_dict``.  The weights and ``rho`` are recorded
        for reference and are not restored by :meth:`load_state_dict`, since
        they belong to the caller.  The accumulation buffer and ``z`` are zero
        between steps and are not saved.

        Raises:
            RuntimeError: if an earlier step raised partway, since the state is
                then not worth saving.
        """
        if self._inconsistent:
            raise RuntimeError('an earlier step raised partway through its update, so the state '
                               'is inconsistent and is not saved.')
        return {
            'W': [w.clone() for w in self.W],
            'x_bar': self._x_bar.clone(),
            'iteration': self.iteration,
            'info': copy.deepcopy(self.info),
            'mu': list(self.mu),
            'rho': self.rho,
            'agents': [agent.state_dict() if hasattr(agent, 'state_dict') else None
                       for agent in self.agents],
        }

    def load_state_dict(self, state):
        """Restore what :meth:`state_dict` saved into this object, whose agents
        and array shape must match.  The loop keeps its own ``mu`` and
        ``rho``; when they differ from the saved values, a warning names both.
        """
        if len(state['W']) != len(self.W):
            raise ValueError(f"the state holds {len(state['W'])} agent inputs; this loop has {len(self.W)} agents.")
        for w, saved in zip(self.W, state['W']):
            if tuple(saved.shape) != tuple(w.shape):
                raise ValueError(f'the state has shape {tuple(saved.shape)}; this loop has {tuple(w.shape)}.')
            w.copy_(saved)
        self._x_bar.copy_(state['x_bar'])
        self._x_bar_accum.zero_()
        self._z.zero_()
        self.iteration = int(state['iteration'])
        self.info = copy.deepcopy(state['info'])
        for agent, saved in zip(self.agents, state['agents']):
            if saved is not None and hasattr(agent, 'load_state_dict'):
                agent.load_state_dict(saved)
        self._inconsistent = False
        saved_mu = [float(m) for m in state.get('mu', self.mu)]
        saved_rho = float(state.get('rho', self.rho))
        if saved_mu != self.mu or saved_rho != self.rho:
            warnings.warn(
                f'the loaded state was saved with mu={saved_mu} and rho={saved_rho}; this loop '
                f'keeps its own mu={self.mu} and rho={self.rho}.  Set them before stepping if '
                'the saved values were meant.')


def mace(agents, x0, mu=None, rho=0.5, num_iterations=30, callback=None):
    """
    Run the MACE iteration for a fixed number of steps and return the
    consensus average.

    This is :class:`MACE` constructed and run in one call, with every task
    inline.  Besides the loop's traces, ``info`` carries two more.
    ``consensus_change`` is the norm of the change of the average over the
    norm of the previous average, as a fraction.  ``consensus_spread`` is the
    largest of the agents' spreads at each step, each the norm of the agent's
    output minus the previous average over the norm of that average.

    Args:
        agents (list): the agents, as for :class:`MACE`.
        x0 (numpy or tensor): the initial array.
        mu (list of float, optional): the agent weights.  Defaults to equal.
        rho (float, optional): the Mann step.  Defaults to 0.5.
        num_iterations (int, optional): steps to run.  Defaults to 30.
        callback (callable, optional): ``callback(iteration, x_bar)`` after
            each step.

    Returns:
        (x_bar, info)
    """
    with MACE(agents, x0, mu=mu, rho=rho) as loop:
        x_bar, info = loop.run(max_iterations=num_iterations, stop_threshold_change_pct=0.0,
                               callback=callback)
    info['consensus_change'] = [change / 100.0 for change in info['change_pct']]
    info['consensus_spread'] = [max(spread) for spread in info['spread']]
    return x_bar, info


def _reject_divided_output(output, agent_name):
    """Raise when an agent output is divided across devices.  An agent must
    return one tensor, which the loop can fold."""
    if isinstance(output, _sharding.Shards):
        raise ValueError(f'{agent_name} needs a model configured on one device; this model is '
                         f'configured on {output.placement.n_devices} and returned a divided array.')


class ForwardProxAgent:
    """
    The proximal map of the tomographic data-fit term, through
    :meth:`TomographyModel.prox_map`.

    The sinogram, the weights, and ``sigma_prox`` are bound at construction.
    Each call runs ``inner_iterations`` VCD iterations with a stop threshold
    of zero, starting at entry ``floor(iteration * partition_advance)`` of
    the model's partition sequence, so the partitions go from coarse to fine
    across the loop.  The model's prox initialization runs at iteration 0
    only.  Use one agent per model, and a model on one device.

    With a seed, every random draw the agent makes comes from a generator
    named by the seed and what the draw is for, so a run makes the same draws
    whatever thread runs it and in whatever order.  Without one the draws come
    from the global np.random state.

    Args:
        model (TomographyModel): the projection model, configured on one
            device.
        sinogram (numpy or tensor): the measured sinogram.
        weights (optional): sinogram weights, as for ``prox_map``.
        sigma_prox (float, optional): the proximal strength.  None uses the
            model's automatic value.
        inner_iterations (int, optional): VCD iterations per call.  Defaults
            to 3.
        init_recon (optional): the starting volume of the first call.  None
            lets ``prox_map`` build its own.
        device (optional): a device to pin the model to.  The sinogram and
            the weights are then placed on it once.  None leaves the model's
            device layout as it is.
        partition_advance (float, optional): entries of the partition sequence
            the agent moves forward per loop iteration; it may be below one.
            Defaults to 1.0.
        use_warm_start (bool, optional): start each call from the agent's own
            previous output when True, and from the input when False.  The
            warm start keeps one volume and saves inner iterations.  Defaults
            to True.
        seed (int, optional): the seed every draw of this agent is derived
            from.  None draws from the global np.random state.
    """

    def __init__(self, model, sinogram, weights=None, sigma_prox=None, inner_iterations=3,
                 init_recon=None, device=None, partition_advance=1.0, use_warm_start=True,
                 seed=None):
        self.model = model
        self.device = None if device is None else torch.device(device)
        if self.device is not None:
            model.configure_devices(devices=[self.device])
            placed = model.prepare_sino_for_devices(sinogram, weights)
            if weights is None:
                sinogram = placed
            else:
                sinogram, weights = placed
        self.sinogram = sinogram
        self.weights = weights
        self.sigma_prox = sigma_prox
        self.inner_iterations = int(inner_iterations)
        self.partition_advance = float(partition_advance)
        self.use_warm_start = bool(use_warm_start)
        self.seed = None if seed is None else int(seed)
        self._previous_output = init_recon

    def rng_for(self, name):
        """Return the generator of one named draw of this agent, or None when
        the agent has no seed."""
        return named_rng(self.seed, name)

    def __call__(self, w, iteration=0):
        first = int(math.floor(iteration * self.partition_advance))
        if self.use_warm_start:
            init_recon = self._previous_output
        else:
            init_recon = w
        do_initialization = iteration == 0
        if self.seed is not None:
            # The initialization draws its partitions here rather than inside
            # the sweep, so that they do not depend on which worker runs the
            # first iteration.
            do_initialization = False
            if self.model.prox_data is None:
                self.model.initialize_prox(
                    self.sinogram, weights=self.weights, init_recon=init_recon,
                    max_iterations=first + self.inner_iterations,
                    first_iteration=first, logfile_path=None, print_logs=False,
                    rng=self.rng_for('init'))
        output, _ = self.model.prox_map(
            w, self.sinogram, sigma_prox=self.sigma_prox, weights=self.weights,
            init_recon=init_recon, do_initialization=do_initialization,
            max_iterations=first + self.inner_iterations, first_iteration=first,
            stop_threshold_change_pct=0.0, logfile_path=None, print_logs=False,
            output_sharded=True, rng=self.rng_for(iteration))
        _reject_divided_output(output, 'ForwardProxAgent')
        if self.use_warm_start:
            self._previous_output = output
        return output.to(w.device)

    def state_dict(self):
        return {'previous_output': None if not self.use_warm_start or self._previous_output is None
                else torch.as_tensor(self._previous_output).clone(),
                'seed': self.seed}

    def load_state_dict(self, state):
        if state.get('previous_output') is not None:
            self._previous_output = state['previous_output']
        # The 4D data-fit agent installs a warm start with a dict that holds
        # that key alone, so the seed is restored only when it is there.
        if 'seed' in state:
            self.seed = state['seed']


class QGGMRFDenoiserAgent:
    """
    The qGGMRF prior on one volume, through :meth:`QGGMRFDenoiser.denoise`.

    The prior parameters are pinned at construction and auto-regularization
    is turned off, so the agent is the same operator on every call and
    ``sigma_noise`` is its one strength.  The pixel partition is settled on
    the first call and reused, which makes the agent a fixed operator down to
    its pixel grouping; with a seed that partition is drawn from a generator
    named by the seed rather than from the global state, so it does not depend
    on which worker makes the first call.

    Args:
        image_shape (tuple of int): the volume shape.
        sigma_noise (float): the denoising strength.
        pinned_params (dict, optional): prior parameters to fix, such as
            ``{'sigma_x': value}``.
        inner_iterations (int, optional): VCD iterations per call.  Defaults
            to 8.
        like_model (TomographyModel, optional): a model whose device layout
            the denoiser copies, so volumes pass between them on the devices.
        use_ror_mask (bool, optional): restrict the update to the inscribed
            ellipse.  Defaults to False.
        device (optional): a device to pin the denoiser to.  Takes precedence
            over ``like_model``.
        use_warm_start (bool, optional): start each call from the agent's own
            previous output when True, and from the input when False.
            Defaults to True.
        seed (int, optional): the seed the pixel partition is derived from.
            None draws it from the global np.random state.
    """

    def __init__(self, image_shape, sigma_noise, pinned_params=None, inner_iterations=8,
                 like_model=None, use_ror_mask=False, device=None, use_warm_start=True,
                 seed=None):
        self.model = QGGMRFDenoiser(tuple(int(n) for n in image_shape))
        self.device = None if device is None else torch.device(device)
        if self.device is not None:
            self.model.configure_devices(devices=[self.device])
        elif like_model is not None:
            self.model.configure_devices(like=like_model)
        self.model.set_params(no_warning=True, verbose=0)
        if pinned_params:
            self.model.set_params(no_warning=True, **pinned_params)
        self.model.set_params(no_warning=True, auto_regularize_flag=False)
        # The mask is set here, so that the partition settled on the first
        # call is drawn over the pixels every call updates.
        self.model.set_params(no_warning=True, use_ror_mask=use_ror_mask)
        self.sigma_noise = float(sigma_noise)
        self.inner_iterations = int(inner_iterations)
        self.use_ror_mask = use_ror_mask
        self.use_warm_start = bool(use_warm_start)
        self.seed = None if seed is None else int(seed)
        self._previous_output = None

    def __call__(self, w, iteration=0):
        init_image = self._previous_output if self.use_warm_start else None
        if self.model.denoise_data is None:
            self.model.initialize_denoiser(image=w, sigma_noise=self.sigma_noise,
                                           rng=named_rng(self.seed, 'denoiser'))
        output, _ = self.model.denoise(
            w, sigma_noise=self.sigma_noise, use_ror_mask=self.use_ror_mask,
            init_image=init_image, max_iterations=self.inner_iterations,
            stop_threshold_change_pct=0.0, logfile_path=None, print_logs=False,
            output_sharded=True, do_initialization=False)
        _reject_divided_output(output, 'QGGMRFDenoiserAgent')
        if self.use_warm_start:
            self._previous_output = output
        return output.to(w.device)

    def state_dict(self):
        return {'previous_output': None if self._previous_output is None
                else self._previous_output.clone()}

    def load_state_dict(self, state):
        if state.get('previous_output') is not None:
            self._previous_output = state['previous_output']


class HyperplaneAgent:
    """
    A prior on the hyperplane volumes of a 4D array.

    The array has shape ``(frames, x, y, z)``.  Fixing one spatial axis at
    one index gives a 3D volume that holds every frame, a hyperplane volume.
    The agent denoises the stack of all such volumes with a stack denoiser,
    so the prior couples neighboring frames as well as neighboring voxels.
    The work is split into tasks of ``batch_size`` hyperplanes, and any
    device of the loop's pool may take a task.  When a frame-axis filter is
    given, each stack is filtered before it is denoised.

    Args:
        axis (int): the spatial axis that is fixed: 1, 2, or 3 in the
            ``(frames, x, y, z)`` order.
        make_stack_denoiser (callable): ``make_stack_denoiser(device)`` returns
            the stack denoiser for that device, a callable ``denoise(stack)``,
            or ``denoise(stack, init_stack=...)`` when the warm start is on.
            The stack is a tensor of shape ``(hyperplanes, frames, d1, d2)``
            on that device, and the result has the same shape.  The stack
            and the initial stack are copies the agent made and does not
            read again, so the denoiser may write them in place.  One stack
            denoiser is made per worker thread and kept.
        batch_size (int, optional): hyperplanes per task.  None puts the whole
            orientation in one task, which copies the whole array to one
            device.  Pass the number of hyperplanes whose volumes fit the
            memory one task may use, so that a task holds one slab and the
            tasks spread over the pool.
        filter_matrix (torch.Tensor, optional): a square matrix whose size is
            the number of frames, applied along the frame axis of each batch
            before denoising, such as the matrix from
            :func:`mbirtorch.mace4d.temporal_filter_matrix`.
        use_warm_start (bool, optional): keep the previous call's output, one
            array of the input's shape on the input's device, and pass the
            matching slab to the stack denoiser as ``init_stack``.  The first
            call passes no ``init_stack``.  Defaults to False.
    """

    fold_after_all = False

    def __init__(self, axis, make_stack_denoiser, batch_size=None, filter_matrix=None,
                 use_warm_start=False):
        axis = int(axis)
        if axis not in (1, 2, 3):
            raise ValueError(f'axis must be 1, 2, or 3, a spatial axis of a (frames, x, y, z) array; got {axis}.')
        self.axis = axis
        self.make_stack_denoiser = make_stack_denoiser
        self.batch_size = None if batch_size is None else int(batch_size)
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f'batch_size must be at least 1; got {batch_size}.')
        self.filter_matrix = filter_matrix
        self.use_warm_start = bool(use_warm_start)
        # These hold the previous call's output and whether a whole call has completed
        # into it.  The flag turns on when the last task of a call finishes.
        self._previous_output = None
        self._have_previous = False
        self._pending_tasks = 0
        self._denoisers = {}
        self._lock = threading.Lock()

    def _denoiser_for(self, device):
        """Return the stack denoiser of the calling worker on ``device``.  It
        is made on first use."""
        key = (threading.get_ident(), torch.device(device))
        with self._lock:
            denoiser = self._denoisers.get(key)
            if denoiser is None:
                denoiser = self.make_stack_denoiser(torch.device(device))
                self._denoisers[key] = denoiser
        return denoiser

    def tasks(self, w, iteration=0):
        num_planes = int(w.shape[self.axis])
        batch = num_planes if self.batch_size is None else self.batch_size
        tasks = []
        for start in range(0, num_planes, batch):
            stop = min(start + batch, num_planes)
            region = tuple(slice(start, stop) if d == self.axis else slice(None) for d in range(w.ndim))
            tasks.append(Task(functools.partial(self._run_slab, w, region), device=None, region=region))
        if self.use_warm_start:
            with self._lock:
                if self._previous_output is None:
                    self._previous_output = torch.zeros_like(w)
                self._pending_tasks = len(tasks)
        return tasks

    def __call__(self, w, iteration=0):
        """The whole array in one call, task by task on the input's device."""
        output = torch.empty_like(w)
        for task in self.tasks(w, iteration):
            output[task.region] = task.run(w.device)
        return output

    def _slab_copy(self, source, region, device):
        """Return the slab of ``source`` over ``region`` as a new contiguous
        tensor on ``device``, with the hyperplane axis first.

        The result is always a copy, because the stack denoiser may write
        what it is handed and ``source`` must not be written."""
        slab = source[region].to(device).movedim(self.axis, 0)
        return slab.contiguous() if not slab.is_contiguous() else slab.clone()

    def _run_slab(self, w, region, device):
        device = torch.device(device)
        stack = self._slab_copy(w, region, device)
        if self.filter_matrix is not None:
            # The filter returns a permuted view.  It is made contiguous so that the
            # reshape to (volumes, pixels, slices) is a view rather than a copy.
            stack = _filter_along_axis(stack, self.filter_matrix, axis=1).contiguous()
        denoiser = self._denoiser_for(device)
        if self.use_warm_start and self._have_previous:
            init_stack = self._slab_copy(self._previous_output, region, device)
            denoised = denoiser(stack, init_stack=init_stack)
        else:
            denoised = denoiser(stack)
        result = denoised.movedim(0, self.axis).to(w.device)
        if self.use_warm_start:
            # Tasks of one call write disjoint slabs, so this write needs no
            # lock.
            self._previous_output[region] = result
            with self._lock:
                self._pending_tasks -= 1
                if self._pending_tasks == 0:
                    self._have_previous = True
        return result

    def state_dict(self):
        return {'previous_output': None if not (self.use_warm_start and self._previous_output is not None)
                else self._previous_output.clone()}

    def load_state_dict(self, state):
        if state.get('previous_output') is not None:
            self._previous_output = state['previous_output'].clone()
            self._have_previous = True
