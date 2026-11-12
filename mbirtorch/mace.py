"""This module implements multi-agent consensus equilibrium (MACE) and several MACE agents.

MACE computes one array that balances several agents.  An agent is a callable
``agent(w, iteration)`` that maps a tensor to a tensor of the same shape.  For example, an
agent can be a data-fit proximal map or a denoiser.  Let X_k be the output of agent k on its
input W_k.  Each MACE step computes ``z = sum_k mu_k (2 X_k - W_k)``.  It then updates each
input by ``W_k += 2 rho (z - X_k)``.  It also computes the average ``x_bar = sum_k mu_k X_k``.
The class :class:`MACE` runs the iteration, and the function :func:`mace` runs it in one call.
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
from ._memory_estimate import (DEVICE_COUNT_ENV_VAR, L1_CHUNK_BYTES, L1_MAX_CHUNKS,
                               env_device_count)
from .denoising import QGGMRFDenoiser
from .tomography_model import cpu_devices, default_devices, gpu_devices
from .vcd_utils import named_rng


def _canonical_device(device):
    """This function returns a ``torch.device`` with an explicit index.  As a result, ``'cuda'``
    and ``'cuda:0'`` give the same device."""
    device = torch.device(device)
    if device.type in ('cuda', 'mps', 'xpu') and device.index is None:
        return torch.device(device.type, 0)
    return device


def _filter_along_axis(x, matrix, axis):
    """This function applies a square matrix along one axis of a tensor.  It computes on the
    device and in the dtype of the tensor."""
    matrix = matrix.to(device=x.device, dtype=x.dtype)
    moved = x.movedim(axis, 0)
    filtered = torch.tensordot(matrix, moved, dims=1)
    return filtered.movedim(0, axis)


def resolve_device_pool(devices=None):
    """
    This function converts a description of a set of devices into a list of
    ``torch.device``.  If ``devices`` is None or an integer, the function uses at most
    ``MBIRTORCH_NUM_DEVICES`` devices.  If ``devices`` is a sequence, the function does not
    apply this limit.

    Args:
        devices: One of the following values.

            * None: every GPU, or the CPU when there is no GPU.
            * ``'cpu'``: the one CPU device.
            * ``'gpu'``: every GPU.  The function raises an error when there is no GPU.
            * An integer ``n``: the first ``n`` default devices.  The function raises an
              error when ``n`` is larger than the number of default devices.
            * A sequence of integers: the default devices with those indices.
            * A sequence of devices (``torch.device`` or device strings):
              those devices, in that order.  The CPU may appear more than once,
              and each entry gets one worker thread.  A GPU may not appear more
              than once.

    Returns:
        list of torch.device: The devices.

    Raises:
        ValueError: If a string is not ``'cpu'`` or ``'gpu'``, if GPUs are
            requested when there are none, if a count or index is beyond the
            available devices, or if a GPU appears more than once.
    """
    return _reject_repeated_gpu(_device_pool(devices))


def _reject_repeated_gpu(pool):
    """This function returns ``pool`` unchanged.  It raises an error if a GPU appears in
    ``pool`` more than once.  A CPU may appear more than once.
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
    """This function returns the default devices.  It returns at most the number of devices
    that ``MBIRTORCH_NUM_DEVICES`` sets."""
    pool = [_canonical_device(d) for d in default_devices()]
    env_count = env_device_count()
    return pool if env_count is None else pool[:env_count]


def _device_pool(devices):
    """This function returns the same devices as :func:`resolve_device_pool`.  It does not
    check for a GPU that appears more than once."""
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
            env_count = env_device_count()
            env_note = ('' if env_count is None else
                        f'  {DEVICE_COUNT_ENV_VAR} sets the count to {env_count}.')
            raise ValueError(f'{count} device(s) were requested, but {len(pool)} '
                             f'are available.{env_note}')
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
    This class is used to hold one unit of an agent's work.

    Args:
        run (callable): The call ``run(device)`` computes the agent's output
            over the region of the task on ``device``.  It returns the output on
            the device of the agent's input.
        device (torch.device, optional): The device that the task must run on.
            If None, the task may run on any device of the MACE object.
        region (tuple of slice, optional): The part of the array that the
            output covers.  If None, the output covers the whole array.  Each
            entry is a slice or an integer.

    Raises:
        ValueError: If an entry of ``region`` is neither a slice nor an
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
    """This function returns the tasks of an agent.  If the agent does not provide tasks, the
    function returns one task that covers the whole array on the agent's device."""
    if hasattr(agent, 'tasks'):
        return list(agent.tasks(w, iteration))
    device = getattr(agent, 'device', None)
    return [Task(lambda _device, agent=agent, w=w: agent(w, iteration), device=device, region=None)]


def _chunk_count(tensor):
    """This function returns the number of chunks to use for a sum over a tensor.  The
    chunks keep the temporary arrays near a fixed size."""
    n_bytes = tensor.numel() * tensor.element_size()
    return min(L1_MAX_CHUNKS, max(1, round(n_bytes / L1_CHUNK_BYTES)))


def _norms_of_change(new, previous):
    """This function returns the L2 norms ``(||new - previous||, ||previous||)``.  It computes
    the sums one chunk at a time."""
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
    """This function returns ``sum((x - reference)^2)``.  It computes the sum one chunk at a
    time along the first axis."""
    n_chunks = min(_chunk_count(x), max(1, int(x.shape[0]))) if x.ndim > 0 else 1
    total = 0.0
    for a, b in zip(torch.chunk(x, n_chunks, dim=0), torch.chunk(reference, n_chunks, dim=0)):
        total += float(torch.sum((a - b) ** 2))
    return total


class _Worker(threading.Thread):
    """This class is used to run jobs on one thread that is assigned to one device."""

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
    This class is used to run the MACE iteration over a list of agents.

    The object holds one input ``W_k`` for each agent and the consensus average.  These arrays
    have the shape and the device of ``x0``.  The method :meth:`step` runs one iteration, and
    the method :meth:`run` runs many iterations.  The methods :meth:`state_dict` and
    :meth:`load_state_dict` save and restore the state.  The user may change ``mu`` and ``rho``
    between steps.

    Args:
        agents (list): The agents.  Each agent is a callable ``agent(w, iteration)``
            that returns a tensor with the shape and the device of ``w``.  An
            agent may provide ``tasks(w, iteration)``, which returns a list of
            :class:`Task`.  The tasks split the agent's work over regions and
            devices.  An agent may also set ``fold_after_all = True`` and
            provide ``pieces()``.  In that case, MACE ignores the outputs of the
            tasks.  After all tasks of the agent have run, MACE adds the
            ``(region, output)`` pairs that ``pieces()`` yields.  An agent may
            provide ``state_dict`` and ``load_state_dict`` to save its own state.
        x0 (numpy or tensor): The initial array.  Every agent input starts at
            ``x0``, and the object holds its state on the device of ``x0``.
        mu (list of float, optional): The agent weights, which should sum to
            one.  Defaults to equal weights.
        rho (float, optional): The step size of the Mann iteration, in (0, 1).
            Defaults to 0.5.
        devices (optional): If None, the object runs every task in the calling
            thread, in order.  Otherwise, ``devices`` is any value that
            :func:`resolve_device_pool` accepts, and the object starts one
            worker thread for each device.  A GPU may not appear twice.  A task
            with a fixed device must name one of these devices.  The method
            :meth:`close` stops the worker threads.

    Attributes:
        W (list of torch.Tensor): The agent inputs.
        x_bar (torch.Tensor): The consensus average after the last step.  The
            next step overwrites this tensor, so a caller who keeps it must
            clone it.
        iteration (int): The number of steps completed.
        info (dict): The history of each quantity, with one list entry per step.

            * ``change_pct``: the percent change of the average.
            * ``spread``: one value per agent.  The value is the norm of the
              agent's output minus the previous average, divided by the norm
              of that average.
            * ``time``: the wall time of the step in seconds.
            * ``tasks``: one row per task, ``(agent_index, task_index,
              device_index, start, end)``.  The times are in seconds from the
              start of the step.  When the tasks run in the calling thread, the
              device index is 0.
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
        # This is True after a step raised an error partway.  Then step() raises an error until a
        # state is loaded.
        self._inconsistent = False

    @property
    def x_bar(self):
        return self._x_bar

    @property
    def devices(self):
        """This property returns the list of devices.  It returns None when the tasks run in
        the calling thread."""
        return None if self._pool is None else list(self._pool)

    def _start_workers(self):
        self._workers = [_Worker(index, device) for index, device in enumerate(self._pool)]
        for worker in self._workers:
            worker.start()

    def close(self):
        """This method stops the worker threads.  The state can still be read after this
        method runs."""
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
        This method runs one iteration and returns the percent change of the average.

        Raises:
            RuntimeError: If an earlier step raised an error.  In that case,
                some agent inputs hold a partial update.  The method
                :meth:`load_state_dict` restores a consistent state.
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
            # The copy keeps x_bar in the same memory for the life of the object.
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
        """This method adds the output of one region to the consensus arrays in place."""
        index = (Ellipsis,) if region is None else tuple(region)
        x = x.detach().to(device=self._device, dtype=self._x_bar.dtype)
        mu_k = context['mu'][k]
        rho = context['rho']
        spread = _sum_of_squared_difference(x, self._x_bar[index])
        with self._lock:
            w = self.W[k][index]
            # The update of z must read the agent's input before the next lines change it.
            self._z[index].add_(x, alpha=2.0 * mu_k).sub_(w, alpha=mu_k)
            self._x_bar_accum[index].add_(x, alpha=mu_k)
            w.sub_(x, alpha=2.0 * rho)
            context['spread'][k] += spread

    def run(self, max_iterations=30, stop_threshold_change_pct=0.0, callback=None):
        """
        This method runs steps until the percent change of the average falls
        below the threshold or until ``max_iterations`` steps have run in total.

        Args:
            max_iterations (int, optional): The total number of steps.  The
                count starts at the first step of this object or of the saved
                state that it loaded.  Defaults to 30.
            stop_threshold_change_pct (float, optional): The method stops when
                the percent change of the average falls below this value.  The
                value 0 runs every step.  Defaults to 0.0.
            callback (callable, optional): The method calls
                ``callback(iteration, x_bar)`` after each step.  The next step
                overwrites ``x_bar``, so a callback that keeps it must clone it.

        Returns:
            (x_bar, info): The consensus average and the history dict.
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
        This method returns the state needed to resume the iteration.  The
        state holds the agent inputs, the consensus average, the iteration
        count, and the history.  It also holds the state of each agent that
        provides ``state_dict``.  The state records ``mu`` and ``rho``, but
        :meth:`load_state_dict` does not restore them.

        Raises:
            RuntimeError: If an earlier step raised an error partway.
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
        """This method loads a state that :meth:`state_dict` saved.  The agents and the array
        shape must match.  The object keeps its own ``mu`` and ``rho``.  The method issues a warning
        when they differ from the saved values.
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
    This function runs the MACE iteration for a fixed number of steps and returns the
    consensus average.  It runs every task in the calling thread.  The dict ``info`` holds the
    history from :class:`MACE` and two more lists.  The list ``consensus_change`` holds the
    relative change of the average at each step.  The list ``consensus_spread`` holds the
    largest agent spread at each step.

    Args:
        agents (list): The agents, as for :class:`MACE`.
        x0 (numpy or tensor): The initial array.
        mu (list of float, optional): The agent weights.  Defaults to equal weights.
        rho (float, optional): The step size of the Mann iteration.  Defaults to 0.5.
        num_iterations (int, optional): The number of steps to run.  Defaults to 30.
        callback (callable, optional): The function calls
            ``callback(iteration, x_bar)`` after each step.

    Returns:
        (x_bar, info): The consensus average and the history dict.
    """
    with MACE(agents, x0, mu=mu, rho=rho) as loop:
        x_bar, info = loop.run(max_iterations=num_iterations, stop_threshold_change_pct=0.0,
                               callback=callback)
    info['consensus_change'] = [change / 100.0 for change in info['change_pct']]
    info['consensus_spread'] = [max(spread) for spread in info['spread']]
    return x_bar, info


def _reject_divided_output(output, agent_name):
    """This function raises an error when an agent output is split across devices."""
    if isinstance(output, _sharding.Shards):
        raise ValueError(f'{agent_name} needs a model configured on one device; this model is '
                         f'configured on {output.placement.n_devices} and returned a divided array.')


class ForwardProxAgent:
    """
    This class is used to apply the proximal map of the tomographic data-fit
    term.  It calls :meth:`TomographyModel.prox_map`.

    Each call runs ``inner_iterations`` VCD iterations.  The first of these iterations uses
    entry ``floor(iteration * partition_advance)`` of the model's sequence of partitions.  Each
    agent has its own model on one device.  If a seed is given, the random draws of the agent
    do not depend on which thread runs the agent.

    Args:
        model (TomographyModel): The projection model, configured on one
            device.
        sinogram (numpy or tensor): The measured sinogram.
        weights (optional): The sinogram weights, as for ``prox_map``.
        sigma_prox (float, optional): The proximal strength.  If None, the
            agent uses the model's automatic value.
        inner_iterations (int, optional): The number of VCD iterations per
            call.  Defaults to 3.
        init_recon (optional): The starting volume of the first call.  If None,
            ``prox_map`` computes its own starting volume.
        device (optional): The device for the model.  If it is given, the agent
            copies the sinogram and the weights to this device once.  If None,
            the agent does not change the model's devices.
        partition_advance (float, optional): The number of entries of the
            partition sequence that the agent advances per MACE iteration.  The
            value may be less than one.  Defaults to 1.0.
        use_warm_start (bool, optional): If True, each call starts from the
            agent's previous output.  If False, each call starts from the input.
            Defaults to True.
        seed (int, optional): The seed from which the agent derives every
            random draw.  If None, the agent draws from the global np.random
            state.
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
        """This method returns the random number generator for one named draw of this agent.
        It returns None when the agent has no seed."""
        return named_rng(self.seed, name)

    def __call__(self, w, iteration=0):
        first = int(math.floor(iteration * self.partition_advance))
        if self.use_warm_start:
            init_recon = self._previous_output
        else:
            init_recon = w
        do_initialization = iteration == 0
        if self.seed is not None:
            # The partitions are drawn here, so they do not depend on which thread runs first.
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
        # A saved state may have no 'seed' entry.
        if 'seed' in state:
            self.seed = state['seed']


class QGGMRFDenoiserAgent:
    """
    This class is used to apply the qGGMRF prior to one volume.  It calls
    :meth:`QGGMRFDenoiser.denoise`.  The first call fixes the prior parameters and the pixel
    partition.  As a result, the agent applies the same operator on every call.

    Args:
        image_shape (tuple of int): The volume shape.
        sigma_noise (float): The denoising strength.
        pinned_params (dict, optional): The prior parameters to fix, such as
            ``{'sigma_x': value}``.
        inner_iterations (int, optional): The number of VCD iterations per
            call.  Defaults to 8.
        like_model (TomographyModel, optional): A model whose devices the
            denoiser uses.  Volumes then pass between the model and the
            denoiser without leaving the devices.
        use_ror_mask (bool, optional): If True, the update is restricted to the
            inscribed ellipse.  Defaults to False.
        device (optional): The device for the denoiser.  This argument takes
            precedence over ``like_model``.
        use_warm_start (bool, optional): If True, each call starts from the
            agent's previous output.  If False, each call starts from the input.
            Defaults to True.
        seed (int, optional): The seed from which the agent derives the pixel
            partition.  If None, the agent draws the partition from the global
            np.random state.
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
        # The mask must be set before the first call draws the partition.
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
    This class is used to apply a prior to the hyperplane volumes of a 4D array.

    Consider an array with axes ``(frames, x, y, z)``.  If one spatial axis is fixed at one
    index, the result is a 3D hyperplane volume that holds every frame.  The agent denoises the
    stack of these volumes.  As a result, the prior couples neighboring frames as well as
    neighboring voxels.

    Args:
        axis (int): The spatial axis that is fixed.  It is 1, 2, or 3 in the
            ``(frames, x, y, z)`` order.
        make_stack_denoiser (callable): The call ``make_stack_denoiser(device)``
            returns the stack denoiser for that device.  The stack denoiser is a
            callable ``denoise(stack)``, or ``denoise(stack, init_stack=...)``
            when the warm start is on.  The stack is a tensor with shape
            ``(hyperplanes, frames, d1, d2)`` on that device, and the result has
            the same shape.  The denoiser may write into the stack in place.
            The agent makes one stack denoiser for each worker thread.
        batch_size (int, optional): The number of hyperplanes per task.  If
            None, one task holds all hyperplanes, and that task copies the whole
            array to one device.
        filter_matrix (torch.Tensor, optional): A square matrix whose size is
            the number of frames.  The agent applies this matrix along the frame
            axis of each batch before it denoises the batch.  One example is the
            matrix from :func:`mbirtorch.mace4d.sor_filter_matrix`.
        use_warm_start (bool, optional): If True, the agent keeps the output of
            the previous call.  It passes the matching part of that output to
            the stack denoiser as ``init_stack``.  Defaults to False.
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
        # These hold the output of the previous call and whether every task of that call has
        # finished.
        self._previous_output = None
        self._have_previous = False
        self._pending_tasks = 0
        self._denoisers = {}
        self._lock = threading.Lock()

    def _denoiser_for(self, device):
        """This method returns the stack denoiser of the calling thread on ``device``."""
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
        """This method returns the whole result in one call.  It runs the tasks one at a time
        on the device of the input."""
        output = torch.empty_like(w)
        for task in self.tasks(w, iteration):
            output[task.region] = task.run(w.device)
        return output

    def _slab_copy(self, source, region, device):
        """This method returns a contiguous copy of the part of ``source`` in ``region``.  The
        copy is on ``device``, and its first axis is the hyperplane axis."""
        slab = source[region].to(device).movedim(self.axis, 0)
        return slab.contiguous() if not slab.is_contiguous() else slab.clone()

    def _run_slab(self, w, region, device):
        device = torch.device(device)
        stack = self._slab_copy(w, region, device)
        if self.filter_matrix is not None:
            # The stack is made contiguous, so the reshape to (volumes, pixels, slices) is a view.
            stack = _filter_along_axis(stack, self.filter_matrix, axis=1).contiguous()
        denoiser = self._denoiser_for(device)
        if self.use_warm_start and self._have_previous:
            init_stack = self._slab_copy(self._previous_output, region, device)
            denoised = denoiser(stack, init_stack=init_stack)
        else:
            denoised = denoiser(stack)
        result = denoised.movedim(0, self.axis).to(w.device)
        if self.use_warm_start:
            # The tasks of one call write to disjoint regions, so this write needs no lock.
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
