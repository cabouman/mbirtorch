"""The host thread count torch uses, against the cores the process may run on.

Every host tensor operation in this package runs on torch's intra-op
threads: the consensus folds of the MACE loops, CPU reconstructions, the
denoiser statistics, the preprocessing, and the host side of every copy
from a device.  Torch takes the count from ``OMP_NUM_THREADS`` and
``MKL_NUM_THREADS`` at import, and honors the smaller of the two.  A shell
or a cluster module that sets them to one leaves every such operation on
one core, however many the allocation holds.
"""

import os
import warnings

import torch

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')


def available_cores():
    """Return the number of cores this process may run on: the affinity mask
    where the platform reports one, which respects a scheduler's allocation,
    and the machine's count otherwise."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def host_thread_report():
    """Return ``(threads, cores, variables)``: torch's intra-op thread count,
    the cores available, and the thread variables set in the environment as
    a dict."""
    variables = {name: os.environ[name] for name in THREAD_VARIABLES if name in os.environ}
    return torch.get_num_threads(), available_cores(), variables


def warn_if_threads_unused():
    """Warn once when torch runs on one host thread while more cores are
    available.  Returns the warning text, or None when there is nothing to
    say."""
    threads, cores, variables = host_thread_report()
    if threads > 1 or cores <= 1:
        return None
    if variables:
        cause = ('the environment sets ' +
                 ' and '.join(f'{name}={value}' for name, value in variables.items()))
    else:
        cause = 'the thread count was set to one'
    text = (f'torch runs host tensor operations on one thread, and this process may use '
            f'{cores} cores; {cause}.  Host-side work in mbirtorch, such as the consensus '
            f'update of a MACE reconstruction, then runs on one core.  Unset '
            f'{" and ".join(THREAD_VARIABLES)} before starting Python, or call '
            f'torch.set_num_threads({cores}).')
    warnings.warn(text, RuntimeWarning, stacklevel=2)
    return text
