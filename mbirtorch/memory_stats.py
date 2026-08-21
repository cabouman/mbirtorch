"""Per-processor memory statistics.

``get_memory_stats`` returns one dict per processor with the bytes in use,
the peak, and the limit, printed in GB.  The device numbers come from each
torch backend's own allocator introspection (torch.cuda / torch.mps) and the
host numbers from psutil.  Backends differ in what they can report, so the
per-device dicts carry the keys their backend actually tracks (the printer
iterates whatever is present):

- CUDA: ``bytes_in_use`` (allocated), ``peak_bytes_in_use`` (resettable via
  torch.cuda.reset_peak_memory_stats), ``reserved_bytes`` (the caching
  allocator's pool, the analog of jax's pool-vs-in-use distinction), and
  ``bytes_limit`` (the device's total memory).
- MPS: ``bytes_in_use`` (current allocated), ``driver_bytes_in_use`` (the
  Metal driver's total footprint), and ``bytes_limit`` (the recommended
  working-set maximum).  MPS does not track a peak.
- CPU: the psutil convention is deliberate -- ``bytes_in_use`` is the
  unique set size (USS), ``peak_bytes_in_use`` the resident set size (RSS),
  and ``bytes_limit`` the currently available physical memory.

There is no inventory of live arrays here, because torch keeps no registry of
them (torch.cuda.memory_summary is the closest native deep-dive tool).
"""

import os

import psutil
import torch


def get_memory_stats(print_results=True, file=None):
    """Collect (and optionally print) memory statistics per processor.

    Args:
        print_results (bool, optional): If True (default), print a formatted
            GB summary of each processor's stats.
        file (file-like, optional): destination for the printout (None means
            stdout), e.g. an io.StringIO for routing into a logger.

    Returns:
        list of dict: one dict per processor (devices first, then 'CPU'),
        each with an 'id' plus the byte counts described in the module
        docstring.
    """
    memory_stats_per_processor = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_stats = dict()
            memory_stats['id'] = 'GPU ' + str(i)
            memory_stats['bytes_in_use'] = torch.cuda.memory_allocated(i)
            memory_stats['peak_bytes_in_use'] = torch.cuda.max_memory_allocated(i)
            memory_stats['reserved_bytes'] = torch.cuda.memory_reserved(i)
            memory_stats['bytes_limit'] = torch.cuda.get_device_properties(i).total_memory
            memory_stats_per_processor.append(memory_stats)
    elif torch.backends.mps.is_available():
        memory_stats = dict()
        memory_stats['id'] = 'MPS'
        memory_stats['bytes_in_use'] = torch.mps.current_allocated_memory()
        memory_stats['driver_bytes_in_use'] = torch.mps.driver_allocated_memory()
        memory_stats['bytes_limit'] = torch.mps.recommended_max_memory()
        memory_stats_per_processor.append(memory_stats)

    # Then add info for the CPU
    memory_stats = dict()
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_full_info()
    memory_stats['id'] = 'CPU'
    # memory_info.rss is the Resident Set Size (the non-swapped physical
    # memory the process has used); uss is the Unique Set Size.  The
    # peak<-rss / in_use<-uss assignment is intentional.
    memory_stats['bytes_in_use'] = memory_info.uss
    memory_stats['peak_bytes_in_use'] = memory_info.rss
    # Available physical memory (excluding swap)
    memory_stats['bytes_limit'] = psutil.virtual_memory().available
    memory_stats_per_processor.append(memory_stats)

    if print_results:
        for memory_stats in memory_stats_per_processor:
            print(memory_stats['id'], file=file)
            for tag, value in memory_stats.items():
                if tag == 'id':
                    continue
                cur_value = value / (1024 ** 3)
                extra_space = ' ' * max(1, 21 - len(tag) - len(str(int(cur_value))))
                print(f'  {tag}:{extra_space}{cur_value:.3f}GB', file=file)

    return memory_stats_per_processor
