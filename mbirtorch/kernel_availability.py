"""Once-per-process availability probe for the hand-written (Triton) kernel
paths.

Modeled on mbirjax's pallas ``availability()`` gate: the kernel paths are an
OPTIMIZATION, so an environment that cannot run them must fall back to the
torch.compile paths silently -- but not silently-unexplained.  The probe
returns (usable, reason), and the reason string is the record of WHY a node is
not using the custom kernels (the same question a benchmark asks first).

The probe compiles and runs a trivial kernel end to end rather than trusting
version checks: it is the toolchain, not the version number, that breaks.  It
is exception-safe by construction -- every failure becomes a (False, reason)
pair -- because a broken optional path must never break a caller who only
wanted to know whether it was there.
"""

import os

import torch

# Kill switch: set to 1 to force the fallback paths (a bisection handle, and
# the escape hatch when a toolchain compiles the probe but miscompiles the
# real kernels).
DISABLE_ENV_VAR = 'MBIRTORCH_DISABLE_TRITON'

# Module-level cache of the single probe result: compiling even a trivial
# kernel costs real time, and the answer cannot change within a process.
_PROBE_RESULT = None


def triton_available():
    """(usable, reason): whether the hand-written Triton kernel paths may be
    used, and why not when they cannot.  Cached per process.

    Returns:
        (bool, str): True with a short description of what was probed, or
        False with the reason -- the kill switch, no CUDA platform, no triton
        import, or a trivial kernel that failed to compile, failed to run, or
        returned the wrong value.
    """
    global _PROBE_RESULT
    if _PROBE_RESULT is None:
        _PROBE_RESULT = _probe_triton()
    return _PROBE_RESULT


def _probe_triton():
    """Run the probe once (see :func:`triton_available`); never raises."""
    if os.environ.get(DISABLE_ENV_VAR, '0') == '1':
        result = (False, f'disabled by {DISABLE_ENV_VAR}=1')
    else:
        try:
            if not torch.cuda.is_available():
                result = (False, 'not a CUDA platform (torch.cuda is '
                                 'unavailable)')
            else:
                import triton
                import triton.language as tl

                @triton.jit
                def probe_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
                    offsets = tl.arange(0, BLOCK)
                    tl.store(out_ptr + offsets,
                             tl.load(x_ptr + offsets) + tl.load(y_ptr + offsets))

                block = 32
                x = torch.ones(block, dtype=torch.float32, device='cuda')
                y = torch.full((block,), 2.0, dtype=torch.float32, device='cuda')
                out = torch.empty_like(x)
                probe_kernel[(1,)](x, y, out, BLOCK=block)
                if bool(torch.all(out == 3.0)):
                    result = (True, f'available (triton {triton.__version__}, '
                                    f'{torch.cuda.get_device_name(0)})')
                else:
                    result = (False, 'probe kernel returned a wrong value')
        except Exception as e:                                    # noqa: BLE001
            result = (False, f'probe kernel failed to compile/run: {e}')
    return result


def _reset_probe_cache():
    """Drop the cached probe result so the next call probes again (tests:
    the kill switch is read INSIDE the probe, so flipping it takes effect
    only across a reset)."""
    global _PROBE_RESULT
    _PROBE_RESULT = None
