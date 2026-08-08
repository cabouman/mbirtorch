import os

import pytest
import torch


def available_devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=available_devices())
def device(request):
    return request.param


@pytest.fixture(autouse=True, scope="session")
def pin_device_count():
    """Pin the automatic device count to 1 for the whole suite.

    A CUDA model spreads a reconstruction across the devices that can hold
    their share, so on a multi-GPU machine an unpinned suite would run some
    tests at a device count that depends on the machine.  That would make
    results, memory, and float trajectories vary by host.

    The pin is a visible, auditable line rather than a monkeypatch, and it
    uses the same environment variable a nightly or a measurement script
    would use.  Tests that WANT more than one device call configure_devices
    explicitly, which the pin does not affect.
    """
    previous = os.environ.get("MBIRTORCH_NUM_DEVICES")
    os.environ["MBIRTORCH_NUM_DEVICES"] = "1"
    yield
    if previous is None:
        os.environ.pop("MBIRTORCH_NUM_DEVICES", None)
    else:
        os.environ["MBIRTORCH_NUM_DEVICES"] = previous
