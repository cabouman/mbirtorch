import os

import pytest
import torch

# Start CUDA before any test runs.  torch checks every device in
# range(device_count()) at the first CUDA use, and tests below fake
# device_count, so a first use inside one of them asks for a device that does
# not exist and fails.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")


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
def redirect_default_log_location(tmp_path_factory):
    """Send the default '~/.mbirtorch' paths into the pytest tmp tree.

    recon and prox_map default logfile_path to '~/.mbirtorch/logs/recon.log'
    (prox.log), so every test that takes the default would otherwise leave a
    real file in the user's home.  On a cluster that home is quota'd, and a
    job that fills it fails with an empty log.

    Two things narrow the redirect.  It maps only paths at or under
    '~/.mbirtorch' and hands every other path to the real expanduser, so
    HOME itself is untouched and torch's caches -- expensive to rebuild --
    stay where they are.  And it patches the expansion rather than the
    logging setup, because the composite runs (recon_split_sino,
    recon_plastic_metal) expand the path themselves to name their per-part
    temp files; one patch covers every site that turns a '~' into a real
    directory.
    """
    base = tmp_path_factory.mktemp("mbirtorch_home")
    real_expanduser = os.path.expanduser
    prefix = os.path.join("~", ".mbirtorch")

    def expanduser(path):
        redirected = (isinstance(path, str)
                      and (path == prefix or path.startswith(prefix + os.sep)))
        if redirected:
            expanded = os.path.join(str(base), path[2:])
        else:
            expanded = real_expanduser(path)
        return expanded

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os.path, "expanduser", expanduser)
        yield base


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
