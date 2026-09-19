"""Test for the lazy package exports.

A bare ``import mbirtorch`` in a fresh process must still load none of the
lazy modules.
"""

import subprocess
import sys


def test_bare_import_loads_no_lazy_module():
    # -W error turns any warning into a failure, and the module listing proves that
    # none of the lazy modules loaded.
    code = (
        "import sys; import mbirtorch; "
        "lazy = ('preprocess', 'hsnt', 'vcls', 'bn256'); "
        "loaded = sorted(m for m in sys.modules "
        "if m.startswith('mbirtorch.') and m.split('.')[1] in lazy); "
        "print(','.join(loaded) or 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', code],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'CLEAN'
