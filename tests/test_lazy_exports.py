"""Tests for the lazy package exports and the guarded import block that describes them.

The first test parses ``mbirtorch/__init__.py`` and checks that the block guarded by
``TYPE_CHECKING`` names exactly the lazily resolved names, each imported from the module
that supplies it.  The second test checks that a bare ``import mbirtorch`` in a fresh
process still loads none of the lazy modules.
"""

import ast
import subprocess
import sys

import mbirtorch


def test_guarded_block_names_every_lazy_export():
    with open(mbirtorch.__file__) as f:
        tree = ast.parse(f.read())

    guarded = [node for node in tree.body
               if isinstance(node, ast.If)
               and isinstance(node.test, ast.Name)
               and node.test.id == 'TYPE_CHECKING']
    assert len(guarded) == 1, 'expected exactly one TYPE_CHECKING block'

    imported = {}
    for node in guarded[0].body:
        assert isinstance(node, ast.ImportFrom), 'the block holds only from-imports'
        module = node.module or ''
        for alias in node.names:
            assert alias.asname is None, \
                f'{alias.name} must be imported under its own name'
            imported[alias.name] = module

    expected = {name: 'view_utils' for name in mbirtorch._VIEWER_EXPORTS}
    expected.update({name: '' for name in mbirtorch._LAZY_MODULES})
    expected.update(mbirtorch._LAZY_NAMES)

    assert imported == expected


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
