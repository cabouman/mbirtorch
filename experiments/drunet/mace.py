"""The MACE loop used by the scripts in this directory.

The loop is defined in the package module :mod:`mbirtorch.mace`.  This
module imports the two names the scripts use from there: ``mace``, the
one-call function, and ``MACE``, the class it is built on.
"""

from mbirtorch.mace import MACE, mace  # noqa: F401
