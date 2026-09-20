"""``python -m mbirtorch.hsnt`` runs the command-line interface (see mbirtorch.hsnt.cli)."""
import sys

from .cli import main

sys.exit(main())
