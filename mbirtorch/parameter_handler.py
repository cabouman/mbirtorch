"""Parameter storage and access, ported (lean) from mbirjax.parameter_handler.

The public surface the engine and users rely on is kept: ``get_params`` (one
name or a list), ``set_params(**kwargs)`` with recompile-on-geometry-change,
``verify_valid_params``, ``print_params``, and the shared geometry-params
namedtuple cache.  Not ported in Phase 1: YAML save/load, the ParamNames
Literal typing machinery, and the use_gpu deprecation shim.
"""

from collections import namedtuple

import numpy as np

from . import _utils


class ParameterHandler:

    def __init__(self):
        self.params = _utils.get_default_params()

    # ── access ────────────────────────────────────────────────────────────────
    def get_params(self, parameter_names):
        """Return the value of one parameter (a string) or a list of values."""
        if isinstance(parameter_names, str):
            if parameter_names not in self.params:
                raise NameError(f"'{parameter_names}' not a recognized parameter")
            return self.params[parameter_names].val
        values = []
        for name in parameter_names:
            if name not in self.params:
                raise NameError(f"'{name}' not a recognized parameter")
            values.append(self.params[name].val)
        return values

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """Update parameters; recompile the projectors if a geometry parameter changed.

        Mirrors mbirjax: each parameter carries a recompile flag; setting any
        flagged parameter triggers ``create_projectors`` (unless ``no_compile``),
        and validity checking runs unless ``no_warning``.
        """
        recompile = False
        for key, val in kwargs.items():
            if key in self.params:
                self.params[key].val = val
                if self.params[key].recompile_flag:
                    recompile = True
            else:
                raise NameError(f"'{key}' not a recognized parameter")

        if recompile and not no_compile:
            self.create_projectors()
        if not no_warning:
            self.verify_valid_params()

    # ── hooks implemented by TomographyModel / geometry classes ──────────────
    def create_projectors(self):
        raise NotImplementedError

    def verify_valid_params(self):
        """Check parameter consistency; geometry classes extend this."""
        sinogram_shape = self.get_params('sinogram_shape')
        if len(sinogram_shape) != 3:
            raise ValueError(f'sinogram_shape must be (views, rows, channels); '
                             f'got {sinogram_shape}')

    def print_params(self):
        print('----')
        for key, entry in self.params.items():
            print(f'{key} = {entry.val}')
        print('----')

    # ── shared geometry-params namedtuple cache ───────────────────────────────
    # The namedtuple CLASS is cached per field-name tuple (module-level), matching
    # mbirjax's make_geometry_params.  In torch there is no jit-cache reason, but a
    # shared class keeps equality and repr behavior consistent across instances.
    _geometry_param_classes = {}

    @classmethod
    def make_geometry_params(cls, names, values):
        key = tuple(names)
        if key not in cls._geometry_param_classes:
            cls._geometry_param_classes[key] = namedtuple('GeometryParams', names)
        return cls._geometry_param_classes[key](*values)

    @staticmethod
    def get_cluster_boundaries(*args, **kwargs):   # placeholder parity hook
        raise NotImplementedError
