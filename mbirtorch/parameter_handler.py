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
        """
        Update parameters using keyword arguments.

        This method updates internal model parameters.  If any key
        geometry-related parameters are modified, it triggers a rebuild of the
        projector system unless suppressed via the `no_compile` flag (each
        parameter carries a recompile flag, mirroring mbirjax; in torch the
        rebuild is cheap -- there is no jit compilation -- but it refreshes the
        projectors' cached view-parameter array).

        Args:
            no_warning (bool, optional): If True, disables validity checking.
                Defaults to False.
            no_compile (bool, optional): If True, suppresses the projector
                rebuild after updates.  Defaults to False.
            **kwargs: parameter names and values to update.

        Example:
            >>> model.set_params(recon_shape=(128, 128, 128), sharpness=0.7)
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
