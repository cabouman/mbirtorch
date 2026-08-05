"""Parameter storage and access, ported (lean) from mbirjax.parameter_handler.

The public surface the engine and users rely on is kept: ``get_params`` (one
name or a list), ``set_params(**kwargs)`` with mbirjax's recompile and
auto-regularization semantics, ``verify_valid_params``, ``print_params``, and
the shared geometry-params namedtuple cache.  Not ported in Phase 1/2: YAML
save/load and the ParamNames Literal typing machinery.
"""

import warnings
from collections import namedtuple

import numpy as np

from . import _utils
from ._utils import Param


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

    @staticmethod
    def normalize_scalar(val):
        """Convert numpy scalar types to plain python scalars (mbirjax's
        normalize_scalar); arrays and other values pass through."""
        if isinstance(val, np.generic):
            return val.item()
        return val

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Update parameters using keyword arguments.

        This method updates internal model parameters.  If any key
        geometry-related parameters are modified, it triggers a rebuild of the
        projector system unless suppressed via the `no_compile` flag.

        The mbirjax special-case semantics are reproduced exactly:

        - Directly setting a regularization parameter (``sigma_y``, ``sigma_x``,
          or ``sigma_prox``) DISABLES auto-regularization (with a warning unless
          ``no_warning``), so the user's value is actually used by ``recon``.
        - Setting ``sharpness`` or ``snr_db`` RE-ENABLES a disabled
          auto-regularization (with a warning), so those parameters take effect.
        - An unknown parameter name raises ValueError listing the valid names,
          except under ``no_warning`` (the construction path), where it is
          ADDED as a new recompile-flagged parameter (how the geometry's own
          parameters, e.g. ``angles``, enter).
        - No validity check runs here: like mbirjax, validation is deferred to
          reconstruction entry (``verify_valid_params`` in ``vcd_recon``), so
          multi-step geometry changes (set a new sinogram shape, then call
          ``auto_set_recon_geometry``) work without a transiently-inconsistent
          state raising.

        Args:
            no_warning (bool, optional): If True, disables warnings and the
                unknown-parameter check.  Defaults to False.
            no_compile (bool, optional): If True, suppresses the projector
                rebuild after updates.  Defaults to False.
            **kwargs: parameter names and values to update.

        Example:
            >>> model.set_params(recon_shape=(128, 128, 128), sharpness=0.7)
        """
        recompile = False
        regularization_parameter_change = False
        meta_parameter_change = False

        for key, val in kwargs.items():
            # Default to forcing a recompile for new parameters.
            recompile_flag = True
            if key in self.params:
                recompile_flag = self.params[key].recompile_flag
            elif not no_warning:   # disabled for initialization, as in mbirjax
                error_message = '{} is not a recognized parameter'.format(key)
                error_message += '\nValid parameters are: \n'
                for valid_key in self.params.keys():
                    error_message += '   {}\n'.format(valid_key)
                raise ValueError(error_message)

            clean_val = ParameterHandler.normalize_scalar(val)
            self.params[key] = Param(clean_val, recompile_flag)

            # Handle special cases.
            if recompile_flag:
                recompile = True
            elif key in ["sigma_y", "sigma_x", "sigma_prox"]:
                regularization_parameter_change = True
            elif key in ["sharpness", "snr_db"]:
                meta_parameter_change = True

        # Directly-set regularization parameters disable auto-regularization,
        # so the user's value survives recon's auto_set pass.
        if regularization_parameter_change:
            if not no_warning:
                self.set_params(auto_regularize_flag=False)
                warnings.warn('You are directly setting regularization parameters, '
                              'sigma_x, sigma_y or sigma_prox. This is an advanced '
                              'feature that will disable auto-regularization.')

        # Setting sharpness/snr_db re-enables a disabled auto-regularization,
        # so those parameters take effect.
        if meta_parameter_change:
            if self.get_params('auto_regularize_flag') is False:
                self.set_params(auto_regularize_flag=True)
                if not no_warning:
                    warnings.warn('You have re-enabled auto-regularization by '
                                  'setting sharpness or snr_db. It was previously '
                                  'disabled')

        if recompile and not no_compile:
            self.create_projectors()

    # ── hooks implemented by TomographyModel / geometry classes ──────────────
    def create_projectors(self):
        raise NotImplementedError

    def verify_valid_params(self):
        """Check parameter consistency; geometry classes extend this.  Called at
        reconstruction entry (not from set_params), matching mbirjax."""
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
