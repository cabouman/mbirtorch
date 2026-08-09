"""Parameter storage and access, ported (lean) from mbirjax.parameter_handler.

The public surface the reconstruction methods and users rely on is kept: ``get_params`` (one
name or a list), ``set_params(**kwargs)`` with mbirjax's recompile and
auto-regularization semantics, ``verify_valid_params``, ``print_params``, and
the shared geometry-params namedtuple cache.  Not yet ported: YAML save/load
and the ParamNames Literal typing machinery.
"""

import io
import logging
import os
import warnings
from collections import namedtuple

import numpy as np

from . import _utils
from ._utils import Param


class ParameterHandler:

    def __init__(self):
        self.params = _utils.get_default_params()
        self.logger = None
        self.log_buffer = None

    def setup_logger(self, *, logfile_path: str = "~/.mbirtorch/logs/recon.log", print_logs: bool = True):
        """
        Initialize self.logger and self.log_buffer.
        The logging level comes from the model's 'verbose' parameter (0 -> WARNING, 1 -> INFO, 2+ -> DEBUG).

        Args:
            logfile_path: Path to the log file ('~' is expanded to the user's home, so the
                default lands in the per-user mbirtorch directory rather than littering the
                current working directory). If None or empty, file logging is skipped.
            print_logs: If True, emit logs to console.

        Raises:
            Exception: If logfile_path directory cannot be created.
        """
        if logfile_path:
            logfile_path = os.path.expanduser(logfile_path)
        # Map verbosity to logging level
        verbose = self.get_params('verbose')
        if verbose < 1:
            level = logging.WARNING
        elif verbose < 2:
            level = logging.INFO
        else:
            level = logging.DEBUG

        # Configure logger
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(level)
        # The handlers attached below are the complete set of intended outputs:
        # the in-memory buffer, the console when print_logs is on, and the file
        # when a path is given.  Without this, records also reach whatever the
        # application configured on the root logger, so print_logs=False would
        # still print the whole run log through a caller's logging.basicConfig.
        logger.propagate = False
        # Close and remove any existing handlers to prevent leaked file descriptors
        for h in list(logger.handlers):
            try:
                h.flush()
            finally:
                h.close()
                logger.removeHandler(h)

        # In-memory buffer handler (always enabled)
        self.log_buffer = io.StringIO()
        buffer_handler = logging.StreamHandler(self.log_buffer)
        buffer_handler.setLevel(level)
        buffer_formatter = logging.Formatter('%(message)s')
        buffer_handler.setFormatter(buffer_formatter)
        logger.addHandler(buffer_handler)

        # Console handler
        if print_logs:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler (optional)
        if logfile_path:
            from .utilities import makedirs
            makedirs(logfile_path)
            file_handler = logging.FileHandler(logfile_path, mode='w')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def _log_run_header(self, first_iteration, logfile_path, print_logs):
        """Set up the run logger (on the first iteration, or whenever none has been set up) and log
        the MBIRTorch version.

        Shared by recon and prox_map.  The devices are logged separately, by
        :meth:`_log_device_report`, because mbirtorch chooses the device layout
        only once the reconstruction is about to start; see that method.
        """
        # log_buffer, not logger, is what says "setup_logger has never run".
        # mbirjax tests self.logger here, which works there because its logger
        # stays None until setup.  TomographyModel fills that slot at
        # construction with a console-only logger for messages that happen
        # before a recon starts, so testing it would skip setup on every
        # resumed run (first_iteration > 0) and drop the log entirely.  Only
        # setup_logger ever creates the buffer.
        if first_iteration == 0 or self.log_buffer is None:
            self.setup_logger(logfile_path=logfile_path, print_logs=print_logs)
        from . import __version__
        self.logger.info('MBIRTorch Version = {}'.format(__version__))

    def _log_device_report(self):
        """Log the devices the reconstruction will actually use.

        Called once the device layout is final.  mbirjax logs this in the run
        header, which it can do because its layout is fixed when the model is
        built.  Here the automatic layout is chosen when a reconstruction
        starts, so a header-time report would name the placement the run was
        about to leave, and a widened run would log '1 x CUDA'.
        """
        self.logger.info('Reconstruction devices: {}'.format(
            self._device_report()))

    def _device_report(self):
        """An 'N x PLATFORM (sharded)' summary of the recon devices, for the
        recon log, noting any padding of the sharded axes.

        Padding is invisible in the results (it is kept exactly inert), so the
        log says so rather than leaving the device-form shapes a surprise.
        """
        devices = self.recon_placement.devices
        platform = devices[0].type.upper()
        report = '{} x {} (sharded)'.format(len(devices), platform)
        if self.sino_placement is not None and self.sino_placement.is_padded:
            report += ' (views padded {}->{})'.format(
                self.sino_placement.real_size, self.sino_placement.padded_size)
        if self.recon_placement.is_padded:
            report += ' (slices padded {}->{})'.format(
                self.recon_placement.real_size,
                self.recon_placement.padded_size)
        # Automatic selection that used fewer than the visible devices: say
        # which counts were turned down and why, so idle hardware is never
        # silent.  Only a layout the library chose can have a search to
        # explain, so an explicitly configured layout never carries this
        # clause -- an explicit call also clears the recorded rejections.
        rejected = getattr(self, 'device_choice_rejections', None)
        automatic = getattr(self, 'device_layout_is_automatic', False)
        if rejected and automatic:
            visible = max([count for count, _why in rejected] + [len(devices)])
            report += ' (using {} of {} {} devices: {})'.format(
                len(devices), visible, platform,
                '; '.join('{} rejected, {}'.format(count, why)
                          for count, why in rejected))
        return report

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
            self.refresh_device_bindings()

    # ── hooks implemented by TomographyModel / geometry classes ──────────────
    def create_projectors(self):
        raise NotImplementedError

    def refresh_device_bindings(self):
        # TomographyModel overrides: rebuilds the device placements from the
        # CURRENT shapes before recreating the projectors, so a
        # geometry-changing set_params can never leave a stale placement
        # silently truncating sharded arrays.
        self.create_projectors()

    def verify_valid_params(self):
        """Check parameter consistency; geometry classes extend this.  Called at
        reconstruction entry (not from set_params), matching mbirjax."""
        sinogram_shape = self.get_params('sinogram_shape')
        if len(sinogram_shape) != 3:
            raise ValueError(f'sinogram_shape must be (views, rows, channels); '
                             f'got {sinogram_shape}')

    def print_params(self):
        """
        Print the current parameter values in the model.

        This method prints all parameters stored in the model's internal
        dictionary.  If the model's verbosity level is less than 3, the view
        parameter array (e.g. the angles) is summarized rather than printed
        in full.

        Example:
            >>> ct_model = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
            >>> ct_model.set_params(sharpness=0.7)
            >>> ct_model.print_params()
        """
        verbose, view_params_name = self.get_params(['verbose',
                                                     'view_params_name'])
        print('----')
        for key, entry in self.params.items():
            if verbose < 3 and key == view_params_name:
                val = np.asarray(entry.val)
                print(f'{key} = array(shape={val.shape}, '
                      f'dtype={val.dtype})')
            else:
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
