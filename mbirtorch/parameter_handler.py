"""Parameter storage and access.

The public surface is ``get_params`` (one name or a list),
``set_params(**kwargs)`` with its recompile and auto-regularization semantics,
``verify_valid_params``, ``print_params``, and the shared geometry-params
namedtuple cache.  Not implemented: YAML save/load and the ParamNames Literal
typing machinery.
"""

import io
import itertools
import logging
import os
import warnings
from collections import namedtuple

import numpy as np

from . import _utils
from ._utils import Param

# This counter gives each model instance a unique logger name.
_instance_counter = itertools.count(1)


class ParameterHandler:
    """Store and access model parameters.  ``TomographyModel`` inherits its
    parameter interface -- ``get_params``, ``set_params``, and
    ``print_params`` -- from this class."""

    def __init__(self):
        self.params = _utils.get_default_params()
        # The logger name must be different for each instance, because Python logging
        # keeps one logger per name and two instances would share handlers.  The counter
        # is used instead of id(self), because Python reuses an id after a free.
        self.logger_name = 'mbirtorch.{}.{}'.format(type(self).__name__,
                                                    next(_instance_counter))
        # Until a run calls setup_logger, messages go to the console only.
        self.logger = logging.getLogger(self.logger_name)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        self.log_buffer = None
        # This holds the log file of the run in progress, so that a call which
        # continues the run can reopen it.
        self._logfile_path = None

    def setup_logger(self, *, logfile_path: str = "~/.mbirtorch/logs/recon.log", print_logs: bool = True):
        """
        Set up this instance's logger (self.logger) and self.log_buffer for a run.
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
        verbose = self.get_params('verbose')
        if verbose < 1:
            level = logging.WARNING
        elif verbose < 2:
            level = logging.INFO
        else:
            level = logging.DEBUG

        logger = self.logger
        logger.setLevel(level)
        # The handlers attached below are the complete set of outputs.  Setting
        # propagate to False keeps records from also reaching the root logger.
        logger.propagate = False
        # Close existing handlers to avoid leaking file descriptors.
        for h in list(logger.handlers):
            try:
                h.flush()
            finally:
                h.close()
                logger.removeHandler(h)

        self.log_buffer = io.StringIO()
        buffer_handler = logging.StreamHandler(self.log_buffer)
        buffer_handler.setLevel(level)
        buffer_formatter = logging.Formatter('%(message)s')
        buffer_handler.setFormatter(buffer_formatter)
        logger.addHandler(buffer_handler)

        if print_logs:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Mode 'w' starts a new log for a new run.
        self._logfile_path = logfile_path if logfile_path else None
        if logfile_path:
            self._add_log_file_handler(logfile_path, mode='w', level=level)

    def _add_log_file_handler(self, logfile_path, mode, level):
        """Attach a handler that copies the log to ``logfile_path``.

        The handler uses delay=True, so a run that logs nothing creates no file.
        """
        from .utilities import makedirs
        makedirs(logfile_path)
        file_handler = logging.FileHandler(logfile_path, mode=mode, delay=True)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

    def close_log_file(self):
        """Finish the log file: flush it, close it, and detach its handler.

        Called when a run has written its last line.  An open file handler
        keeps writing to whatever file it opened, even after that file has
        been renamed or deleted, and on Windows it would block the delete
        outright.  So the composite runs that merge and then delete the log of
        each part (recon_split_sino, recon_plastic_metal) need each part's
        file closed first.

        The in-memory buffer handler is left alone, so the run's log text is
        still available for recon_dict['recon_log'].  A later call continuing
        the same run reopens the file in append mode; see ``_reopen_log_file``.
        """
        if self.logger is None:
            return
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    h.flush()
                finally:
                    h.close()
                    self.logger.removeHandler(h)

    def _reopen_log_file(self):
        """Reattach the log file that ``close_log_file`` closed, opening it in
        append mode.  Do nothing if the run has no log file, or if a file
        handler is already attached."""
        if self.logger is None or not self._logfile_path:
            return
        if any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            return
        self._add_log_file_handler(self._logfile_path, mode='a',
                                   level=self.logger.level)

    def _log_run_header(self, first_iteration, logfile_path, print_logs):
        """Set up the run logger if needed and log the MBIRTorch version.

        Both recon and prox_map call this method.  The devices are logged
        separately by :meth:`_log_device_report`, once the layout is final.
        """
        # The log buffer, not the logger, tells whether setup_logger has run.
        # The constructor always fills the logger slot with a console logger.
        if first_iteration == 0 or self.log_buffer is None:
            self.setup_logger(logfile_path=logfile_path, print_logs=print_logs)
        else:
            self._reopen_log_file()
        from . import __version__
        self.logger.info('MBIRTorch Version = {}'.format(__version__))

    def _log_device_report(self):
        """Log the devices the reconstruction will use.  Call only once the
        device layout is final, which is after the run header is written."""
        self.logger.info('Reconstruction devices: {}'.format(
            self._device_report()))

    def _device_report(self):
        """Return a summary of the reconstruction devices for the log, in the
        form 'N x PLATFORM (sharded)'."""
        devices = self.recon_placement.devices
        platform = devices[0].type.upper()
        report = '{} x {} (sharded)'.format(len(devices), platform)
        # When the library chose the layout itself and used fewer than the visible
        # devices, the report names each rejected device count and the reason.
        rejected = getattr(self, 'device_choice_rejections', None)
        automatic = getattr(self, 'device_layout_is_automatic', False)
        if rejected and automatic:
            visible = max([count for count, _why in rejected] + [len(devices)])
            # One recorded entry can name the count that is actually in use, so
            # each entry is labelled either 'used' or 'rejected'.
            report += ' (using {} of {} {} devices: {})'.format(
                len(devices), visible, platform,
                '; '.join('{} {}, {}'.format(
                    count, 'used' if count == len(devices) else 'rejected',
                    why) for count, why in rejected))
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
        """Convert numpy scalar types to plain python scalars; arrays and
        other values pass through."""
        if isinstance(val, np.generic):
            return val.item()
        return val

    def set_params(self, no_warning=False, no_compile=False, **kwargs):
        """
        Update parameters using keyword arguments.

        This method updates internal model parameters.  If any key
        geometry-related parameters are modified, it triggers a rebuild of the
        projector system unless suppressed via the `no_compile` flag.

        Four special cases apply:

        - Directly setting a regularization parameter (``sigma_y``, ``sigma_x``,
          or ``sigma_prox``) disables auto-regularization and warns, so the
          user's value is actually used by ``recon``.  With ``no_warning=True``
          neither happens: the value is stored and auto-regularization stays
          on.  (The automatic setters use that path internally.)
        - Setting ``sharpness`` or ``snr_db`` re-enables a disabled
          auto-regularization, with a warning unless ``no_warning``.
        - An unknown parameter name raises ValueError listing the valid names,
          except under ``no_warning`` (the construction path), where it is
          ADDED as a new recompile-flagged parameter (how the geometry's own
          parameters, e.g. ``angles``, enter).
        - No validity check runs here.  Validation is deferred to
          reconstruction entry (``verify_valid_params`` in ``_vcd_recon``), so
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
            # New parameters force a recompile.
            recompile_flag = True
            if key in self.params:
                recompile_flag = self.params[key].recompile_flag
            elif not no_warning:   # disabled for initialization
                error_message = '{} is not a recognized parameter'.format(key)
                error_message += '\nValid parameters are: \n'
                for valid_key in self.params.keys():
                    error_message += '   {}\n'.format(valid_key)
                raise ValueError(error_message)

            clean_val = ParameterHandler.normalize_scalar(val)
            self.params[key] = Param(clean_val, recompile_flag)

            if recompile_flag:
                recompile = True
            elif key in ["sigma_y", "sigma_x", "sigma_prox"]:
                regularization_parameter_change = True
            elif key in ["sharpness", "snr_db"]:
                meta_parameter_change = True

        if regularization_parameter_change:
            if not no_warning:
                self.set_params(auto_regularize_flag=False)
                warnings.warn('You are directly setting regularization parameters, '
                              'sigma_x, sigma_y or sigma_prox. This is an advanced '
                              'feature that will disable auto-regularization.')

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
        # TomographyModel overrides this to rebuild device placements from the
        # current shapes before recreating the projectors.
        self.create_projectors()

    def verify_valid_params(self):
        """Check parameter consistency; geometry classes extend this.  Called at
        reconstruction entry, not from set_params."""
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

    # The namedtuple class is cached and shared for each set of field names, so
    # that equality and repr behave the same way for all instances.
    _geometry_param_classes = {}

    @classmethod
    def make_geometry_params(cls, names, values):
        key = tuple(names)
        if key not in cls._geometry_param_classes:
            cls._geometry_param_classes[key] = namedtuple('GeometryParams', names)
        return cls._geometry_param_classes[key](*values)
