"""Interactive viewer for 4D volumes, built on the slice viewer.

The slice viewer in ``slice_figure`` is not changed.  This module subclasses its two
classes.  ``VolumeStack4D`` adds a frame axis to the data model, and ``SliceViewer4D``
adds a frame slider, playback, and a plot of the ROI mean against frame.  A 4D array is
``(t, x, y, z)``.  Like ``slice_figure``, this module imports numpy and the matplotlib
base package at most, so it loads without a GUI toolkit.
"""

import matplotlib
import numpy as np

from . import slice_figure as sf
from .slice_figure import Mode, SliceViewer, VolumeStack

__all__ = ['SliceViewer4D', 'VolumeStack4D', 'slice_viewer4d']

# Names of the axes of one frame, in axis order.
AXIS_NAMES = ('x', 'y', 'z')

ROI_PLOT_FONT_SIZE = 8
ROI_PLOT_HINT = 'Draw an ROI on an image to plot its mean against frame'


class VolumeStack4D(VolumeStack):
    """Pure-numpy data model for the 4D viewer.

    Holds 2D, 3D, and 4D arrays with one shared frame index.  A 4D array is
    ``(t, x, y, z)``.  A 2D or 3D array has one frame, so it stays fixed in time, and a
    4D array with fewer frames than the longest one holds its last frame.  Everything
    the 3D model does, such as slicing, reorienting, and ROI statistics, acts on the
    current frame.

    Args:
        datasets (sequence of ndarray or None): One or more 2D, 3D, or 4D arrays.  2D
            arrays gain a trailing singleton axis, and None becomes a placeholder zero
            volume.
        data_dicts (None, dict, or list of dict/None, optional): String-valued dict(s)
            associated with the volumes.
        vmin (float, optional): Display minimum.  Defaults to the minimum over every
            voxel of every volume.
        vmax (float, optional): Display maximum.  Defaults to the maximum over every
            voxel of every volume.
        slice_label (str or list of str, optional): Label(s) at the start of each panel
            title.  Defaults to no label.
        slice_axis (int or list of int, optional): Axis to slice along, counted in each
            array's own axes: 1, 2, or 3 for a 4D array and 0, 1, or 2 for a 3D array.
            Defaults to z.  With 3D and 4D arrays together, give a list with one entry
            per volume.
    """

    def __init__(self, datasets, data_dicts=None, vmin=None, vmax=None,
                 slice_label=None, slice_axis=None):
        datasets = list(datasets)
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")
        self.n_volumes = len(datasets)
        self.labels = self._normalize_labels(slice_label)
        self.data_dicts = self._normalize_data_dicts(data_dicts)

        # original_data holds each array as it was passed in, and Save writes it.
        # frames holds every volume as a 4D array, with one frame for a 2D or 3D
        # input, and a difference image replaces it.  data holds the displayed view
        # of the current frame with the slice axis last, as in the 3D model, so the
        # inherited slicing, reorienting, and ROI methods act on the current frame.
        self.original_data = []
        self.frames = []
        for dataset in datasets:
            if dataset is None:
                dataset = np.zeros(sf.PLACEHOLDER_SHAPE)
            dataset = np.asarray(dataset)
            if dataset.ndim == 2:
                dataset = dataset[..., np.newaxis]
            elif dataset.ndim not in (3, 4):
                raise ValueError("Each input data must be a 2D, 3D, or 4D array")
            self.original_data.append(dataset)
            self.frames.append(self._as_frames(dataset))

        self.axes_perms = self._normalize_frame_slice_axes(slice_axis)
        self._difference_info = [None] * self.n_volumes

        self.master_frame = 0
        self.cur_frames = [0] * self.n_volumes
        self.data = [self._frame_view(i) for i in range(self.n_volumes)]

        # Each volume opens at its own middle slice, and the shared slider starts
        # at volume 0's middle slice, as in the 3D model.
        self.cur_slices = [d.shape[2] // 2 for d in self.data]
        self.master_index = self.cur_slices[0]

        self.vmin, self.vmax = self.resolve_range(vmin, vmax)

    # --- Input normalization ---

    @staticmethod
    def _as_frames(array):
        """Return ``array`` as a 4D view; a 3D array becomes one frame."""
        return array if array.ndim == 4 else array[np.newaxis]

    def _normalize_labels(self, slice_label):
        # A 4D panel title names its position, so no label is needed by default.
        return super()._normalize_labels('' if slice_label is None else slice_label)

    def _normalize_frame_slice_axes(self, slice_axis):
        # Each array counts its own axes, so axis 3 of a 4D array and axis 2 of a 3D
        # array both mean z.  The model keeps the axis of the frame, 0, 1, or 2.
        if slice_axis is None:
            return [self.perm_from_slice_axis(2) for _ in range(self.n_volumes)]
        if isinstance(slice_axis, (int, np.integer)):
            slice_axis = [slice_axis] * self.n_volumes
        slice_axis = [int(s) for s in slice_axis]
        if len(slice_axis) != self.n_volumes:
            raise ValueError(
                "slice_axis must be a single int or a list of ints of the "
                "same length as the number of datasets")
        frame_axes = []
        for axis, dataset in zip(slice_axis, self.original_data):
            if dataset.ndim == 4:
                if axis not in (1, 2, 3):
                    raise ValueError("slice_axis of a 4D array must be 1, 2, or 3")
                axis -= 1
            frame_axes.append(axis)
        return [self.perm_from_slice_axis(axis) for axis in frame_axes]

    # --- Frames: one shared frame index; a shorter volume holds its last frame ---

    @property
    def frame_counts(self):
        """Per-volume number of frames; 1 for a 2D or 3D volume."""
        return [frames.shape[0] for frames in self.frames]

    @property
    def max_frames(self):
        """Number of frames of the longest volume."""
        return max(self.frame_counts)

    def has_time(self, i):
        """Return True if volume ``i`` changes with the frame index."""
        return self.original_data[i].ndim == 4 or self.frames[i].shape[0] > 1

    def holds_last_frame(self, i):
        """Return True if volume ``i`` has fewer frames than the frame index needs."""
        return self.has_time(i) and self.master_frame > self.frames[i].shape[0] - 1

    def _frame_view(self, i):
        return np.transpose(self.frames[i][self.cur_frames[i]], self.axes_perms[i])

    def _show_current_frame(self, i):
        self.cur_frames[i] = min(self.master_frame, self.frames[i].shape[0] - 1)
        self.data[i] = self._frame_view(i)

    def set_frame(self, index):
        """Set the shared frame index; return the volumes whose displayed frame changed.

        The index runs from 0 to the last frame of the longest volume, and values
        outside that range are clipped.  A shorter volume holds its last frame, and a
        2D or 3D volume keeps its only frame.
        """
        self.master_frame = int(np.clip(int(np.round(index)), 0, self.max_frames - 1))
        changed = []
        for i, frames in enumerate(self.frames):
            frame = min(self.master_frame, frames.shape[0] - 1)
            if frame != self.cur_frames[i]:
                self.cur_frames[i] = frame
                self.data[i] = self._frame_view(i)
                changed.append(i)
        return changed

    # --- Intensity range ---

    def data_range(self):
        """Return (min, max) over every frame of every volume."""
        lo = min(float(np.min(frames)) for frames in self.frames)
        hi = max(float(np.max(frames)) for frames in self.frames)
        return lo, hi

    # --- Difference images ---

    def can_difference(self, baseline_index, comparison_index):
        """Return True if the pair is a valid difference: distinct volumes with the
        same x, y, z shape, and either the same number of frames or one frame in one
        of them."""
        if baseline_index == comparison_index:
            return False
        base_shape = self.frames[baseline_index].shape
        comparison_shape = self.frames[comparison_index].shape
        return (base_shape[1:] == comparison_shape[1:]
                and (base_shape[0] == comparison_shape[0]
                     or 1 in (base_shape[0], comparison_shape[0])))

    def apply_difference(self, baseline_index, comparison_index, use_abs=False):
        """Replace volume ``baseline_index`` with (comparison - baseline) for every frame.

        The difference is computed for the whole volume at once.  It uses the volumes'
        current data, so a difference against an already-differenced volume compares
        against what is displayed.  A 3D volume is subtracted from every frame of a 4D
        volume.  With ``use_abs`` the absolute difference is shown.  The previous label
        is saved and restored by :meth:`restore`.
        """
        if not self.can_difference(baseline_index, comparison_index):
            raise ValueError(
                "Difference requires two distinct volumes with the same x, y, z shape")
        # The frames are stored in their original axis order, so the two volumes need
        # no reorienting, and a one-frame volume broadcasts over the frames.
        difference = self.frames[comparison_index] - self.frames[baseline_index]
        if use_abs:
            difference = np.abs(difference)
        self._difference_info[baseline_index] = {
            'comparison_index': comparison_index,
            'use_abs': use_abs,
            'prev_label': self.labels[baseline_index],
        }
        self.frames[baseline_index] = difference
        self._show_current_frame(baseline_index)
        if use_abs:
            label_prepend = 'abs(Image {} minus current): '.format(comparison_index)
        else:
            label_prepend = 'Image {} minus current: '.format(comparison_index)
        self.labels[baseline_index] = label_prepend + self.labels[baseline_index]

    def restore(self, i):
        """Restore volume ``i`` from its original data, ending any difference."""
        self.frames[i] = self._as_frames(self.original_data[i])
        self._show_current_frame(i)
        info = self._difference_info[i]
        if info is not None:
            self.labels[i] = info['prev_label']
        self._difference_info[i] = None

    # --- ROI mean against frame ---

    def roi_frame_means(self, i, x, y, radius):
        """Mean of volume ``i`` inside a circle, for every frame, at the current slice.

        ``x`` and ``y`` are data coordinates (column, row) of the circle center, as in
        :meth:`roi_stats`.  Returns an array with one value per frame, or None when no
        pixel center falls inside the circle.
        """
        perm = [0] + [axis + 1 for axis in self.axes_perms[i]]
        # The displayed plane of every frame, as one (frame, row, column) view.
        planes = np.transpose(self.frames[i], perm)[..., self.cur_slices[i]]
        ny, nx = planes.shape[1:]
        yv, xv = np.ogrid[:ny, :nx]
        mask = (xv - x) ** 2 + (yv - y) ** 2 <= radius ** 2
        if not mask.any():
            return None
        return planes[:, mask].mean(axis=1)

    # --- File load ---

    def load_array(self, image_index, new_array, data_dict=None):
        """Replace volume ``image_index`` with a loaded array; return [image_index].

        A 2D array gains a trailing singleton axis, and a 3D or 4D array becomes one
        volume.  Anything else raises ValueError.  The volume leaves any difference
        state and returns to the standard display order for its slice axis.  The frame
        index is kept where the new longest volume allows, and the slice position
        moves to the loaded volume's middle slice.
        """
        new_array = np.asarray(new_array)
        if new_array.ndim == 2:
            new_array = new_array[..., np.newaxis]
        if new_array.ndim not in (3, 4):
            raise ValueError("Loaded array must be 2D, 3D, or 4D")
        i = image_index
        self.original_data[i] = new_array
        self.frames[i] = self._as_frames(new_array)
        self.data_dicts[i] = data_dict
        self.axes_perms[i] = self.perm_from_slice_axis(self.axes_perms[i][-1])
        if self._difference_info[i] is not None:
            self.labels[i] = self._difference_info[i]['prev_label']
            self._difference_info[i] = None
        self.set_frame(self.master_frame)
        self._show_current_frame(i)

        depth = self.data[i].shape[2]
        middle_fraction = (depth // 2) / (depth - 1) if depth > 1 else 0.0
        self._set_master_fraction(middle_fraction)
        return [i]


class SliceViewer4D(SliceViewer):
    """Interactive viewer for 4D volumes, with every feature of the slice viewer (matplotlib).

    The window of :class:`SliceViewer` gains a frame slider with a Play button and a
    plot of the ROI mean against frame, both shown when at least one input is 4D.
    Space plays and pauses, and comma and period step one frame back and forward.
    Construction builds the figure but does not display it; call :meth:`show` to
    display.  All data logic lives in :class:`VolumeStack4D` (``self.stack``).

    Args:
        *datasets (ndarray or None): One or more 2D, 3D, or 4D arrays to display.  A 4D
            array is ``(t, x, y, z)``.  A 2D or 3D array stays fixed in time.
        data_dicts (None, dict, or list of dict/None, optional): String-valued dict(s)
            associated with the volumes, viewable in the viewer.
        title (str, optional): Figure title.  Defaults to ''.
        vmin (float, optional): Minimum display intensity.  Defaults to the minimum
            over every voxel of every volume.
        vmax (float, optional): Maximum display intensity.  Defaults to the maximum
            over every voxel of every volume.
        slice_label (str or list of str, optional): Label(s) at the start of each
            panel title.  Defaults to no label.
        slice_axis (int or list of int, optional): Axis to slice along, counted in each
            array's own axes: 1, 2, or 3 for a 4D array and 0, 1, or 2 for a 3D array.
            Defaults to z.  With 3D and 4D arrays together, give a list.
        cmap (str, optional): Colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): Show the "Press h for help" hint.
            Defaults to True.
        save_fn (callable, optional): Replacement for the built-in HDF5 writer, called
            as ``save_fn(file_path, array, array_name, attributes_dict)``.
        fps (float, optional): Playback speed in frames per second.  Defaults to 5.
    """

    def __init__(self, *datasets, data_dicts=None, title='', vmin=None, vmax=None,
                 slice_label=None, slice_axis=None, cmap='gray',
                 show_instructions=True, save_fn=None, fps=5):
        if fps <= 0:
            raise ValueError('fps must be positive; got {}.'.format(fps))
        sf._load_pyplot()
        self.stack = VolumeStack4D(datasets, data_dicts=data_dicts, vmin=vmin,
                                   vmax=vmax, slice_label=slice_label,
                                   slice_axis=slice_axis)
        self.title = title
        self.cmap = cmap
        self.show_instructions = show_instructions
        self.save_fn = save_fn if save_fn is not None else sf._save_data_hdf5
        self.fps = float(fps)

        self._init_interaction_state()
        self._build_figure()
        self._connect_events()

    # --- Construction ---

    def _init_interaction_state(self):
        super()._init_interaction_state()
        self.playing = False
        self._play_timer = None
        self._play_regions = None
        self.frame_slider = None
        self.play_button = None
        self._frame_label = None
        self.roi_plot_ax = None
        self._roi_lines = []
        self._roi_marker = None
        self._roi_hint = None

    def _build_figure(self):
        n = self.stack.n_volumes
        self.fig = sf.plt.figure(figsize=(6 * n, 10.5))
        self.fig.suptitle(self.title)
        # The rows are the panels, the slice-axis radios, the slice slider, the
        # intensity slider, the frame row, and the plot of the ROI mean against
        # frame.  Rows 2 and 3 hold the sliders, where the inherited methods put them.
        self.gs = sf.gridspec.GridSpec(nrows=6, ncols=n,
                                       height_ratios=[15, 1.7, 1, 1, 1, 4],
                                       left=0.12, right=0.95, top=0.92,
                                       bottom=0.05, hspace=0.5, figure=self.fig)
        self.axes = [None] * n
        self.caxes = [None] * n
        self.images = [None] * n
        self._create_panels()
        self._create_tooltips()
        self._create_axis_row()
        self._create_slice_slider()
        self._create_intensity_slider()
        self._create_frame_row()
        self._create_roi_plot()

        # This opaque rectangle is drawn under partial redraws.  Marking it
        # animated keeps it out of ordinary full draws.
        self._clear_rect = sf.Rectangle((0, 0), 1, 1,
                                        facecolor=self.fig.get_facecolor(),
                                        edgecolor='none', animated=True,
                                        transform=sf.IdentityTransform())
        self.fig.add_artist(self._clear_rect)

        if self.show_instructions:
            self.fig.text(0.01, 0.25, sf.multiline('Press h', 'for help'),
                          fontdict={'color': 'red'})

    def _title_text(self, i):
        stack = self.stack
        head = stack.labels[i].strip()
        if head and not head.endswith(':'):
            head += ':'
        position = '{} = {}'.format(AXIS_NAMES[stack.slice_axes[i]], stack.cur_slices[i])
        if stack.has_time(i):
            frame = 't = {}'.format(stack.cur_frames[i])
            if stack.holds_last_frame(i):
                frame += ' (last)'
            position = frame + ', ' + position
        rows, columns = (AXIS_NAMES[axis] for axis in stack.axes_perms[i][:2])
        return sf.multiline(
            '{} {}'.format(head, position).strip(),
            'Shape: {}, plane {}-{}'.format(stack.original_data[i].shape, rows, columns))

    def _rebuild_axis_radios(self):
        # This follows the inherited method, with the radios labeled x, y, z.
        for ax in self._radio_axes:
            ax.remove()
        self._radio_axes = []
        self.axis_radios = []

        def make_radio(slot, active):
            ax = self.fig.add_subplot(slot)
            ax.set_title("Slice axis", loc='left', fontsize=sf.SLICE_AXIS_FONT_SIZE)
            radio = sf.RadioButtons(ax, labels=list(AXIS_NAMES), active=active,
                                    radio_props={'s': [sf.SLICE_AXIS_RADIO_SIZE]})
            for label in radio.labels:
                label.set_fontsize(sf.SLICE_AXIS_LABEL_FONT_SIZE)
            self._radio_axes.append(ax)
            self.axis_radios.append(radio)
            return radio

        if self.sync_axes or self.stack.n_volumes == 1:
            radio = make_radio(self._radio_slots[0], self.stack.slice_axes[0])
            radio.on_clicked(
                lambda label: self._on_axis_selected(None, AXIS_NAMES.index(label)))
        else:
            for i in range(self.stack.n_volumes):
                radio = make_radio(self._radio_slots[i], self.stack.slice_axes[i])
                radio.on_clicked(
                    lambda label, i=i: self._on_axis_selected(i, AXIS_NAMES.index(label)))

    def _create_frame_row(self):
        # The first two columns together match the label column of the other
        # sliders, so the frame slider lines up with them.  The last column holds
        # the frame label that playback shows in place of the slider's value.
        sub = sf.gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=self.gs[4, :], width_ratios=[1.0, 0.7, 8.0, 2.0])
        self._play_ax = self.fig.add_subplot(sub[0, 0])
        self.play_button = sf.Button(self._play_ax, 'Play')
        self.play_button.label.set_fontsize(sf.STRIP_FONT_SIZE)
        self.play_button.on_clicked(lambda _event: self._toggle_play())
        self._frame_slider_ax = self.fig.add_subplot(sub[0, 2])
        self._frame_label_ax = self.fig.add_subplot(sub[0, 3])
        self._frame_label_ax.axis('off')
        self._frame_label = self._frame_label_ax.text(
            0.02, 0.5, '', va='center', fontsize=10, visible=False,
            transform=self._frame_label_ax.transAxes)
        self._update_frame_slider()

    def _make_frame_slider(self):
        self.frame_slider = sf.Slider(self._frame_slider_ax, label='t', valmin=0,
                                      valmax=self.stack.max_frames - 1,
                                      valinit=self.stack.master_frame,
                                      valstep=1, valfmt='%0.0f')
        self.frame_slider.drawon = False
        self.frame_slider.on_changed(self._on_frame_slider)

    def _update_frame_slider(self):
        """Track the longest volume's frame count: bounds, value, and visibility."""
        max_frames = self.stack.max_frames
        visible = max_frames > 1
        for ax in (self._play_ax, self._frame_slider_ax):
            ax.set_visible(visible)
        if self.roi_plot_ax is not None:
            self.roi_plot_ax.set_visible(visible)
            if visible:
                self.roi_plot_ax.set_xlim(0, max_frames - 1)
        if not visible:
            return
        if self.frame_slider is None:
            self._make_frame_slider()
            return
        self.frame_slider.valmax = max_frames - 1
        self.frame_slider.ax.set_xlim(0, max_frames - 1)
        self.frame_slider.set_val(self.stack.master_frame)

    def _create_roi_plot(self):
        # The plot spans the sliders' columns, so its frame axis lines up with the
        # frame slider.
        self.roi_plot_ax = self.fig.add_subplot(self._slider_slot(5))
        ax = self.roi_plot_ax
        ax.set_xlabel('t', fontsize=ROI_PLOT_FONT_SIZE)
        ax.set_ylabel('ROI mean', fontsize=ROI_PLOT_FONT_SIZE)
        ax.tick_params(labelsize=ROI_PLOT_FONT_SIZE)
        ax.set_xlim(0, max(self.stack.max_frames - 1, 1))
        self._roi_marker = ax.axvline(self.stack.master_frame, color='0.5', lw=1)
        self._roi_hint = ax.text(0.5, 0.5, ROI_PLOT_HINT, transform=ax.transAxes,
                                 ha='center', va='center', color='0.5',
                                 fontsize=ROI_PLOT_FONT_SIZE)
        ax.set_visible(self.stack.max_frames > 1)

    # --- The plot of the ROI mean against frame ---

    def _update_roi_plot(self):
        """Recompute the ROI mean against frame for every volume."""
        ax = self.roi_plot_ax
        for line in self._roi_lines:
            line.remove()
        self._roi_lines = []
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        circle = self.circles[0] if self.circles else None
        if circle is not None and ax.get_visible():
            x, y = circle.center
            radius = circle.get_radius()
            last_frame = self.stack.max_frames - 1
            for i in range(self.stack.n_volumes):
                means = self.stack.roi_frame_means(i, x, y, radius)
                if means is None:
                    continue
                if means.size == 1:
                    # A volume with one frame is fixed in time: a flat line.
                    frames, means = [0, last_frame], [means[0], means[0]]
                else:
                    frames = np.arange(means.size)
                label = self.stack.labels[i].strip() or 'Image {}'.format(i)
                (line,) = ax.plot(frames, means, lw=1.2, label=label)
                self._roi_lines.append(line)
        self._roi_hint.set_visible(not self._roi_lines)
        if self._roi_lines:
            ax.relim()
            ax.autoscale_view(scalex=False)
            ax.legend(fontsize=ROI_PLOT_FONT_SIZE, loc='upper left',
                      bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    def _move_roi_marker(self):
        frame = self.stack.master_frame
        self._roi_marker.set_xdata([frame, frame])

    def _remove_roi_graphics(self):
        super()._remove_roi_graphics()
        if self.roi_plot_ax is not None:
            self._update_roi_plot()

    # --- Frame slider, stepping, and playback ---

    def _on_frame_slider(self, value):
        self._pause_if_playing()
        self.stack.set_frame(value)
        # Every panel with a time axis shows the frame in its title.
        for i in range(self.stack.n_volumes):
            if self.stack.has_time(i):
                self.refresh(i)
        self._display_roi_stats()
        self._move_roi_marker()
        self.fig.canvas.draw_idle()

    def _step_frame(self, step):
        self._pause_if_playing()
        self.frame_slider.set_val(
            int(np.clip(self.stack.master_frame + step, 0, self.stack.max_frames - 1)))

    def _toggle_play(self):
        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _pause_if_playing(self):
        if self.playing:
            self._stop_playback()

    def _playback_regions(self):
        """(bbox, artists) pairs that playback redraws on every frame."""
        regions = []
        for i in range(self.stack.n_volumes):
            if not self.stack.has_time(i):
                continue
            # The ROI circle and its statistics sit above the image, so they are
            # redrawn after it.
            artists = [self.images[i]] + [artist for artist in
                                          (self.circles[i], self.stats_texts[i])
                                          if artist is not None]
            regions.append((self.axes[i].bbox, artists))
        regions.append((self._frame_label_ax.bbox, [self._frame_label]))
        if self.roi_plot_ax.get_visible():
            regions.append((self.roi_plot_ax.bbox, [self._roi_marker]))
        return regions

    def _start_playback(self):
        if self.frame_slider is None or self._dialog is not None:
            return
        self.playing = True
        self._hide_tooltips()
        self.play_button.label.set_text('Pause')
        # Playback redraws only the images, the ROI graphics above them, the frame
        # label, and the plot marker.  They are marked animated, so ordinary draws
        # leave them out, and their backgrounds are saved after one full draw.
        # Nothing else changes while the viewer plays, which is what keeps the
        # macosx backend from redrawing the whole figure on each frame.
        regions = self._playback_regions()
        for _bbox, artists in regions:
            for artist in artists:
                artist.set_animated(True)
        self.frame_slider.valtext.set_visible(False)
        self._frame_label.set_text('t = {}'.format(self.stack.master_frame))
        self._frame_label.set_visible(True)
        canvas = self.fig.canvas
        canvas.draw()
        if getattr(canvas, 'supports_blit', False):
            self._play_regions = [(bbox, canvas.copy_from_bbox(bbox), artists)
                                  for bbox, artists in regions]
        else:
            self._play_regions = None
        self._blit_playback()
        self._play_timer = canvas.new_timer(interval=max(1, int(round(1000 / self.fps))))
        self._play_timer.add_callback(self._play_step)
        self._play_timer.start()

    def _play_step(self):
        """Advance playback by one frame, looping at the end."""
        if not self.playing:
            return
        stack = self.stack
        changed = stack.set_frame((stack.master_frame + 1) % stack.max_frames)
        for i in changed:
            self.images[i].set_data(stack.slice_image(i))
        self._frame_label.set_text('t = {}'.format(stack.master_frame))
        self._move_roi_marker()
        self._blit_playback()

    def _blit_playback(self):
        canvas = self.fig.canvas
        if self._play_regions is None:
            canvas.draw_idle()
            return
        for bbox, background, artists in self._play_regions:
            canvas.restore_region(background)
            for artist in artists:
                self.fig.draw_artist(artist)
            canvas.blit(bbox)
        try:
            canvas.flush_events()
        except NotImplementedError:
            pass

    def _stop_playback(self):
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None
        self.playing = False
        if self._play_regions is not None:
            artists = [artist for _bbox, _background, region_artists in self._play_regions
                       for artist in region_artists]
        else:
            artists = [artist for _bbox, region_artists in self._playback_regions()
                       for artist in region_artists]
        for artist in artists:
            artist.set_animated(False)
        self._play_regions = None
        self._frame_label.set_visible(False)
        self.frame_slider.valtext.set_visible(True)
        self.play_button.label.set_text('Play')
        # The slider and the titles were left alone during playback, so they catch
        # up here.  Setting the slider refreshes the panels and the ROI statistics.
        self.frame_slider.set_val(self.stack.master_frame)
        self._display_roi_stats(force=True)
        self.fig.canvas.draw_idle()

    # --- Event wiring: playback stops before anything else changes the figure ---

    def _connect_events(self):
        super()._connect_events()
        canvas = self.fig.canvas
        # A resize makes the saved backgrounds the wrong size.
        canvas.mpl_connect('resize_event', lambda _event: self._pause_if_playing())
        canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, _event):
        # The window is going away, so only the timer is stopped.
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None
        self.playing = False

    def _on_button_press(self, event):
        # The Play button handles its own clicks.
        if event.inaxes is not getattr(self, '_play_ax', None):
            self._pause_if_playing()
        super()._on_button_press(event)

    def _on_motion(self, event):
        # Hover tooltips would change the figure under the playing images.
        if self.playing:
            return
        super()._on_motion(event)

    def _on_release(self, event):
        roi_changed = self.mode in (Mode.DRAW_ROI, Mode.MOVE_ROI, Mode.RESIZE_ROI)
        super()._on_release(event)
        if roi_changed:
            self._update_roi_plot()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if self._dialog is None and self.frame_slider is not None:
            if event.key == ' ':
                self._toggle_play()
                return
            if event.key in (',', '.'):
                self._step_frame(-1 if event.key == ',' else 1)
                return
        self._pause_if_playing()
        super()._on_key(event)

    def _show_message(self, show, message_type=None, message=None):
        if show and message_type == 'help':
            message = sf.multiline(
                'Left-click and drag on an image for an ROI',
                'Right-click an image for the menu',
                'Right-click the intensity slider or press Set range '
                'for exact bounds',
                'Press [space] to play or pause, [,] and [.] to step one frame',
                'Press [esc] to remove ROI/messages/dialogs',
                'Close the window to quit')
            message_type = None
        super()._show_message(show, message_type=message_type, message=message)

    def _main_widgets(self):
        widgets = super()._main_widgets()
        for widget in (self.frame_slider, self.play_button):
            if widget is not None:
                widgets.append(widget)
        return widgets

    # --- Actions that change what the ROI plot shows ---

    def _on_slice_slider(self, value):
        self._pause_if_playing()
        super()._on_slice_slider(value)
        if self._roi_lines:
            self._update_roi_plot()
            self.fig.canvas.draw_idle()

    def _on_intensity_slider(self, value):
        self._pause_if_playing()
        super()._on_intensity_slider(value)

    def _on_axis_selected(self, volume_index, axis):
        self._pause_if_playing()
        super()._on_axis_selected(volume_index, axis)
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _on_transpose_button(self, i):
        super()._on_transpose_button(i)
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _apply_difference(self, baseline_index, comparison_index, use_abs):
        super()._apply_difference(baseline_index, comparison_index, use_abs)
        self._update_frame_slider()
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _on_restore(self, i):
        super()._on_restore(i)
        self._update_frame_slider()
        self._update_roi_plot()
        self.fig.canvas.draw_idle()

    def _finish_load(self, i, path, name):
        super()._finish_load(i, path, name)
        self._update_frame_slider()
        self._update_roi_plot()
        self.fig.canvas.draw_idle()


def slice_viewer4d(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
                   slice_label=None, slice_axis=None, cmap='gray',
                   show_instructions=True, block=True, save_fn=None, fps=5):
    """Launch an interactive viewer for one or more 4D volumes.

    This function builds a :class:`SliceViewer4D`, shows it, and returns it.  The
    viewer has every feature of :func:`slice_viewer`, and adds a frame slider, a Play
    button, and a plot of the ROI mean against frame.

    Args:
        *datasets (ndarray or None): One or more 2D, 3D, or 4D arrays to display.  A 4D
            array is ``(t, x, y, z)``.  A 2D or 3D array stays fixed in time.
        data_dicts (None, dict, or list of dict/None, optional): String-valued dict(s)
            associated with the volumes, viewable in the viewer.
        title (str, optional): Figure title.  Defaults to ''.
        vmin (float, optional): Minimum display intensity.  Defaults to the minimum
            over every voxel of every volume.
        vmax (float, optional): Maximum display intensity.  Defaults to the maximum
            over every voxel of every volume.
        slice_label (str or list of str, optional): Label(s) at the start of each
            panel title.  Defaults to no label.
        slice_axis (int or list of int, optional): Axis to slice along, counted in each
            array's own axes: 1, 2, or 3 for a 4D array and 0, 1, or 2 for a 3D array.
            Defaults to z.  With 3D and 4D arrays together, give a list.
        cmap (str, optional): Colormap.  Defaults to 'gray'.
        show_instructions (bool, optional): Show the "Press h for help" hint.
            Defaults to True.
        block (bool, optional): If True (default), block until the window is closed.
            If False, leave the window open and return immediately, as in
            :func:`slice_viewer`.
        save_fn (callable, optional): Replacement for the built-in HDF5 writer, called
            as ``save_fn(file_path, array, array_name, attributes_dict)``.
        fps (float, optional): Playback speed in frames per second.  Defaults to 5.

    Returns:
        SliceViewer4D: the viewer object.
    """
    viewer = SliceViewer4D(*datasets, data_dicts=data_dicts, title=title,
                           vmin=vmin, vmax=vmax, slice_label=slice_label,
                           slice_axis=slice_axis, cmap=cmap,
                           show_instructions=show_instructions, save_fn=save_fn,
                           fps=fps)
    viewer.show(block=block)
    # Nonblocking viewers join the slice viewer's registry, so a later blocking call
    # of either viewer adopts and closes them.
    if not block:
        sf._NONBLOCKING_VIEWERS.append(viewer)
        return viewer
    for nonblocking_viewer in sf._NONBLOCKING_VIEWERS:
        sf.plt.close(nonblocking_viewer.fig)
    sf._NONBLOCKING_VIEWERS.clear()
    if matplotlib.get_backend() == 'TkAgg':
        import gc
        gc.collect()
    return viewer
