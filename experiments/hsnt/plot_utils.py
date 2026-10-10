import warnings

import matplotlib.pyplot as plt


def _save_figure(filename):
    """Save the current figure to filename, warning instead of failing if it cannot be written."""
    try:
        plt.savefig(filename, dpi=100)
    except Exception:
        warnings.warn("Can't write to file.")


def plot_images(images, titles=None, vmin=None, vmax=None, cb_orientation='vertical', filename=None):
    """
    Function to display and save multiple 2D arrays as images.

    Args:
        images(list): list of 2D numpy arrays to display
        titles(list,optional): titles for the images
        vmin(float,optional): value mapped to black
        vmax(float,optional): value mapped to white
        cb_orientation(str,optional): orientation of the color bar (must be 'horizontal' or 'vertical')
        filename(str,optional): path to save the image
    """
    num_images = len(images)

    if titles is None:
        titles = [f'Image: {i}' for i in range(num_images)]

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=20)
    fig = plt.figure(figsize=(10 * num_images, 10), dpi=160 / num_images)

    for idx, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1, num_images, idx + 1)
        ax.set_title(title)
        img = ax.imshow(image, vmin=vmin, vmax=vmax, cmap='gray')
        fig.colorbar(img, orientation=cb_orientation)

    if filename is not None:
        _save_figure(filename)


def plot_spectra(spectra, labels=None, title=None, x_label=None, y_label=None, x_lim=None, y_lim=None, wavelengths=None,
                 filename=None):
    """
    Function to display and save multiple spectra in one plot.

    Args:
        spectra(list): list of spectra to display
        labels(list,optional): labels for different spectra
        title(str,optional): title for the image
        x_label(str,optional): X axis label
        y_label(str,optional): Y axis label
        x_lim(tuple,optional): (x_min, x_max) to set x-axis display range
        y_lim(tuple,optional): (y_min, y_max) to set y-axis display range
        wavelengths(list,optional): list of wavelength values for the spectra
        filename(str,optional): path to save the image
    """
    num_spectra = len(spectra)

    if labels is None:
        labels = [f'Spectrum: {i + 1}' for i in range(num_spectra)]

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=20)
    plt.figure(figsize=(12, 10), dpi=80)

    for i, (spectrum, label) in enumerate(zip(spectra, labels)):
        if wavelengths is None:
            plt.plot(spectrum, label=label, zorder=i + 1)
        else:
            plt.plot(wavelengths, spectrum, label=label, zorder=i + 1)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend(loc='upper left')

    if filename is not None:
        _save_figure(filename)
