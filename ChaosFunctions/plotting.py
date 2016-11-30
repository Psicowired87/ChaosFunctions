
"""
plotting
--------
module which contains the plotting functions and utils

"""

import matplotlib.pyplot as plt


def plot_iteration(sequence):
    """The plot iteration of a sequence.

    Parameters
    ----------
    sequence: np.ndarray
        the sequence information.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure object which contains the plot.

    """
    fig = plt.figure(True)
    plt.plot(sequence, '.')
    return fig


def image_building(I, name=None):
    """Image building.

    Parameters
    ----------
    I: np.ndarray
        the image building.
    name: str (default=None)
        the name we want to give to the file saved.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        the figure object which contains the plot.

    """
    I[I == 0] = I.max() + 1
    fig = plt.figure()
    img = plt.imshow(I.T, origin='lower left')
    if name is not None:
        img.write_png(name+'.png', noscale=True)
    return fig
