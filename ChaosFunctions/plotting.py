
"""
plotting
--------
module which contains the plotting functions and utils

"""

import matplotlib.pyplot as plt


def plot_iteration(sequence):
    fig = plt.figure(True)
    plt.plot(sequence, '.')
    return fig


def image_building(I, name):
    I[I == 0] = I.max() + 1
    fig = plt.figure()
    img = plt.imshow(I.T, origin='lower left')
    if name is not None:
        img.write_png(name+'.png', noscale=True)
    return fig
