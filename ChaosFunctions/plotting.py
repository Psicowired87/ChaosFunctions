
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
