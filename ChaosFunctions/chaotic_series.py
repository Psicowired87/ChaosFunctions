
"""
Chaotic series

"""

from plotting import plot_iteration
from generic_iteration import generic_iteration


class Iterator:
    """Chaotic iterator.
    """

    def __init(self, iter_f, stop_f):
        self.iter_f, self.stop_f = iter_f, stop_f

    def iterate_sequence(self, p0):
        sequence = generic_iteration(p0, self.iter_f, self.stop_f)
        return sequence

    def plot_sequence(self, sequence):
        fig = plot_iteration(sequence)
        return fig
