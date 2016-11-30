
"""
Chaotic series

"""

from plotting import plot_iteration
from generic_iteration import generic_iteration


class Iterator:
    """Iterator object to compute iterative processes or magnitudes.
    """

    def __init(self, iter_f, stop_f):
        """Instantiation of the iteration.

        Parameters
        ----------
        iter_f: function
            the iteration function.
        stop_f: function
            the conditions to stop the iteration.

        """
        self.iter_f, self.stop_f = iter_f, stop_f

    def iterate_sequence(self, p0):
        """Comput the iteration from the initial point given.

        Parameters
        ----------
        p0: optional
            initial point to start the iteration.

        Returns
        -------
        sequence: np.ndarray
            the sequence information.

        """
        sequence = generic_iteration(p0, self.iter_f, self.stop_f)
        return sequence

    def plot_sequence(self, sequence):
        """Plot a 1d sequence.

        Parameters
        ----------
        sequence: np.ndarray
            the sequence information.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            the figure object which contains the plot.

        """
        fig = plot_iteration(sequence)
        return fig
