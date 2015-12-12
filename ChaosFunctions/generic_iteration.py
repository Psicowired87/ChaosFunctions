
"""This module contains a generic way to implement an iteration procedure and
other known interations using generic iteration function.
"""

import numpy as np


def generic_iteration(p0, iter_f, stop_f):
    """This functions implements a generic iterations. Repeat the given funcion
    while the stopping condition is not fulfilled.

    Parameters
    ---------
    p0 : float
        intial point of the iteration
    iter_f: function
        function which receives a number and return a number. Decides the next
        state of the system.
    stop_f: function
        function which receives a list of numbers and return a boolean.
        Decides the stoping condition.

    TODO
    ----
    Processes with memory.
    """

    sequence = []
    p = p0
    complete = False
    while not complete:
        sequence.append(p)
        # Stop clause
        complete = stop_f(sequence)
        # Transformation
        p = iter_f(p)

    sequence = np.array(sequence)
    return sequence


def hasse_collatz(n):
    """Collatz conjeture is a conjeture that determine that the algorithm known
    as Hasse's algorithm always ends in 1.

    Parameters
    ----------
    n : int
        natural number

    Returns
    -------
    sequence: array_like
        path over the natural numbers walked by the algorithm.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Collatz_conjecture

    """

    iter_f = lambda x: x/2 if x%2==0 else 3*x+1
    stop_f = lambda x: x[-1] == 1
    sequence = generic_iteration(n, iter_f, stop_f)
    return sequence


def iterator_x2minus1(p0, n):
    """Stable two cycle attractor.
    """
    iter_f = lambda x: x**2-1
    stop_f = lambda x: x[-1] == -1 and len(x) > n
    sequence = generic_iteration(p0, iter_f, stop_f)
    return sequence
    

def iterator_2xminus1(p0, n):
    """Chaotic iteration.
    """
    iter_f = lambda x: 2*x**2-1
    stop_f = lambda x: len(x) == n+1
    sequence = generic_iteration(p0, iter_f, stop_f)
    return sequence


def logistic_map(p0, n, r):
    iter_f = lambda x: r*x*(1-x)
    stop_f = lambda x: len(x) == n+1
    sequence = generic_iteration(p0, iter_f, stop_f)
    return sequence


## Reformulate: two array iterations for the division
def division_iter(p0, n, div):
    i_cond = lambda y: 10*y[0] if y[0] < div else y[0]
    iter_f = lambda x: np.array([int(i_cond(x))%int(div), int(i_cond(x))/int(div)])
    stop_f = lambda x: x[-1][0] == 0 or len(x) == n+1
    sequence = generic_iteration(p0, iter_f, stop_f)[:, 1]
    return sequence

