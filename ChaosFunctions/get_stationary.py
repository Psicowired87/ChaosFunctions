
"""This module contains the tools needed to extract the stationary points of
a dynamics and to stop when it is considered enough.
"""

import numpy as np

seq = logistic_map_bif_diagram(np.linspace(0,3,31), stationary_fixed_points)
seq = logistic_map_bif_diagram(np.linspace(1,3.2,23), stationary_fixed_points)


def logistic_map_bif_diagram(range_par, stop_f):
    iter_f = lambda r: lambda x: r*x*(1-x)
    sequence = obtain_bifurcation_diagram(iter_f, range_par, stop_f)
    sequence = np.array(sequence)
    sequence = sequence.reshape((sequence.reshape(-1).shape[0]/2, 2))
    return sequence


def obtain_bifurcation_diagram(iter_f, range_par, stop_f):

    fixedp = []
    for par in range_par:
        print par
        p0 = np.random.random()
        iter_ff = iter_f(par)
        sequence, fixed_points = generic_iteration_4_fixed_points(p0,
                                                                  iter_ff,
                                                                  stop_f)
        fixedp.append([[par, fp] for fp in fixed_points])

    return fixedp


def generic_iteration_4_fixed_points(p0, iter_f, stop_f_and_fixedp):
    """This functions implements a generic iterations. Repeat the given funcion
    while the stopping condition is not fulfilled.

    Parameters
    ---------
    p0 : float
        intial point of the iteration
    iter_f: function
        function which receives a number and return a number. Decides the next
        state of the system.
    stop_f_and_fixedp: function
        function which receives a list of numbers and return a boolean and a
        fixed points. Decides the stoping condition.

    """

    sequence = []
    fixed_points = None
    p = p0
    complete = False
    while not complete:
        sequence.append(p)
        # Stop clause
        complete, fixed_points = stop_f_and_fixedp(np.array(sequence))
        # Transformation
        p = iter_f(p)

    sequence = np.array(sequence)
    return sequence, fixed_points


def stationary_fixed_points(history):

    stationary = False
    n_limit = int(np.sqrt(history.shape[0]))
    fixed_points = np.array([])
    if n_limit > 100:
        return True, fixed_points
    for order in range(1,n_limit+1):
        s = embedding_matrix(history, order)
        stationary = decision_stationarity(s)
        if stationary:
            fixed_points = s[-1, :]
            break
    return stationary, fixed_points


def decision_stationarity(seq):
    if seq.shape[0] <= 100:
        decision = False
    else:
        decision = np.all(np.std(seq[-100:, ]) < 0.01)
    return decision


def embedding_matrix(seq, order):

    embeded_m = sliding_embeded_transf(seq, 1, order, order)
    embeded_m = embeded_m[embeded_m[:, 0] != 0, :]
    return embeded_m


def sliding_embeded_transf(X, tau, D,  step=1, f=lambda x: x):
    """Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension D. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    Parameters
    ----------
    X : array_like, shape(N,)
        a time series
    tau : int
        the lag or delay when building embedding sequence
    D : integer
        the embedding dimension
    step: int
        the step for which we compute the sequence.
    f: function
        transformation function to be applied to each element of the sequence.

    Returns
    -------
    Y : 2-D list
        embedding matrix built

    """

    N = X.shape[0]

    # Check inputs
    if D * tau > N:
        message = "Cannot build such a matrix, because D * tau > N"
        raise Exception(message)
    if tau < 1:
        message = "Tau has to be at least 1"
        raise Exception(message)

    Y = np.zeros((N - (D - 1) * tau, D))
    for i in xrange(0, N - (D - 1) * tau, step):
        for j in xrange(0, D):
            Y[i][j] = f(X[i + j * tau])
    return Y
