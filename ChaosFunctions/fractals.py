
"""
Fractals
--------
Fractals module which groups all the fractals functions and classes.


TODO
----

"""

import numpy as np
from pythonUtils.parallel_tools import distribute_tasks, reshape_limits
from plotting import image_building


class FractalBuilder:
    """Object wihch is able to build a fractal from a iteration in the complex
    plane.

    Example
    -------
    >>> import numpy as np
    >>> from ChaosFunctions.fractals import FractalBuilder, mandelbrot_iter,\
    >>>     p0_mandelbrot
    >>> ns = 1000, 1000
    >>> itermax = 100
    >>> limits = np.array([[-2, .5], [-1.25, 1.25]])
    >>> fb = FractalBuilder(mandelbrot_iter, p0_mandelbrot)
    >>> img = fb.build_fractal(ns, limits, itermax)
    >>> fig = fb.export_fractal(img, 'prueba')

    """

    def __init__(self, iteration, p0_init):
        """Instatiation of the fractal builder.

        Parameters
        ----------
        iteration: function
            the function which iterates from ones state to the next.
        p0_init: function
            the function which creates the initial state.

        """
        self.iter, self.p0_init = iteration, p0_init

    def build_fractal(self, ns, limits, itermax, memlim=None):
        """Main function to build the fractal.

        Parameters
        ----------
        ns: tuple
            ths size of the image.
        limits: np.ndarray
            the matrix of the limits.
        itermax: int
            the maximum number of iterations.
        memlim: int (default=None)
            the information about the limits of memory. It splits the
            computation task in parts in order to save RAM memory.

        Returns
        -------
        img: np.ndarray
            the final 2d state image.

        """
        img = iterator2d(self.iter, self.p0_init, ns, limits, itermax,
                         memlim=None)
        return img

    def export_fractal(self, img, namefile):
        """Export the fractal to a image file.

        Parameters
        ----------
        img: np.ndarray
            the final 2d state image.
        namefile: str
            the name of the file we want to export the image of the fractal.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            the figure object which contains the plot.

        """
        fig = image_building(img, namefile)
        return fig

    def example_mandelbrot(self, ns=None, limits=None, itermax=100,
                           memlim=None):
        """Specific example of the mandelbrot iteration.

        Parameters
        ----------
        ns: tuple (default=None)
            ths size of the image.
        limits: np.ndarray (default=None)
            the matrix of the limits.
        itermax: int (default=100)
            the maximum number of iterations.
        memlim: int (default=None)
            the information about the limits of memory. It splits the
            computation task in parts in order to save RAM memory.

        Returns
        -------
        img: np.ndarray
            the final 2d state image.

        """
        ## 0. Set initial variables
        ns = (1000, 1000) if ns is None else ns
        if limits is None:
            limits = np.array([[-2, .5], [-1.25, 1.25]])
        itermax = 100 if itermax is None else itermax
        ## 1. Build fractal
        img = iterator2d(mandelbrot_iter, p0_mandelbrot, ns, limits,
                         itermax, 1)
        return img


def iterator2d(f, p0_init, ns, limits, itermax, memlim=None):
    """Function to iterate function f in the space determined by limits and ns.
    The maximum number of iterations it is bounded with itermax.

    Parameters
    ----------
    f: function
        the iteration function. 2 parameters, the state and a parameter.
    p0_init: function
        generator of the initial state from which we start the iteration.
    ns: tuple
        ths size of the image.
    limits: np.ndarray
        the matrix of the limits.
    itermax: int
        the maximum number of iterations.
    memlim: int (default=None)
        the information about the limits of memory. It splits the computation
        task in parts in order to save RAM memory.

    Returns
    -------
    img: np.ndarray
        the final 2d state image.

    """
    ## 00. Fragmentation of computation
    if memlim is not None:
        lims = distribute_tasks(ns[0], memlim)
        new_limits = reshape_limits(lims, [limits[0, 0], limits[0, 1]])
        img = np.zeros(ns)
        for i in xrange(len(lims)):
            n_limits = np.array([[new_limits[i][0], new_limits[i][1]],
                                [limits[1, 0], limits[1, 1]]])
            ms = lims[i][1]-lims[i][0]
            img[lims[i][0]:lims[i][1], :] = iterator2d(f, (ms, ns[1]),
                                                       n_limits, itermax)
        return img

    ## 0. Creation of the c complex number vectors
    c, z, ix, iy = p0_init(ns, limits)
    ## 1. Computation of the image
    img = np.zeros(ns, dtype=int)
    for i in xrange(itermax):
        if not len(z):
            break
        z, rem = f(z, c)
        # Add one for the points which have escaped
        img[ix[rem], iy[rem]] = i+1
        # Only use the points which haven't escaped
        rem = -rem
        z = z[rem]
        ix, iy = ix[rem], iy[rem]
        c = c[rem]
    return img


def mandelbrot_iter(z, c):
    """Mandelbrot iteration.

    Parameters
    ----------
    z: np.ndarray
        the state of the system to iterate.
    c: np.ndarray
        the possible complex numbers.

    Returns
    -------
    z: np.ndarray
        the iterated state of the system.
    rem: boolean np.ndarray
        the indices of the escaped points.

    """
    np.multiply(z, z, z)
    np.add(z, c, z)
    # these are the points that have escaped
    rem = np.abs(z) > 2.0
    return z, rem


def julia_iter(z, c):
    """Julia iteration.

    Parameters
    ----------
    z: np.ndarray
        the state of the system to iterate.
    c: np.ndarray
        the possible complex numbers.

    Returns
    -------
    z: np.ndarray
        the iterated state of the system.
    rem: boolean np.ndarray
        the indices of the escaped points.

    """
    z = (z*z) + c
    rem = np.abs(z) > 4.0
    return z, rem


def p0_mandelbrot(ns, limits):
    """Creation of the initial state for the mandelbrot set.

    Parameters
    ----------
    ns: tuple
        ths size of the image.
    limits: np.ndarray
        the matrix of the limits.

    Returns
    -------
    c: np.ndarray
        the possible complex numbers.
    z: np.ndarray
        the state of the system to iterate.
    ix: np.ndarray
        the possible x complex elements.
    iy: np.ndarray
        the possible y complex elements.

    """
    n, m = ns
    xmin, xmax = limits[0, 0], limits[0, 1]
    ymin, ymax = limits[1, 0], limits[1, 1]
    ix, iy = np.mgrid[0:n, 0:m]
    x = np.linspace(xmin, xmax, n)[ix]
    y = np.linspace(ymin, ymax, m)[iy]
    c = x+complex(0, 1)*y
    del x, y
    ix.shape = n*m
    iy.shape = n*m
    c.shape = n*m
    z = np.copy(c)
    return c, z, ix, iy


def p0_julia(ns, limits, parameter=0):
    """Creation of the initial state for the julia set.

    Parameters
    ----------
    ns: tuple
        ths size of the image.
    limits: np.ndarray
        the matrix of the limits.

    Returns
    -------
    c: np.ndarray
        the possible complex numbers.
    z: np.ndarray
        the state of the system to iterate.
    ix: np.ndarray
        the possible x complex elements.
    iy: np.ndarray
        the possible y complex elements.

    """
    n, m = ns
    xmin, xmax = limits[0, 0], limits[0, 1]
    ymin, ymax = limits[1, 0], limits[1, 1]
    ix, iy = np.mgrid[0:n, 0:m]
    x = np.linspace(xmin, xmax, n)[ix]
    y = np.linspace(ymin, ymax, m)[iy]
    z = x+complex(0, 1)*y
    del x, y
    ix.shape = n*m
    iy.shape = n*m
    z.shape = n*m
    c = np.ones(n*m) * parameter
    return c, z, ix, iy
