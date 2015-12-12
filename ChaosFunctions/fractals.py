
"""
Fractals
--------
Fractals module which groups all the fractals functions and classes.


TODO
----

"""

import numpy as np
import matplotlib.pyplot as plt
from pythonUtils.parallel_tools import distribute_tasks, reshape_limits
from plotting import image_building


class FractalBuilder:
    """Object wihch is able to build a fractal from a iteration in the complex
    plane.

    Example
    -------
    >>> import numpy as np
    >>> from ChaosFunctions.fractals import FractalBuilder, mandelbrot_iter
    >>> ns = 10000, 10000
    >>> itermax = 400
    >>> limits = np.array([[-2, .5], [-1.25, 1.25]])
    >>> fb = FractalBuilder(mandelbrot_iter)
    >>> img = fb.build_fractal(ns, limits, itermax)
    >>> fig = image_building(img, 'prueba')

    """

    def __init__(self, iteration):
        self.iter = iteration

    def build_fractal(self, ns, limits, itermax, memlim=None):
        img = iterator2d(self.iter, ns, limits, itermax, memlim=None)
        return img

    def export_fractal(self, img, namefile):
        fig = image_building(img, namefile)
        return fig

    def example_mandelbrot(self, ns=None, limits=None, itermax=100,
                           memlim=None):
        ## 0. Set initial variables
        ns = (1000, 1000) if ns is None else ns
        if limits is None:
            limits = np.array([[-2, .5], [-1.25, 1.25]])
        itermax = 100 if itermax is None else itermax
        ## 1. Build fractal
        img = iterator2d(mandelbrot_iter, ns, limits, itermax, 1)
        return img


def iterator2d(f, ns, limits, itermax, memlim=None):
    """Function to iterate function f in the space determined by limits and ns.
    The maximum number of iterations it is bounded with itermax.
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
    n, m = ns
    xmin, xmax = limits[0, 0], limits[0, 1]
    ymin, ymax = limits[1, 0], limits[1, 1]
    ix, iy = np.mgrid[0:n, 0:m]
    x = np.linspace(xmin, xmax, n)[ix]
    y = np.linspace(ymin, ymax, m)[iy]
    c = x+complex(0, 1)*y
    del x, y
    ## 1. Computation of the image
    img = np.zeros(c.shape, dtype=int)
    ix.shape = n*m
    iy.shape = n*m
    c.shape = n*m
    ## 2. Computation of the image
    z = np.copy(c)
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
    np.multiply(z, z, z)
    np.add(z, c, z)
    # these are the points that have escaped
    rem = np.abs(z) > 2.0
    return z, rem

