
"""
Example: Julia
-------
>>> import numpy as np
>>> from ChaosFunctions.fractals import FractalBuilder, julia_iter,\
>>>     p0_julia
>>> ns, itermax, c = (1000, 1000), 100, complex(-0.768662, 0.0930477)
>>> limits = np.array([[-2., 2.], [-2., 2.]])
>>> fb = FractalBuilder(julia_iter, lambda x, y: p0_julia(x, y, c))
>>> img = fb.build_fractal(ns, limits, itermax)
>>> fig = fb.export_fractal(img, 'prueba')
"""
