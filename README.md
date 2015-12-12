# ChaosFunctions
Collection of functions related with dynamical systems

The main utilities of this package are

* Create chaotic iterations
* Build fractals in complex plane

# Installation

```Bash
git clone https://github.com/tgquintela/ChaosFunctions
.\install.sh
```

Required extra packages
* pythonUtils
```Bash
git clone https://github.com/tgquintela/pythonUtils
```


# Examples

* Chaotic series

```python
from chaotic_evolution import Iterator
iter_f = lambda x: x/2 if x%2==0 else 3*x+1
stop_f = lambda x: x[-1] == 1
it = Iterator(iter_f, stop_f)
sequence = it.iterate_sequence(124)
fig = plot_sequence(sequence)
```

* Build a fractal

```python
import numpy as np
from ChaosFunctions.fractals import FractalBuilder, mandelbrot_iter
ns, itermax, limits = (10000, 10000), 400, np.array([[-2, .5], [-1.25, 1.25]])
fb = FractalBuilder(mandelbrot_iter)
img = fb.build_fractal(ns, limits, itermax)
fig = image_building(img, 'prueba')
```


