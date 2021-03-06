#!/usr/bin/env/python

"""
Installation script
"""


import sys
import os
import warnings

## Definition of useful functions
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def write_version_py(filename=None):
    cnt = """\
version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'ChaosFunctions', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (VERSION))
    finally:
        a.close()


## Check problems with the setuptools
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

## Quantify the version
MAJOR = 0
MINOR = 0
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

write_version_py()

## Setup
setup(name='ChaosFunctions',
      version=VERSION,
      description='Utils for chaos and fractals studies in python.',
      license='BSD',
      author='T. Gonzalez Quintela',
      author_email='tgq.spm@gmail.com',
      url='',
      long_description=read('README.md'),
      packages=[],
      install_requires=['numpy', 'scipy', 'matplotlib']#, 'pandas',
			#'datetime', #'python-mpltoolkits.basemap'
			#'python-dateutil'],
)

