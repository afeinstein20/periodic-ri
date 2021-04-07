#!/usr/bin/env python  
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
from setuptools import setup

sys.path.insert(0, "periodic-ri")
from version import __version__


long_description = \
    """
A python implementation of randomization inference to detect periodic
signals in astronomy data sets.

"""

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='', # we need a name
      version=__version__,
      description='Detecting periodic signals in astronomy data', 
      packages=['periodic-ri'],
      install_requires=install_requires,
      author='Adina Feinstein',
      author_email='adina.d.feinstein@gmail.com',
      license='MIT',
      long_description = long_description,
      url='https://github.com/afeinstein20/periodic-ri',
      package_data={'': ['README.md', 'LICENSE']},
      include_package_data=True,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
      )
