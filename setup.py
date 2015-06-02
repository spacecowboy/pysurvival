#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name = 'pysurvival',
      version = '1.2',
      description = 'Cox model from R.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = '',
      packages = ['pysurvival'],
      package_dir = {'pysurvival': 'pysurvival'},
      install_requires = ['numpy>=1.7', 'rpy2', 'lifelines>=0.6'],
     )
