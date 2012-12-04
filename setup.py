#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name = 'pysurvival',
      version = '1.1',
      description = 'Cox model from R.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = '',
      packages = ['pysurvival'],
      package_dir = {'pysurvival': 'pysurvival'},
      requires = ['numpy'],
     )
