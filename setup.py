#!/usr/bin/env python

# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.

'''
setup.py

uses numpy.distutils instead of standard distutils so that we can
build fortran extensions

Notes:

The t-matrix and mie scattering codes require working Fortran 90 and
C compilers, as well as f2py and cython. On Ubuntu, you will need the
"gfortran" and "python-dev" packages installed.
'''

from numpy.distutils.core import setup, Extension
import numpy as np


# this will automatically build the scattering extensions, using the
# setup.py files located in their subdirectories
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)
    config.add_subpackage('holopy')
    config.add_subpackage('holopy.analyze')
    config.add_subpackage('holopy.io')
    config.add_subpackage('scatterpy')
    config.add_subpackage('scatterpy.theory')
    config.add_subpackage('scatterpy.theory.mie_f')
    config.add_subpackage('scatterpy.scatterer')
    config.add_subpackage('holopy.process')
    config.add_subpackage('holopy.utility')
    config.add_subpackage('holopy.tests')
    config.add_subpackage('holopy.third_party')
    config.add_scripts('./holopy/bin/fit')
    config.add_data_files(['.',['AUTHORS']])
    config.add_data_dir('./holopy/tests')
    
    config.get_version()
    return config

__version__ = 'unknown'
try:
    from holopy._version import __version__
except ImportError:
    # no version specified, or file got deleted in bzr
    pass

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration,
          name='holopy',
          version=__version__,
          description='Holography in Python',
          author='Manoharan Lab, Harvard University',
          author_email='vnm@seas.harvard.edu',
          url='http://manoharan.seas.harvard.edu/',
          package=['holopy', 'holopy.io'])
