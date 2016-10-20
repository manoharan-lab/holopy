# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.

# Copyright 2013, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.

'''
setup.py

uses numpy.distutils instead of standard distutils so that we can
build fortran extensions

Notes:

The t-matrix and mie scattering codes require working Fortran 90 and
C compilers, as well as f2py and cython. On Ubuntu, you will need the
"gfortran" and "python-dev" packages installed.
'''

import setuptools
from numpy.distutils.core import setup, Extension


# this will automatically build the scattering extensions, using the
# setup.py files located in their subdirectories
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)

    pkglist=setuptools.find_packages()    
    for i in pkglist:
        config.add_subpackage(i)    
    
    config.add_data_files(['.',['AUTHORS']])
    config.add_data_files('holopy/propagation/tests/gold/full_data/*.h5')
    config.add_data_files('holopy/scattering/tests/gold/full_data/*.h5')
    config.add_data_files('holopy/core/tests/exampledata/*.h5')
    config.add_data_files('holopy/core/tests/exampledata/*.jpg')

    config.get_version()
    return config

__version__ = 'unknown'
try:
    from holopy import __version__
except ImportError:
    # no version specified, or file got deleted in bzr
    pass

if __name__ == "__main__":
    setup(configuration=configuration,
          name='HoloPy',
          version=__version__,
          description='Holography in Python',
          requires=['numpy', 'scipy', 'PyYAML', 'pillow','pandas','h5py','emcee','matplotlib','xarray','h5netcdf'],
          author='Manoharan Lab, Harvard University',
          author_email='vnm@seas.harvard.edu',
          url='http://manoharan.seas.harvard.edu/holopy',
          license='GNU GPL',
          package=['HoloPy'])
