# Copyright 2011-2019, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
# Ronald Alexander
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

from os.path import join, dirname, realpath
import setuptools
import sys

import nose
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

from post_install import PostDevelopCommand, PostInstallCommand

try:
    from holopy import __version__
except ImportError:
    __version__ = 'unknown'

HOLOPY_NOSE_PLUGIN_LOCATION = ('holopycatchwarnings = '
                               'holopy.core.tests.common:HoloPyCatchWarnings')

hp_root = dirname(realpath(sys.argv[0]))


def configuration(parent_package='', top_path=''):
    # this will automatically build the scattering extensions, using the
    # setup.py files located in their subdirectories
    config = Configuration(None, parent_package, top_path)

    pkglist = setuptools.find_packages(hp_root)
    for i in pkglist:
        config.add_subpackage(i)

    config.add_data_files(['.', [join(hp_root, 'AUTHORS')]])
    config.add_data_files(join(hp_root, 'holopy', 'scattering', 'tests',
                               'gold', 'full_data', '*.h5'))
    config.add_data_files(join(hp_root, 'holopy', 'scattering', 'tests',
                               'gold', '*.yaml'))
    config.add_data_files(join(hp_root, 'holopy', 'core', 'tests',
                               'exampledata', '*.h5'))
    config.add_data_files(join(hp_root, 'holopy', 'core', 'tests',
                               'exampledata', '*.jpg'))
    config.add_data_files(join(hp_root, 'holopy', 'propagation', 'tests',
                               'gold', 'full_data', '*.h5'))
    config.add_data_files(join(hp_root, 'holopy', 'propagation', 'tests',
                               'gold', '*.yaml'))
    return config


if __name__ == "__main__":
    requires = [l for l in
                open(join(hp_root, "requirements.txt")).readlines()
                if l[0] != '#']

    tests_require = ['memory_profiler']

    setup(configuration=configuration,
          name='HoloPy',
          version=__version__,
          description='Holography in Python',
          install_requires=requires,
          tests_require=tests_require,
          author='Manoharan Lab, Harvard University',
          author_email='vnm@seas.harvard.edu',
          url='http://manoharan.seas.harvard.edu/holopy',
          license='GNU GPL',
          test_suite='nose.collector',
          entry_points={'nose.plugins.0.10': HOLOPY_NOSE_PLUGIN_LOCATION},
          package=['HoloPy'],
          cmdclass={'develop': PostDevelopCommand,
                    'install': PostInstallCommand})
