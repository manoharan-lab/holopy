# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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

import glob, os, setuptools, shutil, site, sys
from os.path import join

import nose
from numpy.distutils.core import setup, Extension
from setuptools.command.develop import develop
from setuptools.command.install import install

try:
    from holopy import __version__
except ImportError:
    __version__ = 'unknown'
HOLOPY_NOSE_PLUGIN_LOCATION = ('holopycatchwarnings = '
                               'holopy.core.tests.common:HoloPyCatchWarnings')

hp_root = os.path.dirname(os.path.realpath(sys.argv[0]))

class PostDevelopConfig(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        if os.name == 'nt': _move_S_msvc_libs('develop')


class PostInstallConfig(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        if os.name == 'nt': _move_msvc_libs('install')


def configuration(parent_package='',top_path=''):
    # this will automatically build the scattering extensions, using the
    # setup.py files located in their subdirectories
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None,parent_package,top_path)

    pkglist=setuptools.find_packages(hp_root)
    for i in pkglist:
        config.add_subpackage(i)

    config.add_data_files(['.',[join(hp_root,'AUTHORS')]])
    config.add_data_files(join(hp_root,'holopy','scattering','tests','gold','full_data','*.h5'))
    config.add_data_files(join(hp_root,'holopy','scattering','tests','gold','*.yaml'))
    config.add_data_files(join(hp_root,'holopy','core','tests','exampledata','*.h5'))
    config.add_data_files(join(hp_root,'holopy','core','tests','exampledata','*.jpg'))
    config.add_data_files(join(hp_root,'holopy','propagation','tests','gold','full_data','*.h5'))
    config.add_data_files(join(hp_root,'holopy','propagation','tests','gold','*.yaml'))

    return config

def _move_msvc_libs(mode='install'):
    """ These dlls need to be moved if the fortran is complied in an environment
    with MSVC 2015 as the C compiler.
    """
    package_dir = _get_holopy_install_dir(mode)
    lib_dir = os.path.join(package_dir, '.libs')
    sep = os.path.sep

    tmatrix_libs = glob.glob(lib_dir + sep + 'libS.*.dll')
    mie_libs = glob.glob(lib_dir + sep + 'libscsm*.dll')
    mie_libs += glob.glob(lib_dir + sep + 'libmieang*.dll')
    mie_libs += glob.glob(lib_dir + sep + 'libuts*.dll')

    tmatrix_f_dir = os.path.join(package_dir, 'scattering', 'theory', 'tmatrix_f')
    mie_f_dir = os.path.join(package_dir, 'scattering', 'theory', 'mie_f')

    for dll in tmatrix_libs: shutil.move(dll, tmatrix_f_dir)
    for dll in mie_libs: shutil.move(dll, mie_f_dir)


def _get_holopy_install_dir(mode):
    if mode == 'install':
        sitepackages = list(site.getsitepackages())
        dir = [path for path in site.getsitepackages() if 'site-packages' in path]
        assert len(dir) == 1
        dir = os.path.join(dir[0], 'holopy')
    if mode == 'develop':
        dir = os.path.dirname(os.path.realpath(__file__))
        dir = os.path.join(dir, 'holopy')
    return dir

if __name__ == "__main__":
    requires=[l for l in open(os.path.join(hp_root,"requirements.txt")).readlines() if l[0] != '#']

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
          entry_points = {'nose.plugins.0.10': HOLOPY_NOSE_PLUGIN_LOCATION},
          package=['HoloPy'],
          cmdclass={'develop': PostDevelopConfig,
                    'install': PostInstallConfig})
