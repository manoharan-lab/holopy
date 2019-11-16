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
post_install.py

Configuration for the post install of holopy
'''
import glob
import os
from os.path import join
import shutil
import site

from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopConfig(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        if os.name == 'nt':
            _move_msvc_libs('develop')


class PostInstallConfig(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        if os.name == 'nt':
            _move_msvc_libs('install')


def _move_msvc_libs(mode='install'):
    """ These dlls need to be moved if the fortran is complied in an
    environment with MSVC 2015 as the C compiler.
    """
    package_dir = _get_holopy_install_dir(mode)
    lib_dir = join(package_dir, '.libs')

    tmatrix_libs = glob.glob(join(lib_dir, 'libS.*.dll'))
    mie_libs = glob.glob(join(lib_dir, 'libscsm*.dll'))
    mie_libs += glob.glob(join(lib_dir, 'libmieang*.dll'))
    mie_libs += glob.glob(join(lib_dir, 'libuts*.dll'))

    tmatrix_f_dir = join(package_dir, 'scattering', 'theory', 'tmatrix_f')
    mie_f_dir = join(package_dir, 'scattering', 'theory', 'mie_f')

    for dll in tmatrix_libs:
        shutil.move(dll, tmatrix_f_dir)
    for dll in mie_libs:
        shutil.move(dll, mie_f_dir)
    shutil.rmtree(lib_dir)


def _get_holopy_install_dir(mode):
    if mode == 'install':
        hp_dir = [path for path in site.getsitepackages()
                  if 'site-packages' in path]
        hp_dir = os.path.join(dir[0], 'holopy')
    if mode == 'develop':
        hp_dir = os.path.dirname(os.path.realpath(__file__))
        hp_dir = os.path.join(hp_dir, 'holopy')
    return hp_dir
