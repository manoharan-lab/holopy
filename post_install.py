import glob
import os
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
    lib_dir = os.path.join(package_dir, '.libs')
    sep = os.path.sep

    tmatrix_libs = glob.glob(lib_dir + sep + 'libS.*.dll')
    mie_libs = glob.glob(lib_dir + sep + 'libscsm*.dll')
    mie_libs += glob.glob(lib_dir + sep + 'libmieang*.dll')
    mie_libs += glob.glob(lib_dir + sep + 'libuts*.dll')

    tmatrix_f_dir = os.path.join(package_dir, 'scattering', 'theory',
                                 'tmatrix_f')
    mie_f_dir = os.path.join(package_dir, 'scattering', 'theory', 'mie_f')

    for dll in tmatrix_libs:
        shutil.move(dll, tmatrix_f_dir)
    for dll in mie_libs:
        shutil.move(dll, mie_f_dir)
    shutil.rmtree(lib_dir)


def _get_holopy_install_dir(mode):
    if mode == 'install':
        sitepackages = list(site.getsitepackages())
        hp_dir = [path for path in site.getsitepackages()
               if 'site-packages' in path]
        assert len(hp_dir) == 1
        hp_dir = os.path.join(hp_dir[0], 'holopy')
    if mode == 'develop':
        hp_dir = os.path.dirname(os.path.realpath(__file__))
        hp_dir = os.path.join(hp_dir, 'holopy')
    return hp_dir
