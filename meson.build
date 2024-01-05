project('HoloPy', 'c',
  version: '3.6.0pre',
  meson_version: '>= 1.2.3',
)

'''
Build instructions for fortran extensions and holopy package

Notes:

The t-matrix and mie scattering codes require working Fortran 90 and C
compilers, as well as cython. On Ubuntu, you will need the "gfortran"
and "python-dev" packages installed.

meson 1.2.3 is first version to not rely on distutils and is minimum
version to support python>3.12
'''

# It seems to be common to add fortran in this way rather than in the
# project statement -- might avoid some warnings.
add_languages('fortran', native: false)

# (below is from scipy meson.build)
# Don't use the deprecated NumPy C API. Define this to a fixed version
# instead of NPY_API_VERSION in order not to break compilation when
# NumPy introduces a new deprecation. Use in a meson.build file:
#
#   py3.extension_module('_name',
#     'source_fname',
#     numpy_nodepr_api)
#
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Unlike numpy.distutils, meson doesn't run f2py automatically to
# compile the fortran extensions. So we need to tell meson how to find
# fortranobject.c, which is included with f2py and should be compiled
# and linked with any extension.
# See https://numpy.org/doc/stable/f2py/buildtools/meson.html
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
# in numpy 1.22 and later, can replace above with
# incdir_f2py = run_command(py,
#   ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
#    check : true
# ).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# All meson builds are done "out of tree". That means compiled
# extensions are not installed into the same directory as their sources.
# Note that holopy has several extensions that must be installed in a
# way that makes it possible to do (for example):
#
# import holopy.scattering.theory.mie_f.scsmfo_min
#
# To enable this functionality, we need to tell meson to copy the
# extensions to the appropriate point in the installation tree, *and* to
# copy the python files too. In the subdirectories, you'll see
# meson.build files that call "install_sources()", which installs the
# .py files of holopy, and that call "extension_module(subdir=...)"
# which tells meson where to install the fortran extensions. All files
# have to be specified, so if we add a python file somewhere, we need to
# update the relevant meson.build file to include it in the
# installation. This is how scipy uses meson. Having to specify all the
# files is a big switch from how numpy.distutils does things, but it
# makes the build process more efficient.

subdir('holopy')
