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
#!/usr/bin/env python
"""
Installs the MieFieldExtension module (MFE)

To install: python setup.py build_ext --inplace

It'll create the file MFE.so.  You can then import MFE to start
using the functions.

Created on: 12/31/2009

Authors: Ryan McGorty

Based on the example file from Cython: http://docs.cython.org/src/userguide/

If you choose not to use this, you can also create the module with:

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -lm -o MFE.so MFE.c MieFieldExtension.c

This will work on Windows machines. You'll need the MinGW compiler (wingw.org) and Python 2.5 or greater.
You'll also need to have a file called 'distutils.cfg' in the folder Python2X\Lib\distutils
The file should have the lines:
	[build]
	compiler = mingw32

Then you can just do the "python setup.py build_ext --inplace" and it should compile okay.

"""

import numpy

# We detect whether Cython is available, so that below, we can
# eventually ship pre-generated C for users to compile the extension
# without having Cython installed on their systems.
try:
    import Cython.Distutils 
    has_cython = True
except ImportError:
    has_cython = False

# Define a cython-based extension module, using the generated sources
# if cython is not available.
if has_cython:
    pyx_sources = ['MFE.pyx', 'MieFieldExtension.c', 'MieFieldExtension.h']
else:
    # In production work, you can ship the auto-generated C source
    # yourself to your users.  In this case, we do NOT ship the .c
    # file as part of numpy, so you'll need to actually have cython
    # installed at least the first time.  Since this is really just an
    # example to show you how to use *Cython*, it makes more sense NOT
    # to ship the C sources so you can edit the pyx at will with less
    # chances for source update conflicts when you update numpy.

    pyx_sources = ['MFE.c', 'MieFieldExtension.c', 'MieFieldExtension.h']
    

# Declare the extension object
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mie', parent_package, top_path)
    config.add_extension('MFE',
                         [pyx_sources], include_dirs = ['.']
                         )
    return config

from os.path import join as pjoin, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError

from numpy.distutils.misc_util import appendpath
from numpy.distutils import log


# patch to get numpy.distutils to work with Cython from Matthew Brett
# see
# http://www.mail-archive.com/numpy-discussion@scipy.org/msg25279.html
def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method
    
    Uses Cython instead of Pyrex.
    
    Assumes Cython is present

    https://github.com/matthew-brett/du-cy-numpy/blob/master/matthew_monkey.py

    '''
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % (target_file))
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source,
                                                   options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" \
                  % (cython_result.num_errors, source))
    return target_file


from numpy.distutils.command import build_src
build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source
# end monkey-patch

# Call the routine which does the real work
if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
