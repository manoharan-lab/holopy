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

'''
Extensions for T-Matrix scattering calculations (in fortran77 and
fortran90); numpy.distutils should automatically use f2py to compile
these, and f2py should detect your fortran compiler.

The code works with gcc, but has not been tested with other
compilers.  Note that f2py by default compiles with optimization
flags.

Ignore compiler warnings of unused variables, unused dummy
arguments, and variables being used uninitialized from compiling
scsmfo_min. The former is relics of how scsmfo was written which I
am not touching. The latter is likely due to some GOTO statements that
could cause a variable to be referenced before it's initialized. Under
normal usage I wouldn't worry about it.
'''

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.command import build_ext
    config = Configuration('mie_f', parent_package, top_path)
    config.add_extension('uts_scsmfo',
                         ['uts_scsmfo.for',
                          '../../third_party/SBESJY.F']
                         )
    config.add_extension('mieangfuncs',
                         ['mieangfuncs.f90',
                          'uts_scsmfo.for',
                          '../../third_party/SBESJY.F',
                          '../../third_party/csphjy.for']
                         )
    config.add_extension('scsmfo_min',
                         ['scsmfo_min.for']
                         )
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
