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
"""
Holographic imaging and light scattering.  Work with experimental data, compute
holograms or light scattering of objects, and fit modeled scattering to measured
data.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""


import core
from .core import load, save
import scattering
import fitting
from propagation import propagate
from vis import show

__all__ = ['core', 'scattering']

__version__ = 'unknown'
try:
    from _version import __version__
except ImportError:
    # version doesn't exist, or got deleted in bzr
    pass
