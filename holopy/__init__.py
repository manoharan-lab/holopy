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
A python package containing routines to load, reconstruct, fit, and
analyze digital holograms  

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""


# import some names into the top level namespace so we can use,
# e.g. holopy.Hologram or holopy.load
from .hologram import Hologram, subimage
from .optics import Optics
from .analyze.reconstruct import reconstruct
from .analyze.fit import fit, get_target, get_initial_guess, get_fit_result
import process
from .io.image_io import load
from .vis import show

__version__ = 'unknown'
try:
    from _version import __version__
except ImportError:
    # version doesn't exist, or got deleted in bzr
    pass

