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
"""
HoloPy is a set of tools for working with digital holograms and light
scattering. It contains tools for working loading data and associating
it with expermiental metadata, reconstructing holograms, calculating
light scattering, fitting scattering models to data, and visualizing
images and calculations.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""


from holopy import core, scattering, inference
from holopy.core import (load, save, load_image, save_image, show,
                         check_display, detector_grid, detector_points)
from holopy.propagation import propagate
from holopy.inference import fit, sample

__version__ = '3.5.0'
__version_info__ = tuple([int(num) for num in __version__.split('.')])
