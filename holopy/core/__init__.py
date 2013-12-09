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
"""Loading, saving, and basic processing of data.

holopy.core contains code to load images and holopy yamls into
:mod:`.marray` objects. It also contains the machinery for saving
all HoloPy objects as holopy yaml files. Finally, it provides some
basic mathematical operations, mostly as higher level wrappers around
numpy or scipy routines.

Main use cases are

1. Image or other data file + metadata => :class:`.Image` or other
   :class:`.Marray` object

2. Raw :class:`.Image` + processing => processed :class:`.Image` object

3. Any :class:`.HoloPyObject` from calculations or processing => achival
   yaml text or text/binary result

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""

from marray import (Marray, Image, Schema, ImageSchema, Volume,
                    VolumeSchema, VectorGrid, VectorGridSchema,
                    subimage)
from metadata import Optics, Grid, Angles, UnevenGrid
from io import load, load_image, save
import process
import helpers
