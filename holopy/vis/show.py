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

"""
A general show method that can display most holopy and scatterpy objects in a
sensible way.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import numpy as np
from ..scattering.scatterer import Spheres, Scatterer
from ..core import Image
from .vis2d import show2d
from .vis3d import show_sphere_cluster, show_scatterer

class VisualizationNotImplemented(Exception):
    def __init__(self, o):
        self.o = o
    def __str__(self):
        return "Visualization of object of type: {0} not implemented".format(
            self.o.__class__.__name__)


def show(o,color=(.5, .5, .5)):
    """
    Visualize a scatterer, hologram, or reconstruction

    Parameters
    ----------
    o : :class:`.Image`, :class:`.Volume` or :class:`.Spheres`
       Object to visualize

    Notes
    -----
    Loads plotting library the first time it is required (so that we don't have
    to import all of matplotlib or mayavi just to load holopy)
    """

    if isinstance(o, Spheres):
        show_sphere_cluster(o,color)
    elif isinstance(o, (Image, np.ndarray, list, tuple)):
        show2d(o)
    elif isinstance(o, Scatterer):
        show_scatterer(o)
    else:
        raise VisualizationNotImplemented(o)
