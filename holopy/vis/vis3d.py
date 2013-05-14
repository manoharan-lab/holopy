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
Show sphere clusters using mayavi

.. moduleauthor:: Tom Dimduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
"""

from __future__ import division

from numpy import arange
import numpy as np

def import_mayavi():
    # Mayavi moved namespaces in the upgrade to 2.4. This try block will
    # allow using either the new or old namespace.
    try:
        from mayavi import mlab
    except ImportError:
        from enthought.mayavi import mlab
    return mlab

def show_sphere_cluster(s,color):
    # Delayed imports to avoid hard dependencies on plotting packages and to
    # avoid the cost of importing them in noninteractive code
    from matplotlib import cm

    mlab = import_mayavi()

    # scale factor is 2 because mayavi interprets 4th
    # argument as a diameter, we keep track of radii
    # view is chosen to be looking from the incoming laser's point of view
    if color == 'rainbow':
        for i in arange(0,len(s.x)):
            numberofcolors = max(10,len(s.x))
            mlab.points3d(s.x[i], s.y[i], s.z[i], s.r[i],
                scale_factor=2.0, resolution=32,
                color=cm.gist_rainbow(float(i)/numberofcolors)[0:3])
            mlab.view(-90,0,s.z[:].mean())
    else:
        mlab.points3d(s.x, s.y, s.z, s.r, scale_factor=2.0,
            resolution=32, color=color)
        mlab.view(-90,0,s.z[:].mean())

def volume_contour(d):
    mlab = import_mayavi()

    vol = mlab.pipeline.scalar_field(d)
    vol.spacing = d.spacing
    contours = mlab.pipeline.contour_surface(vol)

# TODO: not quite working yet
def show_scatterer(scatterer, spacing=None):
    mlab = import_mayavi()
    if spacing == None:
        # if no spacing given, represent the object with 100 voxels
        # along each dimension
        spacing = [(b[1]-b[0])/100 for b in scatterer.bounds]
    vol = scatterer.voxelate(spacing, 0)
    mlab.contour3d(vol)
