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

'''
Defines SphereDimer, a scatterer consisting of two Spheres

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
from scatterpy.scatterer import Scatterer

class SphereDimer(Scatterer):
    '''
    Contains optical and geometrical properties of a dimer of two
    spheres.  

    Attributes
    ----------
    n1 : float or complex
        Index of refraction of sphere 1
    n2 : float or complex
        Index of refraction of sphere 2
    r1 : float
        Radius of sphere 1
    r2 : float
        Radius of sphere 2
    center1 : 3-tuple, list or ndarray
        specifies coordinates of center of sphere 1
    center2 : 3-tuple, list, or ndarray
        specifies coordinates of center of sphere 2
    com : 3-tuple, list or ndarray (optional)
        specifies center of mass of dimer
    beta : float (optional)
       Euler angle beta (deg) in modified zyz convention (rotation about y).
    gamma : float (optional)
       Euler angle gamma (deg) in modified zyz convention (rotation about z).
    gap_distance : float (optional)
       Interparticle gap distance ( = 0 at hard-sphere contact.) 

    Notes
    -----
    There are two ways of specifying the dimer.  You can either
    specify the location of each sphere or specify the location of
    the center of mass, the Euler angles, and the separation
    distance.  Either way involves giving 6 numbers.  Note that
    although you can specify overlapping spheres, most scattering
    theories cannot handle overlaps.

    SphereDimer is a separate class from the SphereCluster class
    because dimers are a special type of cluster: they have continuous
    rotational symmetry about one axis, which means you can specify
    the orientation using only two Euler angles.  Our Euler angle beta
    differs slightly from the zyz convention in that it is permitted
    to be negative.  It will behave in a mathematically sensible way
    between -180 and 360 degrees.  The reference configuration (beta
    and gamma both = 0) occurs with both particles lying on the
    x-axis, with the x coordinate of particle #1 being positive.
    '''

    def __init__(self, n1=1.59, n2=1.59, r1=0.5e-6, r2=0.5e-6,
                 center1 = [1e-6, 0.0, 0.0],
                 center2 = [1e-6, 0.0, 0.0],
                 com = None, alpha = None, beta = None, gap = None):
        self.n1 = n1
        self.n2 = n2
        self.r1 = r1
        self.r2 = r2
        if com is not None:
            # calculate coordinates of spheres from com, alpha, beta, gap
            pass
        else:
            self.center1 = center1
            self.center2 = center2
