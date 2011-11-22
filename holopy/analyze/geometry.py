# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
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
Routines for common calculations and transformations of groups of spheres.

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>

"""

import numpy as np
import holopy as hp

#calculate the distances between each sphere in a cluster and each
#of the others
def distances(cluster, gaponly=False):
    """
    Parameters
    ----------
    cluster: :class:`scatterpy.scatterer.Scatterer`
        A sphere cluster to determine the interparticle distances of.
    gaponly: bool
        Whether to calculate the distances between particle centers
        or between particle surfaces (gap distances).
        
    Notes
    -----
    The returned array of distances includes redundant information.
    The identical distances between sphere 1 and sphere 2 and between sphere 2
    and sphere 1 are both in the returned array. Calculating and returning
    the full array makes it easy for the user to access all the interparticle 
    distances starting from any sphere of interest.
    
    """
    num = len(cluster.centers)
    dist = np.zeros([num,num])
    for i in np.arange(0,num):
        for j in np.arange(0,num):
            dist[i,j] = hp.process.cartesian_distance(cluster.centers[i,:],cluster.centers[j,:])
            #modification to change center to center distances to gap distances if asked for
            if gaponly==True and i!=j:
                dist[i,j] = dist[i,j]-cluster.r[i]-cluster.r[j]
    return dist

#calculate the angles between one particle and every pair
#of other particles
def angles(cluster, degrees=True):
    """
    Parameters
    ----------
    cluster: :class:`scatterpy.scatterer.Scatterer`
        A sphere cluster to determine the interparticle distances of.
    degrees: bool
        Whether to return angles in degrees (True) or in radians (False).
        
    Notes
    -----
    If a, b, and c are locations of particles (vertices),
    the returned 3D array has non-zero values for angles abc, zeros
    for angles aba, and NAN's for "angles" aab.
    
    """
    num = len(cluster.centers)
    ang = np.zeros([num,num,num])
    dist = distances(cluster)
    for i in np.arange(0,num):
        for j in np.arange(0,num): #this particle is the center of the angle
            for k in np.arange(0,num):
                Adjacent1=dist[i,j]
                Adjacent2=dist[j,k]
                Opposite=dist[i,k]
                #use the law of cosines to determine the angles from the distances
                ang[i,j,k] = np.arccos((Adjacent1**2+Adjacent2**2-Opposite**2)/(2*Adjacent1*Adjacent2))
    if degrees==True:
        ang=ang/np.pi*180.0
    return ang #ang[a,b,c] is the acute angle abc as used in geometry (be in the middle)
