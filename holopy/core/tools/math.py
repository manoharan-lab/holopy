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
Misc utility functions to make coding more convenient

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

from numpy import sin, cos
import numpy as np
from scipy.ndimage import gaussian_filter

def rotate_points(points, theta, phi, psi):
    points = np.array(points)
    rot = rotation_matrix(theta, phi, psi)
    if points.ndim == 1:
        return np.dot(rot, points)
    return np.array([np.dot(rot, c) for c in points])

def rotation_matrix(alpha, beta, gamma, radians = True):
    """
    Return a 3D rotation matrix

    Parameters
    ----------
    alpha, beta, gamma: float
        Euler rotation angles in z, y, z convention
    radians: boolean
        Default True; treats input angles in radians

    Returns
    -------
    rot: array(3,3)
        Rotation matrix. To rotate a column vector x, use np.dot(rot, x.) 

    Notes
    -----
    The Euler angles rotate a vector (in the active picture) by alpha
    clockwise about the fixed lab z axis, beta clockwise about
    the lab y axis, and by gamma about the lab z axis.  Clockwise is
    defined as viewed from the origin, looking in the positive direction
    along an axis.  

    This breaks compatability with previous conventions, which were adopted for 
    compatability with the passive picture used by SCSMFO.

    """
    if not radians:
        alpha *= np.pi/180.
        beta *= np.pi/180.
        gamma *= np.pi/180.

    ca = cos(alpha)
    sa = sin(alpha)
    cb = cos(beta)
    sb = sin(beta)
    cg = cos(gamma)
    sg = sin(gamma)

    return np.array([ca*cb*cg - sa*sg, -sa*cb*cg - ca*sg, sb*cg,
                     ca*cb*sg + sa*cg, -sa*cb*sg + ca*cg, sb*sg,
                     -ca*sb, sa*sb, cb]).reshape((3,3)) # row major

def cartesian_distance(p1, p2):
    """
    Return the Cartesian distance between points p1 and p2.

    Parameters
    ----------
    p1, p2: array or list
        Coordinates of point 1 and point 2 in N-dimensions

    Returns
    -------
    dist: float64
        Cartesian distance between points p1 and p2 

    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1-p2)**2))



def simulate_noise(shape, mean=.1, smoothing=.01, poisson_lambda=1000):
    """Create an array of correlated noise. The noise_mean can be controlled independently of
    the poisson_lambda that controls the shape of the distribution. In
    general, you can stick with our default of a large poisson_lambda
    (ie for imaging conditions not near the shot noise limit).

    Defaults are set to give noise vaguely similar to what we tend to
    see in our holographic imaging.

    Parameters
    ----------
    shape : int or array_like of ints
            shape of noise array
    smoothing : float
        Fraction of the image size to smooth by. Should in general be << 1
    poisson_lambda : float
        Used to compute the shape of the noise distribution. You can generally
        leave this at its default value unless you are simulating shot noise
        limited imaging.

    Returns
    -------
    noisy_image : ndarray
       A copy of the input image with noise added.

    """

    raw_poisson = np.random.poisson(poisson_lambda, shape)
    smoothed = gaussian_filter(raw_poisson, np.array(shape)*smoothing)
    return smoothed/smoothed.mean() * mean
