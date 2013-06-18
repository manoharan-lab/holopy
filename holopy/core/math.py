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
from __future__ import division

import scipy.fftpack as fftpack
from numpy import sin, cos
import numpy as np
from .marray import arr_like, Marray


def fft(a, overwrite=False, shift=True):
    """
    More convenient Fast Fourier Transform

    An easier to use fft function, it will pick the correct fft to do
    based on the shape of the Marray, and do the fftshift for you.  This
    is intended for working with images, and thus for dimensions
    greater than 2 does slicewise transforms of each "image" in a
    multidimensional stack

    Parameters
    ----------
    a : ndarray
       The array to transform
    overwrite : bool
       Allow this function to overwrite the Marry you pass in.  This
       may improve performance slightly.  Default is not to overwrite
    shift : bool
       Whether to preform an fftshift on the Marry to give low
       frequences near the center as you probably expect.  Default is
       to do the fftshift.

    Returns
    -------
    fta : ndarray
       The fourier transform of `a`
    """
    if a.ndim is 1:
        if shift:
            res = fftpack.fftshift(fftpack.fft(a, overwrite_x=overwrite))
        else:
            res = fftpack.fft(a, overwrite_x=overwrite)
    else:
        if shift:
            res = fftpack.fftshift(fftpack.fft2(a, axes=[0, 1],
                                                 overwrite_x=overwrite),
                                    axes=[0,1])
        else:
            res = fftpack.fft2(a, axes=[0, 1], overwrite_x=overwrite)
    if isinstance(a, Marray):
        res = arr_like(res, a)
    return res


def ifft(a, overwrite=False, shift=True):
    """
    More convenient Inverse Fast Fourier Transform

    An easier to use ifft function, it will pick the correct ifft to
    do based on the shape of the Marry, and do the fftshift for you.
    This is indendended for working with images, and thus for
    dimensions greater than 2 does slicewise transforms of each
    "image" in a multidimensional stack

    Parameters
    ----------
    a : ndarray
       The array to transform
    overwrite : bool
       Allow this function to overwrite the Marry you pass in.  This
       may improve performance slightly.  Default is not to overwrite
    shift : bool
       Whether to preform an fftshift on the Marry to give low
       frequences near the center as you probably expect.  Default is to
       do the fftshift.

    Returns
    -------
    ifta : ndarray
       The inverse fourier transform of `a`
    """
    if a.ndim is 1:
        if shift:
            res = fftpack.ifft(fftpack.fftshift(a, overwrite_x=overwrite))
        else:
            res = fftpack.ifft(a, overwrite_x=overwrite)
    else:
        if shift:
            res = fftpack.ifft2(fftpack.fftshift(a, axes=[0,1]), axes=[0, 1],
                                 overwrite_x=overwrite)
        else:
            res = fftpack.ifft2(a, overwrite_x=overwrite)
    if isinstance(a, Marray):
        res = arr_like(res, a)
    return res

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
