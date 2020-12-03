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

import numpy as np
from numpy import sin, cos, arccos, arctan2, sqrt, pi
from holopy.core.utils import repeat_sing_dims


def rotate_points(points, theta, phi, psi):
    """
    Rotate an array of points about Euler angles in a z, y, z convention.

    Parameters
    ----------
    points: array-like (n,3)
        Set of points to be rotated
    theta, phi, psi: float
        Euler rotation angles in z, y, z convention. These are *not* the
        same as angles in spherical coordinates.

    Returns
    -------
    rotated_points: array(n,3)
        Points rotated by Euler angles

    """
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
        alpha *= pi/180.
        beta *= pi/180.
        gamma *= pi/180.

    ca = cos(alpha)
    sa = sin(alpha)
    cb = cos(beta)
    sb = sin(beta)
    cg = cos(gamma)
    sg = sin(gamma)

    return np.array([ca*cb*cg - sa*sg, -sa*cb*cg - ca*sg, sb*cg,
                     ca*cb*sg + sa*cg, -sa*cb*sg + ca*cg, sb*sg,
                     -ca*sb, sa*sb, cb]).reshape((3,3)) # row major


def transform_cartesian_to_spherical(x_y_z):
    x, y, z = x_y_z
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x) % (2*np.pi)
    return np.array([r, theta, phi])


def transform_spherical_to_cartesian(r_theta_phi):
    r, theta, phi = r_theta_phi
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def transform_cartesian_to_cylindrical(x_y_z):
    x, y, z = x_y_z
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) % (2*np.pi)
    z = (np.full(rho.size, z) if np.size(z) == 1 else z)
    return np.array([rho, phi, z])


def transform_cylindrical_to_cartesian(rho_phi_z):
    rho, phi, z = rho_phi_z
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = (np.full(x.size, z) if np.size(z) == 1 else z)
    return np.array([x, y, z])


def transform_cylindrical_to_spherical(rho_phi_z):
    rho, phi, z = rho_phi_z
    r = np.sqrt(rho**2 + z**2)
    theta = np.arctan2(rho, z)
    return np.array([r, theta, phi])


def transform_spherical_to_cylindrical(r_theta_phi):
    r, theta, phi = r_theta_phi
    rho = r * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([rho, phi, z])


def keep_in_same_coordinates(coords): return np.array(coords)


_transformation_lut = {
    'cartesian': {
        'spherical': transform_cartesian_to_spherical,
        'cylindrical': transform_cartesian_to_cylindrical,
        'cartesian': keep_in_same_coordinates,
        },
    'spherical': {
        'cartesian': transform_spherical_to_cartesian,
        'cylindrical': transform_spherical_to_cylindrical,
        'spherical': keep_in_same_coordinates,
        },
    'cylindrical': {
        'cartesian': transform_cylindrical_to_cartesian,
        'cylindrical': keep_in_same_coordinates,
        'spherical': transform_cylindrical_to_spherical,
        },
    }
def find_transformation_function(initial_coordinates, desired_coordinates):
    try:
        method = _transformation_lut[initial_coordinates][desired_coordinates]
    except KeyError:
        msg = "Transformation from {} to {} not implemented.".format(
            initial_coordinates, desired_coordinates)
        raise NotImplementedError(msg)
    return method


def to_cartesian(r, theta, phi):
    x, y, z = transform_spherical_to_cartesian([r, theta, phi])
    return repeat_sing_dims({'x': x, 'y': y, 'z': z})


def cartesian_distance(p1, p2=[0,0,0]):
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


def chisq(fit, data):
    r"""
    Calculate per-point value of chi-squared comparing a best-fit model and
    data.

    Parameters
    ----------
    fit : array_like
        Values of best-fit model to be compared to data
    data : array_like
        Data values to be compared to model

    Returns
    -------
    chisq : float
        Chi-squared per point

    Notes
    -----
    chi-squared is defined as

    .. math::
        \chi^2 = \frac{1}{N}\sum_{\textrm{points}} (\textrm{fit} - \textrm{data})^2

    where :math:`N` is the number of data points.
    """
    return float((((fit-data))**2).sum() / fit.size)


def rsq(fit, data):
    r"""
    Calculate correlation coeffiction R-squared comparing a best-fit model
    and data.

    Parameters
    ----------
    fit : array_like
        Values of best-fit model to be compared to data
    data : array_like
        Data values to be compared to model

    Returns
    -------
    rsq : float
        Correlation coefficient R-squared.

    Notes
    -----
    R-squared is defined as

    .. math::
        R^2 = 1 - \frac{\sum_{\textrm{points}} (\textrm{data} - \textrm{fit})^2}{\sum_{\textrm{points}} (\textrm{data} - \bar{\textrm{data}})^2}

    where :math:`\bar{\textrm{data}}` is the mean value of the data. If the
    model perfectly describes the data, :math:`R^2 = 1`.
    """
    return float(1 - ((data - fit)**2).sum()/((data - data.mean())**2).sum())

