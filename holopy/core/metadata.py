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
Classes for defining metadata about experimental or calculated results.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""


import numpy as np
import xarray as xr
from xarray.ufuncs import sqrt, arctan2
from warnings import warn
import copy
from .tools import _ensure_pair, _ensure_array, is_none, flat, from_flat


vector = 'vector'

def to_vector(c):
    if c is None:
        return c
    if hasattr(c, 'vector'):
        return c
    c = np.array(c)
    if c.shape == (2,):
        c = np.append(c, 0)

    return xr.DataArray(c, coords={vector: ['x', 'y', 'z']})

def to_spherical(a, origin, wavevec=None, include_r=True):
    xo, yo, zo = origin
    if hasattr(a, 'flat'):
        a = from_flat(a)
    x, y, z = a.x - xo, a.y - yo, zo - a.z
    theta = arctan2(np.sqrt(x**2 + y**2), z)
    phi = arctan2(y, x)
    phi = phi + 2*np.pi * (phi < 0)
    if include_r:
        r = sqrt(x**2 + y**2 + z**2)
        if wavevec is not None:
            rname = 'kr'
            kr = r*wavevec
        else:
            rname = 'r'
            kr = r
        return xr.DataArray(a, coords={rname: kr, 'theta': theta, 'phi': phi, 'x': a.x, 'y':a.y})
    else:
        return xr.DataArray(a, coords={'theta': theta, 'phi': phi, 'x': a.x, 'y': a.y})


def r_theta_phi_flat(a, origin):
    f = flat(to_spherical(a, origin))
    return f
    return np.vstack((f.r, f.theta, f.phi)).T

def kr_theta_phi_flat(a, origin, wavevec=None):
    if wavevec is None:
        wavevec = get_wavevec(a)
    return flat(to_spherical(a, origin, wavevec))
    pos = r_theta_phi_flat(a, origin)
    pos['r'] *= wavevec
    return pos

def theta_phi_flat(a, origin=None):
    if hasattr(a, 'theta') and hasattr(a, 'phi'):
        return a
    return flat(to_spherical(a, origin))

def make_coords(shape, spacing, z=0):
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)
    return {'x': np.arange(shape[0])*spacing[0], 'y': np.arange(shape[1])*spacing[1], 'z': 0}

def make_attrs(medium_index, illum_wavelen, illum_polarization, normals):
    if is_none(normals):
        normals = (0, 0, 1)
    return {'medium_index': medium_index, 'illum_wavelen': illum_wavelen, 'illum_polarization': to_vector(illum_polarization), 'normals': to_vector(normals)}

def angles_list(theta, phi, medium_index, illum_wavelen, illum_polarization, normals=(0, 0, 1)):
    # This is a hack that gets the data into a format that we can use
    # elsewhere, but feels like an abuse of xarray, it would be nice to replace this with something more ideomatic
    d = make_attrs(medium_index, illum_wavelen, illum_polarization, normals)

    theta = _ensure_array(theta)
    phi = _ensure_array(phi)
    if len(theta) == 1:
        theta = np.repeat(theta,len(phi))
    elif len(phi) == 1:
        phi = np.repeat(phi,len(theta))

    d['theta'] = theta
    d['phi'] = phi
    return xr.DataArray(np.zeros(len(theta)), dims=['point'], attrs=d)

def ImageSchema(shape, spacing, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=(0, 0, 1)):
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)

    d = np.zeros(shape)
    return Image(d, spacing, medium_index, illum_wavelen, illum_polarization, normals)

def Image(arr, spacing=None, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=(0, 0, 1)):
    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)

    return xr.DataArray(arr, dims=['x', 'y'], coords=make_coords(arr.shape, spacing), attrs=make_attrs(medium_index, illum_wavelen, illum_polarization, normals))

def get_med_wavelen(a):
    return a.illum_wavelen/a.medium_index

def get_wavevec(a):
    return 2*np.pi/get_med_wavelen(a)

def optical_parameters(schema=None, **kwargs):
    d = {}
    for par, val in kwargs.items():
        if val is None:
            d[par] = getattr(schema, par)
        else:
            d[par] = val
    r = {}
    if 'illum_wavelen' in d and 'medium_index' in d:
        r['medium_wavevec'] = 2*np.pi/(d['illum_wavelen']/d['medium_index'])
    if 'medium_index' in d:
        r['medium_index'] = d['medium_index']
    if 'illum_polarization' in d:
        r['illum_polarization'] = to_vector(d['illum_polarization'])

    return r

def get_spacing(im):
    xspacing = np.diff(im.x)
    yspacing = np.diff(im.y)
    if not np.allclose(xspacing[0], xspacing) and np.allclose(yspacing[0], yspacing):
        raise ValueError("array has nonuniform spacing, can't determine a single spacing")
    return np.array((xspacing[0], yspacing[0]))
