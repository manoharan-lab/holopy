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
from xarray.ufuncs import sqrt,arctan2
from warnings import warn
from .utils import _ensure_pair, _ensure_array, is_none, updated

vector = 'vector'

def Image(arr, spacing=None, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=(0, 0, 1)):
    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)

    out = xr.DataArray(arr, dims=['x', 'y'], coords=make_coords(arr.shape, spacing))
    return update_metadata(out, medium_index, illum_wavelen, illum_polarization,normals)    

def ImageSchema(shape, spacing, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=(0, 0, 1)):
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)

    d = np.zeros(shape)
    return Image(d, spacing, medium_index, illum_wavelen, illum_polarization, normals)

def update_metadata(a, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=None):
    attrlist = {'medium_index': medium_index, 'illum_wavelen': illum_wavelen, 'illum_polarization': to_vector(illum_polarization), 'normals': to_vector(normals)}
    b = a.copy()
    b.attrs = updated(b.attrs, attrlist)

    for attr in attrlist:
        if not hasattr(b, attr):
            b.attrs[attr] = None

    if is_none(b.normals):
        b.attrs['normals'] = to_vector((0,0,1))

    return b

def copy_metadata(old, new, do_coords=True):
    def find_and_rename(oldkey, oldval):
        for newkey, newval in new.coords.items():
            if np.array_equal(oldval.values, newval.values):
                return new.rename({newkey: oldkey})
            raise ValueError("Coordinate {} does not appear to have a coresponding coordinate in {}".format(oldkey, new))

    if hasattr(old, 'attrs') and hasattr(old, 'name') and hasattr(old, 'coords'):
        if not hasattr(new,'coords'):
            #new is a numpy array, not xarray
            new=xr.DataArray(new, dims=['x', 'y'])
        new.attrs = old.attrs
        new.name = old.name
        if hasattr(old, 'z') and not hasattr(new, 'z'):
            new.coords['z'] = old.coords['z']
        if hasattr(old, 'flat') and hasattr(new, 'flat'):
            new['flat'] = old['flat']
        if do_coords:
            for key, val in old.coords.items():
                if key not in new.coords:
                    new = find_and_rename(key, val)
    return new

def to_vector(c):
    if c is None:
        return c
    if hasattr(c, vector):
        return c
    c = np.array(c)
    if c.shape == (2,):
        c = np.append(c, 0)

    return xr.DataArray(c, coords={vector: ['x', 'y', 'z']})

def flat(a, keep_xy=True):
    if hasattr(a, 'flat'):
        # TODO handle case where we have flat but not xyz
        return a
    if hasattr(a, 'x') and hasattr(a, 'y') and keep_xy:
        a['x_orig'] = a.x
        a['y_orig'] = a.y
        # TODO: remove *_orig coords from a or avoid adding them
        f = a.stack(flat=a.dims)
        del a['x_orig']
        del a['y_orig']
        return f.rename({'x_orig': 'x', 'y_orig': 'y'})
    else:
        return a.stack(flat=a.dims)

def from_flat(a):
    if hasattr(a, 'flat'):
        return a.unstack('flat')
    return a

def to_spherical(a, origin, wavevec=None, include_r=True):
    xo, yo, zo = origin
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

def kr_theta_phi_flat(a, origin, wavevec=None):
    if wavevec is None:
        wavevec = 2*np.pi/(a.illum_wavelen/a.medium_index)
    return flat(to_spherical(a, origin, wavevec))

def theta_phi_flat(a, origin=None):
    if hasattr(a, 'theta') and hasattr(a, 'phi'):
        return a
    return flat(to_spherical(a, origin))

def get_values(a):
    return getattr(a, 'values', a)

def get_spacing(im):
    xspacing = np.diff(im.x)
    yspacing = np.diff(im.y)
    if not np.allclose(xspacing[0], xspacing) and np.allclose(yspacing[0], yspacing):
        raise ValueError("array has nonuniform spacing, can't determine a single spacing")
    return np.array((xspacing[0], yspacing[0]))

def make_coords(shape, spacing, z=0):
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)
    return {'x': np.arange(shape[0])*spacing[0], 'y': np.arange(shape[1])*spacing[1], 'z': 0}
