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
from warnings import warn
from .utils import ensure_array, is_none, updated, repeat_sing_dims
from .math import to_spherical, to_cartesian


vector = 'vector'

def data_grid(arr, spacing=None, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=None, name=None):
    if spacing is None:
        spacing = 1
    if name is None:
        name = 'data'

    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)
    if arr.ndim==2:
        arr=np.array([arr])
    out = xr.DataArray(arr, dims=['z','x', 'y'], coords=make_coords(arr.shape, spacing), name=name)
    return update_metadata(out, medium_index, illum_wavelen, illum_polarization, normals)

def detector_grid(shape, spacing, normals = None, name = None):
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)

    d = np.zeros(shape)
    return data_grid(d, spacing, normals = normals, name = name)

def detector_points(coords = {}, x = None, y = None, z = None, r = None, theta = None, phi = None, normals = 'auto', name = None):
    updatelist = {'x': x, 'y': y, 'z': z, 'r': r, 'theta': theta, 'phi': phi}
    coords = updated(coords, updatelist)
    if 'x' in coords and 'y' in coords:
        keys = ['x', 'y', 'z']
        if not 'z' in coords or is_none(coords['z']):
            coords['z'] = 0
        
    elif 'theta' in coords and 'phi' in coords:
        keys = ['r', 'theta', 'phi']
        if not 'r' in coords or is_none(coords['r']):
            coords['r'] = np.inf
    else:
        raise CoordSysError()

    if name is None:
        name = 'data'

    coords = repeat_sing_dims(coords,keys)
    coords = updated(coords,{key: ('point', coords[key]) for key in keys})
    attrs = {'normals': default_norms(coords, normals)}
    return xr.DataArray(np.zeros(len(coords[keys[0]][1])), dims = ['point'], coords = coords, attrs = attrs, name = name)

def update_metadata(a, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=None):
    attrlist = {'medium_index': medium_index, 'illum_wavelen': illum_wavelen, 'illum_polarization': to_vector(illum_polarization), 'normals': to_vector(normals)}
    b = a.copy()
    b.attrs = updated(b.attrs, attrlist)

    for attr in attrlist:
        if not hasattr(b, attr):
            b.attrs[attr] = None

    if is_none(b.normals):
        b.attrs['normals'] = default_norms(b.coords, 'auto')

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

        if hasattr(old, 'flat') and hasattr(new, 'flat'):
            new['flat'] = old['flat']
        if do_coords:
            for key, val in old.coords.items():
                if key not in new.coords:
                    new = find_and_rename(key, val)
    return new

def to_vector(c):
    if c is None or c is False:
        return c
    if hasattr(c, vector):
        return c
    c = np.array(c)
    if c.shape == (2,):
        c = np.append(c, 0)

    return xr.DataArray(c, coords={vector: ['x', 'y', 'z']})

def flat(a, keep_dims=True):
    if hasattr(a, 'flat') or hasattr(a, 'point'):
        return a
    if len(a.dims)==3 and keep_dims:
        a['x_orig'] = a.x
        a['y_orig'] = a.y
        a['z_orig'] = a.z
        #want to ensure order is x, y, z
        r = a.stack(flat=('x','y','z'))
        del a['x_orig']
        del a['y_orig']
        del a['z_orig']
        return r.rename({'x_orig': 'x', 'y_orig': 'y', 'z_orig': 'z'})
    else:
        return a.stack(flat=a.dims)

def from_flat(a):
    if hasattr(a, 'flat'):
        return a.unstack('flat')
    return a

def sphere_coords(a, origin=(0,0,0), wavevec=1):
    if hasattr(a,'theta') and hasattr(a, 'phi'):
        out = {'theta': a.theta.values, 'phi': a.phi.values, 'point':a.point.values}
        if hasattr(a, 'r') and any(a.r < np.inf):
            out['r'] = a.r.values * wavevec
        return out

    else:
        f = flat(a)
        dimstr = primdim(f)
        # we define positive z opposite light propagation, so we have to invert
        x, y, z = f.x.values - origin[0], f.y.values - origin[1], origin[2] - f.z.values
        out = to_spherical(x,y,z)
        return updated(out, {'r':out['r'] * wavevec, dimstr:f[dimstr],'x':f.x.values, 'y':f.y.values, 'z':f.z.values})

def get_values(a):
    return getattr(a, 'values', a)

def primdim(a):
    if isinstance(a, xr.DataArray):
        a = a.coords
    if 'flat' in a:
        return 'flat'
    if 'point' in a:
        return 'point'
    raise ValueError('Array is not in the form of a 1D list of coordinates')

def default_norms(coords,n):
    if n is 'auto':    
        if 'x' in coords:
            n = (0,0,1)
        elif 'theta' in coords:
            n = to_cartesian(1, coords['theta'][1], coords['phi'][1])
            n = -np.vstack((n['x'],n['y'],n['z']))
            n = xr.DataArray(n, dims=[vector,'point'], coords={vector: ['x', 'y', 'z']})
        else:
            raise CoordSysError()
    return to_vector(n)

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
    return {'z':np.array([z]), 'x': np.arange(shape[1])*spacing[0], 'y': np.arange(shape[2])*spacing[1]}
