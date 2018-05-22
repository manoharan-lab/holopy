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
from .utils import is_none, updated, repeat_sing_dims
from .math import to_spherical, to_cartesian


vector = 'vector'
illumination = 'illumination'

def data_grid(arr, spacing=None, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=None, noise_sd=None, name=None, extra_dims=None, z=0):
    """
    Create a set of detector points along with other experimental metadata.

    Returns
    -------
    DataArray object

    Notes
    -----
    Use of higher-level detector_grid() and detector_points() functions is 
    recommended.
    """

    if spacing is None:
        spacing = 1
        warn("No pixel spacing provided. Setting spacing to 1, but any subsequent calculations will be wrong.")
    if name is None:
        name = 'data'

    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)
    if np.isscalar(z) and (len(arr) > 1 or arr.ndim==2):
        arr=np.array([arr])
    coords = make_coords(arr.shape, spacing, z)
    if is_none(extra_dims):
        extra_dims={}
    else:
        coords.update(extra_dims)
    out = xr.DataArray(arr, dims=['z','x', 'y']+list(extra_dims.keys()), coords = coords, name=name)
    return update_metadata(out, medium_index, illum_wavelen, illum_polarization, normals, noise_sd)

def detector_grid(shape, spacing, normals = None, name = None, extra_dims=None):
    """
    Create a rectangular grid of pixels to represent a detector on which
    scattering calculations are to be performed.

    Parameters
    ----------
    shape : int or list-like (2)
        If int, detector is a square grid of shape x shape points. 
        If array_like, detector has \ *shape*\ [0] rows and \ *shape*\ [1] columns.
    spacing : int or list-like (2)
        If int, distance between square detector pixels.
        If array_like, \ *spacing*\ [0] between adjacent rows and \ *spacing*\ [1] 
        between adjacent columns.
    normals : list-like or None
        If list-like, detector orientation.
    name : string, optional
    extra_dims : dict, optional
        extra dimension(s) to add to the empty detector grid as {dimname:[coords]}

    Returns
    -------
    grid : DataArray object
        DataArray of zeros with coordinates calculated according to \ *shape* \
        and \ *spacing*\

    Notes
    -----
    Typically used to define a set of points to represent the pixels of a 
    digital camera in scattering calculations.
        
    """
    if np.isscalar(shape):
        shape = [shape]*2
    else:
        shape = list(shape)

    if extra_dims is not None:
        for val in extra_dims.values():
            shape.append(len(val))
    d = np.zeros(shape)
    return data_grid(d, spacing, normals = normals, name = name, extra_dims=extra_dims)

def detector_points(coords = {}, x = None, y = None, z = None, r = None, theta = None, phi = None, normals = 'auto', name = None):
    """
    Returns a one-dimensional set of detector coordinates at which scattering 
    calculations are to be done.

    Parameters
    ----------
    coords : dict, optional
        Dictionary of detector coordinates. Default: empty dictionary.
        Typical usage should not pass this argument, giving other parameters
        (Cartesian `x`, `y`, and `z` or polar `r`, `theta`, and `phi` 
        coordinates) instead.
    x, y : int or array_like, optional
        Cartesian x and y coordinates of detectors.
    z : int or array_like, optional
        Cartesian z coordinates of detectors. If not specified, assume `z` = 0.
    r : int or array_like, optional
        Spherical polar radial coordinates of detectors. If not specified,
        assume `r` = infinity (far-field).
    theta : int or array_like, optional
        Spherical polar coordinates (polar angle from z axis) of detectors.
    phi : int or array_like, optional
        Spherical polar azimuthal coodinates of detectors.
    normals : string, optional
        Default behavior: normal in +z direction for Cartesian coordinates,
        -r direction for polar coordinates. Non-default behavior not currently
        implemented.
    name : string

    Returns
    -------
    grid : DataArray object
        DataArray of zeros with calculated coordinates.

    Notes
    -----
    Specify either the Cartesian or the polar coordinates of your detector. 
    This may be helpful for modeling static light scattering calculations.
    Use detector_grid() to specify coordinates of a grid of pixels (e.g., 
    a digital camera.)

    """
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

def update_metadata(a, medium_index=None, illum_wavelen=None, illum_polarization=None, normals=None, noise_sd=None):
    """Returns a copy of an image with updated metadata in its 'attrs' field.

    Parameters
    ----------
    a : xarray.DataArray
        image to update.
    medium_index : float
        Updated refractive index of the medium in the image.
    illum_wavelen : float
        Updated wavelength of illumination in the image.
    illum_polarization : list-like
        Updated polarization of illumination in the image.
    normals : list-like
        Updated detector orientation of the image.
    noise_sd : float
        standard deviation of Gaussian noise in the image.
   
    Returns
    -------
    b : xarray.DataArray
        copy of input image with updated metadata. The 'normals' field is not allowed to be empty.
    """

    attrlist = {'medium_index': medium_index, 'illum_wavelen': dict_to_array(a,illum_wavelen), 'illum_polarization': dict_to_array(a,to_vector(illum_polarization)), 'normals': to_vector(normals), 'noise_sd': dict_to_array(a,noise_sd)}
    b = a.copy()
    b.attrs = updated(b.attrs, attrlist)

    for attr in attrlist:
        if not hasattr(b, attr):
            b.attrs[attr] = None

    if is_none(b.normals):
        b.attrs['normals'] = default_norms(b.coords, 'auto')

    return b

def copy_metadata(old, data, do_coords=True):

    def find_and_rename(oldkey, oldval):
        for newkey, newval in new.coords.items():
            if np.array_equal(oldval.values, newval.values):
                return new.rename({newkey: oldkey})
            raise ValueError("Coordinate {} does not appear to have a coresponding coordinate in {}".format(oldkey, new))
    
    new=data.copy()

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
    if isinstance(c, dict):
        c = c.copy()
        for key, val in c.items():
            c[key]=to_vector(val)
        return c
    c = np.array(c)
    if c.shape == (2,):
        c = np.append(c, 0)

    return xr.DataArray(c, coords={vector: ['x', 'y', 'z']}, dims=vector)

def flat(a):
    if hasattr(a, 'flat') or hasattr(a, 'point'):
        return a
    else:
        return a.stack(flat=('x','y','z'))

def from_flat(a):
    if hasattr(a, 'flat'):
        return a.unstack('flat')
    return a

def sphere_coords(a, origin=(0,0,0), wavevec=1):
    if hasattr(a,'theta') and hasattr(a, 'phi'):
        out = {'theta': a.theta.values, 'phi': a.phi.values, 'point':a.point.values}
        if hasattr(a, 'r') and any(np.isfinite(a.r)):
            out['r'] = a.r.values * wavevec
        return out

    else:
        if origin is None:
            raise ValueError('Cannot convert detector to spherical coordinates without an origin')
        f = flat(a)
        dimstr = primdim(f)
        # we define positive z opposite light propagation, so we have to invert
        x, y, z = f.x.values - origin[0], f.y.values - origin[1], origin[2] - f.z.values
        out = to_spherical(x,y,z)
        return updated(out, {'r':out['r'] * wavevec, dimstr:f[dimstr]})

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

def get_extents(im):
    def get_extent(d):
        if len(im[d]) < 2:
            return 0
        # Add one extra spacing since the xarray coords are right edge only,
        # but we actually want right edge of first pixel to left edge of last
        # pixel
        return float(im[d][-1] - im[d][0] + np.diff(im[d]).mean())
    return {d: get_extent(d) for d in ['x','y','z'] if d in im.dims}

def make_coords(shape, spacing, z=0):
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)
    return {'z':np.array([z]), 'x': np.arange(shape[1])*spacing[0], 'y': np.arange(shape[2])*spacing[1]}

def clean_concat(arrays, dim):
    attrs = arrays[0].attrs
    arrays = [array.assign_attrs(**{attr:None for attr in array.attrs if isinstance(array.attrs[attr],xr.DataArray)}) for array in arrays]
    arrays = xr.concat(arrays, dim)
    arrays.attrs = attrs
    return arrays.transpose(*np.roll(arrays.dims,-1))

def dict_to_array(schema, inval):
    if isinstance(inval, dict):
        keys = sorted(list(inval.keys()))
        dims = {coord:sorted(list(schema.coords[coord].values)) for coord in schema.coords}
        for name, coords in dims.items():
            if keys == coords:
                if isinstance(list(inval.values())[0], xr.DataArray):
                    dim = xr.DataArray(list(inval.keys()), dims=name, name=name)
                    return xr.concat(list(inval.values()), dim = dim)
                else:
                    return xr.DataArray(list(inval.values()), dims=name, coords={name:list(inval.keys())})
        raise ValueError("Dictionary could not be converted to DataArray because reference grid has no dimensions with matching coords")
    elif hasattr(inval, 'from_parameters'):
        #inval is a Scatterer object
        pars = inval.parameters
        pars = {key:dict_to_array(schema, val) for key, val in pars.items()}
        return(inval.from_parameters(pars))
    else:
        return inval
