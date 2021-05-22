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

from warnings import warn

import yaml
import numpy as np
import xarray as xr

from holopy.core.utils import updated, repeat_sing_dims, ensure_array
from holopy.core.math import to_cartesian
from holopy.core.errors import CoordSysError


vector = 'vector'
illumination = 'illumination'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               Methods part of the Holopy API
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def detector_grid(shape, spacing, name=None, extra_dims=None):
    """
    Create a rectangular grid of pixels to represent a detector on which
    scattering calculations are to be performed.

    Parameters
    ----------
    shape : int or list-like (2)
        If int, detector is a square grid of shape x shape points.
        If array_like, detector has \ *shape*\ [0] rows and
        \ *shape*\ [1] columns.
    spacing : int or list-like (2)
        If int, distance between square detector pixels.
        If array_like, \ *spacing*\ [0] between adjacent rows and
        \ *spacing*\ [1] between adjacent columns.
    name : string, optional
    extra_dims : dict, optional
        extra dimension(s) to add to the empty detector grid as
        {dimname: [coords]}.

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
    return data_grid(d, spacing, name=name, extra_dims=extra_dims)


def detector_points(coords={}, x=None, y=None, z=None, r=None, theta=None,
                    phi=None, name=None):
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
        if coords.get('z') is None:
            coords['z'] = 0

    elif 'theta' in coords and 'phi' in coords:
        keys = ['r', 'theta', 'phi']
        if coords.get('r') is None:
            coords['r'] = np.inf
    else:
        raise CoordSysError()

    if name is None:
        name = 'data'

    coords = repeat_sing_dims(coords, keys)
    coords = updated(coords, {key: ('point', coords[key]) for key in keys})
    return xr.DataArray(np.zeros(len(coords[keys[0]][1])), dims=['point'],
                        coords=coords, name=name)


def clean_concat(arrays, dim):
    """Concatenate a list of xr.DataArray objects along a specified dimension,
    keeping the metadata of the first array.

    Parameters
    ----------
    arrays : list of ``xr.xarray``
    dim : valid dimension (string)

    Returns
    -------
    xarray
    """
    attrs = arrays[0].attrs
    arrays = [
        array.assign_attrs(
            **{attr: None
               for attr in array.attrs
               if isinstance(array.attrs[attr], xr.DataArray)}
            )
        for array in arrays]
    arrays = xr.concat(arrays, dim)
    arrays.attrs = attrs
    return arrays.transpose(*np.roll(arrays.dims, -1))


def update_metadata(a, medium_index=None, illum_wavelen=None,
                    illum_polarization=None, noise_sd=None):
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
    noise_sd : float
        standard deviation of Gaussian noise in the image.

    Returns
    -------
    b : xarray.DataArray
        copy of input image with updated metadata.
    """
    attrlist = {'medium_index': medium_index,
                'illum_wavelen': dict_to_array(a, illum_wavelen),
                'illum_polarization': dict_to_array(
                    a, to_vector(illum_polarization)),
                'noise_sd': dict_to_array(a, noise_sd)}
    b = a.copy()
    b.attrs = updated(b.attrs, attrlist)

    for attr in attrlist:
        if not hasattr(b, attr):
            b.attrs[attr] = None
    return b


def get_spacing(detector_grid):
    """Find the (x, y) spacing for a ```detector_grid```."""
    xspacing = np.diff(detector_grid.x)
    yspacing = np.diff(detector_grid.y)
    if not (np.allclose(xspacing[0], xspacing) and
            np.allclose(yspacing[0], yspacing)):
        msg = "array has nonuniform spacing, can't determine a single spacing"
        raise ValueError(msg)
    return np.array((xspacing[0], yspacing[0]))


def get_extents(detector_grid):
    """Find the x, y, z extent of a ```detector_grid```, as a dict."""
    if np.ndim(detector_grid) == 1:
        raise ValueError("Cannot get extent for detector_points")
    def get_extent(d):
        if len(detector_grid[d]) < 2:
            return 0
        # Add one extra spacing since xarray coords are taken to be at
        # pixel centers, but we actually want right edge of first pixel
        # to left edge of last pixel
        return float(
            detector_grid[d][-1] - detector_grid[d][0] +
            np.diff(detector_grid[d]).mean())
    return {d: get_extent(d)
            for d in ['x', 'y', 'z'] if d in detector_grid.dims}


def copy_metadata(old, data, do_coords=True):
    """
    Create a new `xarray` with data from one input and metadata from another.

    Parameters
    ----------
    old : `xr.DataArray`
        The xarray to copy the metadata from.
    data : `xr.DataArray`
        The xarray to copy the data from.
    do_coords : bool, optional
        Whether or not to copy the coordinates. Default is True

    Returns
    `xr.DataArray`
    """

    def find_and_rename(oldkey, oldval):
        for newkey, newval in new.coords.items():
            if np.array_equal(oldval.values, newval.values):
                return new.rename({newkey: oldkey})
            msg = ("Coordinate {} does not appear to have ".format(oldkey) +
                   "a corresponding coordinate in {}".format(new))
            raise ValueError(msg)

    new = data.copy()

    old_is_xarray = isinstance(old, xr.DataArray)
    if old_is_xarray:
        if not hasattr(new, 'coords'):
            # new is a numpy array, not xarray
            new = xr.DataArray(new, dims=['x', 'y'])
        new.attrs = old.attrs
        new.name = old.name

        if hasattr(old, 'flat') and hasattr(new, 'flat'):
            new['flat'] = old['flat']
        if do_coords:
            for key, val in old.coords.items():
                if key not in new.coords:
                    new = find_and_rename(key, val)
    return new


def make_subset_data(data, pixels=None, return_selection=False, seed=None):
    """Sub-sample a data for faster inference.

    Parameters
    ----------
    data : `xr.DataArray`
        The data to subsample
    pixels : int, optional
        The number of pixels to subsample. Defaults to the entire image.
    return_selection : bool, optional
        Whether to return the pixel indices which were sampled.
        Default is False
    seed : int or None, optional
        If not None, the seed to seed the random number generator with.

    Returns
    -------
    subset : `xr.DataArray`

    [selection : np.ndarray, dtype int]
    """
    if pixels is None:
        return data
    if seed is not None:
        np.random.seed(seed)
    tot_pix = len(data.x) * len(data.y)
    selection = np.random.choice(tot_pix, pixels, replace=False)
    subset = flat(data).isel(flat=selection)
    subset = copy_metadata(data, subset, do_coords=False)

    subset.attrs['original_dims'] = {key: data[key].values for key in data.dims}

    if return_selection:
        return subset, selection
    else:
        return subset


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             Methods not part of the Holopy API
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def data_grid(arr, spacing=None, medium_index=None, illum_wavelen=None,
              illum_polarization=None, noise_sd=None, name=None,
              extra_dims=None, z=0):
    """
    Create a set of detector points along with other experimental metadata.

    Returns
    -------
    DataArray object

    Notes
    -----
    Use the higher-level detector_grid() and detector_points() functions.
    This should be viewed as a factory method.
    """
    if spacing is None:
        spacing = 1
        warn("No pixel spacing provided. Setting spacing to 1, but any"
             "subsequent calculations will be wrong.")
    if name is None:
        name = 'data'

    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)
    if np.isscalar(z) and (len(arr) > 1 or arr.ndim == 2):
        arr = np.expand_dims(arr, axis=0)
    coords = make_coords(arr.shape, spacing, z)
    if extra_dims is None:
        extra_dims = {}
    else:
        coords.update(extra_dims)
    dims = ['z', 'x', 'y'] + list(extra_dims.keys())
    out = xr.DataArray(arr, dims=dims,  coords=coords, name=name)
    out = update_metadata(
        out, medium_index=medium_index, illum_wavelen=illum_wavelen,
        illum_polarization=illum_polarization, noise_sd=noise_sd)
    return out


def to_vector(c):
    if c is None or c is False:
        return c
    if hasattr(c, vector):
        return c
    if isinstance(c, dict):
        c = c.copy()
        for key, val in c.items():
            c[key] = to_vector(val)
        return c

    c = np.array(c)
    if c.shape == (2,):
        c = np.append(c, 0)
    # normalize
    c = c/np.sqrt(np.sum(c**2))

    return xr.DataArray(c, coords={vector: ['x', 'y', 'z']}, dims=vector)


def flat(a):
    if hasattr(a, 'flat') or hasattr(a, 'point'):
        return a
    else:
        return a.stack(flat=('x', 'y', 'z'))


def from_flat(a):
    if hasattr(a, 'flat'):
        return a.unstack('flat')
    return a


def get_values(a):
    return getattr(a, 'values', a)


def make_coords(shape, spacing, z=0):
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    if np.isscalar(spacing):
        spacing = np.repeat(spacing, 2)
    to_return = dict([
        ('z', ensure_array(z)),
        ('x', np.arange(shape[1]) * spacing[0]),
        ('y', np.arange(shape[2]) * spacing[1]),
        ])
    return to_return


def dict_to_array(schema, inval):
    if isinstance(inval, dict):
        keys = sorted(list(inval.keys()))
        dims = {coord: sorted(list(schema.coords[coord].values))
                for coord in schema.coords}
        for name, coords in dims.items():
            if keys == coords:
                if isinstance(list(inval.values())[0], xr.DataArray):
                    dim = xr.DataArray(list(inval.keys()), dims=name, name=name)
                    return xr.concat(list(inval.values()), dim=dim)
                else:
                    return xr.DataArray(
                        list(inval.values()),
                        dims=name,
                        coords={name: list(inval.keys())})
        msg = ("Dictionary could not be converted to DataArray because " +
               "reference grid has no dimensions with matching coords")
        raise ValueError(msg)
    else:
        return inval

