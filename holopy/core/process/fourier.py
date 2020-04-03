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
Handles Fourier transforms of HoloPy images by using scipy's fftpack. Tries to correctly interpret dimensions from xarray.

.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Tom G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
"""
import warnings

from scipy import fftpack
import numpy as np
import xarray as xr

from holopy.core.utils import ensure_array



_overwrite_deprectation_warning = (
    "The `overwrite` keyword is deprectated. Data will not be " +
    "overwritten when calling fft or ifft.")

def fft(data, overwrite=False, shift=True):
    """
    More convenient Fast Fourier Transform

    An easier to use fft function, it will pick the correct fft to do
    based on the shape of the Marray, and do the fftshift for you.  This
    is intended for working with images, and thus for dimensions
    greater than 2 does slicewise transforms of each "image" in a
    multidimensional stack

    Parameters
    ----------
    data : ndarray
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
    if overwrite is True:
        warnings.warn(_overwrite_deprectation_warning)
    data_np = data.values if isinstance(data, xr.DataArray) else data
    if data.ndim is 1:
        res = fftpack.fft(data_np, overwrite_x=overwrite)
        if shift:
            res = np.fft.fftshift(res)
    else:
        res = fftpack.fft2(
            data_np,
            axes=[data.dims.index('x'), data.dims.index('y')],
            overwrite_x=overwrite)
        if shift:
            res = np.fft.fftshift(
                res,
                axes=[data.dims.index('x'), data.dims.index('y')])

    if isinstance(data, xr.DataArray):
        res = xr.DataArray(res, **transform_metadata(data, False))
    return res


def ifft(data, overwrite=False, shift=True):
    """
    More convenient Inverse Fast Fourier Transform

    An easier to use ifft function, it will pick the correct ifft to
    do based on the shape of the Marry, and do the fftshift for you.
    This is intended for working with images, and thus for dimensions
    greater than 2 does slicewise transforms of each "image" in a
    multidimensional stack

    Parameters
    ----------
    data : ndarray
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
    ndarray
       The inverse fourier transform of `data`
    """
    if overwrite is True:
        warnings.warn(_overwrite_deprectation_warning)
    data_np = data.values if isinstance(data, xr.DataArray) else data
    if data_np.ndim is 1:
        res = fftpack.ifft(data_np, overwrite_x=overwrite)
        if shift:
            res = fftpack.fftshift(data_np, overwrite_x=overwrite)
    else:
        if shift:
            shifted = np.fft.fftshift(
                data_np,
                axes=[data.dims.index('m'), data.dims.index('n')])
            res = fftpack.ifft2(
                shifted,
                axes=[data.dims.index('m'), data.dims.index('n')],
                overwrite_x=overwrite)
        else:
            res = fftpack.ifft2(data_np, overwrite_x=overwrite)

    if isinstance(data, xr.DataArray):
        res = xr.DataArray(res, **transform_metadata(data, True))
    return res


#The following handles transforming coordinates for fft/ifft
def transform_metadata(a, inverse):
    dims=list(a.dims)

    if not inverse:
        coords = ft_coords(a.coords)
        dims[dims.index('x')]='m'
        dims[dims.index('y')]='n'
    else:
        dims[dims.index('m')]='x'
        dims[dims.index('n')]='y'
        coords = ift_coords(a.coords)

    return {'dims': dims, 'coords': coords, 'attrs': a.attrs, 'name': a.name}


def get_spacing(c):
    spacing = np.diff(c)
    if not np.allclose(spacing[0], spacing):
        raise ValueError("array has nonuniform spacing, can't determine coordinates for fft")
    return spacing[0]


def ft_coord(c):
    spacing = get_spacing(c)
    dim = len(c)
    ext = spacing * dim
    return np.linspace(-dim/(2*ext), dim/(2*ext), dim)


def ift_coord(c):
    spacing = get_spacing(c)
    dim = len(c)
    ext = spacing * dim
    return np.linspace(0, dim/ext, dim)


def ft_coords(cs):
    d = {k: v.values for k, v in cs.items()}
    d['m'] = ft_coord(d.pop('x'))
    d['n'] = ft_coord(d.pop('y'))
    return d


def ift_coords(cs):
    d = {k: v.values for k, v in cs.items()}
    d['x'] = ift_coord(d.pop('m'))
    d['y'] = ift_coord(d.pop('n'))
    return d

