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
Image enhancement through background subtraction, contrast adjustment,
or detrending

.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Tom G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
"""


from ..errors import BadImage
from .math import simulate_noise
from .utilities import is_none, ensure_3d, copy_metadata, get_values
from scipy.signal import detrend
from scipy import fftpack
import numpy as np
import xarray as xr

def normalize(image):
    """
    Normalize an image (NumPy array) by dividing by the pixel average.
    This gives the image a mean value of 1.

    Parameters
    ----------
    image : ndarray
       The array to normalize

    Returns
    -------
    normalized_image : ndarray
       The normalized image
    """
    return copy_metadata(image, image * 1.0 / image.sum() * image.size)

def detrend(image):
    '''
    Remove linear trends from an image.

    Performs a 2 axis linear detrend using scipy.signal.detrend

    Parameters
    ----------
    image : ndarray
       Image to process

    Returns
    -------
    image : ndarray
       Image with linear trends removed
    '''
    return copy_metadata(image, detrend(detrend(image, 0), 1))

def zero_filter(image):
    '''
    Search for and interpolate pixels equal to 0.
    This is to avoid NaN's when a hologram is divided by a BG with 0's.

    Parameters
    ----------
    image : ndarray
       Image to process

    Returns
    -------
    image : ndimage
       Image where pixels = 0 are instead given values equal to average of
       neighbors.  dtype is the same as the input image
    '''
    zero_pix = np.where(image == 0)
    output = image.copy()

    # check to see if adjacent pixels are 0, if more than 1 dead pixel
    if len(zero_pix[0]) > 1:
        delta_rows = zero_pix[0] - np.roll(zero_pix[0], 1)
        delta_cols = zero_pix[1] - np.roll(zero_pix[1], 1)
        if ((1 in delta_rows[np.where(delta_cols == 0)]) or
            (1 in delta_cols[np.where(delta_rows == 0)])):
            raise BadImage('Image has adjacent dead pixels, cannot remove dead pixels')

    for row, col in zip(zero_pix[0], zero_pix[1]):
        # in the bulk
        if ((row > 0) and (row < (image.shape[0]-1)) and
            (col > 0) and (col < image.shape[1]-1)):
            output[row, col] = np.sum(image[row-1:row+2, col-1:col+2]) / 8.
        else: # deal with edges by padding
            im_avg = image.sum()/(image.size - len(zero_pix[0]))
            padded_im = np.ones((image.shape[0]+2, image.shape[1]+2)) * im_avg
            padded_im[1:-1, 1:-1] = image
            output[row, col] = np.sum(padded_im[row:row+3, col:col+3]) / 8.
        print('Pixel with value 0 reset to nearest neighbor average')

    return copy_metadata(image, output)

def add_noise(image, noise_mean=.1, smoothing=.01, poisson_lambda=1000):
    """Add simulated noise to images. Intended for use with exact
    calculated images to make them look more like noisy 'real'
    measurements.

    Real image noise usually has correlation, so we smooth the raw
    random variable. The noise_mean can be controlled independently of
    the poisson_lambda that controls the shape of the distribution. In
    general, you can stick with our default of a large poisson_lambda
    (ie for imaging conditions not near the shot noise limit).

    Defaults are set to give noise vaguely similar to what we tend to
    see in our holographic imaging.

    Parameters
    ----------
    image : ndarray or Image
        The image to add noise to.
    intensity : float
        How large the noise mean should be relative to the image mean
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
    return copy_metadata(image, image + simulate_noise(image.shape, noise_mean, smoothing,
                                  poisson_lambda) * image.mean())

def subimage(arr, center, shape):
    """
    Pick out a region of an image or other array

    Parameters
    ----------
    arr : numpy.ndarray
        The array to subimage
    center : tuple of ints or floats
        The desired center of the region, should have the same number of
        elements as the arr has dimensions. Floats will be rounded
    shape : int or tuple of ints
        Desired shape of the region.  If a single int is given the region will
        be that dimension in along every axis.  Shape should be even

    Returns
    -------
    sub : numpy.ndarray or :class:`.RegularGrid` marray object
        Subset of shape shape centered at center. For marrays, marray.origin
        will be set such that the upper left corner of the output has
        coordinates relative to the input.
    """
    center = (np.round(center)).astype(int)

    if np.isscalar(shape):
        shape = np.repeat(shape, arr.ndim)
    assert len(shape) == arr.ndim

    def intr(n):
        return intr(np.round(n))

    extent = [slice(int(np.round(c-s/2)), int(np.round(c+s/2))) for c, s in zip(center, shape)]

    return copy_metadata(arr, arr.isel(x=extent[0], y=extent[1]))

def get_spacing(c):
    spacing = np.diff(c)
    if not np.allclose(spacing[0], spacing):
        raise ValueError("array has nonuniform spacing, can't determine coordinates for fft")
    return spacing[0]

def ft_coord(c):
    spacing = get_spacing(c)
    dim = len(c)
    ext = spacing * dim
    return np.linspace(-get_values(dim/(2*ext)), get_values(dim/(2*ext)), dim)

def ift_coord(c):
    spacing = get_spacing(c)
    dim = len(c)
    ext = spacing * dim
    return np.linspace(0, get_values(dim/ext), dim)

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

def fft(a, overwrite=False, shift=True):
    """
    More convenient Fast Fourier Transform

    An easier to use fft function, it will pick the correct fft to dom
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
            res = fftpack.fftshift(fftpack.fft2(a, axes=[a.dims.index('x'), a.dims.index('y')],
                                                 overwrite_x=overwrite),
                                    axes=[a.dims.index('x'), a.dims.index('y')])
        else:
            res = fftpack.fft2(a, axes=[a.dims.index('x'), a.dims.index('y')], overwrite_x=overwrite)

    if hasattr(a, 'coords') and hasattr(a, 'attrs'):
        res = xr.DataArray(res, **transform_metadata(a, False))
        res.name = a.name
    return res

def transform_metadata(a, inverse):
    if not inverse:
        coords = ft_coords(a.coords)
        dims = ['m', 'n']
    else:
        coords = ift_coords(a.coords)
        dims = ['x', 'y']
    if 'z' in coords and coords['z'].shape is not ():
        dims = dims + ['z']
    if 'vector' in coords:
        dims = ['vector'] + dims
    return {'dims': dims, 'coords': coords, 'attrs': a.attrs}

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
            res = fftpack.ifft2(fftpack.fftshift(a, axes=[a.dims.index('m'), a.dims.index('n')]), axes=[a.dims.index('m'), a.dims.index('n')],
                                 overwrite_x=overwrite)
        else:
            res = fftpack.ifft2(a, overwrite_x=overwrite)

    if hasattr(a, 'coords') and hasattr(a, 'attrs'):
        res = xr.DataArray(res, **transform_metadata(a, True))
        res.name = a.name
    return res

