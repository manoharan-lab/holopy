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
.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
"""
from scipy.signal import detrend as dt
from scipy.ndimage import gaussian_filter
import numpy as np
import xarray as xr

from holopy.core.errors import BadImage
from holopy.core.metadata import (
    copy_metadata, update_metadata, detector_grid, get_spacing, get_values)


def normalize(image):
    """
    Normalize an image by dividing by the pixel average.
    This gives the image a mean value of 1.

    Parameters
    ----------
    image : xarray.DataArray
       The array to normalize

    Returns
    -------
    normalized_image : xarray.DataArray
       The normalized image
    """
    return copy_metadata(image, image * 1.0 / image.sum() * image.size)


def detrend(image):
    '''
    Remove linear trends from an image.

    Performs a 2 axis linear detrend using scipy.signal.detrend

    Parameters
    ----------
    image : xarray.DataArray
       Image to process

    Returns
    -------
    image : xarray.DataArray
       Image with linear trends removed
    '''
    return copy_metadata(image, dt(dt(image, image.dims.index('x')), image.dims.index('y')))


def zero_filter(image):
    '''
    Search for and interpolate pixels equal to 0.
    This is to avoid NaN's when a hologram is divided by a BG with 0's.
    Interpolation fails if any of the four corner pixels are 0.

    Parameters
    ----------
    image : xarray.DataArray
       Image to process

    Returns
    -------
    image : xarray.DataArray
       Image where pixels = 0 are instead given values equal to average of
       neighbors.  dtype is the same as the input image
    '''
    filtered = xr.where(image > 0, image, np.nan)
    filtered = [filtered.interpolate_na(dim=xy) for xy in 'xy']
    filtered = xr.concat(filtered, dim='dummy')
    filtered = filtered.mean(dim='dummy', skipna=True)
    if np.isnan(filtered).any().item():
        raise BadImage('Image has dead pixels in corners, cannot interpolate')
    return copy_metadata(image, filtered)


def subimage(arr, center, shape):
    """
    Pick out a region of an image or other array

    Parameters
    ----------
    arr : xarray.DataArray
        The array to subimage
    center : tuple of ints or floats
        The desired center of the region, should have the same number of
        elements as the arr has dimensions. Floats will be rounded
    shape : int or (int, int)
        Desired shape of the region in x & y dimensions. If a single int is
        given it is applied along both axes. Shape values must be even.

    Returns
    -------
    sub : xarray.DataArray
        Subset of shape shape centered at center. DataArray coords
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
    image : xarray.DataArray
        The image to add noise to.
    smoothing : float
        Fraction of the image size to smooth by. Should in general be << 1
    poisson_lambda : float
        Used to compute the shape of the noise distribution. You can generally
        leave this at its default value unless you are simulating shot noise
        limited imaging.

    Returns
    -------
    noisy_image : xarray.DataArray
       A copy of the input image with noise added.

    """
    return copy_metadata(image, image + simulate_noise(image.shape, noise_mean, smoothing,
                                  poisson_lambda) * get_values(image.mean()))


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


def bg_correct(raw, bg, df=None):
    """Correct for noisy images by dividing by a background. The calculation used is (raw-df)/(bg-df). 

    Parameters
    ----------
    raw : xarray.DataArray
        Image to be background divided.
    bg : xarray.DataArray
        background image recorded with the same optical setup.
    df : xarray.DataArray
        dark field image recorded without illumination.

    Returns
    -------
    corrected_image : xarray.DataArray
       A copy of the background divided input image with None values of noise_sd updated to match bg.

    """
    if df is None:
        df = raw.copy()
        df[:] = 0

    if not (raw.shape == bg.shape == df.shape and list(get_spacing(raw)) == list(get_spacing(bg)) == list(get_spacing(df))):
        raise BadImage("raw and background images must have the same shape and spacing")

    holo = (raw - df) / zero_filter(bg - df)
    holo = copy_metadata(raw, holo)

    if hasattr(holo, 'noise_sd') and hasattr(bg, 'noise_sd') and holo.noise_sd is None:
        holo = update_metadata(holo, noise_sd = bg.noise_sd)

    return holo
