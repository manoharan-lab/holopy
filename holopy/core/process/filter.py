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
Various kinds of image filtering operations 

.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Tom G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
"""
from __future__ import division
      
import scipy
import numpy as np
from ..math import fft, ifft
from ..errors import ImageError

def _mask(im_size, mask_rad):
    '''
    Simple circular mask

    Parameters
    ----------
    im_size : int
       size of output ndarray (assumed square)
    mask_rad : int 
       radius of mask, in pixels

    Returns
    -------
    mask : ndarray<Boolean>
       im_size x im_size Boolean ndarray; True for r from center <
       mask_rad  
    '''
    # resulting grid will be symmetric about the center if im_size even
    grid = scipy.mgrid[0:im_size, 0:im_size] - (im_size/2. - 0.5)
    rad = grid[0]**2 + grid[1]**2
    return rad < mask_rad**2

def pillbox(image, rad, optics = None):
    '''
    Do a convolution-based low pass filter with a circular mask

    Parameters
    ----------
    image : ndarray
       Image to filter (assumed square)
    rad : int 
       maximum radius of lowpass filter mask, in pixels

    Returns
    -------
    image : ndarray
       Low pass filtered image
    '''
    filter_mask = _mask(image.shape[0], rad)
    image_fft = fft(image) * filter_mask
    return ifft(image_fft).real

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
            raise ImageError('adjacent dead pixels')

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
        print 'Pixel with value 0 reset to nearest neighbor average'

    return output



# This function taken from http://www.scipy.org/Cookbook/SignalSmooth
# TODO convert to using scipy.ndimage.fourier.fourier_gaussian()
# instead.  
def lowpass(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = _gauss_kern(n, sizey=ny)
    improc = scipy.signal.fftconvolve(im, g, mode='valid')
    return improc

# This function taken from http://www.scipy.org/Cookbook/SignalSmooth
# TODO: delete _gauss_kern() in favor of
# scipy.ndimage.fourier.fourier_gaussian()
def _gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()
