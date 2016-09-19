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
Image enhancement through background subtraction, contrast adjustment,
or detrending

.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Tom G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
"""


from ..errors import BadImage


import scipy
import numpy as np

subtract = 0
divide = 1

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
    return image * 1.0 / image.sum() * image.size

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
    return scipy.signal.detrend(scipy.signal.detrend(image, 0), 1)

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
            raise UnusableImage('Image has adjacent dead pixels, cannot remove dead pixels')

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

    return output
