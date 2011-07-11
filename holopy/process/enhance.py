# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
"""
Image enhancement through background subtraction, contrast adjustment,
or detrending

.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Tom G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
"""

import scipy
import numpy as np
from holopy.hologram import Hologram
from .math import fft, ifft
from .filter import zero_filter
from holopy.utility.helpers import _preserve_holo_type
from holopy.utility.errors import ImageError
subtract = 0
divide = 1

def normalize(image):
    """
    Normalize an image (numpy array) by dividing by the pixel average.
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

def background(holo, bg, kind = divide):
    '''
    Improve an image by eliminating a background.

    Parameters
    ----------
    holo : :class:`holopy.hologram.Hologram`
       Image to process
    bg : ndarray
       Background image to remove
    kind : 'subtract' or 'divide'
       Type of background elimination to perform

    Returns
    -------
    holo : :class:`holopy.hologram.Hologram`
       Hologram with background eliminated
    '''
    if bg.ndim < holo.ndim:
        bg = bg[..., np.newaxis]
    
    if kind is subtract or kind is 'subtract':
        name = "%sbgs%s" % (holo.name, bg.name)
        ar = holo - bg
    else:
        name = "%sbgd%s" % (holo.name, bg.name)
        ar = holo / zero_filter(bg)

    return Hologram(ar, holo.optics, holo.time_scale, name=name)

@_preserve_holo_type
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
    
