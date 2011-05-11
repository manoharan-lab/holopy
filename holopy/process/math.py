# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
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
Misc utility functions to make coding more convenient

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

import scipy.fftpack as fftpack
from holopy.utility.helpers import _preserve_holo_type


@_preserve_holo_type
def fft(a, overwrite=False, shift=True):
    """
    More convienent Fast Fourier Transform
    
    An easier to use fft function, it will pick the correct fft to do
    based on the shape of the data, and do the fftshift for you.  This
    is indendended for working with images, and thus for dimensions
    greater than 2 does slicewise transforms of each "image" in a
    multidimensional stack 

    Parameters
    ----------
    a : ndarray
       The array to transform
    overwrite : bool
       Allow this function to overwrite the data you pass in.  This may improve
       performance slightly.  Default is not to overwrite
    shift : bool
       Whether to preform an fftshift on the data to give low frequences near
       the center as you probably expect.  Default is to do the fftshift.

    Returns
    -------
    fta : ndarray
       The fourier transform of a
    """
    if shift:
        shift = lambda x: fftpack.fftshift(x, axes=[0,1]) #Only words for ndim>1
    else:
        shift = lambda x: x

    if a.ndim is 1:
        return fftpack.fftshift(fftpack.fft(a, overwrite_x=overwrite))
    else:
        return shift(fftpack.fft2(a, axes=[0,1], overwrite_x=overwrite))

@_preserve_holo_type
def ifft(a, overwrite=False, shift=True):
    """
    More convienent Inverse  Fast Fourier Transform
    
    An easier to use ifft function, it will pick the correct ifft to do
    based on the shape of the data, and do the fftshift for you.  This is
    indendended for working with images, and thus for dimensions greater than 2
    does slicewise transforms of each "image" in a multidimensional stack

    Parameters
    ----------
    a : ndarray
       The array to transform
    overwrite : bool
       Allow this function to overwrite the data you pass in.  This may improve
       performance slightly.  Default is not to overwrite
    shift : bool
       Whether to preform an fftshift on the data to give low frequences near
       the center as you probably expect.  Default is to do the fftshift.

    Returns
    -------
    ifta : ndarray
       The inverse fourier transform of a
    """

    if shift:
        shift = lambda x: fftpack.fftshift(x, axes=[0,1])
    else:
        shift = lambda x: x
        
    if a.ndim is 1:
        return fftpack.ifft(fftpack.fftshift(a))
    else:
        return fftpack.ifft2(shift(a), axes=[0,1], overwrite_x=overwrite)
    

