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
Routines for manipulating, reconstructing, and fitting holograms

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
from __future__ import division

import numpy as np
import scipy.signal
from types import NoneType
from holopy.utility import errors

class Hologram(np.ndarray):
    """
    Class to store raw holograms

    All preprocessing, fitting, and reconstruction routines operate on
    Hologram objects.  These objects also contains metadata of the
    optical train used to create the hologram.

    Parameters
    ----------
    arr : numpy.ndarray
        raw data array of hologram.
    optics : :class:`holopy.optics.Optics` object (optional)
        optical train parameters
    time_scale : float or list of float (optional)
        time betwen frames or list of times of each frame
    from_origin : numpy.ndarray (2) (optional)
        upper left corner of hologram: (0,0) unless hologram is
        subimaged from a larger hologram

    Notes
    -----
    Hologram class stores the data in the measured hologram or
    hologram stack as an ndarray
    """
    # The reason I don't separate the hologram class from a separate
    # optics class is that the hologram image itself is only
    # meaningful if we know something about the optics.

    # subclassing ndarray is a little unusual.  I'm following the
    # instructions here:
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    # Normally we'd use an __init__ method, but subclassing ndarray
    # requires a __new__ method and an __array_finalize__ method
    def __new__(cls, arr, optics = None, time_scale = None, name = None,
                from_origin = None):
        if isinstance(arr, np.ndarray):
            # Input array is an already formed ndarray instance
            input_array = arr.copy()
        else:
            # see what happens.  For now I'm not going to throw an
            # exception here because it's possible that image has a
            # type that will be cast to an array by the np.ndarray
            # constructor
            input_array = arr.copy()

        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the metadata to the created instance
        obj.optics = optics
        obj.time_scale = time_scale
        obj.name = name
   
        # origin from which the image comes; reset if subimaged
        if from_origin.__class__ is NoneType:
            obj.from_origin = np.zeros(2, dtype = 'int')
        else:
            obj.from_origin = from_origin

        # Finally, we must return the newly created object:
        return obj

    @property
    def shape3d(self):
        """
        returns shape as a 3 tuple (if the hologram is 2d then the
        third dimension will be 1
        """
        if self.ndim == 2:
            return self.shape[0], self.shape[1], 1
        else:
            return self.shape


    def __array_finalize__(self, obj):
        # this function finishes the construction of our new object
        if obj is None: 
            return
        try:
            for var in obj.__dict__:
                setattr(self, var, getattr(obj, var))
        except AttributeError:
            # somehow sometimes we get something without a __dict__ just
            # ignoring it and waiting until we get something with a __dict__
            # seems to work
            pass


    def __array_wrap__(self, out_arr, context=None):
        # this function is needed so that if we run another numpy
        # function on the hologram (for example, numpy.add), the
        # metadata will be transferred to the new object that is
        # created
        if out_arr.ndim == 0:
            # if the thing we are returning is 0 dimension (a single value) ie
            # from .sum(), we want to return the number, not the number wrapped
            # in a hologram
            return out_arr.max()
        
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def resample(self, shape, window=None):
        """
        Resamples hologram to a given size.

        Use, for example, to downsample a hologram in a way that
        avoids aliasing and ringing.
        
        Parameters
        ----------
        shape : int or 2-tuple of ints
            shape of final resampled hologram
        window : string
            type of smoothing window passed to the scipy.signal.resample
            filter.

        Returns
        -------
        new_image : :class:`holopy.hologram.Hologram` object

        Notes
        -----
        This algorithm does 2 1-D resamplings.  
        
        """
        name = None
        if np.isscalar(shape):
            x, y = (shape, shape)
            if self.name is not None:
                name = self.name + 'r({0},{1})'.format(x, y)
        else:
            x, y = shape
            if self.name is not None:
                name = self.name+'r{0}'.format(x)
        new_image = scipy.signal.resample(self, x, axis=0, window=window)
        new_image = scipy.signal.resample(new_image, y, axis=1, 
                                          window=window)

        # Change the pixel calibration appropriately.  If image is
        # downsampled, pixel scale should increase
        factor = np.array(new_image.shape).astype('float')/self.shape

        # return a new hologram, now divorced from its optical train
        return Hologram(new_image, self.optics.resample(1.0/factor),
                        self.time_scale, name)


def subimage(im, center=None, size=None):
    """
    Pick out a subimage of a given size.

    Moves the subimage center to fit if needed and possible

    Parameters
    ----------
    im : ndarray
        the image from which the subimage is to be extracted
    center : (int, int) tuple (optional)
        The center of the subimage.  Defaults to center of im.
    size : int or (int, int) tuple (optional)
        number of pixels across subimage.  Defaults to largest power
        of 2 that fits within im

    Returns
    -------
    Hologram : :class:`holopy.hologram.Hologram` object

    Raises
    ------
    TooBigError
        if the subimage cannot fit within im
    """
    if center == None:
        # default to center of image
        center = [dim/2 for dim in im.shape]
    
    if size == None:
        # default to larges power of 2 that fits within im
        size = 2**int(np.floor(np.log2(min(im.shape)))) 

    if np.isscalar(size):
        # if only one dimension given, make a square subimage
        lx, ly = size, size
    else:
        lx, ly = size
        
    def constrain(center, length, maxval):
        '''
        Makes sure subimage is within bounds of image.
        Moves center of subimage to be within bounds.
        '''
        # move the center if needed to fit the given subimage within
        # the image 
        x1 = center+(length + 1) // 2
        # Use so that, for example, the middle pixel in length 9 comes
        # out is 5 instead of for.  Truncation will cause 8 to still
        # give a middle pixel of 4 as desired.

        #Re-center subimage
        if x1 > maxval:
            center -= x1 - (maxval-1)
            x1 = maxval-1
            
        x0 = x1 - length
        if x0 < 0:
            x1 = length
            x0 = 0

        if x1 > maxval:
            raise errors.OutOfBounds('Specified subimage is too large to fit ' +
                                     'within image')

        return x0, x1
    

    x0, x1 = constrain(center[0], lx, im.shape[0])
    y0, y1 = constrain(center[1], ly, im.shape[1])

    if hasattr(im, 'name'):
        n = "%s@%sx%s" % (im.name, center, size)
    
    # might go wrong if lx,ly are not divisible by 2, but they
    # probably will be, so I am not worrying about it
    return Hologram(im[x0:x1, y0:y1, ...], optics=im.optics,
                    time_scale=im.time_scale, name=n, 
                    from_origin = np.array([x0, y0]))


