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
Routines for manipulating, reconstructing, and fitting data

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
from __future__ import division

import numpy as np
import scipy.signal
import errors
from .holopy_object import HolopyObject

class DataTarget(HolopyObject):
    """
    Generic specifier desired data

    Attributes
    ----------
    meauserements: np.ndarray, or object
        Specification of the locations of measurements
    **kwargs: varies
        Other metadata
    """
    def __init__(self, positions, *args, **kwargs):
        self._metadata = kwargs
        for key, item in self._metadata.iteritems():
            setattr(self, key, item)
        self.positions = positions
        self._metadata['positions'] = positions

    @property
    def _dict(self):
        return self._metadata

class Grid(HolopyObject):
    """
    Rectangular grid of measurements
    """
    def __init__(self, shape, spacing):
        self.shape = shape
        self.spacing = spacing

class Data(DataTarget, np.ndarray):
    """
    Class to store raw data

    All preprocessing, fitting, and reconstruction routines operate on
    Data objects.  These objects also contains metadata of the
    optical train used to create the data.

    Parameters
    ----------
    arr : numpy.ndarray
        raw data array of data.
    metadata : :class:`holopy.metadata.Metadata` object (optional)
        optical train parameters
    """

    # subclassing ndarray is a little unusual.  I'm following the
    # instructions here:
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    # Normally we'd use an __init__ method, but subclassing ndarray
    # requires a __new__ method and an __array_finalize__ method
    def __new__(cls, arr, *args, **kwargs):
        return np.array(arr).view(cls)

    def __init__(self, arr, positions = None, *args, **kwargs):
        super(Data, self).__init__(positions, *args, **kwargs)

    def __repr__(self):
        array_repr = repr(self.view(np.ndarray))[6:-1]
        holopy_obj_repr = super(Data, self).__repr__()
        tok = holopy_obj_repr.split('(', 1)
        return "{0}({1}, {2}".format(tok[0], array_repr, tok[1])


    def __array_finalize__(self, obj):
        # this function finishes the construction of our new object by copying
        # over the metadata
        for item in getattr(obj, '_metadata', []):
            setattr(self, item, getattr(obj, item))

    def __array_wrap__(self, out_arr, context=None):
        # this function is needed so that if we run another numpy
        # function on the data (for example, numpy.add), the
        # metadata will be transferred to the new object that is
        # created
        if out_arr.ndim == 0:
            # if the thing we are returning is 0 dimension (a single value) ie
            # from .sum(), we want to return the number, not the number wrapped
            # in a data
            return out_arr.max()
        
        return np.ndarray.__array_wrap__(self, out_arr, context)


class Image(Data):
    """
    2D pixel data on a rectangular grid
    """
    def __new__(cls, arr, pixel_size=None, optics=None, **kwargs):
        return super(Image, cls).__new__(cls, arr, **kwargs)

    def __init__(self, arr, pixel_size=None, optics=None, **kwargs):
        if pixel_size is None and optics is not None:
            pixel_size = optics.pixel
        if np.isscalar(pixel_size):
            pixel_size = np.repeat(pixel_size, self.ndim)
        super(Image, self).__init__(arr = arr, positions = Grid(self.shape, pixel_size), optics
                                    = optics, **kwargs)

    def resample(self, shape, window=None):
        """
        Resamples data to a given size.

        Use, for example, to downsample a data in a way that
        avoids aliasing and ringing.
        
        Parameters
        ----------
        shape : int or 2-tuple of ints
            shape of final resampled data
        window : string
            type of smoothing window passed to the scipy.signal.resample
            filter.

        Returns
        -------
        new_image : :class:`holopy.data.Data` object

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

        # return a new data, now divorced from its optical train
        return Image(new_image, self.metadata.resample(1.0/factor),
                        self.time_scale, name)

class Volume(Data):
    """
    3D pixel data on a rectangular grid
    """
    pass

class Timeseries(Data):
    pass


    
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
    Data : :class:`holopy.data.Data` object

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
    return Data(im[x0:x1, y0:y1, ...], metadata=im.metadata,
                    time_scale=im.time_scale, name=n, 
                    from_origin = np.array([x0, y0]))


