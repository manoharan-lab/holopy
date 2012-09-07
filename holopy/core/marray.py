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
Storing measurements and calculated data.  This is done through
Arrays with metadata (Marray and subclases).  It also includes Schema which
specify how data should be computed in an analogous interface to how Marrays are
specified.  

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
from __future__ import division

import warnings
import copy
import numpy as np
import scipy.signal
import errors
from .holopy_object import HolopyObject
from .metadata import Grid, Angles
from .helpers import _ensure_pair, _ensure_array
import inspect

# Ancestor for all array like storage objects for data/calculations.  
class PseudoMarray(HolopyObject):
    def __init__(self, positions = None, optics = None, origin = None,
                 use_random_fraction = None):
        self.positions = positions
        self.optics = optics
        self.origin = origin
        self.use_random_fraction = use_random_fraction

    @property
    def selection(self):
        # we only generate a random selection once
        if not hasattr(self, '_selection'):
            if self.use_random_fraction is not None:
                self._selection = (np.random.random((self.positions.shape)) >
                                   1.0-self.use_random_fraction)
            else:
                self._selection = None

        # if it already exists, we just return the cached _selection
        return self._selection

    # TODO: put this somewhere sensible, make it handle phi as well
    def positions_theta(self):
        if isinstance(self.positions, Angles):
            return self.positions.theta
        else:
            raise UnspecifiedPosition()

    # TODO: need equivalents for set_metadata or to rewrite other code so it
    # doesn't need it.  


def dict_without(d, items):
    d = copy.copy(d)
    for item in _ensure_array(items):
        try:
            del d[item]
        except KeyError:
            pass
    return d


def call_super_init(cls, self, consumed = [], **kwargs):
    # this function uses a little inspect magic to call the superclass's __init__
    # the arguments to the current __init__ modulo the arguments consumed or
    # added in the current __init__
    
    # get the arguments passed to the function that called this function
    call = inspect.getargvalues(inspect.currentframe().f_back)
    call_args = dict([(arg, call.locals[arg]) for arg in call.args])
    # TODO: may want to grab add all the kwargs from call, but right now the
    # only one I think we are using is **kwargs
    call_args.update(call.locals['kwargs'])
    # pull out any of the args that we have been informed have been used and
    # should not be passed up the chain
    del call_args['self']
    call_args = dict_without(call_args, consumed)
    # now add any new args to the call dict
    call_args.update(kwargs)
    
    # and finally call the superclass's __init__
    super(cls, self).__init__(**call_args)
        
class Marray(PseudoMarray, np.ndarray):
    def __new__(cls, arr, positions = None, optics = None, origin = None,
                use_random_fraction = None, dtype = None, **kwargs):
        # things like numpy.std give us 0d arrays, the user probably expects
        # python scalars, so return one instead.  
        if hasattr(arr, 'ndim') and arr.ndim == 0:
            # arr.max pulls out the singular value of the 0d array
            return arr.max()
        return np.asarray(arr, dtype = dtype).view(cls)
    
    def __init__(self, arr, positions = None, optics = None, origin = None,
                 use_random_fraction = None, dtype = None, **kwargs):
        call_super_init(Marray, self, ['arr', 'dtype'])

    def __array_finalize__(self, obj):
        # this function finishes the construction of our new object by copying
        # over the metadata
        for key, item in getattr(obj, '__dict__', {}).iteritems():
            setattr(self, key, item)
        

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

    # we have to implement our own std because the numpy one stubbornly returns
    # 0d Data objects which we don't want
    def std(self, axis=None, dtype=None, out=None, ddof=0):
        result = super(Data, self).std(axis=None, dtype=None, out=None, ddof=0)
        if result.ndim == 0:
            return result.max()
        else:
            return result

    @classmethod
    def zeros_like(cls, marray, dtype = None):
        return cls(np.zeros_like(marray, dtype = dtype), **marray._dict) 


class GenericSchema(PseudoMarray):
    """
    Specification of what data should look like.

    A Schema should specify the positions where data would be measured and
    any other metadata that would be associated with the data.  Data target
    objects are used to specify the output format of various calculation
    functions and to provide needed metadata for the calculation.  

    Attributes
    ----------
    shape : tuple(int)
        shape of the data object.  Should be a tuple of 1 or more integers for
        the shape along each axis.  
    positions : :class:`PositionSpecification` object
        Specification of the locations of measurements
    optics : :class:`metadata.Optics` object
        Information about the optical train associated with this data target
    origin : (float, float, float)
        Offset for the origin of the space represented by this Schema.  
    use_random_fraction: float
        Use only a random fraction of the pixels specified by this DataTarget
        when doing calculations.  This can give a good representation of the
        whole field with much lower computation times
    **kwargs : varies
        Other metadata
    """

    def __init__(self, shape, positions = None, optics = None, origin = None,
                 use_random_fraction = None, **kwargs):
        self.shape = shape
        call_super_init(GenericSchema, self, 'shape')
    

class PseudoRegularGrid(PseudoMarray):
    def __init__(self, spacing, optics = None, origin = None,
                 use_random_fraction = None, **kwargs):
        
        if np.isscalar(spacing):
            spacing = np.repeat(spacing, len(self.shape))
        call_super_init(PseudoRegularGrid, self, consumed = 'spacing',
                        positions = Grid(spacing))

    @property
    def spacing(self):
        return self.positions.spacing

    def resample(self, shape, window=None):
        """
        Resamples data to a given shape.

        Use, for example, to downsample a data in a way that
        avoids aliasing and ringing.
        
        Parameters
        ----------
        shape : int or array_like of ints
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
        shape = _ensure_array(shape)
        new = self
        factors = {}
        for i, s in enumerate(shape):
            if s != self.shape[i]:
                factors[i] = 1.0 * self.shape[i] / s
                new = scipy.signal.resample(new, s, axis=i, window=window)

        new = self.__class__(new, **self._dict)
        new.positions = new.positions.resample_by_factors(factors)
        return new



        
class PseudoImage(PseudoRegularGrid):
    def __init__(self, spacing = None, optics = None, origin = None,
                 use_random_fraction = None, **kwargs):
        # legacy code.  We have allowed specifying spacing in the optics, I am
        # trying to depricate that now, but this will keep it working as people
        # expect.  
        if spacing is None:
            if (hasattr(optics, 'pixel_scale') and
                optics.pixel_scale is not None):
                spacing = optics.pixel_scale
                warnings.warn("Specifying pixel_scale in optics is depricated, "
                              "use Image pixel_size or similar instead")
                optics = copy.copy(optics)
                del optics.pixel_scale

        call_super_init(PseudoImage, self)

    # subclasses must provide a self.shape
                
    # these functions can be generalized for other kinds of marrays (I think),
    # look into changing the algorithms so we can push this up the inheritance tree
    def positions_r_theta_phi(self, origin):
        """
        Returns a list of positions of each data point, in spherical coordinates
        relative to origin.  
        
        Parameters
        ----------
        origin : (real, real, real)
            origin of the spherical cooridate system to return

        Returns
        -------
        theta, phi : 1-D array
            Angles
        r : 2-D array
            Distances
        """
        x, y, z = origin
        px, py = self.positions.spacing
        xdim, ydim = self.shape
        xg, yg = np.ogrid[0:xdim, 0:ydim]
        x = xg*px - x
        y = yg*py - y
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        # get phi between 0 and 2pi
        phi = phi + 2*np.pi * (phi < 0)
        points = np.dstack((r, theta, phi))
        if hasattr(self, 'selection') and self.selection is not None:
            points = points[self.selection]
            if not self.selection.any():
                raise errors.InvalidSelection("No pixels selected, can't compute fields")
        else:
            points = points.reshape((-1, 3))
        return points

    def positions_kr_theta_phi(self, origin):
        pos = self.positions_r_theta_phi(origin)
        pos[:,0] *= self.optics.wavevec
        return pos

    
class ImageSchema(GenericSchema, PseudoImage):
    
    pass

class Image(Marray, PseudoImage):
    pass

class PseudoVolume(PseudoRegularGrid):
    pass

class PseudoVectorImage(PseudoImage):
    def __init__(self, spacing, components = ('x', 'y', 'z'), optics = None,
                 origin = None, use_random_fraction = None, **kwargs):
        self.components = components
        call_super_init(PseudoVectorImage, self, 'components')

class VectorImage(Marray, PseudoVectorImage):
    pass
        

class VectorImageSchema(GenericSchema, PseudoVectorImage):
    def __init__(self, shape, spacing, components = ('x', 'y', 'z'), optics = None,
                 origin = None, use_random_fraction = None, **kwargs):
        self.components = components
        call_super_init(VectorImageSchema, self, 'components')

    @classmethod
    def from_ImageSchema(cls, image_schema, components = ('x', 'y', 'z')):
        if isinstance(image_schema, VectorImageSchema):
            shape = image_schema.shape
        else:
            shape = np.append(image_schema.shape, len(components))
        return cls(components = components, shape = shape,
                   spacing = image_schema.positions.spacing, 
                   **dict_without(image_schema._dict, ['shape', 'positions']))

    def interpret_1d(self, arr):
        if self.selection is None:
            return VectorImage(arr.reshape(self.shape), **dict_without(self._dict, ['shape']))
        else:
            new = VectorImage.zeros_like(self, dtype = arr.dtype)
            new[self.selection] = arr
            return new
        

class VolumeSchema(GenericSchema, PseudoVolume):
    pass

class Volume(Marray, PseudoVolume):
    pass
    

