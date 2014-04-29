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
Storing measurements and calculated results.  This is done through
arrays with metadata (Marray and subclasses).  It also includes Schema which
specifies how results should be computed in an analogous interface to how Marrays are
specified.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
from __future__ import division

import warnings
import copy
import numpy as np
import scipy.signal
from errors import UnspecifiedPosition
from .holopy_object import HoloPyObject
from .metadata import Angles, Positions
from .helpers import _ensure_pair, _ensure_array, dict_without, ensure_3d
import inspect

def zeros_like(obj, dtype=None):
    """
    Make an empty marray with the same shape and metadata as the template.

    Parameters
    ----------
    obj : :class:`.Schema`
        Template object (schema or marray)
    dtype : :class:`numpy.dtype`
        Optional argument to override array type

    Returns
    -------
    arr : :class:`.Marray`
        Empty (all zeros) marray with shape and metadata copied from obj
    """
    if isinstance(obj, np.ndarray):
        return obj.__class__(np.zeros_like(obj, dtype=dtype), **obj._dict)
    else:
        return obj._corresponding_marray(np.zeros(obj.shape, dtype=dtype),
                   **dict_without(obj._dict, 'shape'))

def arr_like(arr, template=None, **override):
    """
    Make a new Marray with metadata like an old one

    Parameters
    ----------
    arr : numpy.ndarray
        Array data to add metadata to
    template : :class:`.Schema` (optional)
        Marray to copy metadata from. If not given, will be copied from arr
        (probably used in this case for overrides)
    **override : kwargs
        Optional additional keyword args. They will be used to override
        specific metadata

    Returns
    -------
    res : :class:`.Marray`
        Marray like template containing data from arr
    """
    if template is None:
        template = arr

    if not hasattr(template, '_dict'):
        return arr
    meta = template._dict
    meta.update(override)
    return template.__class__(arr, **meta)

def _describe_init_signature(cls):
    """
    Decorator to facilitate documentation of Marray subclasses.

    This decorator documents the attributes of a class's constructor using a
    common set of decriptions for arguments.  {attrs} in the class's docstring
    will be replaced with a NumPy docstring formatted decription of the
    arguments the class's __init__ takes.
    """

    # setup a dictionary of all the keyword attrs marray classes use
    # If you use this method with a new subclass that takes a new argument, you
    # will need to add it here.
    attrs = {}
    attrs = {'shape' : """
    shape : tuple(int)
        shape of the desired Marray object.  Should be a tuple of 1 or more
        integers for  the shape along each axis.  """,
             'arr' : """
    arr : array
        raw array data/calculations (without metadata)""",
             'positions' : """
    positions : :class:`.PositionSpecification` object
        Specification of the locations of measurements""",
             'spacing' : """
    spacing : array(dtype=float)
        spacing between values along each axis in the {name}""",
             'optics' : """
    optics : :class:`.Optics` object
        Information about the optical train associated with this schema""",
             'origin' : """
    origin : (float, float, float)
        Offset for the origin of the space represented by this Schema.  """,
             'components' : """
    components : list(string) default 'x', 'y', 'x'
        Names of the vector components for this {name}""",
             'dtype' : """
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method."""}

    for key, val in attrs.iteritems():
        if val[0] == '\n':
            val = val[1:]
        # This lets us referr to the class's name as {name} in the attribute
        # documentation snippits above
        attrs[key] = val.format(name = cls.__name__)

    argspec = inspect.getargspec(cls.__init__)
    # leave off self (the first argument)
    args = argspec.args[1:]
    if argspec.keywords is not None:
        args.append(argspec.keywords)

    attr_sig = "Parameters\n    ----------\n{0}".format(
        '\n'.join([attrs[arg] for arg in args if arg in attrs]))
    cls.__doc__ = cls.__doc__.format(attrs = attr_sig)
    return cls


@_describe_init_signature
class Schema(HoloPyObject):
    """Most general description of a desired array with metadata

    A Schema contains metadata for an array and a description of what
    data the array should hold. Schema themselves do not contain array
    data.

    {attrs}
    """
    def __init__(self, shape=None, positions=None, optics=None,
                 origin=np.zeros(3), metadata={}, **kwargs):
        self._positions = positions
        self.optics = optics
        self.origin = origin
        self.metadata = metadata
        # if we are a np.ndarray subclass (if this constructor is
        # called from Marray or subclass) we will already have a shape
        # and should not try to do anything with our shape argument.
        if not hasattr(self, 'shape'):
            if shape is None and hasattr(positions, 'shape'):
                shape = positions.shape[:-1]
            self.shape = shape
        super(Schema, self).__init__(**kwargs)

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, val):
        self._positions = val

    # TODO: put this somewhere sensible, make it handle phi as well
    def positions_theta_phi(self):
        if isinstance(self.positions, Angles):
            return self.positions.positions_theta_phi()
        else:
            raise UnspecifiedPosition()

    # TODO: need equivalents for set_metadata or to rewrite other code so it
    # doesn't need it.

    def get_metadata_from(self, schema, prefer_self=True):
        """
        Fill in the metadata for this schema from another

        Parameters
        ----------
        schema : Schema
            Other schema from which to get metadata for this one
        prefer_mine : Bool
            If true, use values in self any case they exist and only get values
            from schema where they are absent. If true, overwrite self's values
            with those from schema

        Returns
        -------
        Nothing : updates in place
        """
        if prefer_self:
            newdict = copy.copy(schema._dict)
            newdict.update(self._dict)
        else:
            newdict = self._dict
            newdict.update(schema._dict)
        if isinstance(self, np.ndarray):
            newdict = dict_without(newdict, 'shape')

        for key, item in newdict.iteritems():
            setattr(self, key, item)



def call_super_init(cls, self, consumed=[], **kwargs):
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
    # now add any new args to the call dict.  We add explicitly specified args
    # last so they will overwrite other specifications as desired.
    call_args.update(kwargs)

    # and finally call the superclass's __init__
    super(cls, self).__init__(**call_args)



@_describe_init_signature
class Marray(np.ndarray, Schema):
    """
    Generic Array with metadata

    You usually should use a more appropriate subclass, raw Marrays are somewhat
    cumbersome to use.

    {attrs}
    """
    def __new__(cls, arr, positions=None, optics=None,
                origin=np.zeros(3), dtype=None, **kwargs):
        # things like numpy.std give us 0d arrays, the user probably expects
        # python scalars, so return one instead.
        if hasattr(arr, 'ndim') and arr.ndim == 0:
            # arr.max pulls out the singular value of the 0d array
            return arr.max()
        return np.asarray(arr, dtype = dtype).view(cls)

    def __init__(self, arr, positions=None, optics=None,
                 origin=np.zeros(3), dtype=None, **kwargs):
        call_super_init(Marray, self, ['arr', 'dtype'])

    def __array_finalize__(self, obj):
        # this function finishes the construction of our new object by copying
        # over the metadata
        for key, item in getattr(obj, '__dict__', {}).iteritems():
            setattr(self, key, item)


    def __array_wrap__(self, out_arr, context=None):
        # this function is needed so that if we run another numpy
        # function on the Marray (for example, numpy.add), the
        # metadata will be transferred to the new object that is
        # created
        if out_arr.ndim == 0:
            # if the thing we are returning is 0 dimension (a single value) ie
            # from .sum(), we want to return the number, not the number wrapped
            # in a Marray
            return out_arr.max()

        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __repr__(self):
        keywpairs = ["{0}={1}".format(k[0], repr(k[1])) for k in
                     self._dict.iteritems()]
        # we want te print numpy's repr
        arr = np.ndarray.__repr__(self)
        # but take out the class name since we are including it ourself
        arr = 'array(' + arr.split('(', 1)[1]
        return "{0}(arr={1}, \n{2})".format(self.__class__.__name__, arr,
                                          ", ".join(keywpairs))

    @property
    def _dict(self):
        # I believe in all cases where we use Marray._dict we don't actually
        # want the dtype, it is always safer to infer it from the underlying
        # ndarray.   dtype is only provided as a constructor argument in case we
        # want to override the default selection.
        return dict_without(super(Marray, self)._dict, 'dtype')

    # we have to implement our own std because the numpy one stubbornly returns
    # 0d Marray objects which we don't want
    def std(self, axis=None, dtype=None, out=None, ddof=0):
        result = super(Marray, self).std(axis=None, dtype=None, out=None, ddof=0)
        if result.ndim == 0:
            return result.max()
        else:
            return result


class RegularGridSchema(Schema):
    def __init__(self, shape=None, spacing=None, optics=None,
                 origin=np.zeros(3), **kwargs):

        call_super_init(RegularGridSchema, self, consumed = ['spacing'])

        # the setter will set the positions correctly
        self.spacing = spacing

    @property
    def spacing(self):
        return getattr(self, '_spacing', None)
    @spacing.setter
    def spacing(self, spacing):
        # if we change the spacing, we have to change our positions
        # object to match
        if getattr(self, 'spacing', None) is spacing or self.shape is None:
            # Fast path, since there are a number of things that will
            # set schema attributes to the same as they already are
            return
        else:
            if np.isscalar(spacing):
                spacing = np.repeat(spacing, len(self.shape))

            self._spacing = spacing

    # This could be cached, it only changes when the shape, spacing or
    # origin is changed, which is less common than we call
    # positions. However, that is only a minor optimization, (doing
    # something with the positions almost always takes much longer
    # than computing them) so I am not bothering with it at the moment
    # -tgd 2013-12-09
    @property
    def positions(self):
        if self.shape is None:
            return Positions()
        # Compute the coordinates of each point in the grid

        # make it 3d even if the image is 2d
        if len(self.shape) == 2:
            shape = np.append(self.shape, 1)
            spacing = np.append(self.spacing, 1)
        else:
            shape = self.shape
            spacing = self.spacing
        grid_slice = [slice(0, d) for d in shape]
        xyz = np.mgrid[grid_slice].astype('float64')
        pos = np.zeros(np.append(shape, 3))
        for i, s in enumerate(spacing):
            pos[..., i] = xyz[i, ...] * s
        return Positions(pos + self.origin)

    @positions.setter
    def positions(self, val):
        raise Error("Positions of RegularGrids are determined automatically, "
                    "you should not try to set them directly")


    @property
    def extent(self):
        ext = np.array(self.shape) * self.spacing
        if len(ext) == 2:
            ext = np.append(ext, 0)
        return ext

    @property
    def center(self):
        return self.origin + self.extent/2

    @center.setter
    def center(self, value):
        self.origin = ensure_3d(value) - self.extent/2

    def contains(self, point):
        return ((point >=self.origin).all() and
                (point <= self.origin+self.extent).all())

    @property
    def ndim(self):
        return len(self.shape)

class VectorSchema(Schema):
    def __init__(self, shape=None, positions=None,
                 components=('x', 'y', 'z'), optics=None,
                 origin=np.zeros(3), **kwargs):
        self.components = components
        call_super_init(VectorSchema, self, ['components'])

    def interpret_1d(self, arr):
        return VectorMarray(arr.reshape(self.shape), **dict_without(self._dict, ['shape']))

class VectorGridSchema(RegularGridSchema, VectorSchema):
    def __init__(self, shape=None, spacing=None,
                 components=('x', 'y', 'z'), optics=None,
                 origin=np.zeros(3), **kwargs):
        call_super_init(VectorGridSchema, self)

    @property
    def extent(self):
        # the last dimension of shape is the field components, we need
        # to cut it to have the same dimension as self.shape
        ext = np.array(self.shape)[:-1] * self.spacing
        if len(ext) == 2:
            ext = np.append(ext, 0)
        return ext

    def interpret_1d(self, arr):
        return VectorGrid(arr.reshape(self.shape),
                          **dict_without(self._dict, ['shape']))


def make_vector_schema(schema, components=('x', 'y', 'z')):
    if isinstance(schema, VectorSchema):
        shape = schema.shape
    else:
        shape = np.append(schema.shape, len(components))
    if isinstance(schema, RegularGridSchema):
        new = VectorGridSchema(components=components, shape=shape,
                               **dict_without(schema._dict, ['shape', 'dtype', 'components']))
    else:
        new =  VectorSchema(components = components, shape = shape,
                   **dict_without(schema._dict, ['shape', 'dtype', 'components']))

    return new


class RegularGrid(Marray, RegularGridSchema):
    def __init__(self, arr, spacing=None, optics=None,
                 origin=np.zeros(3), metadata={}, dtype=None, **kwargs):
        call_super_init(RegularGrid, self)

    def resample(self, shape, window=None):
        """
        Resamples Marray to a given shape.

        Use, for example, to downsample a Marray in a way that
        avoids aliasing and ringing.

        Parameters
        ----------
        shape : int or array_like of ints
            shape of final resampled results
        window : string
            type of smoothing window passed to the scipy.signal.resample
            filter.

        Returns
        -------
        new_image : :class:`holopy.marray.Marray` object

        Notes
        -----
        This algorithm does two 1D resamplings.

        """
        shape = _ensure_array(shape)
        new = self
        factors = np.zeros(len(shape))
        for i, s in enumerate(shape):
            if s != self.shape[i]:
                factors[i] = 1.0 * self.shape[i] / s
                new = scipy.signal.resample(new, s, axis=i, window=window)

                new = self.__class__(new, **self._dict)
        new.spacing = self.spacing * factors
        return new


@_describe_init_signature
class ImageSchema(RegularGridSchema):
    """
    Description of a desired Image.

    An ImageSchema contains all of the information needed to calculate an Image

    {attrs}
    """
    def __init__(self, shape=None, spacing=None, optics=None,
                 origin=np.zeros(3), metadata={}, **kwargs):
        if shape is not None:
            shape = _ensure_pair(shape)

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

        call_super_init(ImageSchema, self)

    @property
    def size(self):
        return self.shape[0]*self.shape[1]


@_describe_init_signature
class Image(RegularGrid, ImageSchema):
    """
    2D rectangular grid of measurements or calculations.

    {attrs}
    """
    pass

class VectorMarray(Marray, VectorSchema):
    def __init__(self, arr, positions=None,
                 components=('x', 'y', 'z'),
                 optics=None, origin=np.zeros(3), **kwargs):
        call_super_init(VectorMarray, self, ['components'])

class VectorGrid(RegularGrid, VectorGridSchema, VectorMarray):
    """Vector Data on a Rectangular Grid

    {attrs}
    """
    def __init__(self, arr, spacing=None, components=('x', 'y', 'z'),
                 optics=None, origin=np.zeros(3), dtype=None,
                 **kwargs):
        call_super_init(VectorGrid, self)

    @property
    def x_comp(self):
        return self._component(0)

    @property
    def y_comp(self):
        return self._component(1)

    @property
    def z_comp(self):
        return self._component(2)

    def _component(self, comp):
        if self.ndim == 2:
            return Image(self[...,comp],
                         **dict_without(self._dict, 'components'))
        else:
            return Volume(self[...,comp],
                          **dict_without(self._dict, 'components'))


@_describe_init_signature
class VolumeSchema(RegularGridSchema):
    """
    Description of a desired Volume.

    An VolumeSchema contains all of the information needed to calculate a Volume

    {attrs}
    """
    def __init__(self, shape=None, spacing=None, optics=None,
                 origin=np.zeros(3), **kwargs):
        call_super_init(VolumeSchema, self)


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

    extent = [slice(c-s/2, c+s/2) for c, s in zip(center, shape)] + [Ellipsis]
    output = _checked_cut(arr, extent)

    if isinstance(output, RegularGridSchema):
        if output.spacing != None:
            output.center = arr.origin + ensure_3d(center) * ensure_3d(arr.spacing)
        else:
            output.origin = None
    return output

def resize(arr, center=None, extent=None, spacing=None):
    """
    Resize and resample an marray

    Parameters
    ----------
    arr : :class:`.Marray`
        Marray to resize
    center : array(float) optional
        Desired center of the new marray. Default is the old center
    extent : array(float) optional
        Desired extent of the new marray. Default is the old extent
    spacing : array(float) optional
        Desired spacing of the new marray. Default is the old spacing

    Returns
    -------
    arr : :class:`.Marray`
        Desired cut of arr. Will be a view into the old array unless spacing
        is changed
    """
    if center is None:
        center = arr.center
    if extent is None:
        extent = arr.extent
    center = np.array(center)
    extent = np.array(extent)
    # we need to cut spacing and origin down to two dimensions if working with
    # an Image
    cut_center = (center - arr.origin[:arr.ndim])/arr.spacing[:arr.ndim]
    shape = extent / arr.spacing[:arr.ndim]

    extent = [slice(int(np.round(c -s/2)), int(np.round(c+s/2)))
              for c, s in zip(cut_center, shape)]

    arr = _checked_cut(arr, extent)
    arr.center = center
    if spacing is not None and np.any(spacing != arr.spacing):
        shape = (arr.extent / spacing).astype('int')
        arr = arr.resample(shape)

    return arr


_describe_init_signature
class Volume(RegularGrid, VolumeSchema):
    """
    3D rectangular grid of measurements or calculations.

    {attrs}
    """
    pass


def squeeze(arr):
    """
    Turns an NxMx1 array into an NxM array.
    """
    keep = [i for i, dim in enumerate(arr.shape) if dim != 1]
    if not hasattr(arr,'spacing') or arr.spacing == None:
        spacing = None
    else:
        spacing = np.take(arr.spacing, keep)
    return arr_like(np.squeeze(arr), arr,
                    spacing = spacing)


# common code for subimage and resize
def _checked_cut(arr, extent):
    for i, axis in enumerate(extent):
        if axis is not Ellipsis and (axis.start < 0 or axis.stop > arr.shape[i]):
            raise IndexError

    return arr[extent].copy()

ImageSchema._corresponding_marray = Image
VolumeSchema._corresponding_marray = Volume
VectorGridSchema._corresponding_marray = VectorGrid
