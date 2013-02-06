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
import errors
from errors import UnspecifiedPosition
from .holopy_object import HoloPyObject
from .metadata import Grid, Angles
from .helpers import _ensure_pair, _ensure_array
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
             'use_random_fraction' : """
    use_random_fraction : float
        Use only a random fraction of the pixels specified by this Schema
        when doing calculations.  This can give a good representation of the
        whole field with much lower computation times.""",
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
                 origin=np.zeros(3), use_random_fraction=None,
                 **kwargs):
        self.positions = positions
        self.optics = optics
        self.origin = origin
        self.use_random_fraction = use_random_fraction
        # if we are a np.ndarray subclass (if this constructor is
        # called from Marray or subclass) we will already have a shape
        # and should not try to do anything with our shape argument.
        if not hasattr(self, 'shape'):
            if shape is None and hasattr(positions, 'shape'):
                shape = positions.shape
            self.shape = shape
        super(Schema, self).__init__(**kwargs)

    @property
    def selection(self):
        # we only generate a random selection once
        if not hasattr(self, '_selection'):
            if self.use_random_fraction is not None:
                self._selection = self._make_selection()
            else:
                self._selection = None

        # if it already exists, we just return the cached _selection
        return self._selection

    def _make_selection(self):
        return np.random.random(self.shape) > 1.0-self.use_random_fraction

    # TODO: put this somewhere sensible, make it handle phi as well
    def positions_theta_phi(self):
        if isinstance(self.positions, Angles):
            return self.positions.positions_theta_phi()
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
    def __new__(cls, arr, positions=None, optics=None, origin=np.zeros(3),
                use_random_fraction=None, dtype=None, **kwargs):
        # things like numpy.std give us 0d arrays, the user probably expects
        # python scalars, so return one instead.
        if hasattr(arr, 'ndim') and arr.ndim == 0:
            # arr.max pulls out the singular value of the 0d array
            return arr.max()
        return np.asarray(arr, dtype = dtype).view(cls)

    def __init__(self, arr, positions=None, optics=None, origin=np.zeros(3),
                 use_random_fraction=None, dtype=None, **kwargs):
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
                 origin=np.zeros(3), use_random_fraction=None,
                 **kwargs):

        if np.isscalar(spacing):
            if hasattr(self, 'shape'):
                shape = self.shape
            spacing = np.repeat(spacing, len(shape))
        call_super_init(RegularGridSchema, self, consumed = 'spacing',
                        positions = Grid(spacing))

    def positions_r_theta_phi(self, origin):
        """
        Returns a list of positions of each data point, in spherical coordinates
        relative to origin.

        Parameters
        ----------
        origin : (real, real, real)
            origin of the spherical coordinate system to return

        Returns
        -------
        theta, phi : 1-D array
            Angles
        r : 2-D array
            Distances
        """

        x, y, z = (np.array(origin) - self.origin)

        g = np.ogrid[[slice(0, d*s, s) for d, s in zip(self.shape, self.spacing)]]
        if len(g) == 2:
            xg, yg = g
            zg = 0
        else:
            xg, yg, zg = g

        x = xg - x
        y = yg - y
        # sign is reversed for z because of our choice of image
        # centric rather than particle centric coordinate system
        z = z - zg

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        # get phi between 0 and 2pi
        phi = phi + 2*np.pi * (phi < 0)
        # if z is an array, phi will be the wrong shape. Checking its
        # last dimension will determine this so we can correct it
        if phi.shape[-1] != r.shape[-1]:
            phi = phi.repeat(r.shape[-1], -1)
        points = np.concatenate([a[..., np.newaxis] for a in (r, theta, phi)], -1)
        if hasattr(self, 'selection') and self.selection is not None:
            points = points[self.selection]
            if not self.selection.any():
                raise errors.InvalidSelection("No pixels selected, can't compute fields")
        else:
            points = points.reshape((-1, 3))
        return points
        x, y, z = origin

    def positions_kr_theta_phi(self, origin):
        pos = self.positions_r_theta_phi(origin)
        pos[:,0] *= self.optics.wavevec
        return pos

    @property
    def spacing(self):
        return self.positions.spacing

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
        if len(value) == 2:
            value = np.append(value, 0)
        self.origin = value - self.extent/2

    def contains(self, point):
        return ((point >=self.origin).all() and
                (point <= self.origin+self.extent).all())


class VectorGridSchema(RegularGridSchema):
    def __init__(self, shape=None, spacing=None,
                 components=('x', 'y', 'z'), optics=None,
                 origin=np.zeros(3), use_random_fraction=None, **kwargs):
        self.components = components
        call_super_init(VectorGridSchema, self, 'components')

    @property
    def extent(self):
        # the last dimension of shape is the field components, we need
        # to cut it to have the same dimension as self.shape
        ext = np.array(self.shape)[:-1] * self.spacing
        if len(ext) == 2:
            ext = np.append(ext, 0)
        return ext

    def _make_selection(self):
        return np.random.random(self.shape[:-1]) > 1.0-self.use_random_fraction

    def interpret_1d(self, arr):
        if self.selection is None:
            return VectorGrid(arr.reshape(self.shape),
                                  **dict_without(self._dict, ['shape']))
        else:
            new = zeros_like(self, dtype = arr.dtype)
            new[self.selection] = arr
            return new

    @classmethod
    def from_schema(cls, schema, components=('x', 'y', 'z')):
        if isinstance(schema, VectorGridSchema):
            shape = schema.shape
        else:
            shape = np.append(schema.shape, len(components))

        new =  cls(components = components, shape = shape,
                   spacing = schema.positions.spacing,
                   **dict_without(schema._dict, ['shape', 'positions', 'spacing', 'dtype',
                                                 'components']))
        # we want to use the same random selection as the schema we come from did
        if hasattr(schema, '_selection'):
            new._selection = schema._selection

        return new


class RegularGrid(Marray, RegularGridSchema):
    def __init__(self, arr, spacing=None, optics=None, origin=np.zeros(3),
                 use_random_fraction=None, dtype=None, **kwargs):
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
        factors = {}
        for i, s in enumerate(shape):
            if s != self.shape[i]:
                factors[i] = 1.0 * self.shape[i] / s
                new = scipy.signal.resample(new, s, axis=i, window=window)

                new = self.__class__(new, **self._dict)
        new.positions = self.positions.resample_by_factors(factors)
        return new


@_describe_init_signature
class ImageSchema(RegularGridSchema):
    """
    Description of a desired Image.

    An ImageSchema contains all of the information needed to calculate an Image

    {attrs}
    """
    def __init__(self, shape=None, spacing=None, optics=None,
                 origin=np.zeros(3), use_random_fraction=None, **kwargs):
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


@_describe_init_signature
class Image(RegularGrid, ImageSchema):
    """
    2D rectangular grid of measurements or calculations.

    {attrs}
    """
    pass


class VectorGrid(RegularGrid, VectorGridSchema):
    """Vector Data on a Rectangular Grid

    {attrs}
    """
    def __init__(self, arr, spacing=None, optics=None, origin=np.zeros(3),
                 use_random_fraction=None, dtype=None,
                 components=('x', 'y', 'z'), **kwargs):
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
                 origin=np.zeros(3), use_random_fraction=None, **kwargs):
        call_super_init(VolumeSchema, self)


def subimage(arr, center, shape):
    """
    Pick out a region of an image or other array

    Parameters
    ----------
    arr : numpy.ndarray
        The array to subimage
    center : tuple of ints
        The desired center of the region, should have the same number of
        elements as the arr has dimensions
    shape : int or tuple of ints
        Desired shape of the region.  If a single int is given the region will
        be that dimension in along every axis.  Shape should be even

    Returns
    -------
    sub : numpy.ndarray
        Subset of shape shape centered at center
    """
    assert len(center) == arr.ndim
    if np.isscalar(shape):
        shape = np.repeat(shape, arr.ndim)
    assert len(shape) == arr.ndim

    extent = [slice(c-s/2, c+s/2) for c, s in zip(center, shape)]
    # TODO: BUG: get coordinate offset correct (reset origin)
    return _checked_cut(arr, extent)

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
    keep = [i for i, dim in enumerate(arr.shape) if dim != 1]
    return arr_like(np.squeeze(arr), arr,
                    spacing = np.take(arr.spacing, keep))


# common code for subimage and resize
def _checked_cut(arr, extent):
    for i, axis in enumerate(extent):
        if axis.start < 0 or axis.stop > arr.shape[i]:
            raise IndexError

    return arr[extent]

ImageSchema._corresponding_marray = Image
VolumeSchema._corresponding_marray = Volume
VectorGridSchema._corresponding_marray = VectorGrid
