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
Classes and functions for reconstructing holograms

Currently this uses the convolution method outlined in Schnars and
Juptner [1]_.  In the future it may include other reconstruction
techniques.

.. [1] Ulf Schnars and Werner P. O. Juptner, Digital recording and
   numerical reconstruction of holograms, Measurement Science and
   Technology 13(9): R85, 2002.

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.havard.edu>
.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
"""

import numpy as np
import scipy as sp
import scipy.signal
import os
from holopy.process.math import fft
import io
from holopy.analyze.propagate import propagate


class Reconstruction(np.ndarray):
    """
    Reconstructed volume from a hologram
    
    A reconstructed image or stack of reconstructions.  This is in
    general a 4-dimensional array, (x,y,z,t).  A volume reconstruction
    at a single time point has dimensions (xrange, yrange, zrange, 1).
    A slice at a single time point has dimensions (xrange, yrange, 1,
    1).

    Parameters
    ----------
    arr : ndarray
       The reconstructed data this reconstruction should hold
    holo : :class:`holopy.hologram.Hologram`
       The hologram this reconstruction derives from
    name : string
       Description of where this reconstruction came from and what
       manipulations were applied to its hologram and to it.
    
    Attributes
    ----------
    holo : :class:`holopy.hologram.Hologram`
       The hologram this reconstruction derives from
    time_scale : time (float)
       The time spacing between t slices in this reconstruction
    optics : :class:`holopy.optics.Optics`
       Optics of associated hologram
    name : string
       Description of where this reconstruction came from and what
       manipulations were applied to its hologram and to it.
    """

    def __new__(cls, arr, holo, distances,pixel_scale=None, time_scale=None,
                name=None):
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
        obj.holo = holo
        obj.time_scale = holo.time_scale
        obj.distances = distances
        obj.name = name
        obj.optics = holo.optics
        if pixel_scale is None:
            obj.pixel_scale = holo.optics.pixel_scale
        else:
            obj.pixel_scale=pixel_scale
        obj.time_scale=time_scale

        return obj

    def __array_finalize__(self, obj):
        # this function finishes the construction of our new object
        if obj is None: return
        self.holo = getattr(obj, 'holo', None)
        self.time_scale = getattr(obj, 'time_scale', None)
        self.pixel_scale = getattr(obj, 'pixel_scale', None)
        self.distances = getattr(obj, 'distances', None)
        self.name = getattr(obj, 'name', None)


    def __array_wrap__(self, out_arr, context=None):
        # this function is needed so that if we run another numpy
        # function on the hologram (for example, numpy.add), the
        # metadata will be transferred to the new object that is
        # created
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def resample(self, shape, window=None):
        """
        Resamples reconstruction to a given size.

        Use, for example, to downsample in a way that avoids aliasing
        and ringing.

        Parameters
        ----------
        shape : int or (int, int)
           desired shape of reconstruction after resampling.  If only
           one int is given, a square is assumed. 
        window : scipy.signal.window
           type of smoothing window passed to the
           scipy.signal.resample filter.  

        Returns
        -------
        new_rec : :class:`holopy.analyze.reconstruct.Reconstruction` object

        Notes
        -----
        This algorithm does 2 1-D resamplings.
        """
        if np.isscalar(shape):
            x, y = (shape, shape)
            name = self.name + 'r({0},{1})'.format(x, y)
        else:
            x, y = shape
            name = self.name+'r{0}'.format(x)
            
        new_image = scipy.signal.resample(self, x, axis=0, window=window)
        new_image = scipy.signal.resample(new_image, y, axis=1,
                                          window=window) 

        # Change the pixel calibration appropriately.  If image is
        # downsampled, pixel scale should increase
        factor = np.array(new_image.shape).astype('float')/self.shape
        factor = factor[:1]

        # return a new reconstruction
        new_rec = Reconstruction(new_image, self.holo, self.distances,
                                 self.pixel_scale/factor,
                                 self.time_scale, name) 

        return new_rec       


def reconstruct(holo, distances, fourier_mask=None, gradient_filter=None,
                recon_along_xz=False,zprojection=False):
    """
    Reconstructs a hologram at the given distance or distances

    If distances is an array, this will do a volume reconstruction
    with slices at each distance.

    Parameters
    ----------
    holo : :class:`holopy.hologram.Hologram`
       the hologram to reconstruct
    distances : length (float) or array<length>
       reconstruction distance(s)

    Returns
    -------
    reconstruction : :class:`holopy.analyze.reconstruct.Reconstruction`
       Reconstructed images.  4d matrix (x, y, z, t), len(z) will be 1
       unless distances is an array, len(t) will be 1 unless holo is 3d
       (time series) 
    
    Other Parameters
    ----------------
    fourier_mask : function(holo.shape)
       Function generating a fourier mask which you want applied to
       reconstrutions.  Default is None, meaning don't apply a mask
    gradient_filter : length (float)
       Use a gradient filter - subtract a reconstruction spaced a distance
       gradient_filter from the reconstruction to suppress slowly varying
       features
    recon_along_xz : bool
       If True reconstruct a 1d slice along z.  Will sum along the
       shorter of x or y.
    zprojection : bool
       If True the transfer matrix for all distances will be added together
       and a single image returned

    See Also
    --------
    holopy.analyze.propagate.propagate : Used here to do numerical propagations

    Notes
    -----
    This function reconstructs the hologram at some distance z. That
    distance should be in the same units specifying the wavelength and
    pixel size that are in the optics class instance associated with
    the  hologram object.

    Examples
    --------
    To reconstruct the hologram at a distance of 10 microns. :
    >>> rec = reconstruction.reconstruct(holo, 10e-6)

    If the hologram, 'holo', was a 256-by-256 array, then the returned
    array will be 256x256x1x1 and complex.
    
    >>> rec.shape
    (256L, 256L, 1L, 1L)
    >>> rec.dtype
    dtype('complex128')

    
    """
    # This lets us always assume distances is an array, if we are
    # reconstructing at only a single distance, then we just have a 1
    # element array 
    if np.isscalar(distances):
        distances = np.array([distances])

    name = holo.name
        
    if fourier_mask:
        fourier_mask=fourier_mask(holo.shape)
    else:
        fourier_mask=None

    r = Reconstruction(propagate(holo, distances,
                                 fourier_filter=fourier_mask,
                                 squeeze=False,
                                 gradient_filter=gradient_filter, 
                                 rec_zx_plane=recon_along_xz,
                                 project_along_z=zprojection),
                       holo=holo, distances=distances, name=name)
    r.gradient_filter_dz = gradient_filter
    return r

