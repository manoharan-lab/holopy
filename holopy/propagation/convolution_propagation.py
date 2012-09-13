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
Code to propagate objects/waves using scattering models.  

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Ryan McGorty <mcgorty@fas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
from __future__ import division

import numpy as np
from ..core.math import fft, ifft
from ..core.helpers import _ensure_pair, _ensure_array
from ..core import Volume, Image, Grid, UnevenGrid, VolumeSchema, Marray
from holopy.core.marray import dict_without

# May eventually want to have this function take a propagation model
# so that we can do things other than convolution

def propagate(data, schema, gradient_filter=False):
    """
    Propagates a hologram a distance d along the optical axis.

    Uses scalar diffraction theory calculate a hologram at a given
    distance away.  If you are doing a series of propagations of the
    same image (like in a reconstruction stack), you can save time by
    doing the fourier transform once outside this function.  If you do
    not provide the fourier transform, it will be computed here.
    Propagate can apply an arbitrary array to the data as a fourier
    filter, this will be multiplied by the hologram in the fourier
    domain.  If you want to apply several, just multiply them
    together.  This functionallity is used for efficiently applying
    contrast enhancing masks to data.

    Parameters
    ----------
    data : :class:`holopy.core.marray.Image`
       Hologram to propagate
    schema : float, list of floats, or :class:`holopy.core.marray.VolumeSchema`
       Distance to propagate, in meters.  A list tells to propagate to several
       distances and return the volume
    gradient_filter : float
       For each distance, compute a second propagation a distance
       gradient_filter away and subtract.  This enhances contrast of
       rapidly varying features 

    Returns
    -------
    data : :class:`holopy.core.marray.Image` or :class:`holopy.core.marray.Volume`
       The hologram progagated to a distance d from its current location.  
        
    """

    if isinstance(schema, VolumeSchema):
        # get the right z slices in the volume
        d = np.arange(schema.shape[2]) * schema.spacing[2]
        d = d + schema.origin[2]
        vol = propagate(data, d, gradient_filter)


        if data.origin is None:
            data_origin = 0, 0, 0
        else:
            data_origin = data.origin

        offset = schema.origin - data_origin

        x0, y0 = offset[:2]
        x1, y1 = np.array((x0, y0)) + (schema.extent[:2] / data.spacing[:2])
        vol = vol[x0:x1, y0:y1 :]
        vol = vol.resample(schema.shape)
        return vol

    # schema is just a list of distances, so we can compute convolution
    # propagation to those distances
    d = schema
    
    G = trans_func(data.shape[:2], data.positions.spacing,
                   data.optics.med_wavelen, d, squeeze=False,
                   gradient_filter=gradient_filter)
    
    ft = fft(data)

    ft = np.repeat(ft[:, :, np.newaxis,...], G.shape[2], axis=2)

    ft = apply_trans_func(ft, G)
    
    arr = np.squeeze(ifft(ft, overwrite=True))

    if arr.ndim == 2:
        res =  Image(arr,  **dict_without(data._dict, 'dtype'))
    elif arr.ndim == 3:
        # check if supplied distances are in a regular grid
        dd = np.diff(d)
        if np.allclose(dd[0], dd):
            # shape of none will have the shape inferred from arr
            spacing = np.append(data.spacing, dd[0])
            res = Volume(arr, spacing = spacing,
                         **dict_without(data._dict, ['spacing', 'dtype']))
        else:
            res = Marray(arr, positions=positions,
                         **dict_without(data._dict, ['spacing', 'position', 'dtype']))
    return res

def apply_trans_func(ft, G):
    mm, nn = [dim/2 for dim in G.shape[:2]]
    m, n = ft.shape[:2]
    
    ft[(m/2-mm):(m/2+mm),(n/2-nn):(n/2+nn)] *= G[:(mm*2),:(nn*2)]
    
    # Transfer function may not cover the whole image, any values
    # outside it need to be set to zero to make the reconstruction
    # correct
    ft[0:n/2-nn,...] = 0
    ft[n/2+nn:n,...] = 0
    ft[:,0:m/2-mm,...] = 0
    ft[:,m/2+mm:m,...] = 0

    return ft


def trans_func(shape, spacing, wavelen, d, cfsp=0, squeeze=True,
               gradient_filter=0):
    """
    Calculates the optical transfer function to use in reconstruction

    This routine uses the analytical form of the transfer function
    found in in Kreis [1]_.  It can optionally do cascaded free-space
    propagation for greater accuracy [2]_, although the code will run
    slightly more slowly.

    Parameters
    ----------
    shape : (int, int)
       maximum dimensions of the transfer function
    spacing : (float, float)
       the spacing between point is the grid to calculate
    wavelen : float
       the wavelength in the medium you are propagating through
    d : float or list of floats
       reconstruction distance.  If list or array, this function will
       return an array of transfer functions, one for each distance 
    cfsp : integer (optional)
       cascaded free-space propagation factor.  If this is an integer
       > 0, the transfer function G will be calculated at d/csf and
       the value returned will be G**csf.
    squeeze : Bool (optional)
       Remove length 1 dimensions (so that if only one distance is
       specified trans_func will be a 2d array) 
    gradient_filter : float (optional)
       Subtract a second transfer function a distance gradient_filter
       from each z 

    Returns
    -------
    trans_func : np.ndarray
       The calculated transfer function.  This will be at most as large as
       shape, but may be smaller if the frequencies outside that are zero

    References
    ----------
    .. [1] Kreis, Handbook of Holographic Interferometry (Wiley,
       2005), equation 3.79 (page 116)

    .. [2] Kreis, Optical Engineering 41(8):1829, section 5
    
    """
    d = np.array([d])

    dx, dy = spacing
    xdim, ydim = _ensure_pair(shape)

    d = d.reshape([1, 1, d.size])
    
    if(cfsp > 0):
        cfsp = int(abs(cfsp)) # should be nonnegative integer
        d = d/cfsp

    # The transfer function is only defined on a finite domain of
    # spatial frequencies; outside this domain the transfer function
    # is zero (see Kreis, Optical Engineering 41(8):1829, page 1836).
    # It is important to set it to zero outside the domain, otherwise
    # the reconstruction is not correct.  Here I save memory by making
    # the size of the array only as large as the domain corresponding
    # to the transfer function at the smallest z-distance

    # for this we need to use the magnitude of d, size of the image
    # should be a positive number
    try:
        max_m = int(np.max(xdim**2*dx**2/np.abs(d)/wavelen/2))+1
        max_n = int(np.max(ydim**2*dy**2/np.abs(d)/wavelen/2))+1
    except OverflowError:
        max_m = xdim/2
        max_n = ydim/2
    
    # make sure that the array is not larger than the hologram if we
    # are using cascaded free space propagation
    max_m = min(xdim, max_m*2)/2
    max_n = min(ydim, max_n*2)/2
   
    m, n = np.ogrid[-max_m:max_m,-max_n:max_n]
    
    
    root = 1.+0j-(wavelen*n/(xdim*dx))**2 - (wavelen*m/(ydim*dy))**2

    root *= (root >= 0)
    
    # add the z axis to this array so it broadcasts correctly
    root = root[..., np.newaxis]

    g = np.exp(-1j*2*np.pi*d/wavelen*np.sqrt(root))

    if gradient_filter:
        g -= np.exp(-1j*2*np.pi*(d+gradient_filter)/wavelen*np.sqrt(root))

    # set the transfer function to zero where the sqrt is imaginary
    # (this is equivalent to making sure that the largest spatial
    # frequency is 1/wavelength).  (root>=0) returns a boolean matrix
    # that is equal to 1 where the condition is true and 0 where it is
    # false.  Multiplying by this boolean matrix masks the array.
    g = g*(root>=0)

    if cfsp > 0:
        g = g**cfsp

    if squeeze:
        return np.squeeze(g)
    else:
        return g


def impulse_response(shape, optics, d):
    """
    Calculates the impulse response response at a distance d

    Parameters
    ----------
    shape : (int, int)
       maximum dimensions of the transfer function
    optics : :class:`holopy.optics.Optics`
       Optics object with pixel and wavelength information
    d : float or list of floats
       reconstruction distance.  If list or array, this function will
       return an array of transfer functions, one for each distance

    Returns
    -------
    trans_func : np.ndarray
       The calculated transfer function.  This will be at most as large as
       shape, but may be smaller if the frequencies outside that are zero  

    References
    ----------
    .. [1] Schnars and Juptner, Digital recording and numerical
       reconstruction of holograms (Meas. Sci. Technol. 13 2002),
       equation 3.18 (pg R91)

    """

    d = np.array([d])

    dx, dy = optics.pixel
    wavelen = optics.med_wavelen
    xdim, ydim = _ensure_pair(shape)

    d = d.reshape([1, 1, d.size])

    # TODO BUG: this will fail for odd hologram shapes (but I am not worrying
    # about this because in practice we never use them - tgd 2011-11-21) 
    max_m = xdim/2
    max_n = ydim/2

    m, n = np.ogrid[-max_m:max_m,-max_n:max_n]
    m = m.reshape([m.shape[0], 1, 1])
    n = n.reshape([1, n.shape[1], 1])
    
    root = np.sqrt(d**2 + (m*dx)**2 + (n*dy)**2)

    return 1.0j/wavelen * np.exp(-1.0j*optics.wavevec*root)/root
