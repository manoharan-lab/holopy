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
from ..core.marray import VectorGrid
from holopy.core.marray import dict_without, resize

# May eventually want to have this function take a propagation model
# so that we can do things other than convolution

def propagate(data, d, gradient_filter=False):
    """
    Propagates a hologram along the optical axis

    Parameters
    ----------
    data : :class:`.Image` or :class:`.VectorGrid`
       Hologram to propagate
    d : float or list of floats
       Distance to propagate, in meters, or desired schema.  A list tells to
       propagate to several distances and return the volume
    gradient_filter : float
       For each distance, compute a second propagation a distance
       gradient_filter away and subtract.  This enhances contrast of
       rapidly varying features

    Returns
    -------
    data : :class:`.Image` or :class:`.Volume`
       The hologram progagated to a distance d from its current location.

    """
    if np.isscalar(d) and d == 0:
        # Propagationg no distance has no effect
        return data

    # Computing the transfer function will fail for d = 0. So, if we
    # are asked to compute a reconstruction for a set of distances
    # containing 0, we pull that distance out and then add in a copy
    # of the input at the end.
    contains_zero = False
    if not np.isscalar(d):
        d = np.array(d)
        if (d == 0).any():
            contains_zero = True
            d_old = d
            d = np.delete(d, np.nonzero(d == 0))

    G = trans_func(data, d, squeeze=False, gradient_filter=gradient_filter)

    ft = fft(data)

    ft = np.repeat(ft[:, :, np.newaxis,...], G.shape[2], axis=2)

    ft = apply_trans_func(ft, G)

    res = ifft(ft, overwrite=True)

    # This will not work correctly if you have 0 in the distances more
    # than once. But why would you do that?
    if contains_zero:
        d = d_old
        res = np.insert(res, np.nonzero(d==0)[0][0], data, axis=2)

    res = np.squeeze(res)

    origin = np.array(data.origin)
    origin[2] += _ensure_array(d)[0]

    if not np.isscalar(d) and not isinstance(data, VectorGrid):
        # check if supplied distances are in a regular grid
        dd = np.diff(d)
        if np.allclose(dd[0], dd):
            # shape of none will have the shape inferred from arr
            spacing = np.append(data.spacing, dd[0])
            res = Volume(res, spacing = spacing, origin = origin,
                         **dict_without(data._dict,
                                        ['spacing', 'origin', 'dtype']))
        else:
            res = Marray(res, positions=positions, origin = origin,
                         **dict_without(data._dict, ['spacing', 'position', 'dtype']))

    return res

def apply_trans_func(ft, G):
    mm, nn = [dim/2 for dim in G.shape[:2]]
    m, n = ft.shape[:2]

    if ft.ndim == 4:
        # vector field input, so we need to add a dimension to G so it
        # broadcasts correctly
        G = G[...,np.newaxis]
    ft[(m/2-mm):(m/2+mm),(n/2-nn):(n/2+nn)] *= G[:(mm*2),:(nn*2)]

    # Transfer function may not cover the whole image, any values
    # outside it need to be set to zero to make the reconstruction
    # correct
    ft[0:n/2-nn,...] = 0
    ft[n/2+nn:n,...] = 0
    ft[:,0:m/2-mm,...] = 0
    ft[:,m/2+mm:m,...] = 0

    return ft


def trans_func(schema, d, cfsp=0, squeeze=True,
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
       the spacing between points is the grid to calculate
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

    wavelen = schema.optics.med_wavelen

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

    m, n = np.ogrid[[slice(-dim/(2*ext), dim/(2*ext), dim*1j) for
                     (dim, ext) in zip(schema.shape[:2], schema.extent[:2])]]

    root = 1.+0j-(wavelen*n)**2 - (wavelen*m)**2

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
