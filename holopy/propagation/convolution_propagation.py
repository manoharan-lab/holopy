# Copyright 2011-2018, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley,
# Andrei Korigodski
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


import numpy as np
import xarray as xr

from ..core.process import fft, ifft
from ..core.utils import ensure_array
from ..core.metadata import update_metadata, copy_metadata
from ..core.process.fourier import ft_coord
from ..scattering.errors import MissingParameter

# May eventually want to have this function take a propagation model
# so that we can do things other than convolution


def propagate(data, d, medium_index=None, illum_wavelen=None, cfsp=0,
              gradient_filter=False):
    """
    Propagates a hologram along the optical axis

    Parameters
    ----------
    data : xarray.DataArray
       Hologram to propagate
    d : float or list of floats
       Distance to propagate or desired schema.  A list tells to
       propagate to several distances and return the volume
    cfsp : integer (optional)
       Cascaded free-space propagation factor.  If this is an integer
       > 0, the transfer function G will be calculated at d/csf and
       the value returned will be G**csf.  This helps avoid artifacts
       related to the limited window of the transfer function
    gradient_filter : float
       For each distance, compute a second propagation a distance
       gradient_filter away and subtract.  This enhances contrast of
       rapidly varying features.  You may wish to use the number that is
       a multiple of the medium wavelength (illum_wavelen / medium_index)

    Returns
    -------
    data : xarray.DataArray
       The hologram progagated to a distance d from its current location.

    Notes
    -----
    `holopy` is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength are in the same units.
    """
    if np.isscalar(d) and d == 0:
        # Propagating no distance has no effect
        return data

    data = update_metadata(
        data, medium_index=medium_index, illum_wavelen=illum_wavelen)

    if data.medium_index is None or data.illum_wavelen is None:
        raise MissingParameter("refractive index and wavelength")

    med_wavelen = data.illum_wavelen / data.medium_index

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

    G = trans_func(
        data, d, med_wavelen, cfsp=cfsp, gradient_filter=gradient_filter)

    ft = fft(data)
    res = ifft(ft.squeeze('z') * G)

    # we may have lost coordinate values to floating point precision
    # during fft/ifft
    res.name = 'propagation'
    res = res.to_dataset().update({'x': data.x, 'y': data.y})[res.name]

    if contains_zero:
        d = d_old
        res = xr.concat([data, res], dim='z')

    return copy_metadata(data, res)


def trans_func(schema, d, med_wavelen, cfsp=0, gradient_filter=0):
    """
    Calculates the optical transfer function to use in reconstruction

    This routine uses the analytical form of the transfer function
    found in in Kreis [1]_.  It can optionally do cascaded free-space
    propagation for greater accuracy [2]_, although the code will run
    slightly more slowly.

    Parameters
    ----------
    schema : xarray.DataArray
       Hologram to obtain the maximum dimensions of the transfer function
    d : float or list of floats
       Reconstruction distance.  If list or array, this function will
       return an array of transfer functions, one for each distance
    med_wavelen : float
       The wavelength in the medium you are propagating through
    cfsp : integer (optional)
       Cascaded free-space propagation factor.  If this is an integer
       > 0, the transfer function G will be calculated at d/csf and
       the value returned will be G**csf
    gradient_filter : float (optional)
       Subtract a second transfer function a distance gradient_filter
       from each z

    Returns
    -------
    trans_func : xarray.DataArray
       The calculated transfer function.  This will be at most as large as
       shape, but may be smaller if the frequencies outside that are zero

    References
    ----------
    .. [1] Kreis, Handbook of Holographic Interferometry (Wiley,
       2005), equation 3.79 (page 116)

    .. [2] Kreis, Optical Engineering 41(8):1829, section 5

    """
    if not hasattr(d, 'z'):
        d = xr.DataArray(ensure_array(d), dims=['z'], coords={'z': ensure_array(d)})

    if(cfsp > 0):
        cfsp = int(abs(cfsp))  # should be nonnegative integer
        d = d / cfsp

    m, n = ft_coord(schema.x), ft_coord(schema.y)
    m = xr.DataArray(m, dims='m', coords={'m': m})
    n = xr.DataArray(n, dims='n', coords={'n': n})

    root = 1+0j - (med_wavelen * n) ** 2 - (med_wavelen * m) ** 2

    root *= (root >= 0)

    g = np.exp(-1j * 2 * np.pi * d / med_wavelen * np.sqrt(root))

    if gradient_filter:
        g -= np.exp(-1j * 2 * np.pi * (d + gradient_filter) / med_wavelen * np.sqrt(root))

    # Set the transfer function to zero where the sqrt is imaginary
    # (this is equivalent to making sure that the largest spatial
    # frequency is 1/wavelength).  (root>=0) returns a boolean matrix
    # that is equal to 1 where the condition is true and 0 where it is
    # false.  Multiplying by this boolean matrix masks the array.
    g = g * (root >= 0)

    if cfsp > 0:
        g = g ** cfsp

    return g
