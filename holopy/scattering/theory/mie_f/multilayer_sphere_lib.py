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
'''
multilayer_sphere_lib.py

Author:
Jerome Fung (fung@physics.harvard.edu)

Functions to calculate the scattering from a spherically symmetric particle 
with an arbitrary number of layers with different refractive indices. 

Key reference for multilayer algorithm:
Yang, "Improved recursive algorithm for light scattering by a multilayered 
sphere," Applied Optics 42, 1710-1720, (1993).

'''

import numpy as np
import miescatlib
from ...errors import ModelInputError

from numpy import exp, sin, cos, real, imag
from mie_specfuncs import Qratio, log_der_13, riccati_psi_xi

def scatcoeffs_multi(marray, xarray):
    '''
    Calculate scattered field expansion coefficients (in the Mie formalism)
    for a particle with an arbitrary number of layers.

    Inputs:
    marray: numpy array of layer indices, innermost first
    xarray: numpy array of layer size parameters (k * radius), innermost first
    '''
    # ensure correct data types
    marray = np.array(marray, dtype = 'complex128')
    xarray = np.array(xarray, dtype = 'float64')

    # sanity check: marray and xarray must be same size
    if marray.size != xarray.size:
        raise ModelInputError('Arrays of layer indices and size parameters must be the same length!')

    # need number of layers L
    nlayers = marray.size

    # calculate nstop based on outermost radius
    nstop = miescatlib.nstop(xarray.max())

    # initialize H_n^a and H_n^b in the core, see eqns. 12a and 13a
    intl = log_der_13(marray[0]*xarray[0], nstop)[0]
    hans = intl
    hbns = intl
	
    for lay in np.arange(1, nlayers): # lay is l-1 (index on layers used by Yang)
        z1 = marray[lay]*xarray[lay-1] # m_l x_{l-1}
        z2 = marray[lay]*xarray[lay]  # m_l x_l

        # calculate logarithmic derivatives D_n^1 and D_n^3
        derz1s = log_der_13(z1, nstop)
        derz2s = log_der_13(z2, nstop)

        # calculate G1, G2, Gtilde1, Gtilde2 according to 
        # eqns 26-29
        # using H^a_n and H^b_n from previous layer
        G1 = marray[lay]*hans - marray[lay-1]*derz1s[0]
        G2 = marray[lay]*hans - marray[lay-1]*derz1s[1]
        Gt1 = marray[lay-1]*hbns - marray[lay]*derz1s[0]
        Gt2 = marray[lay-1]*hbns - marray[lay]*derz1s[1]

        # calculate ratio Q_n^l for this layer
        Qnl = Qratio(z1, z2, nstop, dns1 = derz1s, dns2 = derz2s)

        # now calculate H^a_n and H^b_n in current layer
        # see eqns 24 and 25
        hans = (G2*derz2s[0] - Qnl*G1*derz2s[1]) / (G2 - Qnl*G1)
        hbns = (Gt2*derz2s[0] - Qnl*Gt1*derz2s[1]) / (Gt2 - Qnl*Gt1)
        # repeat for next layer

    # Relate H^a and H^b in the outer layer to the Mie scat coeffs
    # see Yang eqns 14 and 15
    psiandxi = riccati_psi_xi(xarray.max(), nstop) # n = 0 to nstop
    n = np.arange(nstop+1)
    psi = psiandxi[0]
    xi = psiandxi[1]
    # this doesn't bother to calculate psi/xi_{-1} correctly, 
    # but OK since we're throwing out a_0, b_0 where it appears
    psishift = np.concatenate((np.zeros(1), psi))[0:nstop+1]
    xishift = np.concatenate((np.zeros(1), xi))[0:nstop+1]
    an = ((hans/marray[nlayers-1] + n/xarray[nlayers-1])*psi - psishift) / ( 
        (hans/marray[nlayers-1] + n/xarray[nlayers-1])*xi - xishift)
    bn = ((hbns*marray[nlayers-1] + n/xarray[nlayers-1])*psi - psishift) / ( 
        (hbns*marray[nlayers-1] + n/xarray[nlayers-1])*xi - xishift)
    return np.array([an[1:nstop+1], bn[1:nstop+1]]) # output begins at n=1
