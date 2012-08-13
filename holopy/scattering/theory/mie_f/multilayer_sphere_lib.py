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

def LogDer13(z, nstop):
    '''
    Calculate logarithmic derivatives of Riccati-Bessel functions psi
    and xi for complex arguments.  Riccati-Bessel conventions follow
    Bohren & Huffman.

    See Mackowski et al., Applied Optics 29, 1555 (1990).

    Input:
    z: complex number
    nstop: maximum order of computation
    '''
    z = np.complex128(z) # convert to double precision

    # Calculate Dn_1 (based on \psi(z)) using downward recursion.
    # See Mackowski eqn. 62
    nmx = np.maximum(nstop, np.round_(np.absolute(z))) + 15
    dn1 = np.zeros(nmx+1, dtype = 'complex128') # initialize w/zeros
    for i in np.arange(nmx-1, -1, -1): # down recurrence
        dn1[i] = (i+1.)/z - 1.0/(dn1[i+1.] + (i+1.)/z)
	
    # Calculate Dn_3 (based on \xi) by up recurrence
    # initialize
    dn3 = np.zeros(nstop+1, dtype = 'complex128')
    psixi = np.zeros(nstop+1, dtype = 'complex128')
    dn3[0] = 1j
    psixi[0] = -1j*exp(1j*z)*sin(z)
    for dindex in np.arange(1, nstop+1):
        # Mackowski eqn 63
        psixi[dindex] = psixi[dindex-1] * ( (dindex/z) - dn1[dindex-1]) * (
            (dindex/z) - dn3[dindex-1])
        # Mackowski eqn 64
        dn3[dindex] = dn1[dindex] + 1j/psixi[dindex]

    return dn1[0:nstop+1], dn3


# calculate ratio of RB's defined in Yang eqn. 23 by up recursion relation 
def Qratio(z1, z2, nstop, dns1 = None, dns2 = None): 
    '''
    Calculate ratio of Riccati-Bessel functions defined in Yang eq. 23
    by up recursion.

    Logarithmic derivatives calculated automatically if not specified.
    '''
    # convert z1 and z2 to 128 bit complex to prevent division problems
    z1 = np.complex128(z1)
    z2 = np.complex128(z2)

    if dns1 == None:
        logdersz1 = LogDer13(z1, nstop)
        logdersz2 = LogDer13(z2, nstop)
        d1z1 = logdersz1[0]
        d3z1 = logdersz1[1]
        d1z2 = logdersz2[0]
        d3z2 = logdersz2[1]
    else:
        d1z1 = dns1[0]
        d3z1 = dns1[1]
        d1z2 = dns2[0]
        d3z2 = dns2[1]
	
    qns = np.zeros(nstop+1, dtype = 'complex128')
	
    # initialize according to Yang eqn. 34
    a1 = real(z1)
    a2 = real(z2)
    b1 = imag(z1)
    b2 = imag(z2)
    qns[0] = exp(-2.*(b2-b1)) * (exp(-1j*2.*a1)-exp(-2.*b1)) / (exp(-1j*2.*a2)
                                                                - exp(-2.*b2)) 
    # Loop to do upwards recursion in eqn. 33
    for i in np.arange(1, nstop+1):
        qns[i] = qns[i-1]* ( (d3z1[i] + i/z1) * (d1z2[i] + i/z2) 
	       		     )  / ((d3z2[i] + i/z2) * (d1z1[i] + i/z1) )

    return qns


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
    intl = LogDer13(marray[0]*xarray[0], nstop)[0]
    hans = intl
    hbns = intl
	
    for lay in np.arange(1, nlayers): # lay is l-1 (index on layers used by Yang)
        z1 = marray[lay]*xarray[lay-1] # m_l x_{l-1}
        z2 = marray[lay]*xarray[lay]  # m_l x_l

        # calculate logarithmic derivatives D_n^1 and D_n^3
        derz1s = LogDer13(z1, nstop)
        derz2s = LogDer13(z2, nstop)

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
    psiandxi = miescatlib.RicBesHank(xarray.max(), nstop) # n = 0 to nstop
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
