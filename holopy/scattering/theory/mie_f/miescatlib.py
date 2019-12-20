# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang, Solomon Barkley
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
MieScatLib.py

Library of code to do Mie scattering calculations.

.. moduleauthor:: Jerome Fung <jerome.fung@post.harvard.edu>
'''

import numpy as np
from numpy import sin, cos, array

try:
    from . import mie_specfuncs
    from .mieangfuncs import dn_1_down, lentz_dn1
except ImportError:
    pass



def scatcoeffs(m, x, nstop, eps1 = 1e-3, eps2 = 1e-16): # see B/H eqn 4.88
    '''
    Calculate expansion coefficients for scattered field in Lorenz-Mie
    solution.

    Parameters
    ----------
    m : complex
        Sphere relative refractive index (n_sphere / n_medium)
    x : float
        Sphere size parameter (k_med * a)
    nstop : int
        Maximum order of scattered field expansion
    eps1 : float, optional
        In Lentz continued fraction algorithm for logarithmic derivative
        D_n(z), value of continued fraction numerator or denominator
        triggering ill-conditioning workaround.
    eps2 : float, optional
        Convergence criterion for Lentz continued fraction algorithm

    Returns
    -------
    array(2, nstop), complex
        Scattering coefficients a_n and b_n

    Notes
    -----
    Uses formula for scattering coefficients based on logarithmic derivative
    D_n(z) of spherical Bessel function psi_n(z). See [Bohren1983]_ eq.
    4.88.

    Following BHMIE, calculates D_n for complex argument using downward 
    recursion, and Riccati-Bessel functions psi and xi for real argument 
    using upward recursion.

    Initializes downward recursion for D_n using Lentz continued fraction 
    algorithm [Lentz1976]_.
    '''
    Dnmx = dn_1_down(m * x, nstop + 1, nstop,
                                 lentz_dn1(m * x, nstop + 1, eps1, eps2))
    n = np.arange(nstop+1)
    psi, xi = mie_specfuncs.riccati_psi_xi(x, nstop)
    psishift = np.concatenate((np.zeros(1), psi))[0:nstop+1]
    xishift = np.concatenate((np.zeros(1), xi))[0:nstop+1]
    an = ( (Dnmx/m + n/x)*psi - psishift ) / ( (Dnmx/m + n/x)*xi - xishift )
    bn = ( (Dnmx*m + n/x)*psi - psishift ) / ( (Dnmx*m + n/x)*xi - xishift )
    return array([an[1:nstop+1], bn[1:nstop+1]]) # output begins at n=1

def internal_coeffs(m, x, n_max, eps1 = 1e-3, eps2 = 1e-16):
    '''
    Calculate internal Mie coefficients c_n and d_n given
    relative index, size parameter, and maximum order of expansion.

    Parameters
    ----------
    See docstring for scatcoeffs

    Returns
    -------
    ndarray(2,n) complex
        Internal coefficients c_n and d_n
    
    Notes
    -----
    Follow Bohren & Huffman's convention. Note that van de Hulst and Kerker
    have different conventions (labeling of c_n and d_n and factors of m)
    for their internal coefficients.
    '''
    ratio = mie_specfuncs.R_psi(x, m * x, n_max, eps1, eps2)
    D1x, D3x = mie_specfuncs.log_der_13(x, n_max, eps1, eps2)
    D1mx = dn_1_down(m * x, n_max + 1, n_max, lentz_dn1(m * x, n_max + 1,
                                                        eps1, eps2))
    cl = m * ratio * (D3x - D1x) / (D3x - m * D1mx)
    dl = m * ratio * (D3x - D1x) / (m * D3x - D1mx)
    return array([cl[1:], dl[1:]]) # start from l = 1

def nstop(x):
    '''
    Calculate maximum expansion order of Lorenz-Mie solution.

    Parameters
    ----------
    x : float
        Particle size parameter

    Returns
    -------
    nstop : int

    Notes
    -----
    Criterion taken from [Wiscombe1980]_.
    '''
    # 7/7/08: generalize to apply same criterion when x is complex
    return int(np.round_(np.absolute(x+4.05*x**(1./3.)+2)))

def asymmetry_parameter(al, bl):
    '''
    Calculate asymmetry parameter of scattered field.

    Parameters
    ----------
    an, bn : ndarray
        coefficient arrays from Mie solution

    Returns
    -------
    float

    Notes
    -----
    See discussion on Bohren & Huffman p. 120.
    The output of this function omits the prefactor of 4/(x^2 Q_sca).
    '''
    lmax = al.shape[0]
    l = np.arange(lmax) + 1
    selfterm = (l[:-1] * (l[:-1] + 2.) / (l[:-1] + 1.) *
                np.real(al[:-1] * np.conj(al[1:]) +
                        bl[:-1] * np.conj(bl[1:]))).sum()
    crossterm = ((2. * l + 1.)/(l * (l + 1)) *
                 np.real(al * np.conj(bl))).sum()
    return selfterm + crossterm

def cross_sections(al, bl):
    '''
    Calculates scattering and extinction cross sections
    given arrays of Mie scattering coefficients an and bn.

    Parameters
    ----------
    an, bn : ndarray
        coefficient arrays from Mie solution
   
    Returns
    -------
    ndarray(3)
        Scattering, extinction, and radar backscattering cross sections

    Notes
    -----
    See Bohren & Huffman eqns. 4.61 and 4.62.
    The output omits a scaling prefactor of 2 * pi / k^2.
    '''
    lmax = al.shape[0]

    l = np.arange(lmax) + 1
    prefactor = (2. * l + 1.)
    cscat = (prefactor * (np.abs(al)**2 + np.abs(bl)**2)).sum()
    cext = (prefactor * np.real(al + bl)).sum()

    # see p. 122
    alts = 2. * (np.arange(lmax) % 2) - 1
    cback = np.abs((prefactor * alts * (al - bl)).sum())**2

    return array([cscat, cext, cback])
