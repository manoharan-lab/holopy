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
'''
Do forward calculation of a dimer hologram from Fortran subroutines.
Uses full radial dependence of spherical Hankel functions for scattered
field.

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>

'''

import scipy as sp
import numpy as np
import scattering.tmatrix.mieangfuncs as mieangfuncs
import scattering.tmatrix.scsmfo_min as scsmfo_min
import scattering.tmatrix.miescatlib as miescatlib
from holopy.hologram import Hologram

from scipy import array, pi
from scattering.tmatrix.mieangfuncs import singleholo
from scattering.tmatrix.miescatlib import nstop, scatcoeffs


par_ordering = ['n_particle_real', 'n_particle_imag', 'radius', 'x',
                'y', 'z', 'scaling_alpha'] 


def _scaled_by_k(param_name):
    pars = ['radius', 'x', 'y', 'z']
    return param_name in pars

def _scaled_by_med_index(param_name):
    pars = ['n_particle_real', 'n_particle_imag']
    return param_name in pars


def forward_holo(size, opt, n_particle_real, n_particle_imag, radius,
                 x, y, z, scaling_alpha, dimensional = True):
    """
    Compute a hologram of n spheres by mie superposition

    Parameters may be specified in any consistent set of units (make
    sure the optics object is also in the same units).
    
    Parameters
    ----------
    size : int or (int, int)
       dimension in pixels of the hologram to calculate (square if scalar)
    opt : Optics or dict
       Optics class or dictionary describing wavelength and pixel
       information for the calculation 
    n_particle_real : float 
       refractive index of sphere
    n_particle_imag : float or array(float)
       imaginary refractive index of sphere(s)
    radius : float or array(float)
       sphere(s)'s radius
    x : float or array(float) 
       x-position of sphere(s), (0,0) is upper left
    y : float or array(float)
       y-position of sphere(s)
    z : float or array(float) 
       z-position of sphere(s)
    scaling_alpha : float
       hologram scaling alpha
    dimensional: bool
       If False, assume all lengths non-dimensionalized by k and all
       indices relative (divided by medium index).

    Returns
    -------
    calc_holo : Hologram
       Calculated hologram from the given distribution of spheres

    """
    
    if isinstance(opt, dict):
        opt = optics.Optics(**opt)

    # Allow size and pixel size to be either 1 number (square) 
    #    or rectangular
    if np.isscalar(size):
        xdim, ydim = size, size
    else:
        xdim, ydim = size
    if opt.pixel_scale.size == 1: # pixel_scale is an ndarray
        px, py = opt.pixel_scale, opt.pixel_scale
    else:
        px, py = opt.pixel_scale

    # non-dimensionalization
    wavevec = 2.0 * pi / opt.med_wavelen
    if dimensional:
        # multiply all length scales by k
        com_coords = array([x, y, z]) * wavevec
        x_p = radius * wavevec
        # relative indices
        m_real = n_particle_real / opt.index
        m_imag = n_particle_imag / opt.index
    else:
        com_coords = array([x, y, z])
        x_p = radius
        m_real = n_particle_real
        m_imag = n_particle_imag

    # Scattering coefficent calculation (still in Python)
    ns = nstop(x_p)
    scoeffs = scatcoeffs(x_p, m_real + 1j*m_imag, ns)
    
    # hologram grid (new convention)
    gridx = np.mgrid[0:xdim]*px
    gridy = np.mgrid[0:ydim]*py

    holo = Hologram(singleholo(wavevec*gridx, 
                               wavevec*gridy, com_coords, 
                               scoeffs, scaling_alpha, 
                               opt.polarization), 
                    optics = opt)

    return holo


def _forward_holo(size, opt, scat_dict): 
    '''
    Internal use; passes everything to public forward_holo
    non-dimensionally.
    '''
    return forward_holo(size, opt, dimensional = False, **scat_dict)
