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

Do forward calculation of a dimer hologram from T-matrix.
Uses full radial dependence of spherical Hankel functions for scattered
field.

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>

'''

import numpy as np
import mie_f.mieangfuncs as mieangfuncs
import mie_f.scsmfo_min as scsmfo_min
import holopy.optics as optics
from holopy.hologram import Hologram
from holopy.utility.helpers import _ensure_pair

from numpy import array, pi, sqrt, arccos, sin, cos

par_ordering = ['n_particle_real_1', 'n_particle_real_2', 
                'n_particle_real_3',
                'n_particle_imag_1', 'n_particle_imag_2', 
                'n_particle_imag_3', 
                'radius_1', 'radius_2', 'radius_3', 
                'x_com', 'y_com', 'z_com', 
                'scaling_alpha', 'euler_alpha', 'euler_beta',
                'euler_gamma']

def _scaled_by_k(param_name):
    pars = ['radius_1', 'radius_2', 'radius_3', 'x_com', 'y_com', 'z_com']
    return param_name in pars

def _scaled_by_med_index(param_name):
    pars = ['n_particle_real_1', 'n_particle_imag_1', 'n_particle_real_2',
            'n_particle_imag_2', 'n_particle_real_3', 'n_particle_imag_3']
    return param_name in pars

def forward_holo(size, opt, n_particle_real_1,  n_particle_real_2,
                 n_particle_real_3, n_particle_imag_1, n_particle_imag_2,
                 n_particle_imag_3, radius_1, radius_2, radius_3, x_com, y_com,
                 z_com, scaling_alpha, euler_alpha, euler_beta, euler_gamma, 
                 tmat_dict = None, old_coords = False, dimensional = True):
    """
    Calculate hologram of a trimer with all 3 spheres touching from
    T-matrix scattering model.

    Parameters
    ----------
    size : int or (int, int)
       Number of pixels in hologram to calculate (square if scalar)
    opt : Optics or dict
       Optics class or dictionary describing wavelength and pixel 
       information for the calculation
    n_particle_real_1 : float
       Real part of the index of refraction of particle 1.  
    n_particle_real_2, n_particle_3 : float
       Real index of particle 2 or 3. If None, assumed same as 
       n_particle_real_1.
    n_particle_imag_1 : float
       particle 1 imaginary refractive index
    n_particle_imag_2, n_particle_3 : float
       imaginary index of particle 2 or 3. If None, assumed same as
       n_particle_imag_1.
    radius_1, radius_2, radius_3 : float
       particle radii
    x_com, y_com, z_com :
       coordinates of dimer center of mass
    euler_alpha : float
       Euler angle alpha (deg), describes rotation about z axis
    euler_beta : float
       Euler angle beta (deg), describes rotation about y axis
    euler_gamma : float
       Euler angle gamma (deg), describes rotation about z axis
    scaling_alpha : float
       Overall scaling factor for terms containing E_scat
    tmat_dict : float
       dictionary of T-matrix code parameters
    old_coords : bool
       If True, use old coordinate convention of (0,0) at center,
       positive x -rows, positive y cols.
    dimensional : bool
       If False, assume all lengths non-dimensionalized by k and all
       indices relative (divided by medium index). 

    Returns
    -------
    calc_holo : Hologram
       Calculated trimer hologram       

    Notes
    -----
    This code assumes the three particles are all touching. All three
    may be of different sizes.

    Euler angles are defined in the zyz convention. As described
    elsewhere, it is convenient to think in terms of an active
    transformation. In the reference configuration, with all Euler
    angles = 0, all particles lie in the x-y plane. Particle 1 has a
    larger x-coordinate than particles 2 and 3, and particles 3, 1,
    and 2 have progressively larger y-coordinates.

    See details concerning implementation in [1]_.

    References
    ----------
    .. [1] Jerome Fung et al., Optics Express 19, 8051-8065 (2011).
       
    """
    if not tmat_dict:
        # If not set, provide sensible defaults
        tmat_dict = {'niter' : 200, 'eps' : 1e-6, 'qeps1' : 1e-5, 
                     'qeps2' : 1e-8, 'meth' : 1}

    if not n_particle_real_2:
        n_particle_real_2 = n_particle_real_1

    if not n_particle_real_3:
        n_particle_real_3 = n_particle_real_1

    if not n_particle_imag_2:
        n_particle_imag_2 = n_particle_imag_1

    if not n_particle_imag_3:
        n_particle_imag_3 = n_particle_imag_1

    if isinstance(opt, dict):
        opt = optics.Optics(**opt)

    # Allow size and pixel size to be either 1 number (square) or
    # rectangular
    xdim, ydim = _ensure_pair(size)
    px, py = _ensure_pair(opt.pixel)

    # new: now expect gamma in degrees, not radians. Convert here.
    euler_gamma *= pi/180.

    # non-dimensionalization
    wavevec = 2.0 * pi / opt.med_wavelen
    if dimensional:
        # multiply all length scales by k
        com_coords = array([x_com, y_com, z_com]) * wavevec
        x_parts = array([radius_1, radius_2, radius_3]) * wavevec
        # relative indices
        m_reals = array([n_particle_real_1, n_particle_real_2, 
                         n_particle_real_3]) / opt.index
        m_imags = array([n_particle_imag_1, n_particle_imag_2,
                         n_particle_imag_3]) / opt.index
    else:
        com_coords = array([x_com, y_com, z_com])
        x_parts = array([radius_1, radius_2, radius_3])
        m_reals = array([n_particle_real_1, n_particle_real_2, 
                         n_particle_real_3])
        m_imags = array([n_particle_imag_1, n_particle_imag_2,
                         n_particle_imag_3])

    # calculate particle coordinates. Rely on scat. being computed
    # about COM, so to make geometry easier, put particle #1 at the
    # origin. 
    # particle 2 at x = -sqrt(3)/2 * (x1+x2), y = 0.5 * (x1+x2) by fiat
    # angle defs: see p. 127 of JF lab notebook "June 2010"
    # particle 3 at x = -cos(delta - pi/3) * (x1+x3)
    # y = -sin(delta - pi/6)*(x1+x3)
    x_12 = x_parts[0] + x_parts[1]
    x_13 = x_parts[0] + x_parts[2]
    x_23 = x_parts[1] + x_parts[2]
    cos_delta = (x_12**2 + x_13**2 - x_23**2) / (2. * x_12 * x_13)
    delta = arccos(cos_delta)
    
    # Scattering coefficent calc setup
    xarr = array([0., -sqrt(3.)/2 * x_12, -cos(delta - pi/6.) * x_13]) 
    yarr = array([0., 0.5 * x_12, -sin(delta - pi/6.) * x_13])
    zarr = np.zeros(3)
    ea = array([euler_alpha, euler_beta])

    # calculate amn coefficients via SCSMFO
    nodr, nodrt, amn0 = scsmfo_min.amncalc(1, xarr, yarr, zarr, m_reals, 
                                           m_imags, x_parts, 
                                           tmat_dict['niter'], 
                                           tmat_dict['eps'], 
                                           tmat_dict['qeps1'],
                                           tmat_dict['qeps2'], 
                                           tmat_dict['meth'], ea)

    # chop off unused parts of amn0 (2nd dim is nodrt^2 + 2*nodrt) 
    limit = nodrt**2 + 2*nodrt
    amncoeffs = amn0[:, 0:limit, :]

    if old_coords:
        gridx = np.mgrid[(xdim-1) * px/2 : -0.5 * xdim * px : -1 * px]
        gridy = np.mgrid[(-1 * ydim+1) * py/2 : ydim * py/2 : py]
    else:
        gridx = np.mgrid[0:xdim]*px
        gridy = np.mgrid[0:ydim]*py

    holo = Hologram(mieangfuncs.tmholo_nf(wavevec*gridx, 
                wavevec*gridy, com_coords, amncoeffs, nodrt, euler_gamma, 
                scaling_alpha, opt.polarization), optics = opt)

    return holo


def _forward_holo(size, opt, scat_dict): 
    '''
    Internal use; passes everything to public forward_holo
    non-dimensionally.
    '''
    # make sure these params have value of None if they do not exist.  The
    # fitter will assume a value for them in that case, but it will
    # fail if they don't exist at all.
    scat_dict['n_particle_real_2'] = scat_dict.get('n_particle_real_2')
    scat_dict['n_particle_imag_2'] = scat_dict.get('n_particle_imag_2')
    scat_dict['n_particle_real_3'] = scat_dict.get('n_particle_real_3')
    scat_dict['n_particle_imag_3'] = scat_dict.get('n_particle_imag_3')

    return forward_holo(size, opt, dimensional = False, **scat_dict)


