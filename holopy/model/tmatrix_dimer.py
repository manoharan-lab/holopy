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
import scattering.tmatrix.mieangfuncs as mieangfuncs
import scattering.tmatrix.scsmfo_min as scsmfo_min
import holopy.optics as optics
from holopy.hologram import Hologram
from holopy.utility.helpers import _ensure_pair

from numpy import array, pi

par_ordering = ['n_particle_real_1',  'n_particle_real_2', 
                'n_particle_imag_1',  'n_particle_imag_2', 
                'radius_1', 'radius_2', 
                'x_com', 'y_com', 'z_com', 
                'scaling_alpha', 'euler_beta', 'euler_gamma',
                'gap_distance']

def _scaled_by_k(param_name):
    pars = ['radius_1', 'radius_2', 'x_com', 'y_com', 'z_com', 
            'gap_distance']
    return param_name in pars

def _scaled_by_med_index(param_name):
    pars = ['n_particle_real_1', 'n_particle_imag_1', 'n_particle_real_2',
            'n_particle_imag_2']
    return param_name in pars

def forward_holo(size, opt, n_particle_real_1, n_particle_real_2,
                 n_particle_imag_1, n_particle_imag_2,  radius_1, 
                 radius_2, x_com, y_com, z_com, scaling_alpha, euler_beta,
                 euler_gamma, gap_distance,tmat_dict = None, 
                 old_coords = False, dimensional = True):
    """
    Calculate hologram of a dimer from T-matrix scattering model.

    Parameters may be specified in any consistent set of units (make
    sure the optics object is also in the same units).

    Parameters
    ----------
    size : int or (int, int)
       of pixels in hologram to calculate (square if scalar)
    opt : Optics or dict
       Optics class or dictionary describing wavelength and pixel 
       information for the calculation
    n_particle_real_1 : float
       Real part of the index of refraction of particle 1.  
    n_particle_real_2 : float
       Real index of particle 2. If None, assumed same as n_particle_real_1.
    n_particle_imag_1 : float
       particle 1 imaginary refractive index
    n_particle_imag_2 : float
       imaginary index of particle 2. If None, assumed same as 
       n_particle_imag_1.
    radius_1, radius_2 : float
       particle radii
    x_com, y_com, z_com : float
       coordinates of dimer center of mass
    scaling_alpha : float
       Overall scaling factor for terms containing E_scat
    euler_beta : float
       Euler angle beta (deg) in modified zyz convention (rotation
       about y). 
    euler_gamma : float
       Euler angle gamma (deg) in modified zyz convention (rotation
       about z). 
    gap_distance : float
       Interparticle gap distance ( = 0 at hard-sphere contact.) 
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
       Calculated dimer hologram

    Notes
    -----
    The larger the size of the cluster, the more terms are needed in
    the expansion of the scattered fields in a series of vector
    spherical harmonics, and the longer the code takes to run.  The
    cluster size is determined both by the particle radii and by
    gap_distance.  In particular, this code is not recommended for use
    when the gap distance is very large (in particular, more than
    several particle diameters).  Use Lorenz-Mie superposition
    instead.

    Only two Euler angles are necessary because dimers are
    axisymmetric.  Our Euler angle beta differs slightly from the zyz
    convention in that it is permitted to be negative.  It will behave
    in a mathematically sensible way between -180 and 360 degrees.
    The reference configuration (beta and gamma both = 0) occurs wtih
    both particles lying on the x-axis, with the x coordinate of
    particle #1 being positive.

    See details concerning implementation in [1]_.

    References
    ----------
    .. [1] Jerome Fung et al., Optics Express 19, 8051-8065 (2011).

    """
    if not tmat_dict: # Provide sensible defaults
        tmat_dict = {'niter' : 200, 'eps' : 1e-6, 'qeps1' : 1e-5, 
                     'qeps2' : 1e-8, 'meth' : 1}

    if not n_particle_real_2:
        n_particle_real_2 = n_particle_real_1

    if not n_particle_imag_2:
        n_particle_imag_2 = n_particle_imag_1

    if isinstance(opt, dict):
        opt = optics.Optics(**opt)

    # Allow size and pixel size to be either 1 number (square) or
    # rectangular 
    xdim, ydim = _ensure_pair(size)
    px, py = _ensure_pair(opt.pixel)

    # Work around beta not being modulo anything (to ease fitting)
    if euler_beta < 0:
        euler_beta = 180. + euler_beta
    elif euler_beta > 180:
        euler_beta = euler_beta - 180.

    # Convert gamma from degrees to radians
    euler_gamma *= pi/180.

    # non-dimensionalization
    wavevec = 2.0 * pi / opt.med_wavelen
    if dimensional:
        # all length scales by k in medium
        com_coords = array([x_com, y_com, z_com]) * wavevec
        r_eps = gap_distance * wavevec 
        x_parts = array([radius_1, radius_2]) * wavevec 
        xarr = array([radius_1 + 0.5 * gap_distance, -radius_2 - 0.5 * 
                      gap_distance]) * wavevec
        # need relative indices
        m_reals = array([n_particle_real_1, n_particle_real_2]) / opt.index
        m_imags = array([n_particle_imag_1, n_particle_imag_2]) / opt.index
    else: 
        com_coords = array([x_com, y_com, z_com])
        r_eps = gap_distance
        x_parts = array([radius_1, radius_2])
        xarr = array([radius_1 + 0.5 * gap_distance, -radius_2 - 0.5 *
                      gap_distance])
        m_reals = array([n_particle_real_1, n_particle_real_2])
        m_imags = array([n_particle_imag_1, n_particle_imag_2])
        
    # Scattering coefficent calc setup
    yarr = np.zeros(2)
    zarr = np.zeros(2)
    ea = array([0., euler_beta])

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
    amncoeffs = amn0[:,0:limit, :]

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
    Internal use; passes everything to public forward_holo non-dimensionally.
    '''
    # make sure these params have value of None if they do not exist.
    # The fitter will assume a value for them in that case, but it
    # will fail if they don't exist at all.
    scat_dict['n_particle_real_2'] = scat_dict.get('n_particle_real_2')
    scat_dict['n_particle_imag_2'] = scat_dict.get('n_particle_imag_2')
    
    return forward_holo(size, opt, dimensional = False, **scat_dict)



