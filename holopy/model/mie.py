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

'''
Forward calculations for an arbitrary number of spheres by mie superposition

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
'''

import re
import numpy as np
from scattering.mie import MFE
from holopy.hologram import Hologram
import holopy.optics
from holopy.utility.helpers import _ensure_array, _ensure_pair
from holopy.io.fit_io import _split_particle_number, _get_num_particles

par_ordering = ['n_particle_real', 'n_particle_imag', 'radius', 'x', 'y', 'z',
                'scaling_alpha']

def _scaled_by_k(parm_name):
    pars = ['radius', 'x', 'y', 'z']
    return _split_particle_number(parm_name)[0] in pars

def _scaled_by_med_index(parm_name):
    pars = ['n_particle_real', 'n_particle_imag']
    return _split_particle_number(parm_name)[0] in pars

def _forward_holo(size, opt, scat_dict):
    packed_dict = {}
    num_particles = _get_num_particles(scat_dict, par_ordering[0])
    for name in par_ordering:
        packed_dict[name] = [None] * num_particles
    for key, val in scat_dict.iteritems():
        if _scaled_by_k(key):
            # parameter was nondimensianalized by k in input, our code expects
            # that not to have happened, so we divide it out
            val /= opt.wavevec
        if _scaled_by_med_index(key):
            val *= opt.index
        name, number = _split_particle_number(key)
        if number is None:
            # parameter like scaling alpha that there is only one of
            packed_dict[name] = val
        else:
            packed_dict[name][number-1] = val

    return forward_holo(size, opt, **packed_dict)

def forward_holo(size, opt, n_particle_real, n_particle_imag, radius, x, y, z,
                 scaling_alpha):
    """
    Compute a hologram of n spheres by mie superposition

    Parameters may be specified in any consistent set of units (make sure the
    optics object is also in the same units).  
    
    Parameters
    ----------
    size : int or (int, int)
       dimension in pixels of the hologram to calculate (square if scalar)
    opt : Optics
       Optics class describing wavelength and pixel information for the
       caluclation
    n_particle_real : float or array(float)
       refractive index of sphere(s)
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

    Returns
    -------
    calc_holo : Hologram
       Calculated hologram from the given distribution of spheres
    """
    print(n_particle_real, n_particle_imag, radius, x, y, z,
                 scaling_alpha)
    if isinstance(opt, dict):
        opt = holopy.optics.Optics(**opt)

    xdim, ydim = _ensure_pair(size)
    px, py = _ensure_pair(opt.pixel)

    x = _ensure_array(x)
    y = _ensure_array(y)
    z = _ensure_array(z)
    n_particle_real = _ensure_array(n_particle_real)
    n_particle_imag = _ensure_array(n_particle_imag)
    radius = _ensure_array(radius)
    scaling_alpha = _ensure_array(scaling_alpha)

    # The code we use here expects things in terms of pixels, so convert to
    # pixels by dividing by the pixel size
    x /= opt.pixel[0]
    y /= opt.pixel[1]

    xfield_tot = np.zeros((xdim,ydim),dtype='complex128')
    yfield_tot = np.zeros((xdim,ydim),dtype='complex128')
    zfield_tot = np.zeros((xdim,ydim),dtype='complex128')
    interference = np.zeros((xdim,ydim),dtype='complex128')
    
    for i in range(len(x)):
        # assign phase for each particle based on reference wave phase phi=0 at
        # the imaging plane
        xfield, yfield, zfield = calc_mie_fields(size, opt, n_particle_real[i],
                                                 n_particle_imag[i], radius[i],
                                                 x[i], y[i], z[i],
                                                 scaling_alpha[0])
 
        phase = np.exp(1j*np.pi*2*z[i]/opt.med_wavelen)
        phase_dif = np.exp(1j*np.pi*2*(z[i]-z[0])/opt.med_wavelen)
        interference += np.conj(xfield)*phase + np.conj(phase)*xfield
        xfield_tot += xfield*phase_dif
        yfield_tot += yfield*phase_dif
        zfield_tot += zfield*phase_dif

    total_scat_inten = (abs(xfield_tot**2) + abs(yfield_tot**2) +
                        abs(zfield_tot**2))

    holo = 1. + total_scat_inten*(scaling_alpha**2) + interference*scaling_alpha

    return Hologram(abs(holo), optics = opt)
        
        
def calc_mie_fields(size, opt, n_particle_real, n_particle_imag, radius, x, y,
                    z, alpha):
    """
    Calculates the scattered electric field from a spherical
    particle.

    Parameters
    ----------
    size : int or tuple
        Dimension of hologram.
    opt : instance of the :class:`holopy.optics.Optics` class
        Optics class containing information about the optics
        used in generating the hologram.
    n_particle_real : float
        Refractive index of particle.
    n_particle_imag : float
        Refractive index of particle.
    radius : float
        Radius of bead in microns.
    x : float
        x-position of particle in pixels.
    y : float
        y-position of particle in pixels.
    z : float
        z-position of particle in microns

    Returns
    -------
    Returns three arrays: the x-, y-, and z-component of scattered fields.

    Notes
    -----
    x- and y-coordinate of particle are given in pixels where
    (0,0) is at the top left corner of the image. 

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

    n = xdim*ydim

    fld_array = MFE.fields_tonumpy(x,y,z*1e6,n_particle_real, n_particle_imag,
                                           opt.index, radius*1e6,
                                           xdim, ydim, opt.wavelen,
                                           px*1e6)
    
    fld_x = fld_array[0:n] + (1j * fld_array[3*n:4*n])
    fld_y = fld_array[n:2*n] + (1j * fld_array[4*n:5*n])
    fld_z = fld_array[2*n:3*n] + (1j * fld_array[5*n:6*n])
    return fld_x.reshape(xdim,ydim),fld_y.reshape(xdim,ydim),fld_z.reshape(xdim,ydim)
