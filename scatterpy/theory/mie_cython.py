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
Calculates scattering for an arbitrary number of spheres by mie
superposition using the Cython code for Mie field calculation.  Note
that this code can only handle x-polarized incident fields.

This module is DEPRECATED and proposed for deletion.  Use mie.py
instead. 

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.harvard.edu>
.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
'''

import numpy as np
from mie_c import MFE
from holopy.hologram import Hologram
from holopy import Optics
from holopy.utility.helpers import _ensure_array, _ensure_pair
from holopy.io.fit_io import _split_particle_number, _get_num_particles
from scatterpy.scatterer import Sphere, SphereCluster, Composite
from scatterpy.errors import TheoryNotCompatibleError

import warnings
warnings.warn('mie_cython is deprecated; use the mie module instead')

class Mie():
    """
    Class that contains methods and parameters for calculating
    scattering using Mie theory.

    Attributes
    ----------
    imshape : float or tuple (optional)
        Size of grid to calculate scattered fields or
        intensities. This is the shape of the image that calc_field or
        calc_intensity will return
    phis : array 
        Specifies azimuthal scattering angles to calculate (incident
        direction is z)
    thetas : array 
        Specifies polar scattering angles to calculate
    optics : :class:`holopy.optics.Optics` object
        specifies optical train

    Notes
    -----
    If phis and thetas are both 1-D vectors, the calc_ functions
    should return an array where result(i,j) = result(phi(i),
    theta(j))
    """

    def __init__(self, imshape=(256, 256), thetas=None, phis=None,
                 optics=None): 
        self.imshape = _ensure_pair(imshape)
        self.thetas = thetas
        self.phis = phis
        if isinstance(optics, dict):
            optics = Optics(**optics)
        elif optics is None:
            self.optics = Optics()
        else:
            self.optics = optics

    def calc_field(self, scatterer):
        """
        Calculate fields 

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for

        Returns
        -------
        xfield, yfield, zfield : complex arrays with shape `imshape`
            x, y, z components of scattered fields

        Notes
        -----
        For multiple particles, this code superposes the fields
        calculated from each particle (using calc_mie_fields()). The
        Mie field calculation for each particle assumes that the
        incident field phase angle is 0 at each particle's center.  So
        when we superpose the fields, we need to correct for the phase
        differences between particles.  We choose the convention that
        the incident field phase angle will be 0 at z=0.  This makes
        it possible to interfere the total scattered field with the
        incident field to compute the hologram (in calc_holo())

        Short summary: the total scattered field is computed such that
        the phase angle of the incident field is 0 at z=0
        """
        if isinstance(scatterer, Sphere):
            spheres = [scatterer]
        # compatibility check: verify that the cluster only contains
        # spheres 
        elif isinstance(scatterer, Composite):
            spheres = scatterer.get_component_list()
            if not scatterer.contains_only_spheres():
                for s in spheres:
                    if not isinstance(s, Sphere):
                        raise TheoryNotCompatibleError(self, s)
        else: raise TheoryNotCompatibleError(self, scatterer)
            
        xfield_tot = np.zeros(self.imshape, dtype='complex128')
        yfield_tot = np.zeros(self.imshape, dtype='complex128')
        zfield_tot = np.zeros(self.imshape, dtype='complex128')

        for s in spheres:
            # The cython code we use here expects x,y in terms of
            # pixels, so convert to pixels by dividing by the pixel
            # size
            x = s.x/self.optics.pixel[0]
            y = s.y/self.optics.pixel[1]
            z = s.z     # z is not expected to be in pixels

            xfield, yfield, zfield  = \
                calc_mie_fields(self.imshape, self.optics, 
                                np.real(s.n), np.imag(s.n), s.r,
                                x, y, z)
            # see Notes section above for how phase is computed.
            # The - sign in front of the phase is necessary to get the
            # holograms to come out right!  I think this is because in
            # our convention, k points in the -z direction. 
            phase_dif = (np.exp(-1j*np.pi*2*(s.z)/self.optics.med_wavelen))
            xfield_tot += xfield*phase_dif
            yfield_tot += yfield*phase_dif
            zfield_tot += zfield*phase_dif

        return xfield_tot, yfield_tot, zfield_tot

    def calc_intensity(self, scatterer):
        """
        Calculate intensity at focal plane (z=0)

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        alpha : scaling value for intensity

        Returns
        -------
        
        """

        xfield, yfield, zfield = self.calc_field(scatterer)
        return (abs(xfield**2) + abs(yfield**2) + abs(zfield**2))

    def calc_holo(self, scatterer, alpha=1.0):
        """
        Calculate hologram formed by interference between scattered
        fields and a reference wave
        
        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        alpha : scaling value for intensity of reference wave

        Returns
        -------
        holo : :class:`holopy.hologram.Hologram` object
            Calculated hologram from the given distribution of spheres
        """

        xfield, yfield, zfield = self.calc_field(scatterer)
        total_scat_inten = (abs(xfield**2) + abs(yfield**2) + 
                            abs(zfield**2))
        # normally we would have
        # interference = conj(xfield)*phase + conj(phase)*xfield, 
        # but we choose phase angle = 0 at z=0, so phase = 1
        # which gives 2*real(xfield)
        interference = 2*np.real(xfield * self.optics.polarization[0] +
                                 yfield * self.optics.polarization[1])
        holo = (1. + total_scat_inten*(alpha**2) + 
                interference*alpha)     # holo should be purely real

        return Hologram(abs(holo), optics = self.optics)

par_ordering = ['n_particle_real', 'n_particle_imag', 'radius', 'x',
                'y', 'z', 'scaling_alpha']

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
            # parameter was nondimensionalized by k in input; our code
            # expects that not to have happened, so we divide it out
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

# TODO: Need to refactor fitting code so that it no longer relies on
# the legacy functions below.  Then remove.
def forward_holo(size, opt, n_particle_real, n_particle_imag, radius,
                 x, y, z, 
                 scaling_alpha, intensity=False):
    """
    Compute a hologram of N spheres by Mie superposition, using the
    cython Mie code.

    Parameters may be specified in any consistent set of units (make
    sure the optics object is also in the same units).
    
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

    Notes
    -----
    The Cython Mie code assumes that the polarization is in the
    x-direction, so don't use this function unless you're able to
    align your incident field along x.  Also the scattered intensity
    includes the z-component of the scattered fields, so calculations
    will differ from those of codes that use only the x- and y-
    components to calculated the scattered intensity.
    """
    if isinstance(opt, dict):
        opt = Optics(**opt)

    xdim, ydim = _ensure_pair(size)

    xarr = _ensure_array(x).copy()
    yarr = _ensure_array(y).copy()
    zarr = _ensure_array(z).copy()
    n_particle_real = _ensure_array(n_particle_real)
    n_particle_imag = _ensure_array(n_particle_imag)
    radius = _ensure_array(radius)
    scaling_alpha = _ensure_array(scaling_alpha)

    # The code we use here expects things in terms of pixels, so convert to
    # pixels by dividing by the pixel size
    xarr /= opt.pixel[0]
    yarr /= opt.pixel[1]

    xfield_tot = np.zeros((xdim, ydim),dtype='complex128')
    yfield_tot = np.zeros((xdim, ydim),dtype='complex128')
    zfield_tot = np.zeros((xdim, ydim),dtype='complex128')
    interference = np.zeros((xdim, ydim),dtype='complex128')
    
    for i in range(len(xarr)):
        # assign phase for each particle based on reference wave phase
        # phi=0 at the imaging plane
        xfield, yfield, zfield = calc_mie_fields(size, opt, 
                                                 n_particle_real[i],
                                                 n_particle_imag[i], 
                                                 radius[i],
                                                 xarr[i], yarr[i], zarr[i])

        phase = np.exp(1j*np.pi*2*zarr[i]/opt.med_wavelen)
        phase_dif = np.exp(-1j*np.pi*2*(zarr[i]-zarr[0])/opt.med_wavelen)
        interference += phase*np.conj(xfield) + np.conj(phase)*xfield
        xfield_tot += xfield*phase_dif
        yfield_tot += yfield*phase_dif
        zfield_tot += zfield*phase_dif

    total_scat_inten = (abs(xfield_tot**2) + abs(yfield_tot**2) +
                        abs(zfield_tot**2))

    holo = 1. + total_scat_inten*(scaling_alpha**2) + interference*scaling_alpha

    if intensity is True:
        return total_scat_inten
    else:
        return Hologram(abs(holo), optics = opt)
        

def calc_mie_fields(size, opt, n_particle_real, n_particle_imag,
                    radius, x, y, z):
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
    x- and y-coordinate of particle are given in pixels where (0,0) is
    at the top left corner of the image.  Also, the Cython Mie code
    assumes that the polarization is in the x-direction, so don't use
    this function unless you're able to align your incident field
    along x.
    """
        
    if isinstance(opt, dict):
        opt = Optics(**opt)

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

    fld_array = MFE.fields_tonumpy(x, y, z*1e6, n_particle_real, 
                                   n_particle_imag,
                                   opt.index, radius*1e6,
                                   xdim, ydim, opt.wavelen,
                                   px*1e6)
    
    fld_x = fld_array[0:n] + (1j * fld_array[3*n:4*n])
    fld_y = fld_array[n:2*n] + (1j * fld_array[4*n:5*n])
    fld_z = fld_array[2*n:3*n] + (1j * fld_array[5*n:6*n])
    return fld_x.reshape(xdim,ydim), fld_y.reshape(xdim,ydim), fld_z.reshape(xdim,ydim)
