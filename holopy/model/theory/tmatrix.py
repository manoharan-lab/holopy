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
Defines Tmatrix theory class, which calculates scattering for multiple
spheres using the T-matrix superposition method, as implemented in
modified version of Daniel Mackowski's SCSMFO1B.FOR.  Uses full radial
dependence of spherical Hankel functions for the scattered field.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
'''

import scipy as sp
import numpy as np
import tmatrix_scsmfo.mieangfuncs as mieangfuncs
import tmatrix_scsmfo.scsmfo_min as scsmfo_min
import tmatrix_scsmfo.miescatlib as miescatlib
from holopy.hologram import Hologram
from holopy.utility.helpers import _ensure_array, _ensure_pair

from holopy.model.scatterer import Sphere, SphereCluster, Composite
from holopy.model.errors import TheoryNotCompatibleError
from holopy.model.theory.scatteringtheory import ScatteringTheory
from tmatrix_scsmfo.mieangfuncs import singleholo
from tmatrix_scsmfo.miescatlib import nstop, scatcoeffs

class Tmatrix(ScatteringTheory):
    """
    Class that contains methods and parameters for calculating
    scattering using T-matrix superposition method.

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
    niter : integer (optional)
        maximum number of iterations to use in solving the interaction
        equations 
    meth : integer (optional)
        method to use to solve interaction equations.  Set to 0 for
        biconjugate gradient; 1 for order-of-scattering
    eps : float (optional)
        relative error tolerance in solution for interaction equations
    qeps1 : float (optional) 
        error tolerance used to determine at what order the
        single-sphere spherical harmonic expansion should be truncated
    qeps2 : float (optional) 
        error tolerance used to determine at what order the cluster
        spherical harmonic expansion should be truncated 

    Notes
    -----
    According to Mackowski's manual for SCSMFO1B.FOR [1]_ and later
    papers [2]_, the biconjugate gradient is generally the most
    efficient method for solving the interaction equations, especially
    for dense arrays of identical spheres.  Order-of-scattering may
    converge better for non-identical spheres.

    References
    ---------
    [1] Daniel W. Mackowski, SCSMFO.FOR: Calculation of the Scattering
    Properties for a Cluster of Spheres,
    ftp://ftp.eng.auburn.edu/pub/dmckwski/scatcodes/scsmfo.ps 

    [2] D.W. Mackowski, M.I. Mishchenko, A multiple sphere T-matrix
    Fortran code for use on parallel computer clusters, Journal of
    Quantitative Spectroscopy and Radiative Transfer, In Press,
    Corrected Proof, Available online 11 March 2011, ISSN 0022-4073,
    DOI: 10.1016/j.jqsrt.2011.02.019. 
    """

    def __init__(self, imshape=(256,256), thetas=None, phis=None,
                 optics=None,  niter=200, eps=1e-6, meth=1, qeps1=1e-5, 
                 qeps2=1e-8): 

        # call base class constructor
        ScatteringTheory.__init__(self, imshape=imshape,
                                  thetas=thetas, phis=phis,
                                  optics=optics) 

        self.niter = niter
        self.eps = eps
        self.meth = meth
        self.qeps1 = qeps1
        self.qeps2 = qeps2

    def calc_field(self, scatterer):
        """
        Calculate fields for single or multiple spheres

        Parameters
        ----------
        scatterer : :mod:`holopy.model.scatterer` object
            scatterer or list of scatterers to compute field for

        Returns
        -------
        xfield, yfield, zfield : complex arrays with shape `imshape`
            x, y, z components of scattered fields

        Notes
        -----
        For multiple particles, this code superposes the fields
        calculated from each particle (using calc_mie_fields()). 
        """
        if isinstance(scatterer, Sphere):
            s = scatterer
            xfield, yfield, zfield = calc_mie_fields(self.imshape,
                                                     self.optics,
                                                     s.n.real,
                                                     s.n.imag, 
                                                     s.r, 
                                                     s.x, s.y, s.z)
        elif isinstance(scatterer, Composite):
            spheres = scatterer.get_component_list()
            # compatibility check: verify that the cluster only contains
            # spheres 
            if not scatterer._contains_only_spheres():
                for s in spheres:
                    if not isinstance(s, Sphere):
                        raise TheoryNotCompatibleError(self, s)
            # if it passes, superpose the fields
            xfield, yfield, zfield = self.superpose(spheres)
        else: raise TheoryNotCompatibleError(self, scatterer)

        return xfield, yfield, zfield

    def calc_holo(self, scatterer, alpha=1.0):
        """
        Calculate hologram formed by interference between scattered
        fields and a reference wave
        
        Parameters
        ----------
        scatterer : :mod:`holopy.model.scatterer` object
            scatterer or list of scatterers to compute field for
        alpha : scaling value for intensity of reference wave

        Returns
        -------
        holo : :class:`holopy.hologram.Hologram` object
            Calculated hologram from the given distribution of spheres

        Notes
        -----
        For a single particle, this code uses a fast Fortran
        subroutine to calculate the hologram.  Otherwise it uses the
        Fortran subroutine for calculating the fields from each
        particle, then superposes them using numpy.
        """

        if isinstance(scatterer, Sphere):
            s = scatterer
            holo = forward_holo(self.imshape, self.optics,
                                s.n.real, s.n.imag, s.r, 
                                s.x, s.y, s.z, alpha)
        else:   # call base class calc_holo
            holo = ScatteringTheory.calc_holo(self, scatterer, 
                                              alpha=alpha)

        return Hologram(holo, optics = self.optics)

# TODO: Need to refactor fitting code so that it no longer relies on
# the legacy functions below.  Then remove.
def _scaled_by_k(param_name):
    pars = ['radius', 'x', 'y', 'z']
    return param_name in pars

def _scaled_by_med_index(param_name):
    pars = ['n_particle_real', 'n_particle_imag']
    return param_name in pars

def calc_mie_fields(size, opt, n_particle_real, n_particle_imag,
                    radius, x, y, z, dimensional = True):
    '''
    Calculate the scattered electric field from a spherical particle
    using Fortran Mie code.

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
    dimensional: bool
       If False, assume all lengths non-dimensionalized by k and all
       indices relative (divided by medium index).

    Returns
    -------
    Returns three arrays: the x-, y-, and z-component of scattered fields.

    Notes
    -----
    x- and y-coordinate of particle are given in pixels where
    (0,0) is at the top left corner of the image. 
    '''

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

    # Determine particle properties in scattering units
    if dimensional:
        m_p = (n_particle_real + 1.j * n_particle_imag) / opt.index
        x_p = opt.wavevec * radius        
        kcoords = opt.wavevec * np.array([x, y, z])
    else:
        m_p = (n_particle_real + 1.j * n_particle_imag)
        x_p = radius
        kcoords = np.array([x, y, z])

    # Calculate maximum order lmax of Mie series expansion.
    lmax = miescatlib.nstop(x_p)
    # Calculate scattering coefficients a_l and b_l
    albl = miescatlib.scatcoeffs(x_p, m_p, lmax)

    # mieangfuncs.f90 works with everything dimensionless.
    gridx = opt.wavevec * np.mgrid[0:xdim] * px # (0,0) at upper left convention
    gridy = opt.wavevec * np.mgrid[0:ydim] * py

    escat_x, escat_y, escat_z = mieangfuncs.mie_fields(gridx, gridy, 
                                                       kcoords, 
                                                       albl,
                                                       opt.polarization)

    return escat_x, escat_y, escat_z
    
def forward_holo(size, opt, n_particle_real, n_particle_imag, radius,
                 x, y, z, scaling_alpha, dimensional = True, 
                 intensity=False):
    """
    Compute a hologram of N spheres by Mie superposition

    Parameters may be specified in any consistent set of units (make
    sure the optics object is also in the same units).
    
    Parameters
    ----------
    size : int or (int, int)
       dimension in pixels of the hologram to calculate (square if scalar)
    opt : Optics or dict
       Optics class or dictionary describing wavelength and pixel
       information for the calculation 
    n_particle_real : float or array(float)
       refractive index of sphere(s)
    n_particle_imag : float or array(float)
       imaginary refractive index of sphere(s)
    radius : float or array(float)
       radius of sphere(s)
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

    wavevec = 2.0 * np.pi / opt.med_wavelen

    xarr = _ensure_array(x).copy()
    yarr = _ensure_array(y).copy()
    zarr = _ensure_array(z).copy()
    nrarr = _ensure_array(n_particle_real).copy()
    niarr = _ensure_array(n_particle_imag).copy()
    rarr = _ensure_array(radius).copy()

    # For a single particle, use fast fortran subroutine to
    # calculate hologram instead of calculating fields first
    if len(xarr) == 1:
        # non-dimensionalization
        if dimensional:
            # multiply all length scales by k
            com_coords = np.array([xarr[0], yarr[0], zarr[0]]) * wavevec
            x_p = rarr[0] * wavevec
            # relative indices
            m_real = nrarr[0] / opt.index
            m_imag = niarr[0] / opt.index
        else:
            com_coords = np.array([xarr[0], yarr[0], zarr[0]])
            x_p = rarr[0]
            m_real = nrarr[0]
            m_imag = niarr[0]

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

    xfield_tot = np.zeros((xdim,ydim),dtype='complex128')
    yfield_tot = np.zeros((xdim,ydim),dtype='complex128')
    zfield_tot = np.zeros((xdim,ydim),dtype='complex128')
    interference = np.zeros((xdim,ydim),dtype='complex128')

    # for multiple particles, do Mie superposition in Python using
    # Fortran-calculated fields
    for i in range(len(xarr)):
        # assign phase for each particle based on reference wave phase
        # phi=0 at the imaging plane
        xfield, yfield, zfield = calc_mie_fields(size, opt, 
                                                 nrarr[i],
                                                 niarr[i], 
                                                 rarr[i],
                                                 xarr[i], yarr[i], zarr[i],
                                                 dimensional=dimensional)
 
        phase = np.exp(1j*np.pi*2*zarr[i]/opt.med_wavelen)
        phase_dif = np.exp(-1j*np.pi*2*(zarr[i]-zarr[0])/opt.med_wavelen)
        # allow arbitrary linear polarization
        interference += (phase * (np.conj(xfield) * opt.polarization[0] + 
                                  np.conj(yfield) * opt.polarization[1]) + 
                         np.conj(phase) * (xfield * opt.polarization[0] + 
                                           yfield * opt.polarization[1]))
        xfield_tot += xfield*phase_dif
        yfield_tot += yfield*phase_dif
        zfield_tot += zfield*phase_dif

    # ignore z-field in total scattered intensity; the camera's pixels
    # should be sensitive to the z component of the Poynting vector, 
    # E x B, and the z component of E x B cannot depend on Ez.
    total_scat_inten = (abs(xfield_tot**2) + abs(yfield_tot**2))

    holo = 1. + total_scat_inten*(scaling_alpha**2) + interference*scaling_alpha

    if intensity is True:
        return total_scat_inten
    else:
        return Hologram(abs(holo), optics = opt)

def _forward_holo(size, opt, scat_dict): 
    '''
    Internal use; passes everything to public forward_holo
    non-dimensionally.
    '''
    return forward_holo(size, opt, dimensional = False, **scat_dict)
