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
"""
Defines Multisphere theory class, which calculates scattering for multiple
spheres using the (exact) superposition method implemented in
modified version of Daniel Mackowski's SCSMFO1B.FOR.  Uses full radial
dependence of spherical Hankel functions for the scattered field.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""
from __future__ import division

import numpy as np
import mie_f.mieangfuncs as mieangfuncs
import mie_f.scsmfo_min as scsmfo_min

from scatterpy.errors import UnrealizableScatterer

from scatterpy.scatterer import SphereCluster
from scatterpy.errors import TheoryNotCompatibleError
from scatterpy.theory.scatteringtheory import ScatteringTheory, ElectricField

class Multisphere(ScatteringTheory):
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
p c    for dense arrays of identical spheres.  Order-of-scattering may
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

    def __init__(self, optics, imshape=(256, 256), thetas=None, phis=None,
                 niter=200, eps=1e-6, meth=1, qeps1=1e-5, 
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

    def calc_field(self, scatterer, selection=None):
        """
        Calculate fields for single or multiple spheres

        Parameters
        ----------
        scatterer : :mod:`scatterpy.scatterer` object
            scatterer or list of scatterers to compute field for
        selection : array of integers (optional)
            a mask with 1's in the locations of pixels where you
            want to calculate the field, defaults to all pixels
            
        Returns
        -------
        xfield, yfield, zfield : complex arrays with shape `imshape`
            x, y, z components of scattered fields

        """
        
        if not isinstance(scatterer, SphereCluster):
            raise TheoryNotCompatibleError(self, scatterer)

        # check that the parameters are in a range where the multisphere
        # expansion will work
        for s in scatterer.scatterers:
            if s.r * self.optics.wavevec > 1e3:
                raise UnrealizableScatterer(self, s, "radius too large, field "+
                                            "calculation would take forever")

        # switch to centroid weighted coordinate system tmatrix code expects
        # and nondimensionalize
        centers = (scatterer.centers - scatterer.centers.mean(0)) * self.optics.wavevec

        m = scatterer.n / self.optics.index

        if (centers > 1e4).any():
            raise UnrealizableScatterer(self, scatterer, "Particle separation \
 too large, calculation would take forever")
        
        
        _, lmax, amn0, converged = scsmfo_min.amncalc(1, centers[:,0], 
                                                      centers[:,1],
                                                      # The fortran code uses 
                                                      # oppositely
                                                      # directed z axis 
                                                      # (they have laser
                                                      # propagation as positive,
                                                      # we have it
                                                      # negative), so we 
                                                      # multiply the z
                                                      # coordinate by -1 to 
                                                      # correct for that.  
                                                      -1.0 * centers[:,2], 
                                                      m.real, m.imag,
                                                      scatterer.r * 
                                                      self.optics.wavevec,
                                                      self.niter, self.eps, 
                                                      self.qeps1,
                                                      self.qeps2, self.meth, 
                                                      (0,0))

        # converged == 1 if the SCSMFO iterative solver converged
        # f2py converts F77 LOGICAL to int
        if not converged:
            raise ConvergenceFailureMultisphere()

        # chop off unused parts of amn0, the fortran code currently has a hard
        # coded number of parameters so it will return too many coefficients.
        # We truncate here to reduce the length of stuff we have to compute with
        # later.  
        limit = lmax**2 + 2*lmax
        amn = amn0[:, 0:limit, :]

        if np.isnan(amn).any():
            raise MultisphereExpansionNaN()

        if selection == None:
            selection = np.ones(self.imshape,dtype='int')
        e_x, e_y, e_z = mieangfuncs.tmatrix_fields_sph(self._spherical_grid(
                scatterer.x.mean(), scatterer.y.mean(), scatterer.z.mean()),
                                                       amn, lmax, 0,
                                                       self.optics.polarization,
                                                       selection)
        # TODO: Test this.  How do we intentionally get NaN's out of tmatrix?
        if np.isnan(e_x[0,0]):
            raise TMatrixFieldNaN(self, scatterer, '')

        return ElectricField(e_x, e_y, e_z, scatterer.z.mean(),
                             self.optics.med_wavelen) 


class TMatrixFieldNaN(UnrealizableScatterer):
    def __str__(self):
        return "T-matrix field is NaN, this probably represents a failure of \
the code to converge, check your scatterer."


class MultisphereExpansionNaN(Exception):
    def __str__(self):
        return ("Internal expansion for Multisphere coefficients contains "
                "NaN.  This probably means your scatterer is unphysical.")

class ConvergenceFailureMultisphere(Exception):
    def __str__(self):
        return ("Multisphere calculations failed to converge, this probably means "
                "your scatterer is unphysical, or possibly just huge")
