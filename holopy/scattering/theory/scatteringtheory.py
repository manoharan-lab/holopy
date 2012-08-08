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
Base class for scattering theories.  Implements python-based
calc_intensity and calc_holo, based on subclass's calc_field

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

import numpy as np
from ...core.data import Image, VectorData
from ...core.holopy_object import HolopyObject
from ..binding_method import binding, finish_binding

class ScatteringTheory(HolopyObject):
    """
    Base class for scattering theories
    
    Notes
    -----
    A subclasses that do the work of computing scattering should do it by
    implementing a _calc_field(self, scatterer, target) function that returns a
    VectorData electric field.  
    """

    def __init__(self):
        # If the user instantiates a theory, we need to replace the classmethods
        # that instantiate an object with normal methods that reference the
        # theory object
        finish_binding(self)
        
    @classmethod
    @binding
    def calc_field(cls_self, scatterer, target, scaling = 1.0):
        """
        Calculate fields.  Implemented in derived classes only.

        Parameters
        ----------
        scatterer : :mod:`holopy.scattering.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        e_field : :mod:`holopy.core.VectorData`
            scattered electric field


        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.  
        """
        return cls_self._calc_field(scatterer, target) * scaling
       
    @classmethod
    @binding
    def calc_intensity(cls_self, scatterer, target, scaling = 1.0): 
        """
        Calculate intensity at focal plane (z=0)

        Parameters
        ----------
        scatterer : :mod:`holopy.scattering.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        inten : Data
            scattered intensity

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.

        Total scattered intensity only takes into account the x- and
        y-components of the E-field.  The z-component is ignored
        because the detector's pixels should be sensitive to the z
        component of the Poynting vector, E x B, and the z component
        of E x B cannot depend on Ez.
        """
        field = cls_self.calc_field(scatterer, target = target, scaling = scaling)
        normal = np.array([0, 0, 1])
        normal = normal.reshape((1, 1, 3))
        return (abs(field*(1-normal))**2).sum(-1)


    @classmethod
    @binding
    def calc_holo(cls_self, scatterer, target, scaling=1.0):
        """
        Calculate hologram formed by interference between scattered
        fields and a reference wave
        
        Parameters
        ----------
        scatterer : :mod:`holopy.scattering.scatterer` object
            (possibly composite) scatterer for which to compute scattering
        alpha : scaling value for intensity of reference wave

        Returns
        -------
        holo : :class:`holopy.hologram.Hologram` object
            Calculated hologram from the given distribution of spheres

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.
        """
        scat = cls_self.calc_field(scatterer, target = target, scaling = scaling)

        # add the z component to polarization and adjust the shape so that it is
        # broadcast correctly
        p = np.append(target.optics.polarization, 0).reshape(1, 1, 3)

        ref = VectorData(p)

        return Image(interfere_at_detector(scat, ref),
                        optics=target.optics)

    @classmethod
    @binding
    def calc_cross_sections(cls_self, scatterer, optics):
        """
        Calculate scattering, absorption, and extinction 
        cross sections, and asymmetry parameter <cos \theta>. 
        To be implemented by derived classes.

        Parameters
        ----------
        scatterer : :mod:`holopy.scattering.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        cross_sections : array (4)
            Dimensional scattering, absorption, and extinction 
            cross sections, and <cos \theta>

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.
        """
        return cls_self._calc_cross_sections(scatterer, optics)

    @classmethod
    @binding
    def calc_scat_matrix(cls_self, scatterer, target):
        """
        Compute scattering matricies for scatterer

        Parameters
        ----------
        scatterer : :mod:`holopy.scattering.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        scat_matr : :mod:`holopy.core.Data`
            Scattering matricies at specified positions

        Notes
        -----
        calc_* functions can be called on either a theory class or a theory
        object.  If called on a theory class, they use a default theory object
        which is correct for the vast majority of situations.  You only need to
        instantiate a theory object if it has adjustable parameters and you want
        to use non-default values.  
        """

        return cls_self._calc_scat_matrix(scatterer, target)
        

# Subclass of scattering theory, overrides functions that depend on array
# ordering and handles the tranposes for sending data to/from fortran
class FortranTheory(ScatteringTheory):
    def _list_of_sph_coords(self, center, selection=None):
        return super(FortranTheory, self)._list_of_sph_coords(center,
                                                              selection).T

    
    
#TODO: Should this be a method of the Electric field class? - tgd 2011-08-15
def interfere_at_detector(e1, e2, detector_normal = (0, 0, 1)):
    """
    Compute the intensity as detected by a plane sensor normal to z from the
    interference of e1 and e2

    Parameters
    ----------
    e1, e2: :class:`scatterpy.theory.scatteringtheory.ElectricField`
        The two electric fields to superimpose

    Returns
    i: :class:`numpy.ndarray`
        2d array of detected intensity
    """
    # This function assumes the detector is normal to z and planar, these
    # assumptions could be relaxed by adding more parameters if necessary

    # normally we would have
    # interference = conj(xfield)*phase + conj(phase)*xfield, 
    # but we choose phase angle = 0 at z=0, so phase = 1
    # which gives 2*real(xfield)

    detector_normal = np.array(detector_normal).reshape((1, 1, 3))

    new = ((abs(e1)**2 + abs(e2)**2 + 2* np.real(e1*e2)) *
           (1 - detector_normal)).sum(axis=-1)
    new._update_metadata(e1._metadata)

    return new
    
class InvalidElectricFieldComputation(Exception):
    def __init__(self, reason):
        self.reason = reason
    def __str__(self):
        return "Invalid Electric Computation: " + self.reason
    
