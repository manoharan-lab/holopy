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
Classes for defining metadata about experimental or calulated results.  

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
from __future__ import division

import numpy as np
import copy
from .helpers import _ensure_pair, _ensure_array
from holopy_object import HolopyObject


class Optics(HolopyObject):
    """
    Contains details about the source, detector, and optical train used
    to generate a hologram.

    Attributes
    ----------
    wavelen : float (optional)
        Wavelength of imaging light in vacuo.
    index : float (optional)
        Refractive index of medium
    polarization : tuple or array (optional)
        Electric field amplitudes in x and y direction.
    divergence : float (optional)
        Divergence of the incident beam (currently unused)
    pixel_size : tuple (optional)
        Physical size of the camera's pixels.
    mag : float (optional)
        Magnification of optical train. Ignored if pixel_scale
        is specified.
    pixel_scale : tuple (optional)
        Size of pixel in the imaging plane. This is equal to the
        physical size of the pixel divided by the magnification. 

    Notes
    -----
    You don't have to specify all of these parameters, but to get fits and 
    reconstructions to work you should specify at least `pixel_scale`,
    `wavelen` in vacuo, and `index`.  Alternatively you can specify 
    `pixel_size`, `mag`, `wavelen`, and `index`.
    """

    def __init__(self, wavelen=None, index=None, polarization=(1.0, 0),
                 divergence=0., pixel_size=None, train=None,
                 mag=None, pixel_scale = None):
        # source parameters
        self.wavelen = wavelen
        self.index = index
        self.polarization = np.array(polarization)
        self.divergence = divergence

        # optical train parameters
        self.mag = mag          # magnification 
        self.train = train
        # TODO: code here to validate optical train, calculate
        # magnification

        # detector parameters
        self.pixel_size = _ensure_pair(pixel_size)
        if pixel_scale is None:
            if train is not None:
                # calculate from optical train
                pass # TODO: code here #pragma: no cover
            elif mag is not None:
                # calculate from specified magnification
                self.pixel_scale = self.pixel_size/mag
            else:
                self.pixel_scale = None
        else:
            self.pixel_scale = _ensure_pair(pixel_scale)

    @property
    def med_wavelen(self):
        """
        Calculates the wavelength in the medium.
        """
        return self.wavelen/self.index

    def wavelen_in(self, medium_index):
        return self.wavelen/medium_index
    
    @property
    def wavevec(self):
        """
        The wavevector k, 2pi/(wavelength in medium)
        """
        return 2*np.pi/self.med_wavelen

    def wavevec_in(self, medium_index):
        return 2*np.pi/self.wavelen_in(medium_index)
    
    def resample(self, factor):
        """
        Update an optics instance for a resampling.  This has the effect of
        changing the pixel_scale.

        Returns a new instance
        """
        factor = np.array(factor)
        new = copy.copy(self)
        new.pixel_scale = self.pixel_scale * factor
        return new
    
class WavelengthNotSpecified(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return ("Wavelength not specified in Optics instance.")

class MediumIndexNotSpecified(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return ("Medium index not specified in Optics instance.")

class PositionSpecification(HolopyObject):
    """
    Abstract base class for representations of positions.  You should use its
    subclasses
    """
    pass
        
class Grid(PositionSpecification):
    """
    Rectangular grid of measurements
    """
    def __init__(self, spacing):
        self.spacing = spacing
        

    def resample_by_factors(self, factors):
        new = copy.copy(self)
        new.spacing = _ensure_array(new.spacing).astype('float')
        new.spacing[factors.keys()] *= factors.values()
        return new

    
class UnevenGrid(Grid):
    pass
            
class Angles(PositionSpecification):
    def __init__(self, theta = None, phi = None, units = 'radians'):
        self.theta = theta
        self.phi = phi
        self.units = units
        self.shape = theta.shape
        if self.phi is not None:
            assert self.phi.shape == self.shape

    
