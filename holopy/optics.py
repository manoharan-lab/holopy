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
Classes for defining optical train used to produce a hologram.  

In the future this will include routines to automatically calculate
the magnification.  The information might also be used to calculate
the spherical aberration.

Currently only handles inline optical trains.  Could probably extend
to off-axis by making an option to pass a second optical train for the
reference beam to the Optics class constructor.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""

import numpy as np
import copy
from types import NoneType
from .utility.helpers import _ensure_pair
from .io.yaml_io import Serializable

class Optics(Serializable):
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
        self._wavelen = wavelen
        self._index = index
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
                pass # TODO: code here
            elif mag is not None:
                # calculate from specified magnification
                self.pixel_scale = self.pixel_size/mag
            else:
                self.pixel_scale = None
        else:
            self.pixel_scale = _ensure_pair(pixel_scale)

    @property
    def index(self):
        if self._index is not None:
            return self._index
        else:
            raise MediumIndexNotSpecified
    @index.setter
    def index(self, value):
        self._index=value

    @property
    def wavelen(self):
        if self._wavelen is not None:
            return self._wavelen
        else:
            raise WavelengthNotSpecified
    @wavelen.setter
    def wavelen(self, value):
        self._wavelen=value
    
    
    @property
    def med_wavelen(self):
        """
        Calculates the wavelength in the medium.
        """
        return self.wavelen/self.index

    @property
    def wavevec(self):
        """
        The wavevector k, 2pi/(wavelength in medium)
        """
        return 2*np.pi/self.med_wavelen
            
    @property
    def pixel(self):
        try:
            if self.pixel_scale.__class__ == NoneType or \
                    _ensure_pair(self.pixel_scale)[1].__class__ == NoneType:
                raise PixelScaleNotSpecified
            return _ensure_pair(self.pixel_scale)
        except AttributeError:
            raise PixelScaleNotSpecified

    def propagate_ref_wave(self):
        """
        Not yet implemented

        Returns
        -------
        field : ndarray
            A plane reference wave propagated through the optics 
        """
        pass

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
    
    def __repr__(self):
        def f(x):
            if x is None or (hasattr(x, 'shape') and  x.shape is ()):
                return 'None'
            return list(x)
        return "{s.__class__.__name__}(wavelen={s.wavelen}, index={s._index}, \
polarization={polarization}, divergence={s.divergence}, \
pixel_size={pixel_size}, train={s.train}, mag={s.mag}, \
pixel_scale={pixel_scale})".format(s=self, polarization=f(self.polarization),
                                  pixel_size=f(self.pixel_size),
                                  pixel_scale=f(self.pixel_scale))

        
        return ("{0}(wavelen={1}, index={2}, ".format(self.__class__.__name__,
                                                      self._wavelen, self._index)
                +"polarization={0}, divergence={1}, ".format(repr(self.polarization),
                                                            self.divergence)
                +"pixel_size={0}, train={1}, ".format(repr(self.pixel_size),
                                                      self.train)
                +"mag={0}, pixel_scale = {1})".format(self.mag, repr(self.pixel_scale)))

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

class PixelScaleNotSpecified(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return ("Pixel scale not specified in Optics instance.")
