# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
Classes for defining metadata about experimental or calculated results.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
from __future__ import division

import numpy as np
from warnings import warn
import copy
from .helpers import _ensure_pair, _ensure_array
from holopy_object import HoloPyObject


class Optics(HoloPyObject):
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
        Electric field amplitudes in x and y directions.
    divergence : float (optional)
        Divergence of the incident beam (currently unused)
    pixel_size : tuple (optional)  (deprecated)
        Physical size of the camera's pixels.
    mag : float (optional)
        Magnification of optical train. Ignored if pixel_scale
        is specified.
    pixel_scale : tuple (optional) (deprecated)
        Size of pixel in the imaging plane. This is equal to the
        physical size of the pixel divided by the magnification.

    Notes
    -----
    You don't have to specify all of these parameters, but to get fits and
    reconstructions to work you should specify at least,
    `wavelen` in vacuo and `index`.
    """

    def __init__(self, wavelen=None, index=None, polarization=None,
                 divergence=0., pixel_size=None, mag=None, pixel_scale = None):
        # source parameters
        self.wavelen = wavelen
        self.index = index
        if polarization is None:
            warn("Polarization not specified. You will not be able to use this optics"
                    " for most calculations")
        self.polarization = np.array(polarization)
        self.divergence = divergence
        if divergence != 0.0:
            warn("HoloPy calculations currently ignore divergence")

        # optical train parameters
        self.mag = mag          # magnification

        # detector parameters (deprecated: detector information should
        # be in the Marray object since it isn't really associated
        # withthe optical train)
        self.pixel_size = _ensure_pair(pixel_size)
        if pixel_scale is None:
            if mag is not None:
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

class PositionSpecification(HoloPyObject):
    """
    Abstract base class for representations of positions.  You should use its
    subclasses
    """
    pass

class Positions(np.ndarray, HoloPyObject):
    """
    Positions of pixels of an Marray

    Parameters
    ----------
    arr : ndarray
        Pixel positions, in Cartesian xyz coordinates
    """
    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    def __array_wrap(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def xyz(self):
        return Positions(self.reshape(-1, 3))

    def r_theta_phi(self, origin):
        xg, yg, zg = self.xyz().T
        x, y, z = origin

        x = xg - x
        y = yg - y
        # sign is reversed for z because of our choice of image
        # centric rather than particle centric coordinate system
        z = z - zg

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        # get phi between 0 and 2pi
        phi = phi + 2*np.pi * (phi < 0)
        # if z is an array, phi will be the wrong shape. Checking its
        # last dimension will determine this so we can correct it
        if phi.shape[-1] != r.shape[-1]:
            phi = phi.repeat(r.shape[-1], -1)

        return np.vstack((r, theta, phi)).T

    def kr_theta_phi(self, origin, optics):
        pos = self.r_theta_phi(origin)
        pos[:,0] *= optics.wavevec
        return pos

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
    """Specify far field positions as a grid of angles

    The list of thetas and phis are used to construct a grid of
    positions angles should be specified in radians.

    Parameters
    ----------
    theta : list or ndarray
        coordinates for the polar angle
    phi : list or ndarray
        coordinates for the azimuthal angle

    Notes

    """
    def __init__(self, theta, phi = [0], units = 'radians'):
        self.theta = theta
        self.phi = phi
        self.shape = len(self.theta), len(self.phi)

    def positions_theta_phi(self):
        pos = np.zeros((self.shape[0]*self.shape[1], 2))
        for i, theta in enumerate(self.theta):
            for j, phi in enumerate(self.phi):
                pos[i*self.shape[1]+j] = theta, phi
        return pos
