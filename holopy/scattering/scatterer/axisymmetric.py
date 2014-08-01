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

'''
    Defines axisymmetric scatterers for input into T-matrix code.
    
    .. moduleauthor:: Anna Wang
    '''
from __future__ import division

import numpy as np
from ...core.math import rotation_matrix
from numpy.linalg import norm

from .scatterer import CenteredScatterer, Indicators
from ..errors import ScattererDefinitionError


class Axisymmetric(CenteredScatterer):
    """
        Scattering object representing spheroidal scatterers
        
        Parameters
        ----------
        n : complex
        Index of refraction
        r : float or (float, float)
        a, b semi-axes of the spheroid. b is along the z-axis, so a<b is a prolate spheroid, a>b is an oblate spheroid.
        center : 3-tuple, list or numpy array, specifies coordinates of center of the scatterer
        shape:       These specifications are from Mishchenko's amplq.lp.f code.
                     For spheroids shape = -1 and aspect ratio is the ratio of the 
                         horizontal to rotational axes.  AR is larger than 1 for oblate 
                         spheroids and smaller than 1 for prolate spheroids.                                   
                     For cylinders shape =-2 and AR is the ratio of the          
                         diameter to the length.                              
                     For Chebyshev particles shape must be positive and 
                         is the degree of the Chebyshev polynomial, while     
                         AR is the deformation parameter (Ref. 5).                    
                     For generalized Chebyshev particles (describing the shape
                         of distorted water drops) shape=-3.  The coefficients
                         of the Chebyshev polynomial expansion of the particle
                         expansion of the particle shape (Ref. 7) are specified in subroutine DROP.
        """
    
    def __init__(self, n=None, r=None, rotation = (0, 0), center=None, shape=-1, AR = None):
        self.n = n
        self.r = r
        self.rotation = rotation
        self.center = center

