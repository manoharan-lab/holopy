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
Test fortran-based multilayered Mie calculations and python interface.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from __future__ import division

import numpy as np
from numpy import sqrt
from numpy.testing import assert_allclose
import yaml
import os
from ...core import Optics, ImageSchema
from .common import verify
from ..theory import Mie
from ..theory.mie_f import multilayer_sphere_lib, miescatlib
from ..scatterer import Sphere

from nose.plugins.attrib import attr

@attr('medium')
def test_Shell():
    s = Sphere(center=[7.141442573813124, 7.160766866147957, 11.095409800342143],
              n=[(1.27121212428+0j), (1.49+0j)], r=[0.960957713253-0.0055,
                                                    0.960957713253])

    optics = Optics(wavelen=0.658, index=1.36, polarization=[1.0, 0.0])

    t = ImageSchema(200, .071333, optics = optics)

    thry = Mie(False)
    h = thry.calc_holo(s, t, scaling = 0.4826042444701572)

    verify(h, 'shell')

@attr('medium')
def test_sooty_particles():
    '''
    Test multilayered sphere scattering coefficients by comparison of
    radiometric quantities.

    We will use the data in [Yang2003]_ Table 3 on  p. 1717, cases
    2, 3, and 4 as our gold standard.
    '''
    x_L = 100
    m_med = 1.33
    m_abs = 2. + 1.j
    f_v = 0.1

    def efficiencies_from_scat_units(m, x):
        asbs = multilayer_sphere_lib.scatcoeffs_multi(m, x)
        qs = miescatlib.cross_sections(*asbs) * 2 / x_L**2
        # there is a factor of 2 conventional difference between
        # "backscattering" and "radar backscattering" efficiencies.
        return np.array([qs[1], qs[0], qs[2]/2.])

    # first case: absorbing core
    x_ac = np.array([f_v**(1./3.) * x_L, x_L])
    m_ac = np.array([m_abs, m_med])
    
    # second case: absorbing shell
    x_as = np.array([(1. - f_v)**(1./3.), 1.]) * x_L
    m_as = np.array([m_med, m_abs])

    # third case: smooth distribution (900 layers)
    n_layers = 900
    x_sm = np.arange(1, n_layers + 1) * x_L / n_layers
    beta = (m_abs**2 - m_med**2) / (m_abs**2 + 2. * m_med**2)
    f = 4./3. * (x_sm / x_L) * f_v 
    m_sm = m_med * sqrt(1. + 3. * f * beta / (1. - f * beta))

    location = os.path.split(os.path.abspath(__file__))[0]
    gold_name = os.path.join(location, 'gold',
                             'gold_multilayer')
    gold = np.array(yaml.load(file(gold_name + '.yaml')))

    assert_allclose(efficiencies_from_scat_units(m_ac, x_ac), gold[0],
                    rtol = 2e-5)
    assert_allclose(efficiencies_from_scat_units(m_as, x_as), gold[1],
                    rtol = 2e-5)
    assert_allclose(efficiencies_from_scat_units(m_sm, x_sm), gold[2],
                    rtol = 1e-3)

