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
Tests adda based DDA calculations

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

# Disabled until DDA is closer to working

'''
import holopy
from nose.tools import assert_raises
from numpy.testing import (assert_, assert_almost_equal,
                           assert_allclose)
from nose.tools import with_setup
from nose.plugins.attrib import attr

from scatterpy.scatterer import Sphere, CoatedSphere
from scatterpy.scatterer import Composite, SphereCluster

from scatterpy.theory import Mie, DDA
from scatterpy.calculate import calc_field, calc_holo, calc_intensity
from scatterpy.errors import TheoryNotCompatibleError
from holopy.optics import (WavelengthNotSpecified, PixelScaleNotSpecified,
                           MediumIndexNotSpecified)


# nose setup/teardown methods
def setup_optics():
    # set up optics class for use in several test functions
    global optics
    wavelen = 658e-9
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151e-6, .1151e-6]
    index = 1.33
    
    optics = holopy.optics.Optics(wavelen=wavelen, index=index,
                                  pixel_scale=pixel_scale,
                                  polarization=polarization,
                                  divergence=divergence)
    
def teardown_optics():
    global optics
    del optics

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_construction():
    theory = DDA()
    assert_(theory.imshape == (256,256))
    theory = DDA(imshape=(100,100))
    assert_(theory.imshape == (100,100))

    # test with single value instead of tuple
    theory = DDA(imshape=128)
    assert_(theory.imshape == (128,128))

    # construct with optics
    theory = DDA(imshape=256, optics=optics)
    assert_(theory.optics.index == 1.33)

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_single():
    sc = Sphere(n=1.59, r=5e-7, x=1e-6, y=-1e-6, z=10e-6)
    dda = DDA(imshape=128, optics=optics)
    mie = Mie(imshape=128, optics=optics)

    mie_fields = mie.calc_field(sc)
    dda_fields = dda.calc_field(sc)

    assert_allclose(mie_fields.x_comp, dda_fields.x_comp)
    assert_allclose(mie_fields.y_comp, dda_fields.y_comp)
    assert_allclose(mie_fields.z_comp, dda_fields.z_comp)

    mie_intensity = mie.calc_intensity(sc)
    dda_intensity = dda.calc_intensity(sc)
    assert_allclose(mie_intensity, dda_intensity)

    mie_holo = mie.calc_holo(sc)
    dda_holo = dda.calc_holo(sc)
    assert_allclose(mie_holo, dda_holo)
'''
