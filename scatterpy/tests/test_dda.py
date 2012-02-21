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


import holopy
import numpy as np
from nose.tools import assert_raises
from numpy.testing import (assert_, assert_almost_equal,
                           assert_allclose)
from nose.tools import with_setup
from nose.plugins.attrib import attr

from scatterpy.scatterer import Sphere, CoatedSphere
from scatterpy.scatterer import Composite, SphereCluster

from scatterpy.theory import Mie, DDA
import scatterpy
from scatterpy.calculate import calc_field, calc_holo, calc_intensity
from scatterpy.errors import TheoryNotCompatibleError
from holopy.optics import (WavelengthNotSpecified, PixelScaleNotSpecified,
                           MediumIndexNotSpecified)

import os.path

# nose setup/teardown methods
def setup_optics():
    # set up optics class for use in several test functions
    global optics
    wavelen = 658e-3
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151, .1151]
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
    theory = DDA(optics)
    assert_((theory.imshape == (256,256)).all())
    theory = DDA(optics, imshape=(100,100))
    assert_((theory.imshape == (100,100)).all())

    # test with single value instead of tuple
    theory = DDA(optics, imshape=128)
    assert_((theory.imshape == (128,128)).all())

    # construct with optics
    theory = DDA(imshape=256, optics=optics)
    assert_(theory.optics.index == 1.33)

@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_sphere():
    sc = Sphere(n=1.59, r=3e-1, x=1, y=-1, z=30)
    dda = DDA(imshape=128, optics=optics)
    mie = Mie(imshape=128, optics=optics)

    mie_holo = mie.calc_holo(sc)
    dda_holo = dda.calc_holo(sc)
    assert_allclose(mie_holo, dda_holo, rtol=.0015)

@attr('slow')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_general():
    # test that DDA general gets the same results as DDA sphere as a basic
    # sanity check of dda

    n = 1.59
    center = (1, 1, 30)
    r = .3
    
    sc = Sphere(n=n, r=r, center = center)
    dda = DDA(imshape=128, optics=optics)

    sphere_holo = dda.calc_holo(sc)

    geom = np.loadtxt(os.path.join(dda._last_result_dir, 'sphere.geom'),
                      skiprows=3)

    # hardcode size for now.  This is available in the log of the adda output,
    # so we could get it with a parser, but this works for now, not that it
    # could change if we change the size of the scatterer (and thus lead to a
    # fail)
    # FAIL HINT: grid size hardcoded, check that it is what dda sphere outputs
    dpl_dia = 16
    
    sphere = np.zeros((dpl_dia,dpl_dia,dpl_dia))

    for point in geom:
        x, y, z = point
        sphere[x, y, z] = 1

    sphere = sphere.astype('float') * n
    
    dpl = 13.2569 # dpl_dia * optics.med_wavelen / (r*2)
    dpl = dpl_dia * optics.med_wavelen / (r*2)
    
    s = scatterpy.scatterer.general.GeneralScatterer(sphere, center, dpl)

    gen_holo = dda.calc_holo(s)

    assert_allclose(sphere_holo, gen_holo, rtol=1e-3)
