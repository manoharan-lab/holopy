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
from __future__ import division


from nose.tools import assert_raises
from numpy.testing import assert_almost_equal, dec
import numpy as np
from nose.tools import with_setup
from nose.plugins.attrib import attr

from ..scatterer import Sphere, Ellipsoid, CoatedSphere
from ...core import ImageTarget, Optics
from ..theory import Mie, DDA
from ..theory.dda import DependencyMissing
from ..scatterer.voxelated import ScattererByFunction, VoxelatedScatterer
from .common import assert_allclose, verify


import os.path

def dda_external_not_available():
    try:
        DDA()
    except DependencyMissing:
        return True
    return False
    

# nose setup/teardown methods
def setup_optics():
    # set up optics class for use in several test functions
    global optics, target
    wavelen = 658e-3
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151, .1151]
    index = 1.33
    
    optics = Optics(wavelen=wavelen, index=index,
                    pixel_scale=pixel_scale,
                    polarization=polarization,
                    divergence=divergence)
    target = ImageTarget(128, optics = optics)
    
def teardown_optics():
    global optics, target
    del optics, target

@dec.skipif(dda_external_not_available(), "a-dda not installed")
@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_sphere():
    sc = Sphere(n=1.59, r=3e-1, center=(1, -1, 30))

    mie_holo = Mie.calc_holo(sc, target)
    dda_holo = DDA.calc_holo(sc, target)
    assert_allclose(mie_holo, dda_holo, rtol=.0015)

@dec.skipif(dda_external_not_available(), "a-dda not installed")
@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_voxelated():
    # test that DDA voxelated gets the same results as DDA sphere as a basic
    # sanity check of dda

    n = 1.59
    center = (1, 1, 30)
    r = .3

    dda = DDA()
    
    sc = Sphere(n=n, r=r, center = center)

    sphere_holo = dda.calc_holo(sc, target)

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
    
    dpl = 13.2569

    # this would nominally be the correct way to determine dpl, but because of
    #volume correction within adda, this is not as accurate (only 
    #dpl = dpl_dia * optics.med_wavelen / (r*2)
    
    s = VoxelatedScatterer(sphere, center, dpl)

    gen_holo = dda.calc_holo(s, target)

    assert_allclose(sphere_holo, gen_holo, rtol=1e-3)

@attr('fast')
@dec.skipif(dda_external_not_available(), "a-dda not installed")
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_voxelated_complex():
    o = Optics(wavelen=.66, index=1.33, pixel_scale=.1)
    s = Sphere(n = 1.2+2j, r = .2, center = (5,5,5))

    def sphere(r):
        rsq = r**2
        def test(point):
            return (point**2).sum() < rsq
        return test

    sv = ScattererByFunction(sphere(s.r), s.n, [[-s.r, s.r], [-s.r, s.r], [-s.r,
    s.r]], center = s.center)

    target = ImageTarget(50, optics = o)

    holo_dda = DDA.calc_holo(sv, target)
    verify(holo_dda, 'dda_voxelated_complex', rtol=1e-5)

    
@attr('medium')
@dec.skipif(dda_external_not_available(), "a-dda not installed")
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_coated():
    cs = CoatedSphere(
        center=[7.141442573813124, 7.160766866147957, 11.095409800342143],
        n=[(1.27121212428+0j), (1.49+0j)], r=[.1-0.0055, 0.1])

    lmie_holo = Mie.calc_holo(cs, target)
    dda_holo = DDA.calc_holo(cs, target)

    assert_allclose(lmie_holo, dda_holo, rtol = 5e-5)

@dec.skipif(dda_external_not_available(), "a-dda not installed")
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_Ellipsoid_dda():
    e = Ellipsoid(1.5, r = (.5, .1, .1), center = (1, -1, 10))
    target = ImageTarget(100, optics = Optics(wavelen=.66, pixel_scale=.1, index=1.33))
    h = DDA.calc_holo(e, target)

    assert_almost_equal(h.max(), 1.3152766077267062)
    assert_almost_equal(h.mean(), 0.99876620628942114)
    assert_almost_equal(h.std(), 0.06453155384119547)

    
