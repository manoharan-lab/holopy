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
from numpy.testing import assert_almost_equal
import numpy as np
from nose.tools import with_setup
from nose.plugins.attrib import attr

from ..scatterer import Sphere, Ellipsoid, CoatedSphere
from ...core import ImageSchema, Optics, math
from ..theory import Mie, DDA
from ..scatterer.voxelated import ScattererByFunction, MultidomainScattererByFunction, VoxelatedScatterer
from .common import assert_allclose, verify


import os.path

# nose setup/teardown methods
def setup_optics():
    # set up optics class for use in several test functions
    global optics, schema
    wavelen = 658e-3
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151, .1151]
    index = 1.33

    optics = Optics(wavelen=wavelen, index=index,
                    polarization=polarization,
                    divergence=divergence)
    schema = ImageSchema(128, spacing = pixel_scale, optics = optics)

def teardown_optics():
    global optics, schema
    del optics, schema

@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_sphere():
    sc = Sphere(n=1.59, r=3e-1, center=(1, -1, 30))

    mie_holo = Mie.calc_holo(sc, schema)
    dda_holo = DDA.calc_holo(sc, schema)
    assert_allclose(mie_holo, dda_holo, rtol=.0015)

@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_dda_2_cpu():
    sc = Sphere(n=1.59, r=3e-1, center=(1, -1, 30))

    mie_holo = Mie.calc_holo(sc, schema)
    dda_n2 = DDA(n_cpu=2)
    dda_holo = dda_n2.calc_holo(sc, schema)

    # TODO: figure out how to actually test that it runs on multiple cpus

def in_sphere(r):
    rsq = r**2
    def test(point):
        return (point**2).sum() < rsq
    return test

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

    sphere_holo = dda.calc_holo(sc, schema)

    s = ScattererByFunction(in_sphere(r), n, [[-r, r], [-r, r], [-r, r]],
                            center)

    gen_holo = dda.calc_holo(s, schema)

    assert_allclose(sphere_holo, gen_holo, rtol=2e-3)

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_voxelated_complex():
    o = Optics(wavelen=.66, index=1.33)
    s = Sphere(n = 1.2+2j, r = .2, center = (5,5,5))

    def sphere(r):
        rsq = r**2
        def test(point):
            return (point**2).sum() < rsq
        return test

    sv = ScattererByFunction(sphere(s.r), s.n, [[-s.r, s.r], [-s.r, s.r], [-s.r,
    s.r]], center = s.center)

    schema = ImageSchema(50, .1, optics = o)

    holo_dda = DDA.calc_holo(sv, schema)
    verify(holo_dda, 'dda_voxelated_complex', rtol=1e-5)


@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_coated():
    cs = CoatedSphere(
        center=[7.141442573813124, 7.160766866147957, 11.095409800342143],
        n=[(1.27121212428+0j), (1.49+0j)], r=[.1-0.0055, 0.1])

    lmie_holo = Mie.calc_holo(cs, schema)
    dda_holo = DDA.calc_holo(cs, schema)

    assert_allclose(lmie_holo, dda_holo, rtol = 5e-5)

@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_Ellipsoid_dda():
    e = Ellipsoid(1.5, r = (.5, .1, .1), center = (1, -1, 10))
    schema = ImageSchema(100, .1, optics = Optics(wavelen=.66, index=1.33))
    h = DDA.calc_holo(e, schema)

    assert_almost_equal(h.max(), 1.3152766077267062)
    assert_almost_equal(h.mean(), 0.99876620628942114)
    assert_almost_equal(h.std(), 0.06453155384119547)

class HemisphericalShellTest:
    def __init__(self, center, normal, innerRadius, outerRadius):
        #store properties as arrays for easy numerical computation
        self.center = np.array(center)
        self.normal = np.array(normal)
        self.innerRadiusSq = innerRadius*innerRadius
        self.outerRadiusSq = outerRadius*outerRadius

    def isPtIn(self, pt):
        #vector center to pt
        delta = np.array(pt) - self.center
        #check which side of the plane we're on
        if np.dot(delta, self.normal) < 0 :
            return False
        #check if we're within the specified distance from the center
        distSq = np.dot(delta, delta)
        if distSq >= self.innerRadiusSq and distSq <= self.outerRadiusSq:
            return True
        else:
            return False


class SphereTest:
    def __init__(self, center, Radius):
        #store properties as arrays for easy numerical computation
        self.center = np.array(center)
        self.RadiusSq = Radius*Radius

    def isPtIn(self, pt):
        #vector center to pt
        delta = np.array(pt) - self.center

        #check if we're within the specified distance from the center
        distSq = np.dot(delta, delta)
        if distSq <= self.RadiusSq:
            return True
        else:
            return False

def test_janus():
    x = HemisphericalShellTest(np.array([0,0,0]), np.array([1,0,0]), .5, .51)
    schema = ImageSchema(60, .1, Optics(.66, 1.33))
    y = SphereTest(np.array([0,0,0]), .5)
    s = MultidomainScattererByFunction([x.isPtIn, y.isPtIn], [2.0+0j, 1.34+0j],[[-.51,.51],[-.51,.51],[-.51,.51]], (5,5,5))

    holo = DDA.calc_holo(s, schema)
    verify(holo, 'janus_dda')
