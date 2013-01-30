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
Tests adda based DDA calculations

.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''
from __future__ import division


from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
import numpy as np
from nose.tools import with_setup
from nose.plugins.attrib import attr
from ...scattering.errors import ScattererDefinitionError
from ..scatterer import Sphere, Ellipsoid, Scatterer, JanusSphere

from ...core import ImageSchema, Optics
from ..theory import Mie, DDA
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
    sc = Sphere(n=1.59, r=3e-1, center=(0, 0, 0))
    assert_raises(ScattererDefinitionError, Sphere, n=1.59, r=3e-1, center=(0, 0))
    sc = sc.translated(1, -1, 30)
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
def test_DDA_indicator():
    n = 1.59
    center = (1, 1, 30)
    r = .3

    dda = DDA()

    sc = Sphere(n=n, r=r, center = center)

    sphere_holo = dda.calc_holo(sc, schema)

    s = Scatterer(Sphere(r=r, center = (0, 0, 0)).contains, n, center)

    gen_holo = dda.calc_holo(s, schema)

    assert_allclose(sphere_holo, gen_holo, rtol=2e-3)

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_voxelated_complex():
    o = Optics(wavelen=.66, index=1.33, polarization = (1, 0))
    s = Sphere(n = 1.2+2j, r = .2, center = (5,5,5))

    sv = Scatterer(s.indicators, s.n, s.center)

    schema = ImageSchema(50, .1, optics = o)

    holo_dda = DDA.calc_holo(sv, schema)
    verify(holo_dda, 'dda_voxelated_complex', rtol=1e-5)


@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_coated():
    cs = Sphere(
        center=[7.141442573813124, 7.160766866147957, 11.095409800342143],
        n=[(1.27121212428+0j), (1.49+0j)], r=[.1-0.0055, 0.1])

    lmie_holo = Mie.calc_holo(cs, schema)
    dda_holo = DDA.calc_holo(cs, schema)

    assert_allclose(lmie_holo, dda_holo, rtol = 5e-4)

@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_Ellipsoid_dda():
    e = Ellipsoid(1.5, r = (.5, .1, .1), center = (1, -1, 10))
    schema = ImageSchema(100, .1, optics = Optics(wavelen=.66, index=1.33, polarization = (1,0)))
    h = DDA.calc_holo(e, schema)

    assert_almost_equal(h.max(), 1.3152766077267062)
    assert_almost_equal(h.mean(), 0.99876620628942114)
    assert_almost_equal(h.std(), 0.06453155384119547)

def test_janus():
    schema = ImageSchema(60, .1, Optics(.66, 1.33, (1, 0)))
    s = JanusSphere(n = [1.34, 2.0], r = [.5, .51], rotation = (-np.pi/2, 0),
                    center = (5, 5, 5))
    assert_almost_equal(s.index_at([5,5,5]),1.34)
    holo = DDA.calc_holo(s, schema)
    verify(holo, 'janus_dda')
