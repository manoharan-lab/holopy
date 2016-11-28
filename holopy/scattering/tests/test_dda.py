# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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


from numpy.testing import assert_almost_equal, assert_allclose
import numpy as np
from nose.tools import with_setup
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest
import os.path


from ...scattering.errors import DependencyMissing
from ..scatterer import Sphere, Ellipsoid, Scatterer, JanusSphere, Difference
from .. import Mie, DDA, calc_holo as calc_holo_external
from ...core import detector_grid, update_metadata
from ...core.tests.common import verify, assert_obj_close

# nose setup/teardown methods
def setup_optics():
    # set up optics class for use in several test functions
    global schema, wavelen, index
    wavelen = 658e-3
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151, .1151]
    index = 1.33

    schema = update_metadata(detector_grid(128, spacing = pixel_scale), illum_wavelen=wavelen, medium_index=index, illum_polarization=polarization)

def teardown_optics():
    global schema
    del schema

def calc_holo(schema, scatterer, medium_index=None, illum_wavelen=None,**kwargs):
    try:
        return calc_holo_external(schema, scatterer, index, wavelen, **kwargs)
    except DependencyMissing:
        raise SkipTest()

@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_sphere():
    sc = Sphere(n=1.59, r=3e-1, center=(0, 0, 0))
    sc = sc.translated(1, -1, 30)
    mie_holo = calc_holo(schema, sc, index, wavelen)
    dda_holo = calc_holo(schema, sc, index, wavelen, theory=DDA)
    assert_allclose(mie_holo, dda_holo, rtol=.0015)

@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_dda_2_cpu():
    sc = Sphere(n=1.59, r=3e-1, center=(1, -1, 30))
    mie_holo = calc_holo(schema, sc, index, wavelen)
    try:
        dda_n2 = DDA(n_cpu=2)
    except DependencyMissing:
        raise SkipTest()
    dda_holo = calc_holo(schema, sc, index, wavelen, theory=dda_n2)

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

    sc = Sphere(n=n, r=r, center = center)

    sphere_holo = calc_holo(schema, sc, index, wavelen, theory=DDA)

    s = Scatterer(Sphere(r=r, center = (0, 0, 0)).contains, n, center)

    gen_holo = calc_holo(schema, s, index, wavelen, theory=DDA)

    assert_allclose(sphere_holo, gen_holo, rtol=2e-3)

@attr('fast')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_voxelated_complex():
    s = Sphere(n = 1.2+2j, r = .2, center = (5,5,5))

    sv = Scatterer(s.indicators, s.n, s.center)

    schema = detector_grid(50, .1)

    holo_dda = calc_holo(schema, sv, illum_wavelen=.66, medium_index=1.33, illum_polarization = (1, 0), theory=DDA)
    verify(holo_dda, 'dda_voxelated_complex', rtol=1e-5)


@attr('medium')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_coated():
    cs = Sphere(
        center=[7.141442573813124, 7.160766866147957, 11.095409800342143],
        n=[(1.27121212428+0j), (1.49+0j)], r=[.1-0.0055, 0.1])

    lmie_holo = calc_holo(schema, cs, index, wavelen, theory=Mie)
    dda_holo = calc_holo(schema, cs, index, wavelen, theory=DDA)

    assert_allclose(lmie_holo, dda_holo, rtol = 5e-4)

@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_Ellipsoid_dda():
    e = Ellipsoid(1.5, r = (.5, .1, .1), center = (1, -1, 10))
    schema = detector_grid(100, .1)
    h = calc_holo(schema, e, illum_wavelen=.66, medium_index=1.33, illum_polarization = (1,0))

    assert_obj_close(h.max(), 1.3152766077267062)
    assert_obj_close(h.mean(), 0.99876620628942114)
    assert_obj_close(h.std(), 0.06453155384119547)

def test_janus():
    schema = detector_grid(60, .1)
    s = JanusSphere(n = [1.34, 2.0], r = [.5, .51], rotation = (-np.pi/2, 0),
                    center = (5, 5, 5))
    assert_almost_equal(s.index_at([5,5,5]),1.34)
    holo = calc_holo(schema, s, illum_wavelen=.66, medium_index=1.33, illum_polarization=(1, 0))
    verify(holo, 'janus_dda')

def test_csg_dda():
    s = Sphere(n = 1.6, r=.1, center=(5, 5, 5))
    st = s.translated(.03, 0, 0)
    pacman = Difference(s, st)
    sch = detector_grid(10, .1)
    h = calc_holo(sch, pacman, 1.33, .66, illum_polarization=(0, 1))
    verify(h, 'dda_csg')

    hr = calc_holo(sch, pacman.rotated(np.pi/2, 0, 0))
    rotated_pac = pacman.rotated(np.pi/2, 0, 0)
    verify(h/hr, 'dda_csg_rotated_div')
