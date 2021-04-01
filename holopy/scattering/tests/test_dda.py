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
import unittest

from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
import numpy as np
from nose.tools import with_setup
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest
from subprocess import CalledProcessError
import os

from holopy.core.errors import DependencyMissing
from holopy.scattering.scatterer import (Sphere, Ellipsoid, Scatterer,
                                         Spheroid, Capsule, Cylinder, Bisphere,
                                         JanusSphere_Uniform, Difference)
from holopy.scattering import Mie, DDA, calc_holo as calc_holo_external
from holopy.core import detector_grid, update_metadata
from holopy.core.tests.common import verify, assert_obj_close

# nose setup/teardown methods
def setup_optics():
    # set up optics class for use in several test functions
    global schema, wavelen, index
    wavelen = 658e-3
    polarization = [0., 1.0]
    divergence = 0
    pixel_scale = [.1151, .1151]
    index = 1.33
    schema = detector_grid(12, spacing = pixel_scale)
    schema = update_metadata(schema, index, wavelen, polarization)

def teardown_optics():
    global schema
    del schema


class TestDDA(unittest.TestCase):
    @attr('fast')
    def test_can_handle_class_method(self):
        self.assertTrue(DDA.can_handle(Sphere()))

    @attr('fast')
    def test_cannot_handle_things_that_are_not_scatterers(self):
        not_a_scatterer = 'not_a_scatterer'
        self.assertFalse(DDA.can_handle(not_a_scatterer))

    @attr('fast', 'dda')
    def test_theory_from_parameters(self):
        np.random.seed(1332)
        kwargs = {
            'n_cpu': np.random.randint(8),
            'use_gpu': np.random.choice([True, False]),
            'gpu_id': None,
            'max_dpl_size': None,
            'use_indicators': np.random.choice([True, False]),
            'keep_raw_calculations': np.random.choice([True, False]),
            'addacmd': [],
            'suppress_C_output': np.random.choice([True, False]),
            }
        try:
            theory_in = DDA(**kwargs)
        except DependencyMissing:
            raise SkipTest()
        pars = {}
        theory_out = theory_in.from_parameters(pars)

        for k, v in kwargs.items():
            self.assertEqual(getattr(theory_out, k), v)


def calc_holo(schema, scatterer, medium_index=None, illum_wavelen=None,
              **kwargs):
    try:
        return calc_holo_external(
                    schema, scatterer, medium_index, illum_wavelen, **kwargs)
    except DependencyMissing:
        raise SkipTest()


@attr('medium', "dda")
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_sphere():
    sc = Sphere(n=1.59, r=3e-1, center=(0, 0, 0))
    sc = sc.translated(1, -1, 30)
    mie_holo = calc_holo(schema, sc, index, wavelen)
    dda_holo = calc_holo(schema, sc, index, wavelen, theory=DDA)
    assert_allclose(mie_holo, dda_holo, rtol=.0015)


@with_setup(setup=setup_optics, teardown=teardown_optics)
@attr('slow', 'dda')
def test_dda_2_cpu():
    if os.name == 'nt': # windows
        raise SkipTest()
    sc = Sphere(n=1.59, r=3e-1, center=(1, -1, 30))
    mie_holo = calc_holo(schema, sc, index, wavelen)
    try:
        dda_n2 = DDA(n_cpu=2)
    except DependencyMissing:
        raise SkipTest()
    try:
        dda_holo = calc_holo(schema, sc, index, wavelen, theory=dda_n2)
    except CalledProcessError:
        # DDA only compiled for serial calculations
        raise SkipTest
    # TODO: figure out how to actually test that it runs on multiple cpus

def in_sphere(r):
    rsq = r**2
    def test(point):
        return (point**2).sum() < rsq
    return test


@attr('medium', 'dda')
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


@attr('fast', 'dda')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_voxelated_complex():
    s = Sphere(n = 1.2+2j, r = .2, center = (5,5,5))
    sv = Scatterer(s.indicators, s.n, s.center)
    schema = detector_grid(10, .1)
    holo_dda = calc_holo(schema, sv, illum_wavelen=.66, medium_index=1.33,
                         illum_polarization = (1, 0), theory=DDA)
    verify(holo_dda, 'dda_voxelated_complex', rtol=1e-5)


@attr('medium', 'dda')
@with_setup(setup=setup_optics, teardown=teardown_optics)
def test_DDA_coated():
    cs = Sphere(
        center=[7.141442573813124, 7.160766866147957, 11.095409800342143],
        n=[(1.27121212428+0j), (1.49+0j)], r=[.1-0.0055, 0.1])
    lmie_holo = calc_holo(schema, cs, index, wavelen, theory=Mie)
    dda_holo = calc_holo(schema, cs, index, wavelen, theory=DDA)
    assert_allclose(lmie_holo, dda_holo, rtol = 5e-4)


@with_setup(setup=setup_optics, teardown=teardown_optics)
@attr('medium', 'dda')
def test_Ellipsoid_dda():
    e = Ellipsoid(1.5, r = (.5, .1, .1), center = (1, -1, 10))
    schema = detector_grid(10, .1)
    try:
        h = calc_holo(schema, e, illum_wavelen=.66, medium_index=1.33,
            illum_polarization = (1,0), theory=DDA(use_indicators=False))
        cmd = DDA()._adda_predefined(
            e, medium_wavelen=.66, medium_index=1.33, temp_dir='temp_dir')
        cmdlist = ['-eq_rad', '0.5', '-shape', 'ellipsoid', '0.2', '0.2', '-m',
               '1.1278195488721805', '0.0', '-orient', '0.0', '0.0', '0.0']
        assert_equal(cmd, cmdlist)
        verify(h, 'ellipsoid_dda', rtol=3e-4, atol=3e-4)
    except DependencyMissing:
        raise SkipTest()


@attr('slow', 'dda')
def test_predefined_scatterers():
    # note this tests only that the code runs, not that it is correct
    try:
        scatterers = [Ellipsoid(n=1.5, r=(0.5, 1, 2), center=(0,0,1)),
                      Spheroid(n=1.5, r=(0.5, 1), center=(0,0,1)),
                      Capsule(n=1.5, h=1, d=0.5, center=(0,0,1)),
                      Cylinder(n=1.5, h=1, d=0.5, center=(0,0,1)),
                      Bisphere(n=1.5, h=1, d=0.5, center=(0,0,1)),
                      Sphere(n=1.5, r=1, center=(0,0,1))]
        detector = detector_grid(5, .1)
        for s in scatterers:
            calc_holo(detector, s, illum_wavelen=.66, medium_index=1.33,
                illum_polarization = (1,0), theory=DDA(use_indicators=False))
    except DependencyMissing:
        raise SkipTest


@attr('dda')
def test_janus():
    schema = detector_grid(10, .1)
    s = JanusSphere_Uniform(n = [1.34, 2.0], r = [.5, .51],
                            rotation = (0, -np.pi/2, 0), center = (5, 5, 5))
    assert_almost_equal(s.index_at([5,5,5]),1.34)
    holo = calc_holo(schema, s, illum_wavelen=.66, medium_index=1.33,
                     illum_polarization=(1, 0))
    verify(holo, 'janus_dda')


@attr('dda')
def test_csg_dda():
    s = Sphere(n = 1.6, r=.1, center=(5, 5, 5))
    st = s.translated(.03, 0, 0)
    pacman = Difference(s, st)
    sch = detector_grid(10, .1)
    h = calc_holo(sch, pacman, 1.33, .66, illum_polarization=(0, 1))
    verify(h, 'dda_csg')

    rotated_pac = pacman.rotated(np.pi/2, 0, 0)
    hr = calc_holo(sch, rotated_pac, 1.33, .66, illum_polarization=(0, 1))
    verify(h/hr, 'dda_csg_rotated_div', rtol=1e-3, atol=1e-3)
